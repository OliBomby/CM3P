import pytest
import torch
from pathlib import Path
from transformers import WhisperFeatureExtractor

from cm3p import CM3PBeatmapConfig
from cm3p.modeling_cm3p import CM3PForMaskedLM
from cm3p.parsing_cm3p import CM3PBeatmapParser
from cm3p.processing_cm3p import CM3PProcessor
from cm3p.tokenization_cm3p import CM3PBeatmapTokenizer, CM3PMetadataTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"
attn_implementation = "flash_attention_2" if torch.cuda.is_available() else "sdpa"
torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

RESOURCES_DIR = (Path(__file__).parent.parent / "resources").resolve()
AUDIO_PATH = str((RESOURCES_DIR / "audio.mp3").resolve())
BEATMAP_PATH = str((RESOURCES_DIR / "Denkishiki Karen Ongaku Shuudan - Aoki Kotou no Anguis (OliBomby) [Ardens Spes].osu").resolve())


@pytest.fixture(scope="module")
def processor():
    return CM3PProcessor(
        WhisperFeatureExtractor(),
        CM3PBeatmapParser(),
        CM3PBeatmapTokenizer(),
        CM3PMetadataTokenizer(),
    )


@pytest.fixture(scope="module")
def model(processor):
    config = CM3PBeatmapConfig()
    config.vocab_size = processor.beatmap_tokenizer.vocab_size
    config.audio_token_id = processor.beatmap_tokenizer.convert_tokens_to_ids(processor.beatmap_tokenizer.audio_token)
    config.audio_sos_token_id = processor.beatmap_tokenizer.convert_tokens_to_ids(processor.beatmap_tokenizer.audio_eos_token)
    config.audio_eos_token_id = processor.beatmap_tokenizer.convert_tokens_to_ids(processor.beatmap_tokenizer.audio_eos_token)
    m = CM3PForMaskedLM._from_config(config, torch_dtype=torch_dtype, attn_implementation=attn_implementation)
    return m.to(device)


def test_model_initialization_and_forward(processor, model):
    inputs = processor(beatmap=BEATMAP_PATH, audio=AUDIO_PATH, return_tensors="pt", multiply_metadata=True)

    # Build labels mask (simplified): mask 10% of non-special tokens
    to_predict_mask = torch.ones_like(inputs.input_ids, dtype=torch.bool)
    for sid in processor.beatmap_tokenizer.all_special_ids:
        to_predict_mask &= inputs.input_ids != sid
    to_predict_mask &= torch.rand(inputs.input_ids.shape) < 0.1
    labels = inputs.input_ids.masked_fill(~to_predict_mask, -100)
    inputs["labels"] = labels

    # Standard MLM masking behavior
    rand = torch.rand(inputs.input_ids.shape)
    masking_mask = (rand < 0.8) & to_predict_mask
    inputs.input_ids.masked_fill_(masking_mask, processor.beatmap_tokenizer.mask_token_id)

    inputs = inputs.to(device, dtype=torch_dtype)
    with torch.no_grad():
        outputs = model(**inputs)

    assert hasattr(outputs, "loss") and hasattr(outputs, "logits"), "Masked LM outputs must include loss and logits"
    assert outputs.logits.shape[-1] == processor.beatmap_tokenizer.vocab_size
    assert outputs.loss.item() >= 0.0
