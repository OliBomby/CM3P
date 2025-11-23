import pytest
import numpy as np
import torch
from pathlib import Path
from transformers import WhisperFeatureExtractor

from cm3p import CM3PConfig
from cm3p.modeling_cm3p import CM3PModel
from cm3p.parsing_cm3p import CM3PBeatmapParser
from cm3p.processing_cm3p import CM3PProcessor
from cm3p.tokenization_cm3p import CM3PBeatmapTokenizer, CM3PMetadataTokenizer, CM3PMetadata

device = "cuda" if torch.cuda.is_available() else "cpu"
attn_implementation = "flash_attention_2" if torch.cuda.is_available() else "sdpa"
torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

RESOURCES_DIR = (Path(__file__).parent.parent / "resources").resolve()
AUDIO_PATH = str((RESOURCES_DIR / "audio.mp3").resolve())
BEATMAP_PATH = str((RESOURCES_DIR / "Denkishiki Karen Ongaku Shuudan - Aoki Kotou no Anguis (OliBomby) [Ardens Spes].osu").resolve())


@pytest.fixture(scope="module")
def processor():
    modes = {0: "osu", 1: "taiko", 2: "fruits", 3: "mania"}
    mappers = {0: "OliBomby", 1: "Cookiezi", 2: "peppy", 3: "Xenon"}
    return CM3PProcessor(
        WhisperFeatureExtractor(),
        CM3PBeatmapParser(),
        CM3PBeatmapTokenizer(),
        CM3PMetadataTokenizer(modes=modes, mappers=mappers),
    )


@pytest.fixture(scope="module")
def model(processor):
    config = CM3PConfig()
    config.beatmap_config.vocab_size = processor.beatmap_tokenizer.vocab_size
    config.beatmap_config.audio_token_id = processor.beatmap_tokenizer.convert_tokens_to_ids(processor.beatmap_tokenizer.audio_token)
    config.beatmap_config.audio_sos_token_id = processor.beatmap_tokenizer.convert_tokens_to_ids(processor.beatmap_tokenizer.audio_eos_token)
    config.beatmap_config.audio_eos_token_id = processor.beatmap_tokenizer.convert_tokens_to_ids(processor.beatmap_tokenizer.audio_eos_token)
    config.metadata_config.vocab_size = processor.metadata_tokenizer.vocab_size
    m = CM3PModel._from_config(config, torch_dtype=torch_dtype, attn_implementation=attn_implementation)
    return m.to(device)


def test_inference_forward(processor, model):
    labels = [CM3PMetadata(difficulty=1.5, mode="osu", mapper="OliBomby", year=2020)]
    inputs = processor(metadata=labels, beatmap=BEATMAP_PATH, return_tensors="pt", multiply_metadata=True)
    inputs = inputs.to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    assert hasattr(outputs, "logits_per_beatmap"), "Model outputs should contain logits_per_beatmap"
    logits = outputs.logits_per_beatmap
    assert logits.ndim == 2 and logits.shape[0] == logits.shape[1], "Logits should be [num_labels, num_labels] or similar"
