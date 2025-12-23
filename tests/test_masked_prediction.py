import pytest
import torch
from pathlib import Path

from cm3p import CM3PModel
from cm3p.processing_cm3p import CM3PProcessor

device = "cuda" if torch.cuda.is_available() else "cpu"
attn_implementation = "flash_attention_2" if torch.cuda.is_available() else "sdpa"
torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

HF_CHECKPOINT = "OliBomby/CM3P"
RESOURCES_DIR = (Path(__file__).parent.parent / "resources").resolve()
AUDIO_PATH = str((RESOURCES_DIR / "audio.mp3").resolve())
BEATMAP_PATH = str((RESOURCES_DIR / "Denkishiki Karen Ongaku Shuudan - Aoki Kotou no Anguis (OliBomby) [Ardens Spes].osu").resolve())


@pytest.fixture(scope="module")
def processor_and_model():
    try:
        proc = CM3PProcessor.from_pretrained(HF_CHECKPOINT)
        mdl = CM3PModel.from_pretrained(
            HF_CHECKPOINT,
            dtype=torch_dtype,
            device_map=device,
            attn_implementation=attn_implementation,
        )
    except Exception as e:
        pytest.skip(f"Skipping masked prediction test due to HF download error: {e}")
    return proc, mdl


def test_masked_lm_forward(processor_and_model):
    processor, model = processor_and_model
    inputs = processor(beatmap=BEATMAP_PATH, audio=AUDIO_PATH, return_tensors="pt")

    # Build labels (mask 15% of non-special tokens)
    to_predict_mask = torch.ones_like(inputs.input_ids, dtype=torch.bool)
    for sid in processor.beatmap_tokenizer.all_special_ids:
        to_predict_mask &= inputs.input_ids != sid
    to_predict_mask &= torch.rand(inputs.input_ids.shape) < 0.15
    labels = inputs.input_ids.masked_fill(~to_predict_mask, -100)
    inputs["labels"] = labels

    rand = torch.rand(inputs.input_ids.shape)
    masking_mask = (rand < 0.8) & to_predict_mask
    random_replacement_mask = (rand >= 0.8) & (rand < 0.9) & to_predict_mask

    inputs.input_ids.masked_fill_(masking_mask, processor.beatmap_tokenizer.mask_token_id)
    if random_replacement_mask.any():
        random_token_ids = torch.randint(0, processor.beatmap_tokenizer.vocab_size, (random_replacement_mask.sum(),))
        inputs.input_ids[random_replacement_mask] = random_token_ids

    inputs = inputs.to(device, dtype=torch_dtype)

    with torch.no_grad():
        outputs = model(**inputs, output_logits=True)

    assert hasattr(outputs, "logits"), "Outputs should contain logits"
    assert outputs.logits.shape[-1] == processor.beatmap_tokenizer.vocab_size
    # If remote model provides loss ensure it's non-negative
    if hasattr(outputs, "loss") and outputs.loss is not None:
        assert outputs.loss.item() >= 0.0

    # Print some predictions for masked positions
    masked_positions = to_predict_mask.nonzero(as_tuple=True)
    predicted_token_ids = outputs.logits.argmax(dim=-1)
    for i in range(min(5, masked_positions[0].shape[0])):
        batch_idx = masked_positions[0][i].item()
        seq_idx = masked_positions[1][i].item()
        true_id = inputs.labels[batch_idx, seq_idx].item()
        pred_id = predicted_token_ids[batch_idx, seq_idx].item()
        # Decode the tokens for better readability
        true_token = processor.beatmap_tokenizer.decode([true_id])
        pred_token = processor.beatmap_tokenizer.decode([pred_id])
        print(f"Batch {batch_idx}, Seq Pos {seq_idx}: True Token = '{true_token}', Predicted Token = '{pred_token}'")
