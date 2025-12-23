import pytest
import torch
from pathlib import Path

from cm3p import CM3PModel
from cm3p.processing_cm3p import CM3PProcessor
from cm3p.tokenization_cm3p import CM3PMetadata

device = "cuda" if torch.cuda.is_available() else "cpu"
attn_implementation = "flash_attention_2" if torch.cuda.is_available() else "sdpa"
torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

HF_CHECKPOINT = "OliBomby/CM3P"
RESOURCES_DIR = (Path(__file__).parent.parent / "resources").resolve()
AUDIO_PATH = str((RESOURCES_DIR / "audio.mp3").resolve())
BEATMAP_PATH = str((RESOURCES_DIR / "Denkishiki Karen Ongaku Shuudan - Aoki Kotou no Anguis (OliBomby) [Ardens Spes].osu").resolve())

base_metadata = CM3PMetadata(
    difficulty=6.0,
    mode="osu",
    mapper="OliBomby",
    year=2020,
    tags=["rhythm/simple"],
    status="ranked",
    cs=4.0,
    hitsounded=True,
    global_sv=1.4,
    song_length=260,
)


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
        pytest.skip(f"Skipping zero shot classification test due to HF download error: {e}")
    return proc, mdl


def make_metadata(**m_dict):
    base = base_metadata.copy()
    for k, v in m_dict.items():
        # noinspection PyTypedDict
        base[k] = v
    return base


@pytest.mark.parametrize(
    "field, values",
    [
        ("difficulty", [1.0, 6.0, 12.0]),
        ("mode", ["osu", "taiko"]),
        ("mapper", ["OliBomby", "peppy"]),
        ("year", [2010, 2020, 2023]),
        ("status", ["ranked", "loved"]),
    ],
)
def test_attribute_classification(field, values, processor_and_model):
    processor, model = processor_and_model
    labels = [make_metadata(**{field: v}) for v in values]
    inputs = processor(metadata=labels, beatmap=BEATMAP_PATH, audio=AUDIO_PATH, min_window_length_sec=8)
    inputs = inputs.to(device, dtype=torch_dtype)

    with torch.no_grad():
        outputs = model(**inputs, return_loss=False)

    assert hasattr(outputs, "logits_per_beatmap"), "Expected logits_per_beatmap in outputs"
    logits = outputs.logits_per_beatmap
    assert logits.shape[1] == len(labels), "Feature size mismatch"
    probs = logits.softmax(dim=1)

    # Print the predicted classes for inspection
    predicted_indices = probs.argmax(dim=1)
    predicted_values = [values[idx] for idx in predicted_indices.cpu().tolist()]
    print(f"Predicted {field} values: {predicted_values}")

    # Assert that the majority of predictions match the expected value
    expected_value = base_metadata[field]
    match_count = sum(1 for v in predicted_values if v == expected_value)
    assert match_count >= len(predicted_values) // 2, f"Majority of predictions should match expected {field}={expected_value}"
