import pytest
import torch
from pathlib import Path

from cm3p import CM3PForBeatmapClassification
from cm3p.processing_cm3p import CM3PProcessor

device = "cuda" if torch.cuda.is_available() else "cpu"
attn_implementation = "flash_attention_2" if torch.cuda.is_available() else "sdpa"
torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

HF_CHECKPOINT = "OliBomby/CM3P-ranked-classifier"
RESOURCES_DIR = (Path(__file__).parent.parent / "resources").resolve()
AUDIO_PATH = str((RESOURCES_DIR / "audio.mp3").resolve())
BEATMAP_PATH = str((RESOURCES_DIR / "Denkishiki Karen Ongaku Shuudan - Aoki Kotou no Anguis (OliBomby) [Ardens Spes].osu").resolve())


@pytest.mark.parametrize("audio_path, beatmap_path, remote", [
    (AUDIO_PATH, BEATMAP_PATH, False),
    (AUDIO_PATH, BEATMAP_PATH, True)
])
def test_ranked_classifier_runs(audio_path, beatmap_path, remote):
    try:
        if remote:
            processor = CM3PProcessor.from_pretrained(
                HF_CHECKPOINT,
                trust_remote_code=True,
                revision="main",
            )
            model = CM3PForBeatmapClassification.from_pretrained(
                HF_CHECKPOINT,
                torch_dtype=torch_dtype,
                device_map=device,
                attn_implementation=attn_implementation,
                trust_remote_code=True,
                revision="main",
            )
        else:
            processor = CM3PProcessor.from_pretrained(HF_CHECKPOINT)
            model = CM3PForBeatmapClassification.from_pretrained(
                HF_CHECKPOINT,
                torch_dtype=torch_dtype,
                device_map=device,
                attn_implementation=attn_implementation,
            )
    except Exception as e:  # Network / HF hub issues
        pytest.skip(f"Skipping ranked classifier test due to download error: {e}")

    inputs = processor(beatmap=beatmap_path, audio=audio_path, min_window_length_sec=8)
    inputs = inputs.to(device, dtype=torch_dtype)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Assertions
    assert logits.ndim == 2, "Logits should be [batch, num_classes]"
    assert logits.shape[-1] == 2, "Classifier should output 2 logits (binary ranked vs not)"
    probs = logits.softmax(dim=-1)

    ranked_threshold = 0.5
    predicted_ranked_states = probs[:, 1] >= ranked_threshold
    assert predicted_ranked_states.dtype == torch.bool, "Predicted ranked states should be a boolean tensor"
    assert predicted_ranked_states.numel() == logits.shape[0], "Predicted ranked states should match batch size"
    assert predicted_ranked_states.all(), "Expected the provided beatmap to be classified as ranked"
