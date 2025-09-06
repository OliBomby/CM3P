import numpy as np
import torch

from cm3p.modeling_cm3p import CM3PModel
from cm3p.processing_cm3p import CM3PProcessor
from cm3p.tokenization_cm3p import CM3PMetadata

device = "cuda" if torch.cuda.is_available() else "cpu"
save_path = "saved_logs/train_v1/trainer_output/checkpoint-30000"

processor = CM3PProcessor.from_pretrained(save_path)
model = CM3PModel.from_pretrained(save_path, torch_dtype=torch.bfloat16, device_map=device, attn_implementation="flash_attention_2")

# audio = r"resources/audio.mp3"
# beatmap = r"resources/Denkishiki Karen Ongaku Shuudan - Aoki Kotou no Anguis (OliBomby) [Ardens Spes].osu"
audio = r"resources/audio2.mp3"
beatmap = r"resources/POLKADOT STINGRAY - Otoshimae (moph) [Mindmaster's Extra].osu"


def metadata(**m_dict):
    # default_metadata = CM3PMetadata(
    #     difficulty=6.0,
    #     mode="osu",
    #     mapper="OliBomby",
    #     year=2020,
    #     tags=["rhythm/simple", "scene/mapping contest", "style/clean", "style/geometric", "tap/bursts", "tap/stamina"],
    #     status="ranked",
    #     cs=4.0,
    #     hitsounded=True,
    #     global_sv=1.4,
    #     song_length=260,
    # )
    default_metadata = CM3PMetadata(
        difficulty=5.43,
        mode="osu",
        mapper="mindmaster107",
        year=2020,
        status="ranked",
        cs=3.0,
        hitsounded=True,
        global_sv=2.0,
        song_length=191,
    )
    for k, v in m_dict.items():
        # noinspection PyTypedDict
        default_metadata[k] = v
    return default_metadata


def classify_labels(name, labels):
    inputs = processor(metadata=labels, beatmap=beatmap, audio=audio)
    inputs = inputs.to(device, dtype=torch.bfloat16)

    with torch.no_grad():
        outputs = model(**inputs, return_loss=False)
        logits_per_beatmap = outputs.logits_per_beatmap
        probs = logits_per_beatmap.softmax(dim=1).cpu()

    for prob in probs:
        most_likely_idx = prob.argmax().item()
        most_likely_label = labels[most_likely_idx][name]
        print(f"Most likely label: {name}={most_likely_label} with probability: {prob[most_likely_idx].item():.3f}")

classify_labels("difficulty", [metadata(difficulty=float(d)) for d in np.linspace(1.0, 14.0, num=14)])
classify_labels("mode", [metadata(mode=m) for m in ["osu", "taiko", "fruits", "mania"]])
classify_labels("mapper", [metadata(mapper=m) for m in processor.metadata_tokenizer.mapper_ids_to_names.values()])
classify_labels("year", [metadata(year=y) for y in range(2007, 2024)])
classify_labels("tags", [metadata(tags=[t]) for t in processor.metadata_tokenizer.tag_ids_to_names.values()])
classify_labels("status", [metadata(status=s) for s in ["ranked", "approved", "qualified", "loved", "pending", "graveyard"]])
