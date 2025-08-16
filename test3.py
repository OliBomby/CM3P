import torch
from transformers import WhisperFeatureExtractor

from cm3p import CM3PConfig
from cm3p.modeling_cm3p import CM3PModel
from cm3p.parsing_cm3p import CM3PBeatmapParser
from cm3p.processing_cm3p import CM3PProcessor
from cm3p.tokenization_cm3p import CM3PBeatmapTokenizer, CM3PMetadataTokenizer, CM3PMetadata

device = "cuda" if torch.cuda.is_available() else "cpu"

test_beatmap_tokenizer_config = {
    "event_types": [
        "hitcircle",
        "slider",
        "spinner",
    ],
}

test_metadata_tokenizer_config = {
    "modes": [
        "osu",
        "taiko",
        "fruits",
        "mania",
    ],
    "mappers": [
        "OliBomby",
        "Cookiezi",
        "peppy",
        "Xenon",
    ],
}

config = CM3PConfig()
model = CM3PModel._from_config(config, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
model = model.to(device)
processor = CM3PProcessor(
    WhisperFeatureExtractor(),
    CM3PBeatmapParser(),
    CM3PBeatmapTokenizer(vocab_init=test_beatmap_tokenizer_config),
    CM3PMetadataTokenizer(vocab_init=test_metadata_tokenizer_config),
)

# print(model)
# print(model.config)
# print parameter count
# def print_parameters(m):
#     print(f"Model: {m.__class__.__name__}")
#     total_params = sum(p.numel() for p in m.parameters())
#     trainable_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
#     print(f"Total parameters: {total_params:,}")
#     print(f"Trainable parameters: {trainable_params:,}")
#
# print_parameters(model)
# print_parameters(model.beatmap_model)
# print_parameters(model.beatmap_model.audio_encoder)
# print_parameters(model.metadata_model)

audio = r"resources/audio.mp3"
beatmap = r"resources/Denkishiki Karen Ongaku Shuudan - Aoki Kotou no Anguis (OliBomby) [Ardens Spes].osu"
labels = [
    CM3PMetadata(difficulty=1.5, mode="osu", mapper="OliBomby", year=2020),
    CM3PMetadata(difficulty=3.0, mode="taiko", mapper="Cookiezi", year=2018),
    CM3PMetadata(difficulty=5.0, mode="fruits", mapper="peppy", year=2021),
    CM3PMetadata(difficulty=7.0, mode="mania", mapper="Xenon", year=2019),
]

inputs = processor(metadata=labels, beatmap=beatmap, audio=audio, return_tensors="pt")
inputs = inputs.to(device, dtype=torch.bfloat16)

with torch.no_grad():
    outputs = model(**inputs)
    logits_per_beatmap = outputs.logits_per_beatmap
    probs = logits_per_beatmap.softmax(dim=1).cpu()

for prob in probs:
    most_likely_idx = prob.argmax().item()
    most_likely_label = labels[most_likely_idx]
    print(f"Most likely label: {most_likely_label} with probability: {prob[most_likely_idx].item():.3f}")
