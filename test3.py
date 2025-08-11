import torch
from transformers import AutoProcessor, WhisperFeatureExtractor
from transformers.tokenization_mistral_common import MistralCommonTokenizer

from cm3p import CM3PConfig
from cm3p.modeling_cm3p import CM3PModel
from cm3p.processing_cm3p import CM3PProcessor

device = "cuda" if torch.cuda.is_available() else "cpu"

config = CM3PConfig()
model = CM3PModel._from_config(config, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
processor = CM3PProcessor(WhisperFeatureExtractor(), MistralCommonTokenizer(""))

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

url = "beatmap_audio_example.mp3"
beatmap = "beatmap_example.osu"
labels = [
    {"difficulty": 1.5, "mode": "osu", "mapper": "OliBomby", "year": 2020},
    {"difficulty": 3.0, "mode": "taiko", "mapper": "Cookiezi", "year": 2018},
    {"difficulty": 5.0, "mode": "fruits", "mapper": "peppy", "year": 2021},
    {"difficulty": 7.0, "mode": "mania", "mapper": "Xenon", "year": 2019},
]

inputs = processor(metadata=labels, beatmap=beatmap, audio=url, return_tensors="pt", padding=True)

outputs = model(**inputs)
logits_per_beatmap = outputs.logits_per_beatmap
probs = logits_per_beatmap.softmax(dim=1)
most_likely_idx = probs.argmax(dim=1).item()
most_likely_label = labels[most_likely_idx]
print(f"Most likely label: {most_likely_label} with probability: {probs[0][most_likely_idx].item():.3f}")
