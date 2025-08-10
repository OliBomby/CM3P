import torch

from cm3p import CM3PConfig
from cm3p.modeling_cm3p import CM3PModel

device = "cuda" if torch.cuda.is_available() else "cpu"

config = CM3PConfig()
model = CM3PModel._from_config(config, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")

# print(model)
# print(model.config)
# print parameter count
def print_parameters(m):
    print(f"Model: {m.__class__.__name__}")
    total_params = sum(p.numel() for p in m.parameters())
    trainable_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

print_parameters(model)
print_parameters(model.beatmap_model)
print_parameters(model.beatmap_model.audio_encoder)
print_parameters(model.metadata_model)
