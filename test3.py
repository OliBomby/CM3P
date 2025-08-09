import torch

from cm3p import CM3PConfig
from cm3p.modeling_cm3p import CM3PModel

device = "cuda" if torch.cuda.is_available() else "cpu"

config = CM3PConfig()
model = CM3PModel._from_config(config, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")

# print(model)
# print(model.config)
# print parameter count
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
# Check if all parameter weights are initialized
# Print the mean and standard deviation of the weights
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"Parameter: {name}, Mean: {param.data.mean().item():.4f}, Std: {param.data.std().item():.4f}")
        if torch.isnan(param.data).any():
            print(f"Parameter {name} has NaN values.")
        if torch.isinf(param.data).any():
            print(f"Parameter {name} has Inf values.")
