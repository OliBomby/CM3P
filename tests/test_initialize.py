import torch
from transformers import WhisperFeatureExtractor

from cm3p import CM3PBeatmapConfig
from cm3p.modeling_cm3p import CM3PForMaskedLM
from cm3p.parsing_cm3p import CM3PBeatmapParser
from cm3p.processing_cm3p import CM3PProcessor
from cm3p.tokenization_cm3p import CM3PBeatmapTokenizer, CM3PMetadataTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"

modes = {
    0: "osu",
    1: "taiko",
    2: "fruits",
    3: "mania",
}

mappers = {
    0: "OliBomby",
    1: "Cookiezi",
    2: "peppy",
    3: "Xenon",
}

processor = CM3PProcessor(
    WhisperFeatureExtractor(),
    CM3PBeatmapParser(),
    CM3PBeatmapTokenizer(),
    CM3PMetadataTokenizer(),
)

config = CM3PBeatmapConfig()
config.vocab_size = processor.beatmap_tokenizer.vocab_size
config.audio_token_id = processor.beatmap_tokenizer.convert_tokens_to_ids(processor.beatmap_tokenizer.audio_token)
config.audio_sos_token_id = processor.beatmap_tokenizer.convert_tokens_to_ids(processor.beatmap_tokenizer.audio_eos_token)
config.audio_eos_token_id = processor.beatmap_tokenizer.convert_tokens_to_ids(processor.beatmap_tokenizer.audio_eos_token)
model = CM3PForMaskedLM._from_config(config, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
model = model.to(device)

print(model)
print(model.config)
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

audio = r"resources/audio.mp3"
beatmap = r"resources/Denkishiki Karen Ongaku Shuudan - Aoki Kotou no Anguis (OliBomby) [Ardens Spes].osu"

inputs = processor(beatmap=beatmap, audio=audio, return_tensors="pt", multiply_metadata=True)

# Make some labels
to_predict_mask = torch.ones_like(inputs.input_ids, dtype=torch.bool)
special_ids = processor.beatmap_tokenizer.all_special_ids
for sid in special_ids:
    to_predict_mask &= inputs.input_ids != sid
# Set 25% of non-special tokens to be predicted
to_predict_mask &= torch.rand(inputs.input_ids.shape) < 0.25
labels = inputs.input_ids.masked_fill(~to_predict_mask, -100)
inputs["labels"] = labels

# For each position to predict, mask the input ids with an 80% chance, replace with random token with 10% chance, or keep original with 10% chance
rand = torch.rand(inputs.input_ids.shape)
masking_mask = (rand < 0.8) & to_predict_mask
random_replacement_mask = (rand >= 0.8) & (rand < 0.9) & to_predict_mask

inputs.input_ids.masked_fill_(masking_mask, processor.beatmap_tokenizer.mask_token_id)
random_token_ids = torch.randint(0, processor.beatmap_tokenizer.vocab_size, (random_replacement_mask.sum(),))
inputs.input_ids[random_replacement_mask] = random_token_ids

inputs = inputs.to(device, dtype=torch.bfloat16)

with torch.no_grad():
    outputs = model(**inputs)
    print(outputs.loss.item())
    logits = outputs.logits
    probs = logits.softmax(dim=-1).cpu()
    most_likely_idx = probs.argmax(dim=-1)
    most_likely_labels = processor.beatmap_tokenizer.batch_decode(most_likely_idx, skip_special_tokens=True)
    for labels in most_likely_labels:
        print(labels)

