import torch
from transformers import WhisperFeatureExtractor

from cm3p import CM3PBeatmapConfig
from cm3p.modeling_cm3p import CM3PForMaskedLM
from cm3p.parsing_cm3p import CM3PBeatmapParser
from cm3p.processing_cm3p import CM3PProcessor
from cm3p.tokenization_cm3p import CM3PBeatmapTokenizer, CM3PMetadataTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"
save_path = r"saved_logs/train_v6_mask/trainer_output/checkpoint-30000"

processor = CM3PProcessor.from_pretrained(save_path)
model = CM3PForMaskedLM.from_pretrained(save_path, torch_dtype=torch.bfloat16, device_map=device, attn_implementation="flash_attention_2")

audio = r"resources/audio.mp3"
beatmap = r"resources/Denkishiki Karen Ongaku Shuudan - Aoki Kotou no Anguis (OliBomby) [Ardens Spes].osu"

inputs = processor(beatmap=beatmap, audio=audio, return_tensors="pt")

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
    most_likely_labels = processor.beatmap_tokenizer.batch_decode(most_likely_idx, skip_special_tokens=False)
    for predicted_label, real_label in zip(most_likely_labels, processor.beatmap_tokenizer.batch_decode(labels, skip_special_tokens=False)):
        print([(p, r) for p, r in zip(predicted_label.split(' '), real_label.split(' ')) if r != '[UNK]' and p != r])

