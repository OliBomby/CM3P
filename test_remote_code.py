import torch
from transformers import AutoProcessor, AutoModelForSequenceClassification

device = "cuda" if torch.cuda.is_available() else "cpu"
save_path = r"OliBomby/CM3P-ranked-classifier"

processor = AutoProcessor.from_pretrained(save_path, trust_remote_code=True, revision="main")
model = AutoModelForSequenceClassification.from_pretrained(save_path, torch_dtype=torch.bfloat16, device_map=device, attn_implementation="sdpa", trust_remote_code=True, revision="main")

audio = r"resources/audio3.mp3"
# beatmap = input("Path to beatmap file: ")
beatmap = r"C:\Users\Olivier\AppData\Local\osu!\Songs\beatmap-638965817922162241-ANTICLOCK TEA-PARTY\Mitsuki Nakae - ANTICLOCK TEA-PARTY (Mapperatorinator) [Mapperatorinator V30].osu"

inputs = processor(beatmap=beatmap, audio=audio)
inputs = inputs.to(device, dtype=torch.bfloat16)

with torch.no_grad():
    logits = model(**inputs).logits
    probs = logits.softmax(dim=-1).cpu()

ranked_threshold = 0.4
predicted_ranked_states = probs[:, 1] >= ranked_threshold

if predicted_ranked_states.all():
    print("Congratulations! Your beatmap has been approved to the ranked section.")
else:
    print("Unfortunately, your beatmap has some quality issues and could not be approved to the ranked section. Please focus on improving these sections:")

    for i, predicted_ranked_state in enumerate(predicted_ranked_states):
        if not predicted_ranked_state:
            start_time = i * processor.default_kwargs["beatmap_kwargs"]["window_stride_sec"]
            end_time = start_time + processor.default_kwargs["beatmap_kwargs"]["window_length_sec"]
            print(f"- From {start_time:.0f}s to {end_time:.0f}s")
