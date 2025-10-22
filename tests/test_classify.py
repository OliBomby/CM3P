import torch

from cm3p import CM3PForBeatmapClassification
from cm3p.processing_cm3p import CM3PProcessor

device = "cuda" if torch.cuda.is_available() else "cpu"
# save_path = r"../saved_logs/train_v7_classifier/trainer_output/checkpoint-10000"
# ranked_threshold = 0.8
save_path = r"../saved_logs/train_v7_classifier2/trainer_output/checkpoint-10000"
ranked_threshold = 0.4

processor = CM3PProcessor.from_pretrained(save_path)
model = CM3PForBeatmapClassification.from_pretrained(save_path, torch_dtype=torch.bfloat16, device_map=device, attn_implementation="sdpa")

# audio = r"..\resources/audio.mp3"
# beatmap = r"..\resources/Denkishiki Karen Ongaku Shuudan - Aoki Kotou no Anguis (OliBomby) [Ardens Spes].osu"
# audio = r"..\resources/audio2.mp3"
# beatmap = r"..\resources/POLKADOT STINGRAY - Otoshimae (moph) [Mindmaster's Extra].osu"
audio = r"..\resources/audio3.mp3"
# beatmap = r"..\resources/Mitsuki Nakae - ANTICLOCK TEA-PARTY.osu"
# beatmap = r"C:\Users\Olivier\AppData\Local\osu!\Songs\beatmap-638965817922162241-ANTICLOCK TEA-PARTY\Mitsuki Nakae - ANTICLOCK TEA-PARTY (Mapperatorinator) [Mapperatorinator V30].osu"
# beatmap = r"C:\Users\Olivier\AppData\Local\osu!\Songs\beatmap-638965817922162241-ANTICLOCK TEA-PARTY\beatmapa6fab5441dab43569157deee509de435.osu"
# beatmap = r"C:\Users\Olivier\AppData\Local\osu!\Songs\beatmap-638965817922162241-ANTICLOCK TEA-PARTY\beatmap29f409a21875483ead1799ef778328e4.osu"
beatmap = r"C:\Users\Olivier\AppData\Local\osu!\Songs\beatmap-638965817922162241-ANTICLOCK TEA-PARTY\beatmap17ea5e3d365948f9b4996b6900ccffdd.osu"
# beatmap = r"C:\Users\Olivier\AppData\Local\osu!\Songs\1848363 garnet feat F9 - hauynite\garnet feat. F9 - hauynite (Shurelia) [rumination].osu"


inputs = processor(beatmap=beatmap, audio=audio)
inputs = inputs.to(device, dtype=torch.bfloat16)

with torch.no_grad():
    logits = model(**inputs).logits
    probs = logits.softmax(dim=-1).cpu()

predicted_ranked_states = probs[:, 1] >= ranked_threshold

if predicted_ranked_states.all():
    print("Congratulations! Your beatmap has been approved to the ranked section. Your code is: RANKERATORINATOR3000")
else:
    print("Unfortunately, your beatmap has some quality issues and could not be approved to the ranked section. Please focus on improving these sections:")

    for i, predicted_ranked_state in enumerate(predicted_ranked_states):
        if not predicted_ranked_state:
            start_time = i * processor.default_kwargs["beatmap_kwargs"]["window_stride_sec"]
            end_time = start_time + processor.default_kwargs["beatmap_kwargs"]["window_length_sec"]
            print(f"- From {start_time:.0f}s to {end_time:.0f}s")
