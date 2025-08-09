from transformers import VoxtralForConditionalGeneration, AutoProcessor
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
repo_id = "mistralai/Voxtral-Mini-3B-2507"

processor = AutoProcessor.from_pretrained(repo_id)
model = VoxtralForConditionalGeneration.from_pretrained(repo_id, torch_dtype=torch.bfloat16, device_map=device, attn_implementation="flash_attention_2")

conversation = [
    {
        "role": "user",
        "content": [
            {
                "type": "audio",
                "url": "https://huggingface.co/datasets/eustlb/audio-samples/resolve/main/dude_where_is_my_car.wav",
            },
            {"type": "text", "text": "What can you tell me about this audio?"},
        ],
    }
]

inputs = processor.apply_chat_template(conversation)
inputs = inputs.to(device, dtype=torch.bfloat16)

outputs = model.generate(**inputs, max_new_tokens=500)
decoded_outputs = processor.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)

print("\nGenerated response:")
print("=" * 80)
print(decoded_outputs[0])
print("=" * 80)