import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoProcessor

from cm3p import CM3PBeatmapConfig, CM3PForBeatmapClassification
from cm3p.processing_cm3p import CM3PProcessor

CM3PBeatmapConfig.register_for_auto_class()
CM3PForBeatmapClassification.register_for_auto_class("AutoModelForSequenceClassification")
CM3PProcessor.register_for_auto_class()

AutoConfig.register("CM3PBeatmap", CM3PBeatmapConfig)
AutoModelForSequenceClassification.register(CM3PBeatmapConfig, CM3PForBeatmapClassification)
AutoProcessor.register(CM3PBeatmapConfig, CM3PProcessor)

save_path = r"saved_logs/train_v7_classifier2/trainer_output/checkpoint-10000"
processor = CM3PProcessor.from_pretrained(save_path)
model = CM3PForBeatmapClassification.from_pretrained(save_path, torch_dtype=torch.bfloat16, attn_implementation="sdpa")

model.push_to_hub("CM3P-ranked-classifier", private=True)
processor.push_to_hub("CM3P-ranked-classifier", private=True)
