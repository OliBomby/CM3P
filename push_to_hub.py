import argparse
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModel, AutoModelForSequenceClassification, AutoModelForMaskedLM, AutoProcessor

from cm3p import (
    CM3PConfig,
    CM3PMetadataConfig,
    CM3PAudioConfig,
    CM3PBeatmapConfig,
    CM3PModel,
    CM3PMetadataModel,
    CM3PMetadataModelWithProjection,
    CM3PBeatmapModel,
    CM3PBeatmapModelWithProjection,
    CM3PForBeatmapClassification,
    CM3PForMaskedLM,
)
from cm3p.processing_cm3p import CM3PProcessor

# ---------------------------------------------------------------------------
# Auto class registrations (idempotent; safe to call multiple times)
# ---------------------------------------------------------------------------
for cfg_cls in [CM3PConfig, CM3PMetadataConfig, CM3PAudioConfig, CM3PBeatmapConfig]:
    try:
        cfg_cls.register_for_auto_class()
    except Exception:
        pass  # already registered

REGISTRATION_TABLE = [
    (CM3PModel, "AutoModel"),
    (CM3PMetadataModel, "AutoModel"),
    (CM3PMetadataModelWithProjection, "AutoModel"),
    (CM3PBeatmapModel, "AutoModel"),
    (CM3PBeatmapModelWithProjection, "AutoModel"),
    (CM3PForBeatmapClassification, "AutoModelForSequenceClassification"),
    (CM3PForMaskedLM, "AutoModelForMaskedLM"),
    (CM3PProcessor, "AutoProcessor"),
]
for cls, auto_name in REGISTRATION_TABLE:
    try:
        cls.register_for_auto_class(auto_name)
    except Exception:
        pass  # already registered

# Also ensure transformers Auto registries know about config->model mappings
AutoConfig.register("CM3P", CM3PConfig)
AutoConfig.register("CM3PMetadata", CM3PMetadataConfig)
AutoConfig.register("CM3PAudio", CM3PAudioConfig)
AutoConfig.register("CM3PBeatmap", CM3PBeatmapConfig)
AutoModel.register(CM3PConfig, CM3PModel)
AutoModel.register(CM3PMetadataConfig, CM3PMetadataModel)
AutoModel.register(CM3PBeatmapConfig, CM3PBeatmapModel)
AutoModelForSequenceClassification.register(CM3PBeatmapConfig, CM3PForBeatmapClassification)
AutoModelForMaskedLM.register(CM3PBeatmapConfig, CM3PForMaskedLM)
AutoProcessor.register(CM3PConfig, CM3PProcessor)
AutoProcessor.register(CM3PBeatmapConfig, CM3PProcessor)

MODEL_CLASS_CHOICES = {
    "cm3p": CM3PModel,
    "metadata": CM3PMetadataModel,
    "metadata_with_projection": CM3PMetadataModelWithProjection,
    "beatmap": CM3PBeatmapModel,
    "beatmap_with_projection": CM3PBeatmapModelWithProjection,
    "classification": CM3PForBeatmapClassification,
    "masked_lm": CM3PForMaskedLM,
}

dtype_map = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def infer_model_class(save_path: Path) -> type | None:
    """Attempt to infer appropriate model class from config if user didn't specify one."""
    try:
        cfg = AutoConfig.from_pretrained(save_path)
    except Exception:
        return None
    mt = getattr(cfg, "model_type", None)
    if mt == "CM3P":
        return CM3PModel
    if mt == "CM3PBeatmap":
        # Decide between masked LM vs classification vs plain
        if hasattr(cfg, "num_labels") and getattr(cfg, "num_labels") is not None and getattr(cfg, "num_labels") > 0:
            return CM3PForBeatmapClassification
        # heuristic: if has_decoder_head flag stored somewhere for beatmap LM
        if hasattr(cfg, "sparse_prediction") or hasattr(cfg, "repad_logits_with_grad"):
            # We favor masked LM if vocab_size > 10 (skip pure audio config)
            if getattr(cfg, "vocab_size", 0) > 10:
                return CM3PForMaskedLM
        return CM3PBeatmapModel
    if mt == "CM3PMetadata":
        return CM3PMetadataModel
    return None


def parse_args():
    parser = argparse.ArgumentParser(description="Push CM3P model and processor to the Hugging Face Hub.")
    parser.add_argument("--save-path", type=str, required=True, help="Local checkpoint directory (contains config & weights).")
    parser.add_argument("--repo-id", type=str, required=True, help="Target repository id on the Hub (e.g. user/CM3P-model).")
    parser.add_argument("--processor-repo-id", type=str, default=None, help="Optional processor repo id (defaults to --repo-id if not set).")
    parser.add_argument("--model-class", type=str, choices=MODEL_CLASS_CHOICES.keys(), default=None, help="Override which model class to load.")
    parser.add_argument("--private", action="store_true", help="Create/update the Hub repo as private.")
    parser.add_argument("--no-push-processor", action="store_true", help="Skip pushing the processor.")
    parser.add_argument("--dtype", type=str, choices=dtype_map.keys(), default="bfloat16", help="Torch dtype to load weights with.")
    parser.add_argument("--attn-implementation", type=str, choices=["sdpa", "flash_attention_2"], default="sdpa", help="Attention backend to configure during load.")
    parser.add_argument("--commit-message", type=str, default="Add CM3P model", help="Hub commit message.")
    parser.add_argument("--revision", type=str, default="main", help="Target branch/revision on Hub.")
    parser.add_argument("--push-only", action="store_true", help="Only push already existing repo metadata (do not reload model).")
    parser.add_argument("--trust-remote-code", action="store_true", help="Set trust_remote_code=True while loading (if needed).")
    return parser.parse_args()


def main():
    args = parse_args()

    save_path = Path(args.save_path)
    if not save_path.exists():
        raise FileNotFoundError(f"--save-path '{save_path}' does not exist.")

    model_repo_id = args.repo_id
    processor_repo_id = args.processor_repo_id or model_repo_id

    # infer or choose model class
    model_cls = MODEL_CLASS_CHOICES.get(args.model_class)
    if model_cls is None:
        model_cls = infer_model_class(save_path)
        if model_cls is None:
            raise ValueError("Could not infer model class. Please specify --model-class.")
        print(f"[info] Inferred model class: {model_cls.__name__}")
    else:
        print(f"[info] Using user-selected model class: {model_cls.__name__}")

    if args.push_only:
        print("[info] push-only flag set; skipping model/processor loading.")
    else:
        torch_dtype = dtype_map[args.dtype]
        if hasattr(model_cls, "from_pretrained"):
            print(f"[info] Loading model from {save_path} with dtype={torch_dtype} attn={args.attn_implementation}")
            model = model_cls.from_pretrained(
                save_path,
                torch_dtype=torch_dtype,
                attn_implementation=args.attn_implementation,
                trust_remote_code=args.trust_remote_code,
            )
        else:
            raise ValueError(f"Selected model class {model_cls.__name__} does not implement from_pretrained.")

        print(f"[info] Model loaded: {model.__class__.__name__}")

        if not args.no_push_processor:
            try:
                processor = CM3PProcessor.from_pretrained(save_path, trust_remote_code=args.trust_remote_code)
                print(f"[info] Processor loaded: {processor.__class__.__name__}")
            except Exception as e:
                print(f"[warn] Could not load processor from save path: {e}")
                processor = None
        else:
            processor = None
            print("[info] Skipping processor push per flag.")

        # Push model
        print(f"[info] Pushing model to Hub repo '{model_repo_id}' (private={args.private})")
        model.push_to_hub(
            model_repo_id,
            private=args.private,
            commit_message=args.commit_message,
            revision=args.revision,
        )
        print("[success] Model push complete.")

        if processor is not None:
            print(f"[info] Pushing processor to Hub repo '{processor_repo_id}' (private={args.private})")
            processor.push_to_hub(
                processor_repo_id,
                private=args.private,
                commit_message=args.commit_message,
                revision=args.revision,
            )
            print("[success] Processor push complete.")
        else:
            print("[info] Processor not pushed.")

    print("[done] push_to_hub script finished.")


if __name__ == "__main__":
    main()
