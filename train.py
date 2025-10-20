import json
import logging
import os
import sys
from pathlib import Path

import hydra
import torch
from omegaconf import OmegaConf
from transformers import Trainer, EvalPrediction, TrainerCallback
from transformers import TrainingArguments, WhisperFeatureExtractor
from transformers.trainer_utils import get_last_checkpoint, set_seed

from cm3p import CM3PModel, CM3PConfig
from cm3p.modeling_cm3p import CM3PForMaskedLM, CM3PForBeatmapClassification
from cm3p.parsing_cm3p import CM3PBeatmapParser
from cm3p.processing_cm3p import CM3PProcessor
from cm3p.tokenization_cm3p import CM3PBeatmapTokenizer, CM3PMetadataTokenizer
from config import TrainConfig
from utils.data_utils import filter_mmrs_metadata, load_mmrs_metadata
from mmrs_dataset import MmrsDataset

logger = logging.getLogger(__name__)
accumulated_metrics = {}


class UnfreezeBeatmapCallback(TrainerCallback):
    def __init__(self, unfreeze_at_step=1000):
        self.unfreeze_at_step = unfreeze_at_step

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == self.unfreeze_at_step:
            logger.info(f"Unfreezing beatmap_model at step {state.global_step}")
            for param in kwargs["model"].beatmap_model.parameters():
                param.requires_grad = True


def compute_metrics(eval_pred: EvalPrediction, compute_result) -> dict | None:
    global accumulated_metrics

    # Variation classes: -1 padding, 0 original, 1 year, 2 status, 3 tags, 4 mapper
    variation_classes = {
        -200: "classification",
        -100: "masked_lm",
        -1: "padding",
        0: "original",
        1: "year",
        2: "status",
        3: "tags",
        4: "mapper",
    }
    classes_range = range(1, 5)
    classes_with_top5 = [-100, 3, 4]

    if eval_pred.label_ids is not None and len(eval_pred.label_ids) > 0:
        if eval_pred.label_ids.ndim == 1:
            # Classification task
            logits: torch.FloatTensor = eval_pred.predictions if isinstance(eval_pred.predictions, torch.Tensor) else torch.tensor(eval_pred.predictions)
            labels: torch.LongTensor = eval_pred.label_ids if isinstance(eval_pred.label_ids, torch.Tensor) else torch.tensor(eval_pred.label_ids)
            predictions = logits.argmax(-1)

            var_class = -200
            correct = (predictions == labels).sum().item()
            total = labels.size(0)
            top5_indices = torch.topk(logits, k=min(5, logits.size(-1)), dim=-1).indices
            top5_correct = (top5_indices == labels.unsqueeze(-1)).any(dim=-1).sum().item()

            if var_class not in accumulated_metrics:
                accumulated_metrics[var_class] = {"correct": 0, "total": 0, "top5_correct": 0}

            accumulated_metrics[var_class]["correct"] += correct
            accumulated_metrics[var_class]["total"] += total
            accumulated_metrics[var_class]["top5_correct"] += top5_correct
        else:
            # Calculate accuracy for masked LM if labels are provided
            var_class = -100
            logits: torch.FloatTensor = eval_pred.predictions[4] if isinstance(eval_pred.predictions, tuple) else eval_pred.predictions
            labels: torch.LongTensor = eval_pred.label_ids
            mask = labels != -100
            correct = (logits.argmax(-1)[mask] == labels[mask]).sum().item()
            total = mask.sum().item()
            top5_indices = torch.topk(logits, k=min(5, logits.size(-1)), dim=-1).indices
            top5_correct = (top5_indices[mask] == labels[mask].unsqueeze(-1)).any(dim=-1).sum().item()

            if var_class not in accumulated_metrics:
                accumulated_metrics[var_class] = {"correct": 0, "total": 0, "top5_correct": 0}

            accumulated_metrics[var_class]["correct"] += correct
            accumulated_metrics[var_class]["total"] += total
            accumulated_metrics[var_class]["top5_correct"] += top5_correct

    if "metadata_variation_classes" in eval_pred.inputs:
        # eval_pred.inputs["metadata_variation_classes"] (batch_size, num_variations)
        # eval_pred.predictions[0]: logits_per_beatmap (batch_size, batch_size, num_variations)

        # For each variation class, compute accuracy:
        # For each example: Separate out the classes of metadata variations
        # For each class group: Check if the highest logit corresponds to the original metadata (class 0)
        # Compute accuracy for all classes and top-5 accuracy for tags and mappers

        logits_per_beatmap: torch.FloatTensor = eval_pred.predictions[0]
        metadata_variation_classes: torch.LongTensor = eval_pred.inputs["metadata_variation_classes"]
        batch_size = logits_per_beatmap.shape[0]

        for var_class in classes_range:  # Skip padding class -1
            correct = 0
            total = 0
            top5_correct = 0

            for i in range(batch_size):
                # Get indices of examples with the same variation class
                class_mask: torch.BoolTensor = ((metadata_variation_classes[i] == var_class) |
                                                (metadata_variation_classes[i] == 0))  # Include original class 0

                if class_mask.sum() <= 1:  # Skip if there is only one example (no variation)
                    continue

                # Get logits for this group
                group_logits = logits_per_beatmap[i, i][class_mask]
                group_classes = metadata_variation_classes[i][class_mask]

                # Check if the highest logit corresponds to the original metadata (class 0)
                total += 1

                predicted_index = torch.argmax(group_logits).item()
                if group_classes[predicted_index] == 0:
                    correct += 1

                if var_class in classes_with_top5:
                    top5_indices = torch.topk(group_logits, k=min(5, group_logits.size(0))).indices
                    if (group_classes[top5_indices] == 0).any():
                        top5_correct += 1

            if var_class not in accumulated_metrics:
                accumulated_metrics[var_class] = {"correct": 0, "total": 0, "top5_correct": 0}

            accumulated_metrics[var_class]["correct"] += correct
            accumulated_metrics[var_class]["total"] += total
            accumulated_metrics[var_class]["top5_correct"] += top5_correct

    if not compute_result:
        return None

    result = {}
    for var_class, metrics in accumulated_metrics.items():
        class_name = variation_classes.get(var_class, f"class_{var_class}")
        if metrics["total"] > 0:
            accuracy = metrics["correct"] / metrics["total"]
            result[f"accuracy_{class_name}"] = accuracy
            if var_class in classes_with_top5:
                top5_accuracy = metrics["top5_correct"] / metrics["total"]
                result[f"top5_accuracy_{class_name}"] = top5_accuracy
        else:
            result[f"accuracy_{class_name}"] = None
            if var_class in classes_with_top5:
                result[f"top5_accuracy_{class_name}"] = None

    # Reset accumulated metrics after computing final result
    accumulated_metrics = {}
    return result


# noinspection PyArgumentList
@hydra.main(config_path="configs/train", config_name="v1", version_base="1.1")
def main(args: TrainConfig):
    # Parse input arguments
    args = OmegaConf.to_object(args)

    use_muon = False
    if args.training.get("optim", None) == "muon":
        use_muon = True
        del args.training["optim"]

    training_args = TrainingArguments(**args.training)

    if args.wandb_project is not None:
        os.environ["WANDB_PROJECT"] = args.wandb_project
    if args.wandb_entity is not None:
        os.environ["WANDB_ENTITY"] = args.wandb_entity
    if args.wandb_mode is not None:
        os.environ["WANDB_MODE"] = args.wandb_mode

    # Set seed for all RNGs
    set_seed(training_args.seed)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.INFO,
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.bf16}"
    )
    # logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint and eventually continue from last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint

    # Load pretrained model, tokenizer, and image processor
    # Populate metadata tokenizer modes, mappers, and tags configs from dataset if not provided
    if (args.processor.metadata_tokenizer.modes is None
            or args.processor.metadata_tokenizer.statuses is None
            or args.processor.metadata_tokenizer.mappers is None
            or args.processor.metadata_tokenizer.tags is None):
        train_metadata = filter_mmrs_metadata(
            load_mmrs_metadata(args.dataset.train_dataset_paths),
            start=args.dataset.train_dataset_start,
            end=args.dataset.train_dataset_end,
            gamemodes=args.dataset.gamemodes,
            min_year=args.dataset.min_year,
            max_year=args.dataset.max_year,
            min_difficulty=args.dataset.min_difficulty,
            max_difficulty=args.dataset.max_difficulty,
        )
        if args.processor.metadata_tokenizer.modes is None:
            args.processor.metadata_tokenizer.modes = train_metadata.reset_index().set_index(["ModeInt"])["Mode"].to_dict()
        if args.processor.metadata_tokenizer.statuses is None:
            args.processor.metadata_tokenizer.statuses = train_metadata.reset_index().set_index(["Ranked"])["Status"].to_dict()
        if args.processor.metadata_tokenizer.mappers is None:
            args.processor.metadata_tokenizer.mappers = train_metadata.reset_index().set_index(["UserId"])["Creator"].to_dict()
        if args.processor.metadata_tokenizer.tags is None:
            all_tag_ids = train_metadata["TopTagIds"].explode().dropna().unique().tolist()
            tags_info = json.load(open(Path(__file__).parent / "resources" / "tags.json", "r", encoding="utf-8"))["tags"]
            tags_info = {int(tag["id"]): {"name": tag["name"], "ruleset_id": tag["ruleset_id"], "description": tag["description"]} for tag in tags_info}
            args.processor.metadata_tokenizer.tags = {tag_id: tags_info[tag_id] for tag_id in tags_info if tag_id in all_tag_ids}

    processor = CM3PProcessor(
        audio_feature_extractor=WhisperFeatureExtractor(**args.processor.audio_feature_extractor.__dict__),
        beatmap_parser=CM3PBeatmapParser(**args.processor.beatmap_parser.__dict__),
        beatmap_tokenizer=CM3PBeatmapTokenizer(**args.processor.beatmap_tokenizer.__dict__),
        metadata_tokenizer=CM3PMetadataTokenizer(**args.processor.metadata_tokenizer.__dict__),
        default_kwargs=args.processor.default_kwargs,
    )

    # Load dataset
    train_dataset = MmrsDataset(
        args.dataset,
        processor=processor,
        test=False,
    )
    eval_dataset = MmrsDataset(
        args.dataset,
        processor=processor,
        test=True,
    )

    # Populate model config with tokenizer info
    model_config = CM3PConfig(**args.model)
    model_config._attn_implementation = args.attn_implementation

    def assign_token_id(config, tokenizer, token_attr_name):
        token = getattr(tokenizer, token_attr_name, None)
        token_id = tokenizer.convert_tokens_to_ids(token)
        setattr(config, f"{token_attr_name}_id", token_id)

    model_config.beatmap_config.vocab_size = processor.beatmap_tokenizer.vocab_size
    assign_token_id(model_config.beatmap_config, processor.beatmap_tokenizer, "pad_token")
    assign_token_id(model_config.beatmap_config, processor.beatmap_tokenizer, "bos_token")
    assign_token_id(model_config.beatmap_config, processor.beatmap_tokenizer, "eos_token")
    assign_token_id(model_config.beatmap_config, processor.beatmap_tokenizer, "audio_sos_token")
    assign_token_id(model_config.beatmap_config, processor.beatmap_tokenizer, "audio_eos_token")
    assign_token_id(model_config.beatmap_config, processor.beatmap_tokenizer, "audio_token")

    model_config.metadata_config.vocab_size = processor.metadata_tokenizer.vocab_size
    assign_token_id(model_config.metadata_config, processor.metadata_tokenizer, "pad_token")
    assign_token_id(model_config.metadata_config, processor.metadata_tokenizer, "bos_token")
    assign_token_id(model_config.metadata_config, processor.metadata_tokenizer, "eos_token")

    if args.model_cls == "CM3PForMaskedLM":
        model_class = CM3PForMaskedLM
        model_config = model_config.beatmap_config
    elif args.model_cls == "CM3PForBeatmapClassification":
        model_class = CM3PForBeatmapClassification
        model_config = model_config.beatmap_config
    else:
        model_class = CM3PModel

    if not training_args.do_train and checkpoint is not None:
        logger.warning(f"Loading model from checkpoint {checkpoint} for evaluation")
        model = model_class.from_pretrained(checkpoint, config=model_config)
    elif args.from_pretrained is not None:
        logger.warning(f"Loading model from {args.from_pretrained}")
        model = model_class.from_pretrained(args.from_pretrained, config=model_config)
    else:
        model = model_class(model_config)

    def _freeze_params(module):
        for param in module.parameters():
            param.requires_grad = False

    if args.freeze_beatmap_model:
        _freeze_params(model.beatmap_model)

    if args.freeze_metadata_model:
        _freeze_params(model.metadata_model)

    # Configure custom optimizer if needed
    optimizers = (None, None)
    if use_muon:
        from utils.muon_utils import Muon
        """
        Muon is intended to optimize only the internal ≥2D parameters of a network. 
        Embeddings, classifier heads, and scalar or vector parameters should be optimized using AdamW.
        """
        adamw_params = [
            param for name, param in model.named_parameters()
            if (any(kw in name.lower() for kw in {'embed', 'proj_out'}) or param.ndim <= 1)
        ]

        adamw_param_set = set(adamw_params)
        muon_params = [
            param for _, param in model.named_parameters()
            if param not in adamw_param_set
        ]
        logger.info(f"Number of parameters for Muon: {len(muon_params)}")
        logger.info(f"Number of parameters for AdamW: {len(adamw_params)}")

        optimizers = (Muon(
            muon_params=muon_params,
            lr=training_args.learning_rate,
            adamw_lr=training_args.learning_rate / 4,
            adamw_params=adamw_params,
            adamw_betas=(training_args.adam_beta1, training_args.adam_beta2),
            adamw_wd=training_args.weight_decay,
            adamw_eps=training_args.adam_epsilon,
        ), optimizers[1])

    # Configure custom callbacks if needed
    callbacks = []
    if args.freeze_beatmap_model and args.unfreeze_beatmap_model_at_step is not None:
        callbacks.append(UnfreezeBeatmapCallback(unfreeze_at_step=args.unfreeze_beatmap_model_at_step))

    # Initialize our trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        data_collator=None,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=None,
        optimizers=optimizers,
        processing_class=processor,
        callbacks=callbacks,
    )

    # Training
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        processor.save_pretrained(training_args.output_dir)
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Write Training Stats and push to hub.
    kwargs = {"tasks": "contrastive-beatmap-metadata-modeling"}

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)

if __name__ == "__main__":
    main()
