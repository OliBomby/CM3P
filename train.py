import json
import logging
import os
import sys
from pathlib import Path

import hydra
from omegaconf import OmegaConf
from transformers import Trainer
from transformers import TrainingArguments, WhisperFeatureExtractor
from transformers.trainer_utils import get_last_checkpoint, set_seed

from cm3p import CM3PModel, CM3PConfig
from cm3p.parsing_cm3p import CM3PBeatmapParser
from cm3p.processing_cm3p import CM3PProcessor
from cm3p.tokenization_cm3p import CM3PBeatmapTokenizer, CM3PMetadataTokenizer
from config import TrainConfig
from utils.data_utils import filter_mmrs_metadata, load_mmrs_metadata
from mmrs_dataset import MmrsDataset

logger = logging.getLogger(__name__)


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

    model = CM3PModel(model_config)

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
        print(f"Number of parameters for Muon: {len(muon_params)}")
        print(f"Number of parameters for AdamW: {len(adamw_params)}")

        optimizers = (Muon(
            muon_params=muon_params,
            lr=training_args.learning_rate,
            adamw_lr=training_args.learning_rate / 4,
            adamw_params=adamw_params,
            adamw_betas=(training_args.adam_beta1, training_args.adam_beta2),
            adamw_wd=training_args.weight_decay,
            adamw_eps=training_args.adam_epsilon,
        ), optimizers[1])

    # Initialize our trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        data_collator=None,
        compute_metrics=None,
        optimizers=optimizers,
        processing_class=processor,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
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
