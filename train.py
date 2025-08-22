import logging
import os
import sys

import hydra
import transformers
from transformers import TrainingArguments, WhisperFeatureExtractor
from transformers import Trainer
from transformers.trainer_utils import get_last_checkpoint, set_seed

from cm3p import CM3PConfig, CM3PModel
from cm3p.parsing_cm3p import CM3PBeatmapParser
from cm3p.processing_cm3p import CM3PProcessor
from cm3p.tokenization_cm3p import CM3PBeatmapTokenizer, CM3PMetadataTokenizer
from config import TrainConfig


logger = logging.getLogger(__name__)


@hydra.main(config_path="configs/train", config_name="v1", version_base="1.1")
def main(args: TrainConfig):
    # 1. Parse input arguments
    training_args = TrainingArguments(**args.training.to_dict())

    # 2. Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.bf16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # 3. Detecting last checkpoint and eventually continue from last checkpoint
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

    # 4. Load dataset
    train_dataset = None
    eval_dataset = None

    # 5. Load pretrained model, tokenizer, and image processor
    test_beatmap_tokenizer_config = {}

    test_metadata_tokenizer_config = {
        "modes": ["osu", "taiko", "fruits", "mania"],
        "mappers": ["OliBomby", "Cookiezi", "peppy", "Xenon"],
    }

    processor = CM3PProcessor(
        WhisperFeatureExtractor(),
        CM3PBeatmapParser(),
        CM3PBeatmapTokenizer(vocab_init=test_beatmap_tokenizer_config),
        CM3PMetadataTokenizer(vocab_init=test_metadata_tokenizer_config),
    )

    config = CM3PConfig(attn_implementation="flash_attention_2")
    config.beatmap_config.vocab_size = processor.beatmap_tokenizer.vocab_size
    config.beatmap_config.audio_token_id = processor.beatmap_tokenizer.convert_tokens_to_ids(processor.beatmap_tokenizer.audio_token)
    config.beatmap_config.audio_sos_token_id = processor.beatmap_tokenizer.convert_tokens_to_ids(processor.beatmap_tokenizer.audio_eos_token)
    config.beatmap_config.audio_eos_token_id = processor.beatmap_tokenizer.convert_tokens_to_ids(processor.beatmap_tokenizer.audio_eos_token)
    config.metadata_config.vocab_size = processor.metadata_tokenizer.vocab_size
    model = CM3PModel(config)

    def _freeze_params(module):
        for param in module.parameters():
            param.requires_grad = False

    if model_args.freeze_vision_model:
        _freeze_params(model.vision_model)

    if model_args.freeze_text_model:
        _freeze_params(model.text_model)

    # set seed for torch dataloaders
    set_seed(training_args.seed)

    # 8. Initialize our trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        data_collator=None,
        compute_metrics=None,
        processing_class=processor,
    )

    # 9. Training
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

    # 10. Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # 11. Write Training Stats and push to hub.
    kwargs = {"tasks": "contrastive-image-text-modeling"}

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)

if __name__ == "__main__":
    main()