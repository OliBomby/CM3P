import json
import logging
import warnings
import os
import sys
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import TrainingArguments, WhisperFeatureExtractor
from transformers.trainer_utils import set_seed

from cm3p.parsing_cm3p import CM3PBeatmapParser
from cm3p.processing_cm3p import CM3PProcessor
from cm3p.tokenization_cm3p import CM3PBeatmapTokenizer, CM3PMetadataTokenizer
from config import TrainConfig
from utils.data_utils import filter_mmrs_metadata, load_mmrs_metadata
from mmrs_dataset import MmrsDataset, worker_init_fn

logger = logging.getLogger(__name__)


# noinspection PyArgumentList
@hydra.main(config_path="configs/train", config_name="v1", version_base="1.1")
def main(args: TrainConfig):
    # Parse input arguments
    args: TrainConfig = OmegaConf.to_object(args)
    training_args = TrainingArguments(**args.training)

    # Set seed for all RNGs
    set_seed(training_args.seed)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    logger.setLevel(logging.DEBUG)

    # Populate metadata tokenizer modes, mappers, and tags configs from dataset if not provided
    args.dataset.train_dataset_start = None
    args.dataset.train_dataset_end = None
    args.dataset.drop_last = False

    metadata = filter_mmrs_metadata(
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
        args.processor.metadata_tokenizer.modes = metadata.reset_index().set_index(["ModeInt"])["Mode"].to_dict()
    if args.processor.metadata_tokenizer.statuses is None:
        args.processor.metadata_tokenizer.statuses = metadata.reset_index().set_index(["Ranked"])["Status"].to_dict()
    if args.processor.metadata_tokenizer.mappers is None:
        args.processor.metadata_tokenizer.mappers = metadata.reset_index().set_index(["UserId"])["Creator"].to_dict()
    if args.processor.metadata_tokenizer.tags is None:
        all_tag_ids = metadata["TopTagIds"].explode().dropna().unique().tolist()
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

    # Load dataset, one to rule them all
    dataset = MmrsDataset(
        args.dataset,
        processor=processor,
        test=False,
    )

    # Create dataloader
    os.mkdir('dataloader')
    dataloader = DataLoader(
        dataset,
        batch_size=training_args.per_device_train_batch_size,
        num_workers=training_args.dataloader_num_workers,
        timeout=600 if training_args.dataloader_num_workers > 0 else 0,
        worker_init_fn=worker_init_fn,
        drop_last=False,
        in_order=False,
    )

    # Make histogram of the lengths of the sequences
    num_batches = 0
    num_examples = 0
    num_beatmap_tokens = 0
    num_metadata_tokens = 0
    beatmap_tokens = []
    metadata_tokens = []
    est_batches = int(metadata["TotalLength"].sum() // args.processor.default_kwargs["beatmap_kwargs"]["window_stride_sec"] // training_args.per_device_train_batch_size + 1)
    try:
        for b in tqdm(dataloader, smoothing=0.01, total=est_batches):
            num_batches += 1
            num_examples += len(b["input_ids"])
            for i in range(len(b["input_ids"])):  # batch size
                beatmap_length = b['attention_mask'][i].sum().item()
                beatmap_tokens.append(beatmap_length)
                num_beatmap_tokens += beatmap_length
                metadata_length = b['metadata_attention_mask'][i].sum().item()
                metadata_tokens.append(metadata_length)
                num_metadata_tokens += metadata_length
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt detected. Printing results...")
        pass

    logger.info(f"Number of beatmapsets: {len(metadata.index.get_level_values(0).unique())}")
    logger.info(f"Number of beatmaps: {len(metadata)}")
    logger.info(f"Number of batches: {num_batches}")
    logger.info(f"Number of examples: {num_examples}")
    logger.info(f"Number of beatmap tokens: {num_beatmap_tokens}")
    logger.info(f"Number of metadata tokens: {num_metadata_tokens}")
    window_length_sec = args.processor.default_kwargs["beatmap_kwargs"]["window_length_sec"]
    logger.info(f"Average beatmap tokens per second: {num_beatmap_tokens / (num_examples * window_length_sec):.2f}")
    logger.info(f"Average beatmap tokens per example: {num_beatmap_tokens / num_examples:.2f}")
    logger.info(f"Average metadata tokens per example: {num_metadata_tokens / num_examples:.2f}")
    logger.info(f"Max beatmap tokens in an example: {max(beatmap_tokens)}")
    logger.info(f"Max metadata tokens in an example: {max(metadata_tokens)}")

    plt.hist(beatmap_tokens, bins=100)
    plt.title("Histogram of Beatmap Token Lengths")
    plt.show()
    plt.hist(metadata_tokens, bins=100)
    plt.title("Histogram of Metadata Token Lengths")
    plt.show()


if __name__ == "__main__":
    main()
