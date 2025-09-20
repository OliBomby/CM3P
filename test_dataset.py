import json
import logging
import os
import sys
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
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

    if args.training.get("optim", None) == "muon":
        del args.training["optim"]

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
        drop_last=True,
        in_order=True,
    )

    # Make histogram of the lengths of the sequences
    num_batches = 0
    num_examples = 0
    num_beatmap_tokens = 0
    num_metadata_tokens = 0
    beatmap_tokens = []
    metadata_tokens = []
    est_batches = int(metadata["TotalLength"].sum() // args.processor.default_kwargs["beatmap_kwargs"]["window_stride_sec"] // training_args.per_device_train_batch_size + 1)

    # Build YEAR token lookup from the metadata tokenizer
    metadata_tok = processor.metadata_tokenizer
    vocab = metadata_tok.get_vocab()

    year_token_id_to_year: dict[int, int] = {}
    year_unk_id = vocab.get(metadata_tok.year_unk_token, None)
    for tok, tid in vocab.items():
        if tok.startswith("[YEAR_") and tok != metadata_tok.year_unk_token:
            try:
                year_token_id_to_year[tid] = int(tok[len("[YEAR_"):-1])
            except ValueError:
                pass
    year_token_ids = set(year_token_id_to_year.keys())

    min_year = getattr(metadata_tok, "min_year", 2000)
    max_year = getattr(metadata_tok, "max_year", 2023)
    year_bins = np.arange(min_year - 0.5, max_year + 1.5, 1)

    # Configure how many time slices to track across the epoch
    num_year_slices = 6
    slice_size_batches = max(1, est_batches // num_year_slices)
    years_by_slice: list[list[int]] = [[] for _ in range(num_year_slices)]
    unknowns_by_slice: list[int] = [0 for _ in range(num_year_slices)]
    batches_per_slice: list[int] = [0 for _ in range(num_year_slices)]

    try:
        for b in tqdm(dataloader, smoothing=0.01, total=est_batches):
            num_batches += 1
            num_examples += len(b["input_ids"])

            # Determine current slice index by batch index
            slice_idx = min((num_batches - 1) // slice_size_batches, num_year_slices - 1)
            batches_per_slice[slice_idx] += 1

            for i in range(len(b["input_ids"])):  # batch size
                beatmap_length = b['attention_mask'][i].sum().item()
                beatmap_tokens.append(beatmap_length)
                num_beatmap_tokens += beatmap_length
                metadata_length = b['metadata_attention_mask'][i].sum().item()
                metadata_tokens.append(metadata_length)
                num_metadata_tokens += metadata_length

            # Collect year tokens from metadata_input_ids
            if 'metadata_ids' in b:
                meta_ids = b['metadata_ids']  # torch.Tensor [B, L]
                for i in range(meta_ids.size(0)):
                    seq_ids = meta_ids[i, 0].tolist()
                    year_found = False
                    for tid in seq_ids:
                        if tid in year_token_ids:
                            years_by_slice[slice_idx].append(year_token_id_to_year[tid])
                            year_found = True
                            break
                        if year_unk_id is not None and tid == year_unk_id:
                            unknowns_by_slice[slice_idx] += 1
                            year_found = True
                            break
                    if not year_found:
                        # No explicit YEAR token found in this sequence
                        unknowns_by_slice[slice_idx] += 1
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

    # Plot year distributions per slice (normalized) to detect drift over time
    rows = int(np.ceil(num_year_slices / 2))
    cols = 2 if num_year_slices > 1 else 1
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 3 * rows), sharex=True, sharey=True)
    axes: list[plt.Axes] = np.array(axes).reshape(-1)  # flatten for uniform indexing

    for si in range(num_year_slices):
        ax = axes[si] if si < len(axes) else None
        if ax is None:
            break
        data = years_by_slice[si]
        if len(data) > 0:
            ax.hist(data, bins=year_bins, density=True, color="#4C78A8", alpha=0.8, edgecolor="black")
        ax.set_title(f"Years slice {si + 1}/{num_year_slices}\n"
                     f"batches={batches_per_slice[si]}, n={len(data)}, unk={unknowns_by_slice[si]}")
        ax.set_xlabel("Year")
        ax.set_ylabel("Density")

    # Hide any empty subplots
    for j in range(num_year_slices, len(axes)):
        axes[j].axis("off")

    plt.suptitle("Metadata YEAR distribution over time slices in epoch")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
