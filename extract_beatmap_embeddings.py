import logging
import sys
from pathlib import Path

import hydra
import pandas as pd
import torch
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import TrainingArguments
from transformers.trainer_utils import set_seed

from cm3p.modeling_cm3p import CM3PModel
from cm3p.parsing_cm3p import CM3PBeatmapParser
from cm3p.processing_cm3p import CM3PProcessor
from cm3p.tokenization_cm3p import CM3PBeatmapTokenizer, CM3PMetadataTokenizer
from config import TrainConfig
from mmrs_dataset import MmrsDataset, worker_init_fn
from utils.data_utils import filter_mmrs_metadata, load_mmrs_metadata
from transformers import WhisperFeatureExtractor
import json

logger = logging.getLogger(__name__)


@hydra.main(config_path="configs/train", config_name="v1", version_base="1.1")
def main(args: TrainConfig):
    # Convert Hydra config to dataclass instance
    args: TrainConfig = OmegaConf.to_object(args)

    if args.training.get("optim", None) == "muon":
        # remove unsupported optimizer key for TrainingArguments
        del args.training["optim"]

    training_args = TrainingArguments(**args.training)
    set_seed(training_args.seed)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO)

    # Force deterministic dataset behavior for embedding extraction
    args.dataset.dt_augment_prob = 0.0
    args.dataset.beatmap_mismatch_prob = 0.0
    args.dataset.metadata_dropout_prob = 0.0
    args.dataset.train_metadata_variations = 1
    args.dataset.test_metadata_variations = 1
    # args.dataset.labels = "none"
    args.dataset.include_source_metadata = True

    # Prepare dataset-related dynamic tokenizer configs (same pattern as validate_dataset)
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

    # metadata index is MultiIndex [BeatmapSetId, Id]; ensure dynamic maps derive from filtered metadata
    if args.processor.metadata_tokenizer.modes is None:
        args.processor.metadata_tokenizer.modes = metadata.reset_index().set_index(["ModeInt"])["Mode"].to_dict()
    if args.processor.metadata_tokenizer.statuses is None:
        args.processor.metadata_tokenizer.statuses = metadata.reset_index().set_index(["Ranked"])["Status"].to_dict()
    if args.processor.metadata_tokenizer.mappers is None:
        args.processor.metadata_tokenizer.mappers = metadata.reset_index().set_index(["UserId"])["Creator"].to_dict()
    if args.processor.metadata_tokenizer.tags is None:
        all_tag_ids = metadata["TopTagIds"].explode().dropna().unique().tolist()
        tags_info_path = Path(__file__).parent / "resources" / "tags.json"
        tags_info = json.load(open(tags_info_path, "r", encoding="utf-8"))["tags"]
        tags_info = {int(tag["id"]): {"name": tag["name"], "ruleset_id": tag["ruleset_id"], "description": tag["description"]} for tag in tags_info}
        args.processor.metadata_tokenizer.tags = {tag_id: tags_info[tag_id] for tag_id in tags_info if tag_id in all_tag_ids}

    processor = CM3PProcessor(
        audio_feature_extractor=WhisperFeatureExtractor(**args.processor.audio_feature_extractor.__dict__),
        beatmap_parser=CM3PBeatmapParser(**args.processor.beatmap_parser.__dict__),
        beatmap_tokenizer=CM3PBeatmapTokenizer(**args.processor.beatmap_tokenizer.__dict__),
        metadata_tokenizer=CM3PMetadataTokenizer(**args.processor.metadata_tokenizer.__dict__),
        default_kwargs=args.processor.default_kwargs,
    )

    # Load training dataset (non-test for original configuration / ordering). Disabling drop_last.
    dataset = MmrsDataset(
        args.dataset,
        processor=processor,
        test=False,
    )

    # Construct DataLoader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=training_args.per_device_train_batch_size,
        num_workers=training_args.dataloader_num_workers,
        timeout=600 if training_args.dataloader_num_workers > 0 else 0,
        worker_init_fn=worker_init_fn,
        drop_last=False,
        in_order=True,
    )

    # Load pretrained CM3P model
    if args.from_pretrained is None:
        raise ValueError("from_pretrained must be specified in the config to load the CM3P model.")

    logger.info(f"Loading CM3P model from: {args.from_pretrained}")
    model = CM3PModel.from_pretrained(args.from_pretrained)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    est_batches = int(metadata["TotalLength"].sum() // args.processor.default_kwargs["beatmap_kwargs"]["window_stride_sec"] // training_args.per_device_train_batch_size + 1)

    # Accumulators for embeddings per beatmap
    embed_accumulator: dict[int, dict[str, any]] = {}
    # Structure: beatmap_id -> { 'sum': np.ndarray, 'count': int }

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting beatmap embeddings", smoothing=0.01, total=est_batches):
            # Some batches may be empty if dataset yields nothing
            if len(batch.get("input_ids", [])) == 0:
                continue

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            input_features = batch.get("input_features", None)
            if input_features is not None:
                input_features = input_features.to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                input_features=input_features,
                return_loss=False,
            )
            # Normalized beatmap embeddings (forward applies projection + normalization)
            embeds = outputs.beatmap_embeds.detach().cpu().numpy()

            beatmap_ids = batch.get("beatmap_id", None)
            if beatmap_ids is None:
                continue
            # beatmap_ids may be tensor -> convert to list
            if torch.is_tensor(beatmap_ids):
                beatmap_ids = beatmap_ids.tolist()

            for i, bid in enumerate(beatmap_ids):
                if bid is None:
                    continue
                if bid not in embed_accumulator:
                    embed_accumulator[bid] = {
                        'sum': embeds[i].copy(),
                        'count': 1,
                    }
                else:
                    embed_accumulator[bid]['sum'] += embeds[i]
                    embed_accumulator[bid]['count'] += 1

    # Build DataFrame of mean embeddings keyed by beatmap_id
    rows = []
    for bid, info in embed_accumulator.items():
        mean_vec = info['sum'] / info['count']
        # Optional: re-normalize the averaged embedding to unit length for consistency
        norm = (mean_vec ** 2).sum() ** 0.5
        if norm > 0:
            mean_vec = mean_vec / norm
        rows.append({
            'beatmap_id': int(bid),
            'embedding': mean_vec.tolist(),
        })

    embeddings_df = pd.DataFrame(rows)

    # Merge embeddings with full metadata (which has MultiIndex [BeatmapSetId, Id])
    meta_df = metadata.reset_index()  # brings BeatmapSetId and Id into columns
    # Ensure the Id column is int for a robust merge
    if meta_df['Id'].dtype != 'int64' and meta_df['Id'].dtype != 'int32':
        meta_df['Id'] = meta_df['Id'].astype(int)
    merged_df = embeddings_df.merge(meta_df, left_on='beatmap_id', right_on='Id', how='left')

    # Reorder columns to put embedding at the end, keeping all original metadata columns
    cols = ['Artist', 'ArtistUnicode', 'Creator', 'FavouriteCount', 'BeatmapSetId', 'Nsfw', 'Offset', 'BeatmapSetPlayCount', 'Source', 'BeatmapSetStatus', 'Spotlight', 'Title', 'TitleUnicode', 'BeatmapSetUserId', 'Video', 'Description', 'GenreId', 'GenreName', 'LanguageId', 'LanguageName', 'PackTags', 'Ratings', 'DownloadDisabled', 'BeatmapSetBpm', 'CanBeHyped', 'DiscussionLocked', 'BeatmapSetIsScoreable', 'BeatmapSetLastUpdated', 'BeatmapSetRanked', 'RankedDate', 'Storyboard', 'SubmittedDate', 'Tags', 'DifficultyRating', 'Id', 'Mode', 'Status', 'TotalLength', 'UserId', 'Version', 'Checksum', 'MaxCombo', 'Accuracy', 'Ar', 'Bpm', 'CountCircles', 'CountSliders', 'CountSpinners', 'Cs', 'Drain', 'HitLength', 'IsScoreable', 'LastUpdated', 'ModeInt', 'PassCount', 'PlayCount', 'Ranked', 'Owners', 'TopTagIds', 'TopTagCounts', 'StarRating', 'OmdbTags', 'AudioFile', 'BeatmapSetFolder', 'BeatmapFile', 'embedding']
    merged_df = merged_df[cols]

    output_path = Path("beatmap_embeddings.parquet")
    merged_df.to_parquet(output_path, index=False)
    logger.info(f"Final DataFrame has {merged_df.shape[0]} rows.")
    logger.info(f"Final DataFrame has {merged_df.shape[1]} columns.")
    logger.info(f"Columns: {merged_df.columns.tolist()}")
    logger.info(f"Saved {len(merged_df)} beatmap embeddings to {output_path.resolve()}")


if __name__ == "__main__":
    main()
