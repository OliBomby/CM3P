from __future__ import annotations

import logging
import os
import random
import traceback
from pathlib import Path
from typing import Optional, Callable

import numpy as np
import torch
from pandas import Series, DataFrame
from torch.utils.data import IterableDataset, get_worker_info
from transformers.utils import PaddingStrategy

from cm3p.processing_cm3p import CM3PProcessor, get_metadata
from config import DataSetConfig
from utils.data_utils import load_mmrs_metadata, filter_mmrs_metadata, load_audio_file

logger = logging.getLogger(__name__)


def worker_init_fn(worker_id: int) -> None:
    """
    Initializes the logging for each dataloader worker to a separate file.
    Gives each dataloader a unique slice of the full dataset.
    """
    # Set-up logging to a file specific to this worker
    os.mkdir('dataloader') if not os.path.exists('dataloader') else None
    log_file_path = os.path.join('dataloader', f'worker_{worker_id}.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename=log_file_path,
        filemode='w'
    )

    # Configure the logging system to capture warnings
    # This redirects warnings.warn() to the logger
    logging.captureWarnings(True)

    # Example of what you might log, print, or warn from a worker
    logging.info(f"Worker {worker_id} started.")


def get_worker_metadata_subset(metadata: DataFrame) -> DataFrame:
    """Get the metadata subset for the current worker."""
    worker_info = get_worker_info()
    if worker_info is None:
        return metadata
    metadata_subset = metadata[worker_info.id::worker_info.num_workers]
    logger.info(f"Worker {worker_info.id} processing {len(metadata_subset)} beatmaps.")
    return metadata_subset


class MmrsDataset(IterableDataset):
    def __init__(
            self,
            args: DataSetConfig,
            processor: CM3PProcessor,
            subset_ids: Optional[list[int]] = None,
            test: bool = False,
    ):
        """Manage and process MMRS dataset.

        Attributes:
            args: Data loading arguments.
            processor: The data processor.
            subset_ids: List of beatmap set IDs to process. Overrides track index range.
            test: Whether to load the test dataset.
        """
        super().__init__()
        self.args = args
        self.processor = processor
        self.test = test
        self.paths = [Path(p) for p in (args.test_dataset_paths if test else args.train_dataset_paths)]
        self.start = args.test_dataset_start if test else args.train_dataset_start
        self.end = args.test_dataset_end if test else args.train_dataset_end
        self.metadata = load_mmrs_metadata(self.paths)
        self.start = self.start or 0
        self.end = self.end or len(self.metadata.index.get_level_values(0).unique())
        self.subset_ids = subset_ids

    def _get_filtered_metadata(self):
        """Get the subset IDs for the dataset with all filtering applied."""
        return filter_mmrs_metadata(
            self.metadata,
            start=self.start,
            end=self.end,
            subset_ids=self.subset_ids,
            gamemodes=self.args.gamemodes,
            min_year=self.args.min_year,
            max_year=self.args.max_year,
            min_difficulty=self.args.min_difficulty,
            max_difficulty=self.args.max_difficulty,
        )

    def __iter__(self):
        filtered_metadata = self._get_filtered_metadata()
        filtered_metadata = get_worker_metadata_subset(filtered_metadata)

        if not self.test:
            subset_ids = filtered_metadata.index.get_level_values(0).unique().to_numpy()
            np.random.shuffle(subset_ids)
            filtered_metadata = filtered_metadata.loc[subset_ids]

        if self.args.cycle_length > 1:
            return InterleavingBeatmapDatasetIterable(
                filtered_metadata,
                self._iterable_factory,
                self.args.cycle_length,
                self.args.drop_last,
            )

        return self._iterable_factory(filtered_metadata).__iter__()

    def _iterable_factory(self, metadata: DataFrame) -> BeatmapDatasetIterable:
        return BeatmapDatasetIterable(
            metadata,
            self.args,
            self.processor,
            self.test,
        )


class InterleavingBeatmapDatasetIterable:
    __slots__ = ("workers", "cycle_length", "index", "drop_last")

    def __init__(
            self,
            metadata: DataFrame,
            iterable_factory: Callable,
            cycle_length: int,
            drop_last: bool = False,
    ):
        self.workers = [
            iterable_factory(metadata[i::cycle_length]).__iter__()
            for i in range(cycle_length)
        ]
        self.cycle_length = cycle_length
        self.index = 0
        self.drop_last = drop_last

    def __iter__(self) -> "InterleavingBeatmapDatasetIterable":
        return self

    def __next__(self) -> tuple[any, int]:
        num = len(self.workers)
        for _ in range(num):
            try:
                self.index = self.index % len(self.workers)
                item = self.workers[self.index].__next__()
                self.index += 1
                return item
            except StopIteration:
                if self.drop_last:
                    raise StopIteration
                self.workers.remove(self.workers[self.index])
        raise StopIteration


class BeatmapDatasetIterable:
    def __init__(
            self,
            metadata: DataFrame,
            args: DataSetConfig,
            processor: CM3PProcessor,
            test: bool,
    ):
        self.args = args
        self.metadata = metadata
        self.processor = processor
        self.test = test

        if self.args.labels == "masked_lm":
            # Precompute eligible random token IDs for masked LM replacement
            exclude_token_ids = torch.tensor(self.processor.beatmap_tokenizer.convert_tokens_to_ids([
                self.processor.beatmap_tokenizer.audio_token,
            ]))
            all_token_ids = torch.arange(self.processor.beatmap_tokenizer.vocab_size)
            exclude_mask = torch.zeros_like(all_token_ids, dtype=torch.bool)
            exclude_mask[exclude_token_ids] = True
            self.eligible_random_token_ids = all_token_ids[~exclude_mask]

    def _get_speed_augment(self):
        if self.test or random.random() >= self.args.dt_augment_prob:
            return 1.0

        mi, ma = self.args.dt_augment_range
        base = random.random()
        if self.args.dt_augment_sqrt:
            base = np.power(base, 0.5)
        return mi + (ma - mi) * base

    def _process_input_for_masked_lm(self, inputs):
        input_ids = inputs.input_ids
        to_predict_mask = torch.ones_like(input_ids, dtype=torch.bool)
        special_ids = self.processor.beatmap_tokenizer.all_special_ids
        for sid in special_ids:
            to_predict_mask &= input_ids != sid
        to_predict_mask &= torch.rand(input_ids.shape) < self.args.masked_lm_prob
        labels = input_ids.masked_fill(~to_predict_mask, -100)
        inputs["labels"] = labels

        # For each position to predict, mask the input ids with an 80% chance, replace with random token with 10% chance, or keep original with 10% chance
        split_bounds = np.cumsum(self.args.masked_lm_split)
        rand = torch.rand(input_ids.shape)
        masking_mask = (rand < split_bounds[0]) & to_predict_mask
        random_replacement_mask = (rand >= split_bounds[0]) & (rand < split_bounds[1]) & to_predict_mask

        input_ids.masked_fill_(masking_mask, self.processor.beatmap_tokenizer.mask_token_id)

        num_random_tokens = random_replacement_mask.sum().item()
        if num_random_tokens > 0:
            random_indices = torch.randint(0, self.eligible_random_token_ids.size(0), (num_random_tokens,))
            random_token_ids = self.eligible_random_token_ids[random_indices]
            input_ids[random_replacement_mask] = random_token_ids

    def __iter__(self):
        return self._get_next_tracks()

    def _get_next_tracks(self) -> dict:
        for beatmapset_id in self.metadata.index.get_level_values(0).unique():
            logger.info(f"Processing beatmap set ID: {beatmapset_id}")
            metadata = self.metadata.loc[beatmapset_id]
            first_beatmap_metadata = metadata.iloc[0]

            audio_cache = {}
            speed = self._get_speed_augment()
            track_path = Path(first_beatmap_metadata["Path"]) / "data" / first_beatmap_metadata["BeatmapSetFolder"]

            for i, beatmap_metadata in metadata.iterrows():
                audio_path = track_path / beatmap_metadata["AudioFile"]
                is_ranked = beatmap_metadata["Status"] == "ranked"

                if random.random() < 0.5:
                    # Replace with a random beatmap from the dataset to increase robustness
                    beatmap_metadata = self.metadata.sample(n=1).iloc[0]
                    is_ranked = False

                beatmap_path = Path(beatmap_metadata["Path"]) / "data" / beatmap_metadata["BeatmapSetFolder"] / beatmap_metadata["BeatmapFile"]

                for sample in self._get_next_beatmap(beatmap_path, audio_path, is_ranked, beatmap_metadata, speed, audio_cache):
                    yield sample

    def _get_next_beatmap(self, beatmap_path, audio_path, is_ranked, beatmap_metadata: Series, speed: float, audio_cache: dict) -> dict:
        audio_samples = None
        if self.args.include_audio:
            try:
                if audio_path in audio_cache:
                    audio_samples = audio_cache[audio_path]
                else:
                    logger.info(f"Loading audio file: {audio_path}")
                    audio_samples = load_audio_file(audio_path, self.args.sampling_rate, speed)
                    audio_cache[audio_path] = audio_samples
                    logger.info(f"Audio length: {len(audio_samples) / self.args.sampling_rate:.2f} seconds")
            except Exception as e:
                logger.warning(f"Failed to load audio file: {audio_path}")
                logger.warning(e)
                return

        try:
            results = self.processor(
                metadata=get_metadata(beatmap_metadata=beatmap_metadata, speed=speed) if self.args.include_metadata else None,
                beatmap=beatmap_path if self.args.include_beatmap else None,
                audio=audio_samples,
                audio_sampling_rate=self.args.sampling_rate,
                speed=speed,
                multiply_metadata=self.args.include_metadata,
                populate_metadata=self.args.include_metadata,
                metadata_dropout_prob=self.args.metadata_dropout_prob if not self.test else 0.0,
                metadata_variations=self.args.test_metadata_variations if self.test else self.args.train_metadata_variations,
                padding=PaddingStrategy.MAX_LENGTH,
                return_tensors="pt",
            )

            if self.args.labels == "masked_lm":
                self._process_input_for_masked_lm(results)
            elif self.args.labels == "ranked_classification":
                results["labels"] = torch.full((results['input_ids'].size(0),), is_ranked, dtype=torch.long)
        except Exception as e:
            logger.warning(f"Failed to process beatmap: {beatmap_path}")
            logger.warning(e)
            traceback.print_exc()
            return

        # Split the batch feature and yield each individual sample, so we can interleave and create varied batches
        batch_size = len(results["input_ids"])
        assert len(results["attention_mask"]) == batch_size
        assert len(results["input_features"]) == batch_size
        for i in range(batch_size):
            result = {key: results[key][i] for key in results}
            yield result
