from __future__ import annotations

import random
from pathlib import Path
from typing import Optional, Callable

import numpy as np
from pandas import Series, DataFrame
from torch.utils.data import IterableDataset
from transformers.utils import PaddingStrategy

from cm3p.processing_cm3p import CM3PProcessor, get_metadata
from config import DataSetConfig
from data_utils import load_mmrs_metadata, filter_mmrs_metadata, load_audio_file


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

        if not self.test:
            subset_ids = filtered_metadata.index.get_level_values(0).unique().to_numpy()
            np.random.shuffle(subset_ids)
            filtered_metadata = filtered_metadata.loc[subset_ids]

        if self.args.cycle_length > 1 and not self.test:
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
            iterable_factory(df).__iter__()
            for df in np.array_split(metadata, cycle_length)  # Causes swapaxes future warning
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

    def _get_speed_augment(self):
        if self.test or random.random() >= self.args.dt_augment_prob:
            return 1.0

        mi, ma = self.args.dt_augment_range
        base = random.random()
        if self.args.dt_augment_sqrt:
            base = np.power(base, 0.5)
        return mi + (ma - mi) * base

    def __iter__(self):
        return self._get_next_tracks()

    def _get_next_tracks(self) -> dict:
        for beatmapset_id in self.metadata.index.get_level_values(0).unique():
            metadata = self.metadata.loc[beatmapset_id]
            first_beatmap_metadata = metadata.iloc[0]

            audio_cache = {}
            speed = self._get_speed_augment()
            track_path = Path(first_beatmap_metadata["Path"]) / "data" / first_beatmap_metadata["BeatmapSetFolder"]

            for i, beatmap_metadata in metadata.iterrows():
                for sample in self._get_next_beatmap(track_path, beatmap_metadata, speed, audio_cache):
                    yield sample

    def _get_next_beatmap(self, track_path, beatmap_metadata: Series, speed: float, audio_cache: dict) -> dict:
        audio_path = track_path / beatmap_metadata["AudioFile"]
        beatmap_path = track_path / beatmap_metadata["BeatmapFile"]

        try:
            if audio_path in audio_cache:
                audio_samples = audio_cache[audio_path]
            else:
                audio_samples = load_audio_file(audio_path, self.args.sampling_rate, speed)
                audio_cache[audio_path] = audio_samples
        except Exception as e:
            print(f"Failed to load audio file: {audio_path}")
            print(e)
            return

        results = self.processor(
            metadata=get_metadata(beatmap_metadata=beatmap_metadata, speed=speed),
            beatmap=beatmap_path,
            audio=audio_samples,
            audio_sampling_rate=self.args.sampling_rate,
            speed=speed,
            multiply_metadata=True,
            populate_metadata=True,
            metadata_dropout_prob=self.args.metadata_dropout_prob,
            padding=PaddingStrategy.MAX_LENGTH,
            return_tensors="pt",
        )

        # Split the batch feature and yield each individual sample, so we can interleave and create varied batches
        batch_size = len(results["input_ids"])
        for i in range(batch_size):
            result = {key: results[key][i] for key in results}
            yield result
