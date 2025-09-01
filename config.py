from dataclasses import dataclass
from typing import Any, Optional

from hydra.core.config_store import ConfigStore


@dataclass
class AudioFeatureExtractorConfig:
    feature_size: int
    sampling_rate: int
    hop_length: int
    chunk_length: int
    n_fft: int
    padding_value: float
    dither: float
    return_attention_mask: bool


@dataclass
class BeatmapParserConfig:
    add_timing: bool
    add_snapping: bool
    add_timing_points: bool
    add_hitsounds: bool
    add_distances: bool
    add_positions: bool
    add_kiai: bool
    add_sv: bool
    add_mania_sv: bool
    mania_bpm_normalized_scroll_speed: bool
    position_split_axes: bool
    slider_version: int


@dataclass
class BeatmapTokenizerConfig:
    min_time: int
    max_time: int
    time_step: int
    max_distance: int
    distance_step: int
    position_range: tuple[int, int, int, int]
    position_step: int
    position_split_axes: bool


@dataclass
class MetadataTokenizerConfig:
    min_difficculty: float
    max_difficulty: float
    difficulty_step: float
    min_year: int
    max_year: int
    max_song_length: int
    song_length_step: int
    song_position_step: float
    global_sv_step: float
    hold_note_ratio_step: float
    scroll_speed_ratio_step: float
    modes: Optional[dict[int, str]] = None
    mappers: Optional[dict[int, str]] = None
    tags: Optional[dict[int, dict]] = None


@dataclass
class ProcessorConfig:
    audio_feature_extractor: AudioFeatureExtractorConfig
    beatmap_parser: BeatmapParserConfig
    beatmap_tokenizer: BeatmapTokenizerConfig
    metadata_tokenizer: MetadataTokenizerConfig
    default_kwargs: Optional[dict]


@dataclass
class DataSetConfig:
    train_dataset_paths: list[str]
    train_dataset_start: int
    train_dataset_end: int
    test_dataset_paths: list[str]
    test_dataset_start: int
    test_dataset_end: int
    cycle_length: int
    drop_last: bool
    gamemodes: list[int]
    min_year: Optional[int]
    max_year: Optional[int]
    min_difficulty: float
    max_difficulty: float
    metadata_dropout_prob: float
    dt_augment_prob: float
    dt_augment_range: list[float]
    dt_augment_sqrt: bool
    sampling_rate: int


@dataclass
class TrainConfig:
    freeze_beatmap_model: bool
    freeze_metadata_model: bool

    training: dict
    processor: ProcessorConfig
    dataset: DataSetConfig
    model: dict


cs = ConfigStore.instance()
cs.store(group="train", name="base", node=TrainConfig)

