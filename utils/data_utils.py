from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import soxr
from pandas import DataFrame, Series

import numpy.typing as npt
from slider import Beatmap, HoldNote
from transformers.audio_utils import load_audio

from cm3p.tokenization_cm3p import CM3PMetadata


def load_audio_file(file: str, sampling_rate: int, speed: float = 1.0) -> npt.NDArray:
    """Load an audio file as a numpy time-series array

    The signals are resampled, converted to mono channel, and normalized.

    Args:
        file: Path to audio file.
        sampling_rate: Sample rate of the audio.
        speed: Speed multiplier for the audio.

    Returns:
        audio: Audio time series.
    """
    audio = load_audio(str(file), sampling_rate=int(sampling_rate // speed))
    audio = np.asarray(audio)

    # Convert to mono if needed
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    return audio


def load_mmrs_metadata(path: Union[str, list[str]]) -> DataFrame:
    # Loads the metadata parquet from the dataset path(s)
    if isinstance(path, str):
        path = [path]

    df_list = []
    for p in path:
        df = pd.read_parquet(Path(p) / "metadata.parquet")
        df["BeatmapIdx"] = df.index
        df["Path"] = str(p)  # Add a column for the dataset path
        df.set_index(["BeatmapSetId", "Id"], inplace=True)
        df_list.append(df)

    df = pd.concat(df_list, ignore_index=False)
    df.sort_index(inplace=True)
    return df


def filter_mmrs_metadata(
        df: DataFrame,
        *,
        start: Optional[int] = None,
        end: Optional[int] = None,
        subset_ids: Optional[list[int]] = None,
        gamemodes: Optional[list[int]] = None,
        min_year: Optional[int] = None,
        max_year: Optional[int] = None,
        min_difficulty: Optional[float] = None,
        max_difficulty: Optional[float] = None,
) -> DataFrame:
    """Filter the MMRs metadata DataFrame based on the given criteria.

    Args:
        df: DataFrame containing the metadata.
        start: Start split index.
        end: End split index.
        subset_ids: List of beatmap IDs to filter by.
        gamemodes: List of gamemodes to filter by.
        min_year: Minimum year to filter by.
        max_year: Maximum year to filter by.
        min_difficulty: Minimum difficulty star rating to filter by.
        max_difficulty: Maximum difficulty star rating to filter by.

    Returns:
        Filtered DataFrame.
    """
    if start is not None and end is not None:
        first_level_labels = df.index.get_level_values(0).unique()
        start_label = first_level_labels[start]
        end_label = first_level_labels[end - 1]
        df = df.loc[start_label:end_label]

    if subset_ids is not None:
        df = df.loc[subset_ids]

    if gamemodes is not None:
        df = df[df["ModeInt"].isin(gamemodes)]

    if min_year is not None:
        df = df[df["SubmittedDate"] >= datetime(min_year, 1, 1)]

    if max_year is not None:
        df = df[df["SubmittedDate"] < datetime(max_year + 1, 1, 1)]

    if min_difficulty is not None:
        df = df[df["DifficultyRating"] >= min_difficulty]

    if max_difficulty is not None:
        df = df[df["DifficultyRating"] <= max_difficulty]

    return df

