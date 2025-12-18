from __future__ import annotations

import logging
import zipfile
from pathlib import Path

import pandas as pd
from pandas import DataFrame
from torch.utils.data import IterableDataset
from transformers.utils import PaddingStrategy

from cm3p.processing_cm3p import CM3PProcessor, get_metadata
from utils.mmrs_dataset import get_worker_metadata_subset

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ['Artist', 'ArtistUnicode', 'Creator', 'FavouriteCount', 'BeatmapSetId', 'Nsfw', 'Offset',
                    'BeatmapSetPlayCount', 'Source', 'BeatmapSetStatus', 'Spotlight', 'Title', 'TitleUnicode',
                    'BeatmapSetUserId', 'Video', 'Description', 'GenreId', 'GenreName', 'LanguageId', 'LanguageName',
                    'PackTags', 'Ratings', 'DownloadDisabled', 'BeatmapSetBpm', 'CanBeHyped', 'DiscussionLocked',
                    'BeatmapSetIsScoreable', 'BeatmapSetLastUpdated', 'BeatmapSetRanked', 'RankedDate', 'Storyboard',
                    'SubmittedDate', 'Tags', 'DifficultyRating', 'Id', 'Mode', 'Status', 'TotalLength', 'UserId',
                    'Version', 'Checksum', 'MaxCombo', 'Accuracy', 'Ar', 'Bpm', 'CountCircles', 'CountSliders',
                    'CountSpinners', 'Cs', 'Drain', 'HitLength', 'IsScoreable', 'LastUpdated', 'ModeInt', 'PassCount',
                    'PlayCount', 'Ranked', 'Owners', 'TopTagIds', 'TopTagCounts', 'StarRating', 'OmdbTags', 'AudioFile',
                    'BeatmapSetFolder', 'BeatmapFile']


def _collect_paths(paths: list[str]) -> list[Path]:
    collected: list[Path] = []
    for p in paths:
        pth = Path(p)
        if pth.is_file():
            if pth.suffix.lower() in {'.osu', '.osz'}:
                collected.append(pth)
        elif pth.is_dir():
            for fp in pth.rglob('*'):
                if fp.is_file() and fp.suffix.lower() in {'.osu', '.osz'}:
                    collected.append(fp)
    return collected


def _extract_osz(osz_path: Path, extract_root: Path) -> Path:
    target_dir = extract_root / osz_path.stem
    if target_dir.exists():
        return target_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(osz_path, 'r') as zf:
        zf.extractall(target_dir)
    return target_dir


def _kv(section_lines: list[str], key: str) -> str | None:
    for line in section_lines:
        if line.strip().startswith(f"{key}:"):
            return line.split(':', 1)[1].strip()
    return None


def _parse_osu_file(osu_path: Path) -> dict:
    # Minimal parsing: read file, split sections, extract key-value fields
    data: dict = {col: pd.NA for col in REQUIRED_COLUMNS}
    data['BeatmapSetFolder'] = osu_path.parent.name
    data['BeatmapFile'] = osu_path.name
    data['Path'] = str(osu_path.parent.parent)

    try:
        with open(osu_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
    except Exception:
        lines = []

    # Split sections
    sections: dict[str, list[str]] = {}
    current = None
    for line in lines:
        line = line.rstrip('\n')
        if line.startswith('[') and line.endswith(']'):
            current = line.strip('[]')
            sections[current] = []
        elif current is not None:
            sections[current].append(line)

    # General
    general = sections.get('General', [])
    data['AudioFile'] = _kv(general, 'AudioFilename') or pd.NA

    # Metadata
    meta = sections.get('Metadata', [])
    data['Title'] = _kv(meta, 'Title') or pd.NA
    data['TitleUnicode'] = _kv(meta, 'TitleUnicode') or pd.NA
    data['Artist'] = _kv(meta, 'Artist') or pd.NA
    data['ArtistUnicode'] = _kv(meta, 'ArtistUnicode') or pd.NA
    data['Creator'] = _kv(meta, 'Creator') or pd.NA
    data['Version'] = _kv(meta, 'Version') or pd.NA

    # Difficulty
    diff = sections.get('Difficulty', [])
    cs = _kv(diff, 'CircleSize')
    ar = _kv(diff, 'ApproachRate')
    hp = _kv(diff, 'HPDrainRate')
    data['Cs'] = float(cs) if cs is not None else pd.NA
    data['Ar'] = float(ar) if ar is not None else pd.NA
    data['Drain'] = float(hp) if hp is not None else pd.NA

    # TimingPoints to estimate BPM (first uninherited timing point)
    bpm = pd.NA
    for line in sections.get('TimingPoints', []):
        parts = [p.strip() for p in line.split(',')]
        if len(parts) >= 2:
            try:
                ms_per_beat = float(parts[1])
                if ms_per_beat > 0:
                    bpm = 60000.0 / ms_per_beat
                    break
            except Exception:
                pass
    data['Bpm'] = bpm

    # HitObjects length and counts
    first_t = None
    last_t = None
    count_circles = 0
    count_sliders = 0
    count_spinners = 0
    for line in sections.get('HitObjects', []):
        parts = [p.strip() for p in line.split(',')]
        if len(parts) >= 5:
            try:
                t = int(parts[2])
                type_flags = int(parts[3])
                first_t = t if first_t is None else min(first_t, t)
                last_t = t if last_t is None else max(last_t, t)
                # type flags: 1 circle, 2 slider, 8 spinner
                if type_flags & 1:
                    count_circles += 1
                if type_flags & 2:
                    count_sliders += 1
                if type_flags & 8:
                    count_spinners += 1
            except Exception:
                pass
    total_len = ((last_t - first_t) / 1000.0) if (first_t is not None and last_t is not None) else 0.0
    data['TotalLength'] = float(total_len)
    data['HitLength'] = float(total_len)
    data['CountCircles'] = int(count_circles)
    data['CountSliders'] = int(count_sliders)
    data['CountSpinners'] = int(count_spinners)

    # ModeInt from file version header if present; fallback to 0 (osu)
    file_header = sections.get('', [])
    mode_int = 0
    for line in lines:
        if line.startswith('Mode:'):
            try:
                mode_int = int(line.split(':', 1)[1].strip())
                break
            except Exception:
                pass
    data['ModeInt'] = mode_int

    # Defaults / placeholders
    data['Id'] = abs(hash(osu_path)) % (2 ** 31)
    data['BeatmapSetId'] = abs(hash(osu_path.parent)) % (2 ** 31)
    data['Status'] = pd.NA
    data['DifficultyRating'] = pd.NA
    data['UserId'] = pd.NA
    data['SubmittedDate'] = pd.Timestamp.utcnow()
    data['Mode'] = pd.NA
    data['Ranked'] = pd.NA
    data['TopTagIds'] = []
    data['TopTagCounts'] = []
    data['StarRating'] = []
    return data


def build_metadata_dataframe(paths: list[str]) -> DataFrame:
    files = _collect_paths(paths)
    extract_root = Path('./tmp_osz_extract')
    extract_root.mkdir(exist_ok=True)

    rows: list[dict] = []
    for p in files:
        if p.suffix.lower() == '.osu':
            rows.append(_parse_osu_file(p))
        elif p.suffix.lower() == '.osz':
            folder = _extract_osz(p, extract_root)
            for osu in folder.rglob('*.osu'):
                rows.append(_parse_osu_file(osu))
    df = pd.DataFrame(rows)
    # Ensure types
    if 'Id' in df.columns:
        df['Id'] = df['Id'].astype('int64')
    if 'BeatmapSetId' in df.columns:
        df['BeatmapSetId'] = df['BeatmapSetId'].astype('int64')
    # Set MultiIndex similar to dataset
    df.set_index(['BeatmapSetId', 'Id'], inplace=True)
    df.sort_index(inplace=True)
    return df


class BeatmapFilesDataset(IterableDataset):
    def __init__(
            self,
            beatmap_paths: list[str],
            processor: CM3PProcessor,
            sampling_rate: int = 16000,
            include_audio: bool = True,
            include_beatmap: bool = True,
            include_metadata: bool = True,
    ):
        super().__init__()
        self.beatmap_paths = beatmap_paths
        self.metadata = build_metadata_dataframe(beatmap_paths)
        self.processor = processor
        self.sampling_rate = sampling_rate
        self.include_audio = include_audio
        self.include_beatmap = include_beatmap
        self.include_metadata = include_metadata

    def __iter__(self):
        filtered_metadata = get_worker_metadata_subset(self.metadata)
        return self._iter(filtered_metadata)

    def _iter(self, metadata: DataFrame):
        for beatmapset_id in metadata.index.get_level_values(0).unique():
            subset = metadata.loc[beatmapset_id]
            first_beatmap_metadata = subset.iloc[0]
            track_path = Path(first_beatmap_metadata.get("Path", ".")) / first_beatmap_metadata.get("BeatmapSetFolder",
                                                                                                    "")

            audio_cache = {}
            for _, beatmap_metadata in subset.iterrows():
                audio_samples = None
                audio_filename = beatmap_metadata.get("AudioFile", None)
                if self.include_audio and audio_filename:
                    audio_path = track_path / audio_filename
                    try:
                        if audio_path in audio_cache:
                            audio_samples = audio_cache[audio_path]
                        else:
                            from utils.data_utils import load_audio_file
                            audio_samples = load_audio_file(audio_path, self.sampling_rate, 1.0)
                            audio_cache[audio_path] = audio_samples
                    except Exception as e:
                        logger.warning(f"Failed to load audio file: {audio_path}")
                        logger.warning(e)
                        continue

                beatmap_path = track_path / beatmap_metadata.get("BeatmapFile", "")
                try:
                    results = self.processor(
                        metadata=get_metadata(beatmap_metadata=beatmap_metadata,
                                              speed=1.0) if self.include_metadata else None,
                        beatmap=beatmap_path if self.include_beatmap else None,
                        audio=audio_samples,
                        audio_sampling_rate=self.sampling_rate,
                        speed=1.0,
                        multiply_metadata=self.include_metadata,
                        populate_metadata=self.include_metadata,
                        metadata_dropout_prob=0.0,
                        metadata_variations=1,
                        padding=PaddingStrategy.MAX_LENGTH,
                        return_tensors="pt",
                    )
                except Exception as e:
                    logger.warning(f"Failed to process beatmap: {beatmap_path}")
                    logger.warning(e)
                    continue

                # Yield individual samples
                batch_size = len(results["input_ids"])
                for i in range(batch_size):
                    item = {k: results[k][i] for k in results}
                    item['beatmap_id'] = beatmap_metadata.name
                    yield item
