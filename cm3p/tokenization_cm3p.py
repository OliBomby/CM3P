import json
from typing import Optional, Union, TypedDict

import numpy as np
from transformers import PreTrainedTokenizer, BatchEncoding, AutoTokenizer
from transformers.tokenization_utils_base import TruncationStrategy
from transformers.utils import PaddingStrategy

from cm3p import CM3PBeatmapConfig, CM3PMetadataConfig
from cm3p.parsing_cm3p import Group, EventType


class CM3PBeatmapTokenizer(PreTrainedTokenizer):
    model_input_names: list[str] = ["input_ids", "attention_mask"]
    vocab_files_names: dict[str, str] = {"vocab_file": "vocab.json"}

    def __init__(
            self,
            vocab_file: Optional[str] = None,
            min_time: int = 0,
            max_time: int = 30000,
            time_step: int = 10,
            max_distance: int = 640,
            distance_step: int = 4,
            position_range: tuple[int, int, int, int] = (-256, 768, -256, 640),
            position_step: int = 4,
            position_split_axes: bool = True,
            **kwargs,
    ):
        self.min_time = min_time
        self.max_time = max_time
        self.time_step = time_step
        self.max_distance = max_distance
        self.distance_step = distance_step
        self.position_range = position_range
        self.position_step = position_step
        self.position_split_axes = position_split_axes

        self.audio_bos_token = "[AUDIO_BOS]"
        self.audio_eos_token = "[AUDIO_EOS]"
        self.audio_token = "[AUDIO]"

        if vocab_file is None:
            self.vocab = self._build_vocab_from_config()
        else:
            with open(vocab_file, 'r', encoding='utf-8') as f:
                self.vocab = json.load(f)

        self.ids_to_tokens = {i: t for t, i in self.vocab.items()}
        super().__init__(
            bos_token=kwargs.pop("bos_token", "[BOS]"),
            eos_token=kwargs.pop("eos_token", "[EOS]"),
            unk_token=kwargs.pop("unk_token", "[UNK]"),
            sep_token=kwargs.pop("sep_token", "[SEP]"),
            pad_token=kwargs.pop("pad_token", "[PAD]"),
            cls_token=kwargs.pop("cls_token", "[CLS]"),
            mask_token=kwargs.pop("mask_token", "[MASK]"),
            additional_special_tokens=kwargs.pop("additional_special_tokens", [
                self.audio_bos_token,
                self.audio_eos_token,
                self.audio_token,
            ]),
            min_time=min_time,
            max_time=max_time,
            time_step=time_step,
            max_distance=max_distance,
            distance_step=distance_step,
            position_range=position_range,
            position_step=position_step,
            position_split_axes=position_split_axes,
            **kwargs
        )

    def _build_vocab_from_config(self):
        vocab = []

        for event_type in EventType:
            vocab.append(f"[{event_type.value.upper()}]")

        for time in np.arange(self.min_time, self.max_time + 1e-5, self.time_step):
            vocab.append(f"[TIME_SHIFT_{int(time)}]")

        for snapping in range(0, 17):
            vocab.append(f"[SNAPPING_{snapping}]")

        for distance in range(0, self.max_distance + 1):
            vocab.append(f"[DISTANCE_{distance}]")

        if self.position_split_axes:
            for x in np.arange(self.position_range[0], self.position_range[1] + 1e-5, self.position_step):
                vocab.append(f"[POS_X_{int(x)}]")
            for y in np.arange(self.position_range[2], self.position_range[3] + 1e-5, self.position_step):
                vocab.append(f"[POS_Y_{int(y)}]")
        else:
            for x in np.arange(self.position_range[0], self.position_range[1] + 1e-5, self.position_step):
                for y in np.arange(self.position_range[2], self.position_range[3] + 1e-5, self.position_step):
                    vocab.append(f"[POS_{int(x)}_{int(y)}]")

        for mania_column in range(1, 19):
            vocab.append(f"[MANIA_COLUMN_{mania_column}]")

        for scroll_speed in np.arange(0.0, 10.0 + 1e-5, 0.01):
            vocab.append(f"[SCROLL_SPEED_{scroll_speed:.2f}]")

        vocab.append("[NEW_COMBO]")

        for hitsound in range(8):
            for sampleset in range(1, 4):
                for additions in range(1, 4):
                    vocab.append(f"[HITSOUND_{(hitsound << 1)}_{sampleset}_{additions}]")

        for volume in range(101):
            vocab.append(f"[VOLUME_{volume}]")

        return {token: idx for idx, token in enumerate(vocab)}

    def _tokenize_time_shift(self, time: int):
        time = np.clip(time, self.min_time, self.max_time)
        time = round(time / self.time_step) * self.time_step
        return f"[TIME_SHIFT_{int(time)}]"

    def _tokenize_distance(self, distance: int):
        distance = np.clip(distance, 0, self.max_distance)
        distance = round(distance / self.distance_step) * self.distance_step
        return f"[DISTANCE_{distance}]"

    def _tokenize_position(self, pos_x: int, pos_y: int):
        pos_x = np.clip(pos_x, self.position_range[0], self.position_range[1])
        pos_y = np.clip(pos_y, self.position_range[2], self.position_range[3])
        pos_x = round(pos_x / self.position_step) * self.position_step
        pos_y = round(pos_y / self.position_step) * self.position_step

        if self.position_split_axes:
            yield f"[POS_X_{int(pos_x)}]"
            yield f"[POS_Y_{int(pos_y)}]"
        else:
            yield f"[POS_{int(pos_x)}_{int(pos_y)}]"

    def _tokenize_mania_column(self, mania_column: int):
        mania_column = np.clip(mania_column, 1, 18)
        return f"[MANIA_COLUMN_{mania_column}]"

    def _tokenize_scroll_speed(self, scroll_speed: float):
        scroll_speed = np.clip(scroll_speed, 0.0, 10.0)
        scroll_speed = round(scroll_speed / 0.01) * 0.01
        return f"[SCROLL_SPEED_{scroll_speed:.2f}]"

    def _tokenize_hitsound(self, hitsound: int, sampleset: int, addition: int):
        hitsound = np.clip(hitsound >> 1, 0, 7) << 1
        sampleset = np.clip(sampleset, 1, 3)
        addition = np.clip(addition, 1, 3)
        return f"[HITSOUND_{hitsound}_{sampleset}_{addition}]"

    def _tokenize_groups(
            self,
            groups: list[Group],
            window_start_ms: Optional[int] = None,
            **kwargs
    ):
        window_start_ms = window_start_ms or 0
        tokens = [self.bos_token]

        for group in groups:
            tokens.append(f"[{group.event_type.value.upper()}]")
            if group.has_time:
                tokens.append(self._tokenize_time_shift(group.time - window_start_ms))
                if group.snapping is not None:
                    tokens.append(f"[SNAPPING_{group.snapping}]")
            if group.distance is not None:
                tokens.append(self._tokenize_distance(group.distance))
            if group.x is not None and group.y is not None:
                tokens.extend(self._tokenize_position(group.x, group.y))
            if group.mania_column is not None:
                tokens.append(self._tokenize_mania_column(group.mania_column))
            if group.new_combo:
                tokens.append("[NEW_COMBO]")
            if group.scroll_speed is not None:
                tokens.append(self._tokenize_scroll_speed(group.scroll_speed))
            for h, s, a, v, in zip(
                    group.hitsounds,
                    group.samplesets,
                    group.additions,
                    group.volumes,
            ):
                tokens.append(self._tokenize_hitsound(h, s, a))
                tokens.append(f"[VOLUME_{v}]")

        tokens.append(self.eos_token)
        return tokens

    def _encode_single(
            self,
            groups: Optional[Union[list[Group]]] = None,
            window_start_ms: Optional[int] = None,
            num_audio_tokens: Optional[int] = None,
    ):
        token_strings = self._tokenize_groups(groups, window_start_ms=window_start_ms)
        token_ids = self.convert_tokens_to_ids(token_strings)

        if num_audio_tokens is not None and num_audio_tokens > 0:
            audio_tokens = [self.audio_bos_token] + [self.audio_token] * num_audio_tokens + [self.audio_eos_token]
            token_ids = self.convert_tokens_to_ids(audio_tokens) + token_ids

        return token_ids

    def __call__(
            self,
            groups: Optional[Union[list[Group], list[list[Group]]]] = None,
            window_start_ms: Optional[Union[int, list[int]]] = None,
            num_audio_tokens: Optional[Union[int, list[int]]] = None,
            padding: PaddingStrategy = PaddingStrategy.LONGEST,
            truncation: TruncationStrategy = TruncationStrategy.LONGEST_FIRST,
            **kwargs
    ) -> BatchEncoding:
        if len(groups) == 0:
            raise ValueError("Input groups list is empty.")

        if isinstance(groups, list) and all(isinstance(g, Group) for g in groups):
            token_ids = self._encode_single(
                groups=groups,
                window_start_ms=window_start_ms,
                num_audio_tokens=num_audio_tokens,
            )
            encoding = self.prepare_for_model(
                token_ids,
                padding=padding,
                truncation=truncation,
                **kwargs,
            )
        elif isinstance(groups, list):
            if num_audio_tokens is None:
                num_audio_tokens = [None] * len(groups)

            if window_start_ms is None:
                window_start_ms = [None] * len(groups)

            if len(groups) != len(num_audio_tokens):
                raise ValueError("Number of num_audio_tokens inputs must match the number of sequences.")

            if len(window_start_ms) != len(groups):
                raise ValueError("Number of window start times must match the number of sequences.")

            all_token_ids = []
            for g, w, a in zip(groups, window_start_ms, num_audio_tokens):
                token_ids = self._encode_single(
                    groups=g,
                    window_start_ms=w,
                    num_audio_tokens=a,
                )
                all_token_ids.append((token_ids, None))

            encoding = self._batch_prepare_for_model(
                all_token_ids,
                padding_strategy=PaddingStrategy(padding),
                truncation_strategy=TruncationStrategy(truncation),
                **kwargs,
            )
        else:
            raise ValueError("Input must be a list of Group objects or a single Group object.")

        return encoding

    @property
    def vocab_size(self):
        return len(self.vocab) + len(self._added_tokens_encoder)

    def get_vocab(self):
        return self.vocab | self._added_tokens_encoder

    def _convert_token_to_id(self, token):
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index):
        return self.ids_to_tokens.get(index, self.unk_token)

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> tuple[str]:
        if not save_directory:
            raise ValueError("The save_directory must be specified.")

        vocab_file = f"{save_directory}/{filename_prefix or ""}vocab.json"
        with open(vocab_file, 'w', encoding='utf-8') as f:
            json.dump(self.vocab, f, ensure_ascii=False)

        return (vocab_file,)


class CM3PMetadata(TypedDict, total=False):
    """
    Metadata fields for a beatmap.

    difficulty: Star rating, unitless (osu! difficulty)
    year: Year of beatmap creation (YYYY)
    mode: Game mode ID or name (e.g., "osu", "mania")
    mapper: Beatmap creator's ID or username
    cs: Circle size (osu!std), unitless
    hitsounded: Whether the beatmap is hitsounded (True/False)
    song_length: Song length in seconds
    song_position: Relative position in song [0.0-1.0], unitless
    global_sv: Global scroll velocity (osu!mania), multiplier
    mania_keycount: Number of keys in osu!mania [1-18]
    hold_note_ratio: Ratio of hold notes [0.0-1.0], unitless
    scroll_speed_ratio: Ratio of scroll speed changes [0.0-1.0], unitless
    tags: List of beatmap tag IDs or names
    """
    difficulty: float  # Star rating, unitless (osu! difficulty)
    year: int  # Year of beatmap creation (YYYY)
    mode: Union[int, str]  # Game mode ID or name (e.g., "osu", "mania")
    status: Union[int, str]  # Beatmap status (e.g., "ranked", "approved", "loved", "pending", "graveyard")
    mapper: Union[int, str]  # Beatmap creator's ID or username
    cs: float  # Circle size (osu!std), unitless
    hitsounded: bool  # Whether the beatmap is hitsounded (True/False)
    song_length: float  # Song length in seconds
    song_position: float  # Relative position in song [0.0-1.0], unitless
    global_sv: float  # Global scroll velocity (osu!mania), multiplier
    mania_keycount: int  # Number of keys in osu!mania [1-18]
    hold_note_ratio: float  # Ratio of hold notes [0.0-1.0], unitless
    scroll_speed_ratio: float  # Ratio of scroll speed changes [0.0-1.0], unitless
    tags: list[Union[int, str]]  # List of beatmap tag IDs or names


def merge_metadata_dicts(m1, m2):
    if m1 is None:
        return m2
    if m2 is None:
        return m1
    merged = {}
    for key in CM3PMetadata.__annotations__.keys():
        v1 = m1.get(key, None)
        v2 = m2.get(key, None)
        merged[key] = v2 if v1 is None else v1
    return CM3PMetadata(**merged)


class CM3PMetadataTokenizer(PreTrainedTokenizer):
    model_input_names: list[str] = ["input_ids", "attention_mask"]
    vocab_files_names: dict[str, str] = {"vocab_file": "vocab.json"}

    def __init__(
            self,
            vocab_file: Optional[str] = None,
            modes: Optional[dict[int, str]] = None,
            statuses: Optional[dict[int, str]] = None,
            mappers: Optional[dict[int, str]] = None,
            tags: Optional[dict[int, dict]] = None,
            min_difficculty: float = 0.0,
            max_difficulty: float = 14.0,
            difficulty_step: float = 0.1,
            min_year: int = 2000,
            max_year: int = 2023,
            max_song_length: int = 600,
            song_length_step: int = 10,
            song_position_step: float = 0.01,
            global_sv_step: float = 0.01,
            hold_note_ratio_step: float = 0.1,
            scroll_speed_ratio_step: float = 0.1,
            **kwargs,
    ):
        self.min_difficulty = min_difficculty
        self.max_difficulty = max_difficulty
        self.difficulty_step = difficulty_step
        self.min_year = min_year
        self.max_year = max_year
        self.max_song_length = max_song_length
        self.song_length_step = song_length_step
        self.song_position_step = song_position_step
        self.global_sv_step = global_sv_step
        self.hold_note_ratio_step = hold_note_ratio_step
        self.scroll_speed_ratio_step = scroll_speed_ratio_step

        self.difficulty_unk_token = "[DIFFICULTY_UNK]"
        self.year_unk_token = "[YEAR_UNK]"
        self.mode_unk_token = "[MODE_UNK]"
        self.status_unk_token = "[STATUS_UNK]"
        self.mapper_unk_token = "[MAPPER_UNK]"
        self.cs_unk_token = "[CS_UNK]"
        self.hitsounded_unk_token = "[HITSOUNDED_UNK]"
        self.song_length_unk_token = "[SONG_LENGTH_UNK]"
        self.song_position_unk_token = "[SONG_POSITION_UNK]"
        self.global_sv_unk_token = "[GLOBAL_SV_UNK]"
        self.mania_keycount_unk_token = "[MANIA_KEYCOUNT_UNK]"
        self.hold_note_ratio_unk_token = "[HOLD_NOTE_RATIO_UNK]"
        self.scroll_speed_ratio_unk_token = "[SCROLL_SPEED_RATIO_UNK]"
        self.tag_unk_token = "[TAG_UNK]"

        self.modes = modes or {}
        self.statuses = statuses or {}
        self.mappers = mappers or {}
        self.tags = tags or {}
        self.mode_names_to_ids = {v: k for k, v in self.modes.items()}
        self.mode_ids_to_names = self.modes
        self.status_names_to_ids = {v: k for k, v in self.statuses.items()}
        self.status_ids_to_names = self.statuses
        self.mapper_names_to_ids = {v: k for k, v in self.mappers.items()}
        self.mapper_ids_to_names = self.mappers
        self.tag_names_to_ids = {v['name']: k for k, v in self.tags.items()}
        self.tag_ids_to_names = {k: v['name'] for k, v in self.tags.items()}

        if vocab_file is None:
            self.vocab = self._build_vocab_from_config()
        else:
            with open(vocab_file, 'r', encoding='utf-8') as f:
                self.vocab = json.load(f)

        self.ids_to_tokens = {i: t for t, i in self.vocab.items()}

        super().__init__(
            bos_token=kwargs.pop("bos_token", "[BOS]"),
            eos_token=kwargs.pop("eos_token", "[EOS]"),
            pad_token=kwargs.pop("pad_token", "[PAD]"),
            additional_special_tokens=kwargs.pop("additional_special_tokens", [
                self.difficulty_unk_token,
                self.year_unk_token,
                self.mode_unk_token,
                self.status_unk_token,
                self.mapper_unk_token,
                self.cs_unk_token,
                self.hitsounded_unk_token,
                self.song_length_unk_token,
                self.song_position_unk_token,
                self.global_sv_unk_token,
                self.mania_keycount_unk_token,
                self.hold_note_ratio_unk_token,
                self.scroll_speed_ratio_unk_token,
                self.tag_unk_token,
            ]),
            modes=modes,
            statuses=statuses,
            mappers=mappers,
            tags=tags,
            min_difficculty=min_difficculty,
            max_difficulty=max_difficulty,
            difficulty_step=difficulty_step,
            min_year=min_year,
            max_year=max_year,
            max_song_length=max_song_length,
            song_length_step=song_length_step,
            song_position_step=song_position_step,
            global_sv_step=global_sv_step,
            hold_note_ratio_step=hold_note_ratio_step,
            scroll_speed_ratio_step=scroll_speed_ratio_step,
            **kwargs
        )

    def _build_vocab_from_config(self):
        vocab = []

        for difficulty in np.arange(self.min_difficulty, self.max_difficulty + 1e-5, self.difficulty_step):
            vocab.append(f"[DIFFICULTY_{difficulty:.1f}]")

        for year in range(self.min_year, self.max_year + 1):
            vocab.append(f"[YEAR_{year}]")

        for mode in self.mode_ids_to_names.values():
            vocab.append(f"[MODE_{str(mode)}]")

        for status in self.status_ids_to_names.values():
            vocab.append(f"[STATUS_{str(status)}]")

        for mapper in self.mapper_ids_to_names.keys():
            vocab.append(f"[MAPPER_{str(mapper)}]")

        for cs in np.arange(0.0, 10.0 + 1e-5, 0.1):
            vocab.append(f"[CS_{cs:.1f}]")

        for hitsounded in [True, False]:
            vocab.append(f"[HITSOUNDED_{str(hitsounded).upper()}]")

        for song_length in np.arange(0, self.max_song_length + 1e-5, self.song_length_step):
            vocab.append(f"[SONG_LENGTH_{int(song_length)}]")

        for song_position in np.arange(0.0, 1.0 + 1e-5, self.song_position_step):
            vocab.append(f"[SONG_POSITION_{song_position:.2f}]")

        for global_sv in np.arange(0.4, 3.6 + 1e-5, self.global_sv_step):
            vocab.append(f"[GLOBAL_SV_{global_sv:.2f}]")

        for mania_keycount in range(1, 19):
            vocab.append(f"[MANIA_KEYCOUNT_{mania_keycount}]")

        for hold_note_ratio in np.arange(0.0, 1.0 + 1e-5, self.hold_note_ratio_step):
            vocab.append(f"[HOLD_NOTE_RATIO_{hold_note_ratio:.1f}]")

        for scroll_speed_ratio in np.arange(0.0, 1.0 + 1e-5, self.scroll_speed_ratio_step):
            vocab.append(f"[SCROLL_SPEED_RATIO_{scroll_speed_ratio:.1f}]")

        for tag in self.tag_ids_to_names.values():
            vocab.append(f"[TAG_{tag}]")

        return {token: idx for idx, token in enumerate(vocab)}

    def _tokenize_difficulty(self, metadata: CM3PMetadata):
        difficulty = metadata.get('difficulty', None)
        if difficulty is None:
            return self.difficulty_unk_token
        difficulty = np.clip(difficulty, self.min_difficulty, self.max_difficulty)
        difficulty = round(difficulty / self.difficulty_step) * self.difficulty_step
        return f"[DIFFICULTY_{difficulty:.1f}]"

    def _tokenize_year(self, metadata: CM3PMetadata):
        year = metadata.get('year', None)
        if year is None:
            return self.year_unk_token
        year = np.clip(year, self.min_year, self.max_year)
        return f"[YEAR_{year}]"

    def _tokenize_mode(self, metadata: CM3PMetadata):
        mode_str = metadata.get('mode', None)
        if isinstance(mode_str, int):
            mode_str = self.mode_ids_to_names.get(mode_str, None)
        if mode_str is None or mode_str not in self.mode_names_to_ids:
            return self.mode_unk_token
        return f"[MODE_{str(mode_str)}]"

    def _tokenize_status(self, metadata: CM3PMetadata):
        status_str = metadata.get('status', None)
        if isinstance(status_str, int):
            status_str = self.status_ids_to_names.get(status_str, None)
        if status_str is None or status_str not in self.status_names_to_ids:
            return self.status_unk_token
        return f"[STATUS_{str(status_str)}]"

    def _tokenize_mapper(self, metadata: CM3PMetadata):
        mapper_id = metadata.get('mapper', None)
        if isinstance(mapper_id, str):
            mapper_id = self.mapper_names_to_ids.get(mapper_id, None)
        if mapper_id is None or mapper_id not in self.mapper_ids_to_names:
            return self.mapper_unk_token
        return f"[MAPPER_{str(mapper_id)}]"

    def _tokenize_cs(self, metadata: CM3PMetadata):
        cs = metadata.get('cs', None)
        if cs is None:
            return self.cs_unk_token
        cs = np.clip(cs, 0.0, 10.0)
        cs = round(cs / 0.1) * 0.1
        return f"[CS_{cs:.1f}]"

    def _tokenize_hitsounded(self, metadata: CM3PMetadata):
        hitsounded = metadata.get('hitsounded', None)
        if hitsounded is None:
            return self.hitsounded_unk_token
        return f"[HITSOUNDED_{str(hitsounded).upper()}]"

    def _tokenize_song_length(self, metadata: CM3PMetadata):
        song_length = metadata.get('song_length', None)
        if song_length is None:
            return self.song_length_unk_token
        song_length = np.clip(song_length, 0, self.max_song_length)
        song_length = round(song_length / self.song_length_step) * self.song_length_step
        return f"[SONG_LENGTH_{int(song_length)}]"

    def _tokenize_song_position(self, metadata: CM3PMetadata):
        song_position = metadata.get('song_position', None)
        if song_position is None:
            return self.song_position_unk_token
        song_position = np.clip(song_position, 0.0, 1.0)
        song_position = round(song_position / self.song_position_step) * self.song_position_step
        return f"[SONG_POSITION_{song_position:.2f}]"

    def _tokenize_global_sv(self, metadata: CM3PMetadata):
        global_sv = metadata.get('global_sv', None)
        if global_sv is None:
            return self.global_sv_unk_token
        global_sv = np.clip(global_sv, 0.4, 3.6)
        global_sv = round(global_sv / self.global_sv_step) * self.global_sv_step
        return f"[GLOBAL_SV_{global_sv:.2f}]"

    def _tokenize_mania_keycount(self, metadata: CM3PMetadata):
        mania_keycount = metadata.get('mania_keycount', None)
        if mania_keycount is None:
            return self.mania_keycount_unk_token
        mania_keycount = int(mania_keycount)
        mania_keycount = np.clip(mania_keycount, 1, 18)
        return f"[MANIA_KEYCOUNT_{mania_keycount}]"

    def _tokenize_hold_note_ratio(self, metadata: CM3PMetadata):
        hold_note_ratio = metadata.get('hold_note_ratio', None)
        if hold_note_ratio is None:
            return self.hold_note_ratio_unk_token
        hold_note_ratio = np.clip(hold_note_ratio, 0.0, 1.0)
        hold_note_ratio = round(hold_note_ratio / self.hold_note_ratio_step) * self.hold_note_ratio_step
        return f"[HOLD_NOTE_RATIO_{hold_note_ratio:.1f}]"

    def _tokenize_scroll_speed_ratio(self, metadata: CM3PMetadata):
        scroll_speed_ratio = metadata.get('scroll_speed_ratio', None)
        if scroll_speed_ratio is None:
            return self.scroll_speed_ratio_unk_token
        scroll_speed_ratio = np.clip(scroll_speed_ratio, 0.0, 1.0)
        scroll_speed_ratio = round(scroll_speed_ratio / self.scroll_speed_ratio_step) * self.scroll_speed_ratio_step
        return f"[SCROLL_SPEED_RATIO_{scroll_speed_ratio:.1f}]"

    def _tokenize_tags(self, metadata: CM3PMetadata):
        tags = metadata.get('tags', None)
        if tags is None:
            return [self.tag_unk_token]
        new_tags = []
        for tag in tags:
            if isinstance(tag, str) and tag in self.tag_names_to_ids:
                new_tags.append(tag)
            elif tag in self.tag_ids_to_names:
                new_tags.append(self.tag_ids_to_names[tag])
        if not new_tags:
            return [self.tag_unk_token]
        return [f"[TAG_{tag}]" for tag in new_tags]

    def _tokenize_metadata(self, metadata: CM3PMetadata):
        tokens = [
            self.bos_token,
            self._tokenize_difficulty(metadata),
            self._tokenize_year(metadata),
            self._tokenize_mode(metadata),
            self._tokenize_status(metadata),
            self._tokenize_mapper(metadata),
            self._tokenize_cs(metadata),
            self._tokenize_hitsounded(metadata),
            self._tokenize_song_length(metadata),
            self._tokenize_song_position(metadata),
            self._tokenize_global_sv(metadata),
            self._tokenize_mania_keycount(metadata),
            self._tokenize_hold_note_ratio(metadata),
            self._tokenize_scroll_speed_ratio(metadata),
        ]
        tokens.extend(self._tokenize_tags(metadata))
        tokens.append(self.eos_token)
        return tokens

    def __call__(
            self,
            metadata: Optional[Union[CM3PMetadata, list[CM3PMetadata]]] = None,
            padding: PaddingStrategy = PaddingStrategy.LONGEST,
            truncation: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
            max_length: Optional[int] = None,
            return_tensors: Optional[str] = "pt",
            **kwargs
    ) -> BatchEncoding:
        if isinstance(metadata, dict):
            token_strings = self._tokenize_metadata(metadata)
            token_ids = self.convert_tokens_to_ids(token_strings)
            return self.prepare_for_model(
                token_ids,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                return_tensors=return_tensors,
                **kwargs,
            )
        elif isinstance(metadata, list):
            all_token_ids = []
            for m in metadata:
                token_strings = self._tokenize_metadata(m)
                token_ids = self.convert_tokens_to_ids(token_strings)
                all_token_ids.append((token_ids, None))

            return self._batch_prepare_for_model(
                all_token_ids,
                padding_strategy=PaddingStrategy(padding),
                truncation_strategy=TruncationStrategy(truncation),
                max_length=max_length,
                return_tensors=return_tensors,
            )

    @property
    def vocab_size(self):
        return len(self.vocab) + len(self._added_tokens_encoder)

    def get_vocab(self):
        return self.vocab | self._added_tokens_encoder

    def _convert_token_to_id(self, token):
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index):
        return self.ids_to_tokens.get(index, self.unk_token)

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> tuple[str]:
        if not save_directory:
            raise ValueError("The save_directory must be specified.")

        vocab_file = f"{save_directory}/{filename_prefix or ""}vocab.json"
        with open(vocab_file, 'w', encoding='utf-8') as f:
            json.dump(self.vocab, f, ensure_ascii=False)

        return (vocab_file,)

AutoTokenizer.register(CM3PBeatmapConfig, CM3PBeatmapTokenizer)
AutoTokenizer.register(CM3PMetadataConfig, CM3PMetadataTokenizer)

__all__ = ["CM3PBeatmapTokenizer", "CM3PMetadataTokenizer", "CM3PMetadata", "merge_metadata_dicts"]
