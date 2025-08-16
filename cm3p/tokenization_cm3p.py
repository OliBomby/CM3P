import json
from typing import Optional, Union

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
            vocab_init: Optional[dict] = None,
            min_time: int = 0,
            max_time: int = 8000,
            time_step: int = 10,
            **kwargs,
    ):
        self.min_time = min_time
        self.max_time = max_time
        self.time_step = time_step

        self.audio_bos_token = "[AUDIO_BOS]"
        self.audio_eos_token = "[AUDIO_EOS]"
        self.audio_token = "[AUDIO]"

        if vocab_file is None and vocab_init is None:
            raise ValueError("Either vocab_file or vocab_init must be provided.")

        if vocab_init is not None:
            self.vocab = self._build_vocab_from_config(vocab_init)

        if vocab_file is not None:
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
            **kwargs
        )

    def _build_vocab_from_config(self, vocab_init):
        vocab = []

        # Add time tokens
        for time in np.arange(self.min_time, self.max_time + 1e-5, self.time_step):
            vocab.append(f"[TIME_SHIFT_{int(time)}]")

        # Add event type tokens
        for event_type in EventType:
            vocab.append(f"[{event_type.value.upper()}]")

        return {token: idx for idx, token in enumerate(vocab)}

    def _tokenize_groups(
            self,
            groups: list[Group],
            window_start_ms: Optional[int] = None,
            **kwargs
    ):
        window_start_ms = window_start_ms or 0
        tokens = [self.bos_token]

        for group in groups:
            # Calculate time delta relative to the last event
            time_delta = group.time - window_start_ms
            # Quantize time_delta into a time token
            time_delta = np.clip(time_delta, self.min_time, self.max_time)
            time_delta = round(time_delta / self.time_step) * self.time_step
            time_token = f"[TIME_SHIFT_{int(time_delta)}]"
            tokens.append(time_token)

            # Add the event type token
            event_token = f"[{group.event_type.value.upper()}]"
            tokens.append(event_token)

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
                padding_strategy=padding,
                truncation_strategy=truncation,
                **kwargs,
            )
        else:
            raise ValueError("Input must be a list of Group objects or a single Group object.")

        return encoding

    @property
    def vocab_size(self):
        return len(self.vocab)

    def get_vocab(self):
        return self.vocab.copy()

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


class CM3PMetadataTokenizer(PreTrainedTokenizer):
    model_input_names: list[str] = ["input_ids"]
    vocab_files_names: dict[str, str] = {"vocab_file": "vocab.json"}

    def __init__(
            self,
            vocab_file: Optional[str] = None,
            vocab_init: Optional[dict] = None,
            min_difficculty: float = 0.0,
            max_difficulty: float = 10.0,
            difficulty_step: float = 0.1,
            min_year: int = 2000,
            max_year: int = 2030,
            **kwargs,
    ):
        self.min_difficulty = min_difficculty
        self.max_difficulty = max_difficulty
        self.difficulty_step = difficulty_step
        self.min_year = min_year
        self.max_year = max_year

        self.difficulty_unk_token = "[DIFFICULTY_UNK]"
        self.year_unk_token = "[YEAR_UNK]"
        self.mode_unk_token = "[MODE_UNK]"
        self.mapper_unk_token = "[MAPPER_UNK]"

        if vocab_file is None and vocab_init is None:
            raise ValueError("Either vocab_file or vocab_init must be provided.")

        if vocab_init is not None:
            self.vocab = self._build_vocab_from_config(vocab_init)

        if vocab_file is not None:
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
                self.mapper_unk_token,
            ]),
            min_difficculty=min_difficculty,
            max_difficulty=max_difficulty,
            difficulty_step=difficulty_step,
            min_year=min_year,
            max_year=max_year,
            **kwargs
        )

    def _build_vocab_from_config(self, vocab_init):
        vocab = []

        for difficulty in np.arange(
                self.min_difficulty, self.max_difficulty + 1e-5, self.difficulty_step
        ):
            vocab.append(f"[DIFFICULTY_{difficulty:.1f}]")

        for year in range(self.min_year, self.max_year + 1):
            vocab.append(f"[YEAR_{year}]")

        for mode in vocab_init.get('modes', []):
            vocab.append(f"[MODE_{str(mode).upper()}]")

        for mapper in vocab_init.get('mappers', []):
            vocab.append(f"[MAPPER_{mapper}]")

        return {token: idx for idx, token in enumerate(vocab)}

    def _tokenize_diffulty(self, metadata):
        difficulty = metadata.get('difficulty', None)
        if difficulty is None:
            return self.difficulty_unk_token
        difficulty = np.clip(difficulty, self.min_difficulty, self.max_difficulty)
        difficulty = round(difficulty / self.difficulty_step) * self.difficulty_step
        return f"[DIFFICULTY_{difficulty:.1f}]"

    def _tokenize_year(self, metadata):
        year = metadata.get('year', None)
        if year is None:
            return self.year_unk_token
        year = np.clip(year, self.min_year, self.max_year)
        return f"[YEAR_{year}]"

    def _tokenize_mode(self, metadata):
        mode = metadata.get('mode', None)
        if mode is None:
            return self.mode_unk_token
        mode = str(mode).upper()
        return f"[MODE_{mode}]"

    def _tokenize_mapper(self, metadata):
        mapper = metadata.get('mapper', None)
        if mapper is None:
            return self.mapper_unk_token
        return f"[MAPPER_{mapper}]"

    def _tokenize_metadata(self, metadata):
        tokens = [
            self.bos_token,
            self._tokenize_diffulty(metadata),
            self._tokenize_year(metadata),
            self._tokenize_mode(metadata),
            self._tokenize_mapper(metadata),
            self.eos_token
        ]
        return tokens

    def __call__(
            self,
            metadata: Optional[Union[dict, list[dict]]] = None,
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
                padding_strategy=padding,
                truncation_strategy=truncation,
                max_length=max_length,
                return_tensors=return_tensors,
            )

    @property
    def vocab_size(self):
        return len(self.vocab)

    def get_vocab(self):
        return self.vocab.copy()

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

__all__ = ["CM3PBeatmapTokenizer", "CM3PMetadataTokenizer"]
