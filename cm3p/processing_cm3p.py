import copy
import itertools
import math
import os
from os import PathLike
from typing import Optional, Union, IO, TypedDict

import numpy as np
import soxr
from slider import Beatmap
from transformers import WhisperFeatureExtractor, AutoProcessor
from transformers.tokenization_utils_base import TruncationStrategy
from transformers.utils import is_torch_available, PaddingStrategy, PROCESSOR_NAME, logging

from cm3p import CM3PConfig
from cm3p.parsing_cm3p import CM3PBeatmapParser
from cm3p.tokenization_cm3p import CM3PBeatmapTokenizer, CM3PMetadataTokenizer

if is_torch_available():
    import torch

from transformers.audio_utils import AudioInput, make_list_of_audio, load_audio
from transformers.feature_extraction_utils import BatchFeature
from transformers.processing_utils import AudioKwargs, ProcessorMixin, ImagesKwargs, CommonKwargs

logger = logging.get_logger(__name__)


class CM3PTokenizerKwargs(TypedDict, total=False):
    add_special_tokens: Optional[bool]
    padding: Union[bool, str, PaddingStrategy]
    truncation: Union[bool, str, TruncationStrategy]
    max_length: Optional[int]
    pad_to_multiple_of: Optional[int]
    return_token_type_ids: Optional[bool]
    return_attention_mask: Optional[bool]
    return_overflowing_tokens: Optional[bool]
    return_special_tokens_mask: Optional[bool]
    return_offsets_mapping: Optional[bool]
    return_length: Optional[bool]
    verbose: Optional[bool]
    padding_side: Optional[str]
    return_mm_token_type_ids: Optional[bool]


class CM3PBeatmapKwargs(CM3PTokenizerKwargs, total=False):
    window_length_sec: float
    window_stride_sec: float


class CM3PAudioKwargs(AudioKwargs, total=False):
    max_source_positions: Optional[int]
    hop_length: Optional[int]
    window_size: Optional[int]
    audio_length_per_tok: Optional[int]


# noinspection PyTypedDict
class CM3PProcessorKwargs(CM3PTokenizerKwargs, ImagesKwargs, AudioKwargs, CommonKwargs, total=False):
    _defaults = {
        "beatmap_kwargs": {
            "max_length": 8000,
            "padding": PaddingStrategy.LONGEST,
            "truncation": TruncationStrategy.LONGEST_FIRST,
            "window_length_sec": 30.0,
            "window_stride_sec": 30.0,
        },
        "metadata_kwargs": {
            "max_length": 128,
            "padding": PaddingStrategy.LONGEST,
            "truncation": TruncationStrategy.DO_NOT_TRUNCATE,
        },
        "audio_kwargs": {
            "sampling_rate": 16000,
            "padding": True,
            "truncation": False,
            "pad_to_multiple_of": 480000,
            "max_source_positions": 3000,
            "hop_length": 160,
            "window_size": 400,
            "audio_length_per_tok": 8,
        },
        "common_kwargs": {
            "return_tensors": "pt",
        },
    }

    common_kwargs: CommonKwargs = {
        **CommonKwargs.__annotations__,
    }
    beatmap_kwargs: CM3PBeatmapKwargs = {
        **CM3PTokenizerKwargs.__annotations__,
    }
    metadata_kwargs: CM3PTokenizerKwargs = {
        **CM3PTokenizerKwargs.__annotations__,
    }
    audio_kwargs: AudioKwargs = {
        **AudioKwargs.__annotations__,
    }


class CM3PProcessor(ProcessorMixin):
    r"""
    Constructs a CM3P processor which wraps [`WhisperFeatureExtractor`] and
    [`MistralCommonTokenizer`] into a single processor that inherits both the audio feature extraction and
    tokenizer functionalities.

    Args:
        audio_feature_extractor ([`WhisperFeatureExtractor`]):
            The feature extractor is a required input.
        beatmap_parser ([`CM3PBeatmapParser`]):
            The beatmap parser is a required input.
        beatmap_tokenizer ([`CM3PBeatmapTokenizer`]):
            The beatmap tokenizer is a required input.
        metadata_tokenizer ([`CM3PMetadataTokenizer`]):
            The metadata tokenizer is a required input.
        default_kwargs (`CM3PProcessorKwargs`, *optional*):
            Default keyword arguments for the processor. If not provided, the processor will use its own defaults
    """

    attributes = ["audio_feature_extractor", "beatmap_parser", "beatmap_tokenizer", "metadata_tokenizer"]
    audio_feature_extractor_class = "WhisperFeatureExtractor"
    beatmap_parser_class = "CM3PBeatmapParser"
    beatmap_tokenizer_class = "CM3PBeatmapTokenizer"
    metadata_tokenizer_class = "CM3PMetadataTokenizer"

    def __init__(
        self,
        audio_feature_extractor: WhisperFeatureExtractor,
        beatmap_parser: CM3PBeatmapParser,
        beatmap_tokenizer: CM3PBeatmapTokenizer,
        metadata_tokenizer: CM3PMetadataTokenizer,
        default_kwargs: Optional[CM3PProcessorKwargs] = None,
    ):
        self.audio_feature_extractor = audio_feature_extractor
        self.beatmap_parser = beatmap_parser
        self.beatmap_tokenizer = beatmap_tokenizer
        self.metadata_tokenizer = metadata_tokenizer
        self.audio_token = beatmap_tokenizer.audio_token

        # noinspection PyProtectedMember
        self.default_kwargs = default_kwargs or copy.deepcopy(CM3PProcessorKwargs._defaults)

        super().__init__(audio_feature_extractor, beatmap_parser, beatmap_tokenizer, metadata_tokenizer)

    def _pad_audio(
            self,
            audio_array: np.ndarray,
            window_size: int = 400,
            pad_to_multiple_of: Optional[int] = 480000,
            **kwargs,
    ) -> np.ndarray:
        r"""Pad the audio array to the desired length.

        Args:
            audio_array: Audio data as a numpy array.
            sampling_rate: Sampling rate of the audio.

        Returns:
            Padded audio array.
        """
        if pad_to_multiple_of:
            next_multiple_of_chunk_frames = math.ceil(audio_array.shape[-1] / pad_to_multiple_of) * pad_to_multiple_of
            audio_array = np.pad(audio_array, (0, next_multiple_of_chunk_frames - audio_array.shape[-1]))
        elif audio_array.shape[-1] < window_size:
            # minimum length for audios is at least one spectrogram frame
            audio_array = np.pad(audio_array, (0, window_size - audio_array.shape[-1]))

        return audio_array

    def _encode_audio(
            self,
            audio: np.ndarray,
            hop_length: int = 160,
            audio_length_per_tok: int = 8,
            **kwargs,
    ) -> tuple[np.ndarray, int]:
        audio = self._pad_audio(audio, **kwargs)
        signal_length = audio.shape[0]

        # for spectrogram-based models, the waveform is downsampled by the hop_length when computing the log-mel
        if signal_length % hop_length != 0:
            signal_length = math.ceil(signal_length / hop_length - 1)
        else:
            signal_length = signal_length // hop_length

        num_audio_tokens = math.ceil(signal_length / audio_length_per_tok)

        return audio, num_audio_tokens

    def _retrieve_input_features(self, audio, max_source_positions, **kwargs) -> torch.Tensor:
        """
        Handles specific logic of CM3P expected input features: audio arrays should be padded to next multiple of 480000 (duration is a multiple of 30s), see CM3PProcessorKwargs' default audio_kwargs.
        Then mel input features are extracted and stacked along batch dimension, splitting into chunks of max_source_positions.
        """
        input_features_list = []
        for audio_array in audio:
            audio_inputs = self.audio_feature_extractor(audio_array, **kwargs)

            # let's split into chunks of max_source_positions, and then stack them along batch dimension
            input_features = audio_inputs["input_features"].reshape(
                self.audio_feature_extractor.feature_size, -1, max_source_positions
            )

            input_features_list.append(input_features.transpose(0, 1))

        return torch.cat(input_features_list)

    def _load_audio(
        self,
        sampling_rate: int,
        audio: Union[str, list[str], AudioInput],
        audio_sampling_rate: Optional[Union[int, list[int]]] = None,
    ) -> list[np.ndarray]:
        """
        Helper method to load audio from various formats and return a list of audio buffers.
        """

        # validate audio input
        is_str = isinstance(audio, str)
        is_list_of_str = all(isinstance(el, str) for el in audio)
        is_list_of_audio = not (is_str or is_list_of_str)

        if is_list_of_audio:
            if audio_sampling_rate is None:
                logger.warning_once(
                    f"You've provided audio without specifying the sampling rate. It will be assumed to be {sampling_rate}, which can result in silent errors."
                )
                audio_sampling_rate = sampling_rate

        if is_str:
            audio = [load_audio(audio, sampling_rate=sampling_rate)]
            audio_sampling_rate = sampling_rate
        elif is_list_of_str:
            audio = [load_audio(el, sampling_rate=sampling_rate) for el in audio]
            audio_sampling_rate = sampling_rate

        audio = make_list_of_audio(audio)

        if isinstance(audio_sampling_rate, int):
            audio_sampling_rate = [audio_sampling_rate] * len(audio)

        audio_buffers = []
        for array, s in zip(audio, audio_sampling_rate):
            array = np.asarray(array)
            # Convert to mono if needed
            if array.ndim == 2:
                array = array.mean(axis=1)
            # Resample if the sampling rate is different from the expected one
            if s != sampling_rate:
                array = soxr.resample(array, s, sampling_rate, quality="HQ")
            audio_buffers.append(array)

        return audio_buffers

    # noinspection PyTypedDict
    def _merge_kwargs(self, **kwargs) -> CM3PProcessorKwargs:
        output_kwargs = CM3PProcessorKwargs()
        nested_modalities = ["beatmap_kwargs", "metadata_kwargs", "audio_kwargs", "common_kwargs"]
        possible_modality_keywords = {"beatmap", "metadata", "audio"}
        used_keys = set()

        # pass defaults to output dictionary
        output_kwargs.update(self.default_kwargs)

        # update modality kwargs with passed kwargs
        non_modality_kwargs = set(kwargs) - set(output_kwargs)
        for modality, output_kwarg in output_kwargs.items():
            for modality_key in CM3PProcessorKwargs.__annotations__[modality].__annotations__:
                # check if we received a structured kwarg dict or not to handle it correctly
                if modality in kwargs:
                    kwarg_value = kwargs[modality].pop(modality_key, "__empty__")
                    # check if this key was passed as a flat kwarg.
                    if kwarg_value != "__empty__" and modality_key in non_modality_kwargs:
                        raise ValueError(
                            f"Keyword argument {modality_key} was passed two times:\n"
                            f"in a dictionary for {modality} and as a **kwarg."
                        )
                elif modality_key in kwargs:
                    # we get a modality_key instead of popping it because modality-specific processors
                    # can have overlapping kwargs
                    kwarg_value = kwargs.get(modality_key, "__empty__")
                else:
                    kwarg_value = "__empty__"
                if not isinstance(kwarg_value, str) or kwarg_value != "__empty__":
                    output_kwarg[modality_key] = kwarg_value
                    used_keys.add(modality_key)

        # Determine if kwargs is a flat dictionary or contains nested dictionaries
        if any(key in nested_modalities for key in kwargs):
            # kwargs is dictionary-based, and some keys match modality names
            for modality, subdict in kwargs.items():
                if modality in nested_modalities:
                    for subkey, subvalue in subdict.items():
                        if subkey not in used_keys:
                            output_kwargs[modality][subkey] = subvalue
                            used_keys.add(subkey)
        else:
            # kwargs is a flat dictionary
            for key, kwarg in kwargs.items():
                if key not in used_keys:
                    if key in CM3PProcessorKwargs.__annotations__["common_kwargs"].__annotations__:
                        output_kwargs["common_kwargs"][key] = kwarg
                    elif key not in possible_modality_keywords:
                        logger.warning_once(
                            f"Keyword argument `{key}` is not a valid argument for this processor and will be ignored."
                        )

        # all modality-specific kwargs are updated with common kwargs
        for kwarg in output_kwargs.values():
            kwarg.update(output_kwargs["common_kwargs"])
        return output_kwargs

    def __call__(
        self,
        metadata: Optional[Union[dict, list[dict]]] = None,
        beatmap: Optional[Union[str, list[str], PathLike, list[PathLike], IO[str], list[IO[str]], Beatmap, list[Beatmap]]] = None,
        audio: Optional[Union[str, list[str], AudioInput]] = None,
        audio_sampling_rate: Optional[Union[int, list[int]]] = None,
        **kwargs,
    ):
        output_kwargs = self._merge_kwargs(**kwargs)

        beatmap_kwargs: CM3PTokenizerKwargs = output_kwargs["beatmap_kwargs"]
        metadata_kwargs: CM3PTokenizerKwargs = output_kwargs["metadata_kwargs"]
        audio_kwargs: CM3PAudioKwargs = output_kwargs["audio_kwargs"]
        common_kwargs: CommonKwargs = output_kwargs["common_kwargs"]

        window_length_sec = beatmap_kwargs.pop("window_length_sec")
        window_stride_sec = beatmap_kwargs.pop("window_stride_sec")
        sampling_rate = audio_kwargs["sampling_rate"]
        return_tensors = common_kwargs["return_tensors"]

        metadata_encoding, beatmap_encoding, num_audio_tokens = None, None, None

        if return_tensors != "pt":
            raise ValueError(f"{self.__class__.__name__} only supports `return_tensors='pt'`.")

        if metadata is None and beatmap is None:
            raise ValueError("You have to specify either metadata or beatmap. Both cannot be none.")

        if metadata is not None:
            if isinstance(metadata, dict):
                metadata = [metadata]

            metadata_encoding = self.metadata_tokenizer(
                metadata,
                **metadata_kwargs,
            )

        if audio is not None:
            audio = self._load_audio(
                sampling_rate,
                audio,
                audio_sampling_rate=audio_sampling_rate,
            )

        if beatmap is not None:
            if not isinstance(beatmap, list):
                beatmap = [beatmap]

            if audio is not None:
                if len(beatmap) != len(audio):
                    raise ValueError(
                        f"The number of beatmaps ({len(beatmap)}) must match the number of audio ({len(audio)})"
                    )
            else:
                audio = [None] * len(beatmap)

            batch_start_ms = []
            batch_groups = []
            batch_audio = []
            batch_num_audio_tokens = []
            for b, audio_array in zip(beatmap, audio):
                song_length = len(audio_array) / sampling_rate if audio_array is not None else None
                beatmap_groups = self.beatmap_parser.parse_beatmap(b, song_length=song_length)

                # Loop through with sliding window
                groups_search_index = 0
                for start_sec in np.arange(0, song_length - window_length_sec, window_stride_sec):
                    end_sec = start_sec + window_length_sec

                    if audio_array is not None:
                        # Slice audio waveform
                        start_frame = int(start_sec * sampling_rate)
                        end_frame = int(end_sec * sampling_rate)
                        audio_slice = audio_array[start_frame:end_frame]
                        # Pad the audio array and calculate the number of audio tokens
                        audio_slice, num_audio_tokens = self._encode_audio(audio_slice, **kwargs)
                    else:
                        audio_slice = None
                        num_audio_tokens = 0

                    # Find groups that fall within the current window
                    # Groups are sorted by time, so we can use a simple linear search from the last index
                    start_ms = start_sec * 1000
                    end_ms = end_sec * 1000
                    next_start_ms = (start_sec + window_stride_sec) * 1000
                    window_groups = []
                    for group in itertools.islice(beatmap_groups, groups_search_index, None):
                        if group.time < next_start_ms:
                            groups_search_index += 1

                        if group.time < start_ms:
                            continue
                        elif group.time < end_ms:
                            window_groups.append(group)
                        else:
                            break

                    batch_start_ms.append(start_ms)
                    batch_groups.append(window_groups)
                    batch_audio.append(audio_slice)
                    batch_num_audio_tokens.append(num_audio_tokens)

            beatmap_encoding = self.beatmap_tokenizer(
                groups=batch_groups,
                window_start_ms=batch_start_ms,
                num_audio_tokens=batch_num_audio_tokens,
                **beatmap_kwargs,
            )

            if audio is not None:
                data = dict(beatmap_encoding)
                data["input_features"] = self._retrieve_input_features(batch_audio, **audio_kwargs)
                beatmap_encoding = BatchFeature(data, tensor_type=return_tensors)

        if metadata_encoding is not None and beatmap_encoding is not None:
            beatmap_encoding["metadata_ids"] = metadata_encoding["input_ids"]
            return beatmap_encoding
        elif beatmap_encoding is not None:
            return beatmap_encoding
        else:
            return metadata_encoding

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to CM3PBeatmapTokenizer's [`~CM3PBeatmapTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.beatmap_tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to CM3PBeatmapTokenizer's [`~CM3PBeatmapTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.beatmap_tokenizer.decode(*args, **kwargs)

    def save_pretrained(self, save_directory, push_to_hub: bool = False, **kwargs):
        os.makedirs(save_directory, exist_ok=True)

        for attribute_name in self.attributes:
            attribute = getattr(self, attribute_name)
            # Include the processor class in the attribute config so this processor can then be reloaded with the
            # `AutoProcessor` API.
            if hasattr(attribute, "_set_processor_class"):
                attribute._set_processor_class(self.__class__.__name__)
            attribute.save_pretrained(os.path.join(save_directory, attribute_name))

        output_processor_file = os.path.join(save_directory, PROCESSOR_NAME)
        self.to_json_file(output_processor_file)
        logger.warning_once(f"processor saved in {output_processor_file}")

        if push_to_hub:
            commit_message = kwargs.pop("commit_message", None)
            repo_id = kwargs.pop("repo_id", save_directory.split(os.path.sep)[-1])
            repo_id = self._create_repo(repo_id, **kwargs)
            files_timestamps = self._get_files_timestamps(save_directory)

            self._upload_modified_files(
                save_directory,
                repo_id,
                files_timestamps,
                commit_message=commit_message,
                token=kwargs.get("token"),
            )

        return [output_processor_file]

    @classmethod
    def _get_arguments_from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        subfolder = kwargs.pop("subfolder", None)
        args = []
        for attribute_name in cls.attributes:
            class_name = getattr(cls, f"{attribute_name}_class")
            attribute_class = cls.get_possibly_dynamic_module(class_name)
            attribute_subfolder = os.path.join(subfolder, attribute_name) if subfolder else attribute_name

            args.append(attribute_class.from_pretrained(
                pretrained_model_name_or_path,
                subfolder=attribute_subfolder,
                **kwargs
            ))

        return args

AutoProcessor.register(CM3PConfig, CM3PProcessor)

__all__ = ["CM3PProcessor"]
