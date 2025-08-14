import io
import itertools
from os import PathLike
from typing import Optional, Union, IO

import numpy as np
from accelerate import logging
from slider import Beatmap
from transformers import WhisperFeatureExtractor, AutoProcessor
from transformers.utils import is_soundfile_available, is_torch_available, PaddingStrategy

from cm3p import CM3PConfig
from cm3p.parsing_cm3p import CM3PBeatmapParser
from cm3p.tokenization_cm3p import CM3PBeatmapTokenizer, CM3PMetadataTokenizer

if is_torch_available():
    import torch

if is_soundfile_available():
    import soundfile as sf

from transformers.audio_utils import AudioInput, load_audio_as, make_list_of_audio
from transformers.feature_extraction_utils import BatchFeature
from transformers.processing_utils import AudioKwargs, ProcessingKwargs, ProcessorMixin

logger = logging.get_logger(__name__)


class CM3PAudioKwargs(AudioKwargs, total=False):
    max_source_positions: Optional[int]


# noinspection PyTypedDict
class CM3PProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
        },
        "audio_kwargs": {
            "sampling_rate": 16000,
            "padding": True,
            "truncation": False,
            "pad_to_multiple_of": 480000,
            "max_source_positions": 3000,
        },
        "common_kwargs": {
            "return_tensors": "pt",
            "return_dict": True,
            "tokenize": True,
        },
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
        window_length (`float`, *optional*, defaults to 30.0):
            The length of the sliding window in seconds. This is used to process the beatmap events
            and audio in chunks.
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
        window_length: float = 30.0,
    ):
        self.audio_feature_extractor = audio_feature_extractor
        self.beatmap_parser = beatmap_parser
        self.beatmap_tokenizer = beatmap_tokenizer
        self.metadata_tokenizer = metadata_tokenizer
        self.audio_token = beatmap_tokenizer.audio_token
        self.window_length = window_length

        super().__init__(audio_feature_extractor, beatmap_parser, beatmap_tokenizer, metadata_tokenizer)

    def _retrieve_input_features(self, audio, max_source_positions, **kwargs):
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
        audio_kwargs: CM3PAudioKwargs,
        audio: Union[str, list[str], AudioInput],
        sampling_rate: Optional[int] = None,
        audio_format: Optional[Union[str, list[str]]] = None,
    ) -> list[io.BytesIO]:
        """
        Helper method to load audio from various formats and return a list of audio buffers.
        """

        # validate audio input
        is_str = isinstance(audio, str)
        is_list_of_str = all(isinstance(el, str) for el in audio)
        is_list_of_audio = not (is_str or is_list_of_str)

        if is_list_of_audio:
            if sampling_rate is None:
                logger.warning_once(
                    f"You've provided audio without specifying the sampling rate. It will be assumed to be {audio_kwargs["sampling_rate"]}, which can result in silent errors."
                )
            elif sampling_rate != audio_kwargs["sampling_rate"]:
                raise ValueError(
                    f"The sampling rate of the audio ({sampling_rate}) does not match the sampling rate of the processor ({audio_kwargs["sampling_rate"]}). Please provide resampled the audio to the expected sampling rate."
                )

        sampling_rate = audio_kwargs["sampling_rate"]

        if is_str:
            audio = [load_audio_as(audio, return_format="buffer", force_mono=True, sampling_rate=sampling_rate)]
        elif is_list_of_str:
            audio = [
                load_audio_as(el, return_format="buffer", force_mono=True, sampling_rate=sampling_rate) for el in audio
            ]
        else:
            audio = make_list_of_audio(audio)
            if len(audio) != len(audio_format):
                raise ValueError(
                    f"When passed as a list of audio, the length ({len(audio)}) must match the number of format ({len(audio_format)})"
                )
            audio_buffers = []
            for array, f in zip(audio, audio_format):
                # Create new BytesIO object and write audio data to it
                buffer = io.BytesIO()
                # Convert to mono if needed
                if array.ndim == 2:
                    array = array.mean(axis=1)
                # Write to buffer with default format and sampling rate
                sf.write(buffer, array, samplerate=audio_kwargs["sampling_rate"], format=f)
                buffer.seek(0)
                audio_buffers.append(buffer)
            audio = audio_buffers

        return audio

    def __call__(
        self,
        metadata: Optional[Union[dict, list[dict]]] = None,
        beatmap: Optional[Union[str, list[str], PathLike, list[PathLike], IO[str], list[IO[str]], Beatmap, list[Beatmap]]] = None,
        audio: Optional[Union[str, list[str], AudioInput]] = None,
        sampling_rate: Optional[int] = None,
        audio_format: Optional[Union[str, list[str]]] = None,
        **kwargs,
    ):
        output_kwargs = self._merge_kwargs(
            CM3PProcessorKwargs,
            **kwargs,
        )
        text_kwargs = output_kwargs["text_kwargs"]
        audio_kwargs = output_kwargs["audio_kwargs"]
        common_kwargs = output_kwargs["common_kwargs"]

        max_source_positions = audio_kwargs.pop("max_source_positions")
        return_tensors = common_kwargs.pop("return_tensors", None)

        metadata_encoding, beatmap_encoding = None, None

        if return_tensors != "pt":
            raise ValueError(f"{self.__class__.__name__} only supports `return_tensors='pt'`.")

        if metadata is None and beatmap is None:
            raise ValueError("You have to specify either metadata or beatmap. Both cannot be none.")

        if metadata is not None:
            if isinstance(metadata, dict):
                metadata = [metadata]

            metadata_encoding = self.metadata_tokenizer(metadata)

        if audio is not None:
            audio = self._load_audio(
                audio_kwargs,
                audio,
                sampling_rate=sampling_rate,
                audio_format=audio_format,
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

            processed_windows = []
            processed_audio = []
            for b, a in zip(beatmap, audio):
                with sf.SoundFile(a) as f:
                    # Read the entire audio data
                    audio_array = f.read(dtype="float32")
                    sampling_rate = f.samplerate

                song_length = len(audio_array) / sampling_rate
                beatmap_groups = self.beatmap_parser.parse_beatmap(b, song_length=song_length)

                # Loop through with sliding window
                groups_search_index = 0
                for start_sec in np.arange(0, song_length - self.window_length, self.window_length):
                    end_sec = start_sec + self.window_length

                    # Slice audio waveform
                    start_frame = int(start_sec * sampling_rate)
                    end_frame = int(end_sec * sampling_rate)
                    audio_slice = audio_array[start_frame:end_frame]

                    # Find groups that fall within the current window
                    # Groups are sorted by time, so we can use a simple linear search from the last index
                    start_ms = start_sec * 1000
                    end_ms = end_sec * 1000
                    window_groups = []
                    for group in itertools.islice(beatmap_groups, groups_search_index, None):
                        if group.time < start_ms:
                            groups_search_index += 1
                            continue
                        elif group.time < end_ms:
                            window_groups.append(group)
                            groups_search_index += 1
                        else:
                            break

                    window_encoding = self.beatmap_tokenizer(
                        groups=window_groups,
                        audio=audio_slice,
                        sampling_rate=sampling_rate,
                        window_start_ms=start_ms,
                        padding_strategy=PaddingStrategy.DO_NOT_PAD,
                        pad_to_multiple_of=None,
                        padding_side=None,
                        return_attention_mask=False,
                        return_tensors=None,
                        prepend_batch_axis=False,
                    )

                    audio = window_encoding.pop("audio", None)
                    if audio is not None:
                        processed_audio.append(audio)

                    processed_windows.append(window_encoding)

            batch_outputs = self.beatmap_tokenizer.pad(
                processed_windows,
            )

            beatmap_encoding = BatchFeature(data=batch_outputs, tensor_type=return_tensors)
            beatmap_encoding["input_features"] = self._retrieve_input_features(processed_audio, max_source_positions, **audio_kwargs)

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

AutoProcessor.register(CM3PConfig, CM3PBeatmapParser)

__all__ = ["CM3PProcessor"]
