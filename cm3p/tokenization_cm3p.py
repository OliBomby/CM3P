import io
import json
from typing import Optional, Union

from torch import TensorType
from transformers import PreTrainedTokenizer, BatchEncoding
from transformers.tokenization_utils_base import TextInput, PreTokenizedInput, TruncationStrategy, \
    PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy


class CM3PBeatmapTokenizer(PreTrainedTokenizerBase):
    def __init__(
            self,
            vocab_file: Optional[str] = None,
            tokenizer_config: Optional[dict] = None,
            **kwargs
    ):
        if vocab_file is None and tokenizer_config is None:
            raise ValueError("Either vocab_file or tokenizer_config must be provided.")

        if tokenizer_config is not None:
            super().__init__(
                unk_token=tokenizer_config['unk_token'],
                pad_token=tokenizer_config['pad_token'],
                bos_token=tokenizer_config['bos_token'],
                eos_token=tokenizer_config['eos_token'],
                **kwargs
            )
            self.vocab = self._build_vocab_from_config(tokenizer_config)

        if vocab_file is not None:
            with open(vocab_file, 'r', encoding='utf-8') as f:
                self.vocab = json.load(f)

        self.ids_to_tokens = {i: t for t, i in self.vocab.items()}

    def _build_vocab_from_config(self, config):
        vocab = {}
        idx = 0

        # Add special tokens first
        for token in [config['pad_token'], config['unk_token'], config['bos_token'], config['eos_token']]:
            vocab[token] = idx
            idx += 1

        # Add event type tokens
        for event_type in config['event_types']:
            vocab[event_type] = idx
            idx += 1

        # Add quantized time tokens
        for i in range(config['max_time_quanta']):
            vocab[f"[TIME_SHIFT_{i * config['time_step_ms']}ms]"] = idx
            idx += 1

        return vocab

    # You'll need to implement these methods
    @property
    def vocab_size(self):
        return len(self.vocab)

    def get_vocab(self):
        return self.vocab.copy()

    def _tokenize_events(self, events, **kwargs):
        window_start_ms = kwargs.get("window_start_ms", 0)

        tokens = ["[START]"]

        for event in events:
            # Calculate time delta relative to the last event
            time_delta = event['timestamp'] - window_start_ms
            # Quantize time_delta into a time token
            time_token = f"[TIME_SHIFT_{int(time_delta)}ms]"
            tokens.append(time_token)

            # Add the event type token
            event_token = f"[{event['type'].upper()}]"
            tokens.append(event_token)

            last_event_ms = event['timestamp']

        tokens.append("[END]")
        return tokens

    def __call__(
            self,
            events: Optional[list[dict]] = None,
            audio: Optional[Union[io.BytesIO]] = None,
            window_start_ms: int = 0,
            padding: bool = True,
            truncation: bool = True,
            max_length: Optional[int] = None,
            return_tensors: Optional[str] = "pt",
            **kwargs
    ) -> BatchEncoding:
        token_strings = self._tokenize_events(events, window_start_ms=window_start_ms)
        token_ids = self.convert_tokens_to_ids(token_strings)
        return self.prepare_for_model(
            ids=token_ids,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_tensors=return_tensors,
            **kwargs,
        )

    def _convert_token_to_id(self, token):
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index):
        return self.ids_to_tokens.get(index, self.unk_token)

    def _encode_audio(self, audio: Audio) -> AudioEncoding:
        audio.resample(self.audio_config.sampling_rate)

        audio.audio_array = self.pad(audio.audio_array, self.audio_config.sampling_rate)
        signal_length = audio.audio_array.shape[0]

        # for spectrogram-based models, the waveform is downsampled by the hop_length when computing the log-mel
        if signal_length % self.encoding_config.hop_length != 0:
            signal_length = math.ceil(signal_length / self.encoding_config.hop_length - 1)
        else:
            signal_length = signal_length // self.encoding_config.hop_length

        num_audio_tokens = math.ceil(signal_length / self.audio_config.audio_length_per_tok)
        audio_tokens = [self.begin_audio_token] + [self.audio_token] * num_audio_tokens

        return AudioEncoding(
            tokens=audio_tokens,
            audio=audio,
        )


class CM3PMetadataTokenizer(PreTrainedTokenizer):
    def __init__(self, vocab_file, **kwargs):
        super().__init__(**kwargs)
        with open(vocab_file, 'r', encoding='utf-8') as f:
            self.vocab = json.load(f)
        self.ids_to_tokens = {i: t for t, i in self.vocab.items()}

    @property
    def vocab_size(self):
        return len(self.vocab)

    def get_vocab(self):
        return self.vocab.copy()

    def _tokenize(self, metadata):
        # Your logic to convert metadata into tokens
        tokens = []
        # ... your conversion logic here ...
        return tokens

    def _convert_token_to_id(self, token):
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index):
        return self.ids_to_tokens.get(index, self.unk_token)
