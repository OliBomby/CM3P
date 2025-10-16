"""PyTorch CM3P model."""
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Optional, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import ModernBertModel, AutoModel, AutoModelForSequenceClassification, AutoModelForMaskedLM
from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPooling, MaskedLMOutput,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput, auto_docstring, can_return_tuple, logging

from .configuration_cm3p import CM3PConfig, CM3PMetadataConfig, CM3PBeatmapConfig, CM3PAudioConfig


logger = logging.get_logger(__name__)


# contrastive loss function, adapted from
# https://sachinruk.github.io/blog/2021-03-07-clip.html
def contrastive_loss(logits: torch.Tensor, target: torch.LongTensor = None) -> torch.Tensor:
    target = target if target is not None else torch.arange(len(logits), device=logits.device)
    return nn.functional.cross_entropy(logits, target)


# CM3P loss function, adapted from CLIP
def cm3p_loss(similarity: torch.Tensor, metadata_variation_classes: torch.LongTensor = None) -> torch.Tensor:
    if similarity.dim() == 3:  # (metadata_batch_size, variations, beatmap_batch_size)
        metadata_batch_size = similarity.size(0)
        num_variations = similarity.size(1)
        beatmap_batch_size = similarity.size(2)
        assert metadata_batch_size == beatmap_batch_size

        true_metadata_indices = (metadata_variation_classes == 0).int().argmax(dim=1)
        metadata_loss = contrastive_loss(similarity[torch.arange(metadata_batch_size), true_metadata_indices])  # only use original metadata for loss

        beatmap_similarity = similarity.permute(2, 0, 1)  # (beatmap_batch_size, metadata_batch_size, variations)
        beatmap_similarity = beatmap_similarity.reshape(beatmap_batch_size, -1)  # (beatmap_batch_size, metadata_batch_size * variations)
        target = torch.arange(0, beatmap_similarity.size(1), num_variations, device=similarity.device)  # (metadata_batch_size,)
        target += true_metadata_indices
        beatmap_loss = contrastive_loss(beatmap_similarity, target=target)
    else:
        metadata_loss = contrastive_loss(similarity)
        beatmap_loss = contrastive_loss(similarity.t())
    return (metadata_loss + beatmap_loss) / 2.0


def _get_vector_norm(tensor: torch.Tensor) -> torch.Tensor:
    """
    This method is equivalent to tensor.norm(p=2, dim=-1, keepdim=True) and used to make
    model `executorch` exportable. See issue https://github.com/pytorch/executorch/issues/3566
    """
    square_tensor = torch.pow(tensor, 2)
    sum_tensor = torch.sum(square_tensor, dim=-1, keepdim=True)
    normed_tensor = torch.pow(sum_tensor, 0.5)
    return normed_tensor


def _unpad_cm3p_input(
    inputs: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Remove padding from input sequences.

    Args:
        inputs: (batch, seqlen, ...) or (batch, seqlen)
        attention_mask: (batch, seqlen), bool / int, 1 means valid and 0 means not valid.
        position_ids: (batch, seqlen), int, position ids
        labels: (batch, seqlen), int, labels

    Returns:
        unpadded_inputs: (total_nnz, ...), where total_nnz = number of tokens selected in attention_mask.
        indices: (total_nnz)
        cu_seqlens: (batch + 1), the cumulative sequence lengths
        max_seqlen_in_batch: int
        unpadded_position_ids: (total_nnz) or None
        unpadded_labels: (total_nnz) or None
    """
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = int(seqlens_in_batch.max().item())
    cu_seqlens = torch.nn.functional.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))

    if inputs.dim() == 2:
        unpadded_inputs = inputs.flatten()[indices]
    else:
        batch, seqlen, *rest = inputs.shape
        shape = batch * seqlen
        unpadded_inputs = inputs.view(shape, *rest)[indices]

    unpadded_position_ids = position_ids.flatten()[indices] if position_ids is not None else None
    unpadded_labels = labels.flatten()[indices] if labels is not None else None

    return unpadded_inputs, indices, cu_seqlens, max_seqlen_in_batch, unpadded_position_ids, unpadded_labels


def _pad_cm3p_output(
    inputs: torch.Tensor,
    indices: torch.Tensor,
    batch: int,
    seqlen: int,
) -> torch.Tensor:
    """
    Add padding to sequences.

    Args:
        inputs: (total_nnz, ...) or (total_nnz,), where total_nnz = number of tokens selected in attention_mask.
        indices: (total_nnz)
        batch: int, batch size
        seqlen: int, max sequence length

    Returns:
        padded_inputs: (batch, seqlen, ...) or (batch, seqlen)
    """
    if inputs.dim() == 1:
        output = torch.zeros(batch * seqlen, dtype=inputs.dtype, device=inputs.device)
        output[indices] = inputs
        padded_inputs = output.view(batch, seqlen)
    else:
        _, *rest = inputs.shape
        output = torch.zeros(batch * seqlen, *rest, dtype=inputs.dtype, device=inputs.device)
        output[indices] = inputs
        padded_inputs = output.view(batch, seqlen, *rest)

    return padded_inputs


@dataclass
class BeatmapClassifierOutput(ModelOutput):
    """
    Base class for outputs of beatmap classification models.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each stage) of shape `(batch_size, sequence_length, hidden_size)`. Hidden-states
            (also called feature maps) of the model at the output of each stage.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, patch_size,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[tuple[torch.FloatTensor, ...]] = None


@dataclass
@auto_docstring(
    custom_intro="""
    Base class for audio model's outputs that also contains a pooling of the last hidden states.
    """
)
class CM3PAudioModelOutput(BaseModelOutput):
    r"""
    audio_embeds (`torch.FloatTensor` of shape `(batch_size * sequence_length, output_dim)` *optional* returned when model is initialized with `with_projection=True`):
        The audio embeddings obtained by applying the projection layer to the last hidden state.
    """

    audio_embeds: Optional[torch.FloatTensor] = None


@dataclass
@auto_docstring(
    custom_intro="""
    Base class for beatmap model's outputs that also contains beatmap embeddings of the pooling of the last hidden states.
    """
)
class CM3PBeatmapModelOutput(BaseModelOutputWithPooling):
    r"""
    audio_model_output (`BaseModelOutput`):
        The output of the audio model, which contains the last hidden state, hidden states, and attentions.
    beatmap_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim)` *optional* returned when model is initialized with `with_projection=True`):
        The beatmap embeddings obtained by applying the projection layer to the pooler_output.
    """

    beatmap_embeds: Optional[torch.FloatTensor] = None
    audio_model_output: CM3PAudioModelOutput = None


@dataclass
@auto_docstring(
    custom_intro="""
    Base class for metadata model's outputs that also contains a pooling of the last hidden states.
    """
)
class CM3PMetadataModelOutput(BaseModelOutput):
    r"""
    metadata_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim)` *optional* returned when model is initialized with `with_projection=True`):
        The metadata embeddings obtained by applying the projection layer to the pooler_output.
    """

    metadata_embeds: Optional[torch.FloatTensor] = None


@dataclass
@auto_docstring
class CM3POutput(ModelOutput):
    r"""
    loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `return_loss` is `True`):
        Contrastive loss for beatmap-metadata similarity.
    logits_per_beatmap (`torch.FloatTensor` of shape `(beatmap_batch_size, metadata_batch_size)`):
        The scaled dot product scores between `beatmap_embeds` and `metadata_embeds`. This represents the beatmap-metadata
        similarity scores.
    logits_per_metadata (`torch.FloatTensor` of shape `(metadata_batch_size, beatmap_batch_size)`):
        The scaled dot product scores between `metadata_embeds` and `beatmap_embeds`. This represents the metadata-beatmap
        similarity scores.
    metadata_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim`):
        The metadata embeddings obtained by applying the projection layer to the pooled output of [`CM3PMetadataModel`].
    beatmap_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim`):
        The beatmap embeddings obtained by applying the projection layer to the pooled output of [`CM3PBeatmapModel`].
    logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, vocab_size)`, *optional*, returned when `labels` is provided):
        Prediction scores of the masked language modeling head. Only computed if `labels` is provided.
    metadata_model_output (`BaseModelOutputWithPooling`):
        The output of the [`CM3PMetadataModel`].
    beatmap_model_output (`BaseModelOutputWithPooling`):
        The output of the [`CM3PBeatmapModel`].
    """

    loss: Optional[torch.FloatTensor] = None
    logits_per_beatmap: Optional[torch.FloatTensor] = None
    logits_per_metadata: Optional[torch.FloatTensor] = None
    metadata_embeds: Optional[torch.FloatTensor] = None
    beatmap_embeds: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    metadata_model_output: BaseModelOutputWithPooling = None
    beatmap_model_output: BaseModelOutputWithPooling = None

    def to_tuple(self) -> tuple[Any]:
        return tuple(
            self[k] if k not in ["metadata_model_output", "beatmap_model_output"] else getattr(self, k).to_tuple()
            for k in self.keys()
        )


@auto_docstring
class CM3PPreTrainedModel(PreTrainedModel):
    config_class = CM3PConfig
    base_model_prefix = "cm3p"
    supports_gradient_checkpointing = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_flex_attn = False

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            nn.init.normal_(module.weight, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.weight.data.fill_(1.0)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, ModernBertModel):
            module.initialize_weights()
        elif isinstance(module, CM3PModel):
            nn.init.normal_(
                module.metadata_projection.weight,
                std=module.metadata_embed_dim**-0.5 * self.config.initializer_factor,
            )
            nn.init.normal_(
                module.beatmap_projection.weight,
                std=module.beatmap_embed_dim**-0.5 * self.config.initializer_factor,
            )
        elif isinstance(module, CM3PBeatmapModelWithProjection):
            nn.init.normal_(
                module.beatmap_projection.weight,
                std=self.config.hidden_size**-0.5 * self.config.initializer_factor,
            )
        elif isinstance(module, CM3PMetadataModelWithProjection):
            nn.init.normal_(
                module.metadata_projection.weight,
                std=self.config.hidden_size**-0.5 * self.config.initializer_factor,
            )
        elif isinstance(module, CM3PForBeatmapClassification):
            nn.init.normal_(
                module.classifier.weight,
                std=self.config.beatmap_config.hidden_size**-0.5 * self.config.initializer_factor,
            )


class CM3PMetadataTransformer(nn.Module):
    def __init__(self, config: CM3PMetadataConfig):
        super().__init__()
        self.config = config
        self.encoder = ModernBertModel(config)

    def get_input_embeddings(self):
        return self.encoder.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.encoder.set_input_embeddings(value)

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        indices: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        batch_size: Optional[int] = None,
        seq_len: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_pooler: bool = True,
    ) -> BaseModelOutputWithPooling:
        r"""
        indices (`torch.Tensor` of shape `(total_unpadded_tokens,)`, *optional*):
            Indices of the non-padding tokens in the input sequence. Used for unpadding the output.
        cu_seqlens (`torch.Tensor` of shape `(batch + 1,)`, *optional*):
            Cumulative sequence lengths of the input sequences. Used to index the unpadded tensors.
        max_seqlen (`int`, *optional*):
            Maximum sequence length in the batch excluding padding tokens. Used to unpad input_ids and pad output tensors.
        batch_size (`int`, *optional*):
            Batch size of the input sequences. Used to pad the output tensors.
        seq_len (`int`, *optional*):
            Sequence length of the input sequences including padding tokens. Used to pad the output tensors.
        output_pooler (`bool`, *optional*, defaults to `True`):
            Whether to return the pooled output of the model. The pooled output is usually the representation of
            the first token (CLS) or the mean of the token representations.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        if input_ids is None:
            raise ValueError("You have to specify input_ids")

        is_3d = input_ids.dim() == 3
        batch_size_3d = input_ids.size(0)
        if is_3d:
            # flatten to 2D batch if multiple metadata variations are provided
            input_ids = input_ids.view(-1, input_ids.size(-1))
            if attention_mask is not None:
                attention_mask = attention_mask.view(-1, attention_mask.size(-1))

        encoder_outputs: BaseModelOutput = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            indices=indices,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            batch_size=batch_size,
            seq_len=seq_len,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        last_hidden_state = encoder_outputs.last_hidden_state
        pooled_output = None

        if is_3d:
            # un-flatten back to 3D batch (batch_size, variations, seq_length, hidden_size)
            last_hidden_state = last_hidden_state.view(
                batch_size_3d, -1, last_hidden_state.size(-2), last_hidden_state.size(-1)
            )
            if attention_mask is not None:
                attention_mask = attention_mask.view(batch_size_3d, -1, attention_mask.size(-1))

        if output_pooler:
            if indices is not None:
                raise NotImplementedError("Pooling with unpadded input is not implemented yet.")
            if self.config.cls_embed:
                pooled_output = last_hidden_state[..., 0, :]
            elif attention_mask is not None:
                # Use the attention mask to exclude padding tokens
                expanded_attention_mask = attention_mask.unsqueeze(-1).float()
                masked_hidden_states = last_hidden_state * expanded_attention_mask
                sum_hidden_states = torch.sum(masked_hidden_states, dim=-2)
                sum_attention_mask = torch.sum(expanded_attention_mask, dim=-2)
                pooled_output = sum_hidden_states / torch.clamp(sum_attention_mask, min=1e-9)
                pooled_output = pooled_output.to(dtype=last_hidden_state.dtype)
            else:
                pooled_output = torch.mean(last_hidden_state, dim=-2)

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


@auto_docstring(
    custom_intro="""
    The metadata model from CM3P without any head or projection on top.
    """
)
class CM3PMetadataModel(CM3PPreTrainedModel):
    config_class = CM3PMetadataConfig

    def __init__(self, config: CM3PMetadataConfig):
        super().__init__(config)
        self.metadata_model = CM3PMetadataTransformer(config)
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.metadata_model.encoder.embeddings.tok_embeddings

    def set_input_embeddings(self, value):
        self.metadata_model.encoder.embeddings.tok_embeddings = value

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        indices: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        batch_size: Optional[int] = None,
        seq_len: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_pooler: bool = True,
    ) -> BaseModelOutputWithPooling:
        r"""
        indices (`torch.Tensor` of shape `(total_unpadded_tokens,)`, *optional*):
            Indices of the non-padding tokens in the input sequence. Used for unpadding the output.
        cu_seqlens (`torch.Tensor` of shape `(batch + 1,)`, *optional*):
            Cumulative sequence lengths of the input sequences. Used to index the unpadded tensors.
        max_seqlen (`int`, *optional*):
            Maximum sequence length in the batch excluding padding tokens. Used to unpad input_ids and pad output tensors.
        batch_size (`int`, *optional*):
            Batch size of the input sequences. Used to pad the output tensors.
        seq_len (`int`, *optional*):
            Sequence length of the input sequences including padding tokens. Used to pad the output tensors.
        output_pooler (`bool`, *optional*, defaults to `True`):
            Whether to return the pooled output of the model. The pooled output is usually the representation of
            the first token (CLS) or the mean of the token representations.
        """
        return self.metadata_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            indices=indices,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            batch_size=batch_size,
            seq_len=seq_len,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_pooler=output_pooler,
        )


class CM3PMultiModalProjector(nn.Module):
    def __init__(self, config: CM3PAudioConfig):
        super().__init__()
        self.linear_1 = nn.Linear(config.projector_intermediate_size, config.projector_dim, bias=False)
        self.act = ACT2FN[config.projector_hidden_act]
        self.linear_2 = nn.Linear(config.projector_dim, config.projector_dim, bias=False)

    def forward(self, audio_features):
        hidden_states = self.linear_1(audio_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class CM3PAudioEncoder(nn.Module):
    def __init__(self, config: CM3PAudioConfig):
        super().__init__()
        self.config = config
        self.conv1 = nn.Conv1d(config.n_mels, config.hidden_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(config.hidden_size, config.hidden_size, kernel_size=3, stride=2, padding=1)
        self.encoder = ModernBertModel(config)
        self.multi_modal_projector = CM3PMultiModalProjector(config)

    def forward(
            self,
            input_features: torch.FloatTensor,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
    ) -> CM3PAudioModelOutput:
        # Conv layers from Whisper followed by an modern Bert encoder
        inputs_embeds = nn.functional.gelu(self.conv1(input_features))
        inputs_embeds = nn.functional.gelu(self.conv2(inputs_embeds))

        inputs_embeds = inputs_embeds.permute(0, 2, 1).contiguous()

        position_ids = torch.arange(inputs_embeds.size(1), device=inputs_embeds.device).unsqueeze(0).repeat(
            inputs_embeds.size(0), 1)

        encoder_outputs: BaseModelOutput = self.encoder(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        # Reduce the sequence length and project to the beatmap hidden size
        audio_hidden_states = encoder_outputs.last_hidden_state
        audio_hidden_states = audio_hidden_states.reshape(-1, self.config.projector_intermediate_size)
        audio_embeds = self.multi_modal_projector(audio_hidden_states)

        audio_outputs = CM3PAudioModelOutput(
            audio_embeds=audio_embeds,
            last_hidden_state=encoder_outputs.last_hidden_state,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

        return audio_outputs


class CM3PBeatmapTransformer(nn.Module):
    def __init__(self, config: CM3PBeatmapConfig):
        super().__init__()
        self.config = config
        self.audio_encoder = CM3PAudioEncoder(config.audio_config)
        self.encoder = ModernBertModel(config)

    def get_input_embeddings(self):
        return self.encoder.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.encoder.set_input_embeddings(value)

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        input_features: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        sliding_window_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        indices: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        batch_size: Optional[int] = None,
        seq_len: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_pooler: bool = True,
    ) -> CM3PBeatmapModelOutput:
        r"""
        input_features (`torch.FloatTensor` of shape `(batch_size, num_frames, num_mels)`, *optional*):
            The audio frames to be processed by the audio encoder. If provided, the model will use these frames to
            compute the beatmap embeddings.
        sliding_window_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding or far-away tokens. In ModernBert, only every few layers
            perform global attention, while the rest perform local attention. This mask is used to avoid attending to
            far-away tokens in the local attention layers when not using Flash Attention.
        indices (`torch.Tensor` of shape `(total_unpadded_tokens,)`, *optional*):
            Indices of the non-padding tokens in the input sequence. Used for unpadding the output.
        cu_seqlens (`torch.Tensor` of shape `(batch + 1,)`, *optional*):
            Cumulative sequence lengths of the input sequences. Used to index the unpadded tensors.
        max_seqlen (`int`, *optional*):
            Maximum sequence length in the batch excluding padding tokens. Used to unpad input_ids and pad output tensors.
        batch_size (`int`, *optional*):
            Batch size of the input sequences. Used to pad the output tensors.
        seq_len (`int`, *optional*):
            Sequence length of the input sequences including padding tokens. Used to pad the output tensors.
        output_pooler (`bool`, *optional*, defaults to `True`):
            Whether to return the pooled output of the model. The pooled output is usually the representation of
            the first token (CLS) or the mean of the token representations.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        audio_model_outputs = None
        if input_features is not None:
            audio_model_outputs: CM3PAudioModelOutput = self.audio_encoder(
                input_features=input_features,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            # replace text-audio token placeholders with audio embeddings
            audio_embeds = audio_model_outputs.audio_embeds.to(dtype=inputs_embeds.dtype)
            audio_token_mask = input_ids == self.config.audio_token_id
            inputs_embeds[audio_token_mask] = audio_embeds

        encoder_outputs: BaseModelOutput = self.encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            sliding_window_mask=sliding_window_mask,
            position_ids=position_ids,
            indices=indices,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            batch_size=batch_size,
            seq_len=seq_len,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        last_hidden_state = encoder_outputs.last_hidden_state
        pooled_output = None

        if output_pooler:
            if indices is not None:
                if self.config.cls_embed:
                    pooled_output = last_hidden_state[cu_seqlens[:-1]]
                else:
                    raise NotImplementedError("Pooling with unpadded input is not implemented yet.")
            else:
                if self.config.cls_embed:
                    pooled_output = last_hidden_state[:, 0]
                elif attention_mask is not None:
                    # Use the attention mask to exclude padding tokens
                    expanded_attention_mask = attention_mask.unsqueeze(-1).float()
                    masked_hidden_states = last_hidden_state * expanded_attention_mask
                    sum_hidden_states = torch.sum(masked_hidden_states, dim=1)
                    sum_attention_mask = torch.sum(expanded_attention_mask, dim=1)
                    pooled_output = sum_hidden_states / torch.clamp(sum_attention_mask, min=1e-9)
                    pooled_output = pooled_output.to(dtype=last_hidden_state.dtype)
                else:
                    pooled_output = torch.mean(last_hidden_state, dim=1)

        return CM3PBeatmapModelOutput(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            audio_model_output=audio_model_outputs,
        )


@auto_docstring(
    custom_intro="""
    The beatmap model from CM3P without any head or projection on top.
    """
)
class CM3PBeatmapModel(CM3PPreTrainedModel):
    config_class = CM3PBeatmapConfig
    main_input_name = "input_ids"

    def __init__(self, config: CM3PBeatmapConfig):
        super().__init__(config)
        self.beatmap_model = CM3PBeatmapTransformer(config)
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.beatmap_model.encoder.embeddings.tok_embeddings

    def set_input_embeddings(self, value):
        self.beatmap_model.encoder.embeddings.tok_embeddings = value

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        input_features: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        indices: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        batch_size: Optional[int] = None,
        seq_len: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_pooler: bool = True,
    ) -> CM3PBeatmapModelOutput:
        r"""
        input_features (`torch.FloatTensor` of shape `(batch_size, num_frames, num_mels)`, *optional*):
            The audio frames to be processed by the audio encoder. If provided, the model will use these frames to
            compute the beatmap embeddings.
        indices (`torch.Tensor` of shape `(total_unpadded_tokens,)`, *optional*):
            Indices of the non-padding tokens in the input sequence. Used for unpadding the output.
        cu_seqlens (`torch.Tensor` of shape `(batch + 1,)`, *optional*):
            Cumulative sequence lengths of the input sequences. Used to index the unpadded tensors.
        max_seqlen (`int`, *optional*):
            Maximum sequence length in the batch excluding padding tokens. Used to unpad input_ids and pad output tensors.
        batch_size (`int`, *optional*):
            Batch size of the input sequences. Used to pad the output tensors.
        seq_len (`int`, *optional*):
            Sequence length of the input sequences including padding tokens. Used to pad the output tensors.
        output_pooler (`bool`, *optional*, defaults to `True`):
            Whether to return the pooled output of the model. The pooled output is usually the representation of
            the first token (CLS) or the mean of the token representations.
        """

        return self.beatmap_model(
            input_ids=input_ids,
            input_features=input_features,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            indices=indices,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            batch_size=batch_size,
            seq_len=seq_len,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_pooler=output_pooler,
        )


@auto_docstring
class CM3PModel(CM3PPreTrainedModel):
    config_class = CM3PConfig

    def __init__(self, config: CM3PConfig):
        super().__init__(config)

        if not isinstance(config.metadata_config, CM3PMetadataConfig):
            raise TypeError(
                "config.metadata_config is expected to be of type CM3PMetadataConfig but is of type"
                f" {type(config.metadata_config)}."
            )

        if not isinstance(config.beatmap_config, CM3PBeatmapConfig):
            raise TypeError(
                "config.beatmap_config is expected to be of type CM3PBeatmapConfig but is of type"
                f" {type(config.beatmap_config)}."
            )

        metadata_config = config.metadata_config
        beatmap_config = config.beatmap_config

        self.projection_dim = config.projection_dim
        self.metadata_embed_dim = metadata_config.hidden_size
        self.beatmap_embed_dim = beatmap_config.hidden_size
        self.loss_type = config.loss_type

        metadata_model = CM3PMetadataModel._from_config(metadata_config)
        self.metadata_model = metadata_model.metadata_model

        beatmap_model = CM3PBeatmapModel._from_config(beatmap_config)
        self.beatmap_model = beatmap_model.beatmap_model

        self.beatmap_projection = nn.Linear(self.beatmap_embed_dim, self.projection_dim, bias=False)
        self.metadata_projection = nn.Linear(self.metadata_embed_dim, self.projection_dim, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(self.config.logit_scale_init_value))

        self.head = CM3PPredictionHead(beatmap_config)
        self.decoder = nn.Linear(beatmap_config.hidden_size, beatmap_config.vocab_size, bias=beatmap_config.decoder_bias)

        # Initialize weights and apply final processing
        self.post_init()

    @auto_docstring
    def get_metadata_features(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> torch.FloatTensor:
        r"""
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            The input IDs for the metadata model. The model will use these IDs to compute the metadata embeddings.
        Returns:
            metadata_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The metadata embeddings obtained by
            applying the projection layer to the pooled output of [`CM3PMetadataModel`].
        """
        # Use CM3P model's config for some fields (if specified) instead of those of beatmap & metadata components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        metadata_outputs: BaseModelOutputWithPooling = self.metadata_model(
            input_ids=input_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        pooled_output = metadata_outputs.pooler_output
        metadata_features = self.metadata_projection(pooled_output)

        return metadata_features

    @auto_docstring
    def get_beatmap_features(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        input_features: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> torch.FloatTensor:
        r"""
        input_features (`torch.FloatTensor` of shape `(batch_size, num_frames, num_mels)`, *optional*):
            The audio frames to be processed by the audio encoder. If provided, the model will use these frames to
            compute the beatmap embeddings.
        Returns:
            beatmap_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The beatmap embeddings obtained by
            applying the projection layer to the pooled output of [`CM3PBeatmapModel`].
        """
        # Use CM3P model's config for some fields (if specified) instead of those of beatmap & metadata components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        beatmap_outputs: BaseModelOutputWithPooling = self.beatmap_model(
            input_ids=input_ids,
            input_features=input_features,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        pooled_output = beatmap_outputs.pooler_output
        beatmap_features = self.beatmap_projection(pooled_output)

        return beatmap_features

    @torch.compile(dynamic=True)
    def compiled_head(self, output: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.head(output))

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        input_features: Optional[torch.FloatTensor] = None,
        metadata_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        metadata_attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        metadata_variation_classes: Optional[torch.LongTensor] = None,
        labels: Optional[torch.Tensor] = None,
        indices: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        batch_size: Optional[int] = None,
        seq_len: Optional[int] = None,
        return_loss: Optional[bool] = True,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        **kwargs,
    ) -> CM3POutput:
        r"""
        input_features (`torch.FloatTensor` of shape `(batch_size, num_frames, num_mels)`, *optional*):
            The audio frames to be processed by the audio encoder. If provided, the model will use these frames to
            compute the beatmap embeddings.
        metadata_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)` or `(batch_size, variations, sequence_length)`):
            The input IDs for the metadata model. The model will use these IDs to compute the metadata embeddings.
        metadata_attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)` or `(batch_size, variations, sequence_length)`, *optional*):
            The attention mask for the metadata model. If provided, the model will not attend to the padded tokens.
        metadata_variation_classes (`torch.LongTensor` of shape `(batch_size, variations)`, *optional*):
            Tells the model what kind of variation each metadata sequence is.
            0 indicates the original metadata, -1 indicates paddidng, and any positive integer indicates a specific variation class.
        indices (`torch.Tensor` of shape `(total_unpadded_tokens,)`, *optional*):
            Indices of the non-padding tokens in the input sequence. Used for unpadding the output.
        cu_seqlens (`torch.Tensor` of shape `(batch + 1,)`, *optional*):
            Cumulative sequence lengths of the input sequences. Used to index the unpadded tensors.
        max_seqlen (`int`, *optional*):
            Maximum sequence length in the batch excluding padding tokens. Used to unpad input_ids and pad output tensors.
        batch_size (`int`, *optional*):
            Batch size of the input sequences. Used to pad the output tensors.
        seq_len (`int`, *optional*):
            Sequence length of the input sequences including padding tokens. Used to pad the output tensors.
        return_loss (`bool`, *optional*):
            Whether to return the contrastive loss.
        """
        # Use CM3P model's config for some fields (if specified) instead of those of beatmap & metadata components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        if metadata_ids.dim() == 3 and return_loss and metadata_variation_classes is None:
            raise ValueError("When providing multiple metadata variations, metadata_variation_classes must be provided in order to compute loss correctly.")

        # noinspection PyProtectedMember
        if self.config._attn_implementation == "flash_attention_2":
            if indices is None and cu_seqlens is None and max_seqlen is None:
                if batch_size is None and seq_len is None:
                    if inputs_embeds is not None:
                        batch_size, seq_len = inputs_embeds.shape[:2]
                    else:
                        batch_size, seq_len = input_ids.shape[:2]
                device = input_ids.device if input_ids is not None else inputs_embeds.device

                if attention_mask is None:
                    attention_mask = torch.ones((batch_size, seq_len), device=device, dtype=torch.bool)

                if inputs_embeds is None:
                    with torch.no_grad():
                        input_ids, indices, cu_seqlens, max_seqlen, position_ids, labels = _unpad_cm3p_input(
                            inputs=input_ids, attention_mask=attention_mask, position_ids=position_ids, labels=labels
                        )
                else:
                    inputs_embeds, indices, cu_seqlens, max_seqlen, position_ids, labels = _unpad_cm3p_input(
                        inputs=inputs_embeds, attention_mask=attention_mask, position_ids=position_ids, labels=labels
                    )

        beatmap_outputs: BaseModelOutputWithPooling = self.beatmap_model(
            input_ids=input_ids,
            input_features=input_features,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            indices=indices,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            batch_size=batch_size,
            seq_len=seq_len,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        metadata_outputs: BaseModelOutputWithPooling = self.metadata_model(
            input_ids=metadata_ids,
            attention_mask=metadata_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        beatmap_embeds = beatmap_outputs.pooler_output
        beatmap_embeds = self.beatmap_projection(beatmap_embeds)

        metadata_embeds = metadata_outputs.pooler_output
        metadata_embeds = self.metadata_projection(metadata_embeds)

        # normalized features
        beatmap_embeds = beatmap_embeds / _get_vector_norm(beatmap_embeds)
        metadata_embeds = metadata_embeds / _get_vector_norm(metadata_embeds)

        # cosine similarity as logits
        logits_per_metadata = torch.matmul(metadata_embeds, beatmap_embeds.t().to(metadata_embeds.device))
        logits_per_metadata = logits_per_metadata * self.logit_scale.exp().to(metadata_embeds.device)

        if logits_per_metadata.dim() == 3:
            logits_per_beatmap = logits_per_metadata.permute(2, 0, 1)
        else:
            logits_per_beatmap = logits_per_metadata.t()

        loss = None
        if return_loss:
            loss = cm3p_loss(logits_per_metadata, metadata_variation_classes)

        logits = (
            self.compiled_head(beatmap_outputs.last_hidden_state)
            if self.config.beatmap_config.reference_compile
            else self.decoder(self.head(beatmap_outputs.last_hidden_state))
        )

        if labels is not None and return_loss:
            mlm_loss = self.loss_function(logits, labels, vocab_size=self.config.beatmap_config.vocab_size, **kwargs)
            loss += 0.5 * mlm_loss

        # noinspection PyProtectedMember
        if self.config._attn_implementation == "flash_attention_2":
            with nullcontext() if self.config.beatmap_config.repad_logits_with_grad or labels is None else torch.no_grad():
                logits = _pad_cm3p_output(inputs=logits, indices=indices, batch=batch_size, seqlen=seq_len)

        return CM3POutput(
            loss=loss,
            logits_per_beatmap=logits_per_beatmap,
            logits_per_metadata=logits_per_metadata,
            metadata_embeds=metadata_embeds,
            beatmap_embeds=beatmap_embeds,
            logits=logits,
            metadata_model_output=metadata_outputs,
            beatmap_model_output=beatmap_outputs,
        )


@auto_docstring
class CM3PMetadataModelWithProjection(CM3PPreTrainedModel):
    config_class = CM3PMetadataConfig

    def __init__(self, config: CM3PMetadataConfig):
        super().__init__(config)

        metadata_model = CM3PMetadataModel._from_config(config)
        self.metadata_model = metadata_model.metadata_model

        self.metadata_projection = nn.Linear(config.hidden_size, config.projection_dim, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.metadata_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.metadata_model.set_input_embeddings(value)

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> CM3PMetadataModelOutput:
        r"""
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            The input IDs for the metadata model. The model will use these IDs to compute the metadata embeddings.
        Returns:
            metadata_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The metadata embeddings obtained by
            applying the projection layer to the pooled output of [`CM3PMetadataModel`].
        """
        metadata_outputs: BaseModelOutputWithPooling = self.metadata_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        pooled_output = metadata_outputs.pooler_output
        metadata_embeds = self.metadata_projection(pooled_output)

        return CM3PMetadataModelOutput(
            metadata_embeds=metadata_embeds,
            last_hidden_state=metadata_outputs.last_hidden_state,
            hidden_states=metadata_outputs.hidden_states,
            attentions=metadata_outputs.attentions,
        )


@auto_docstring
class CM3PBeatmapModelWithProjection(CM3PPreTrainedModel):
    config_class = CM3PBeatmapConfig

    def __init__(self, config: CM3PBeatmapConfig):
        super().__init__(config)

        beatmap_model = CM3PBeatmapModel._from_config(config)
        self.beatmap_model = beatmap_model.beatmap_model

        self.beatmap_projection = nn.Linear(config.hidden_size, config.projection_dim, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.beatmap_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.beatmap_model.set_input_embeddings(value)

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        input_features: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> CM3PBeatmapModelOutput:
        r"""
        input_features (`torch.FloatTensor` of shape `(batch_size, num_frames, num_mels)`, *optional*):
            The audio frames to be processed by the audio encoder. If provided, the model will use these frames to
            compute the beatmap embeddings.
        Returns:
            beatmap_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The beatmap embeddings obtained by
            applying the projection layer to the pooled output of [`CM3PBeatmapModel`].
        """
        beatmap_outputs: BaseModelOutputWithPooling = self.beatmap_model(
            input_ids=input_ids,
            input_features=input_features,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        pooled_output = beatmap_outputs.pooler_output
        beatmap_embeds = self.beatmap_projection(pooled_output)

        return CM3PBeatmapModelOutput(
            beatmap_embeds=beatmap_embeds,
            pooler_output=pooled_output,
            last_hidden_state=beatmap_outputs.last_hidden_state,
            hidden_states=beatmap_outputs.hidden_states,
            attentions=beatmap_outputs.attentions,
        )


@auto_docstring(
    custom_intro="""
    CM3P beatmap encoder with an beatmap classification head on top (a linear layer on top of the pooled final hidden states of
    the beatmap embeddings) e.g. for BeatmapNet.
    """
)
class CM3PForBeatmapClassification(CM3PPreTrainedModel):
    config_class = CM3PBeatmapConfig
    base_model_prefix = "beatmap_model"

    def __init__(self, config: CM3PBeatmapConfig) -> None:
        super().__init__(config)

        self.num_labels = config.num_labels
        beatmap_model = CM3PBeatmapModel._from_config(config)
        self.beatmap_model = beatmap_model.beatmap_model

        # Classifier head
        self.classifier = (
            nn.Linear(config.hidden_size, config.num_labels) if config.num_labels > 0 else nn.Identity()
        )

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        input_features: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> BeatmapClassifierOutput:
        r"""
        input_features (`torch.FloatTensor` of shape `(batch_size, num_frames, num_mels)`, *optional*):
            The audio frames to be processed by the audio encoder. If provided, the model will use these frames to
            compute the beatmap embeddings.
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the beatmap classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        outputs: BaseModelOutputWithPooling = self.beatmap_model(
            input_ids=input_ids,
            input_features=input_features,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        return BeatmapClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class CM3PPredictionHead(nn.Module):
    def __init__(self, config: CM3PBeatmapConfig):
        super().__init__()
        self.config = config
        self.dense = nn.Linear(config.hidden_size, config.hidden_size, config.classifier_bias)
        self.act = ACT2FN[config.classifier_activation]
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=config.norm_bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.norm(self.act(self.dense(hidden_states)))


class CM3PForMaskedLM(CM3PPreTrainedModel):
    config_class = CM3PBeatmapConfig
    base_model_prefix = "beatmap_model"
    _tied_weights_keys = ["decoder.weight"]

    def __init__(self, config: CM3PBeatmapConfig):
        super().__init__(config)
        self.config = config
        beatmap_model = CM3PBeatmapModel._from_config(config)
        self.beatmap_model = beatmap_model.beatmap_model
        self.head = CM3PPredictionHead(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=config.decoder_bias)

        self.sparse_prediction = self.config.sparse_prediction
        self.sparse_pred_ignore_index = self.config.sparse_pred_ignore_index

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.decoder

    def set_output_embeddings(self, new_embeddings: nn.Linear):
        self.decoder = new_embeddings

    @torch.compile(dynamic=True)
    def compiled_head(self, output: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.head(output))

    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        input_features: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        sliding_window_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        indices: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        batch_size: Optional[int] = None,
        seq_len: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        **kwargs,
    ) -> Union[tuple[torch.Tensor], MaskedLMOutput]:
        r"""
        input_features (`torch.FloatTensor` of shape `(batch_size, num_frames, num_mels)`, *optional*):
            The audio frames to be processed by the audio encoder. If provided, the model will use these frames to
            compute the beatmap embeddings.
        sliding_window_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding or far-away tokens. In ModernBert, only every few layers
            perform global attention, while the rest perform local attention. This mask is used to avoid attending to
            far-away tokens in the local attention layers when not using Flash Attention.
        indices (`torch.Tensor` of shape `(total_unpadded_tokens,)`, *optional*):
            Indices of the non-padding tokens in the input sequence. Used for unpadding the output.
        cu_seqlens (`torch.Tensor` of shape `(batch + 1,)`, *optional*):
            Cumulative sequence lengths of the input sequences. Used to index the unpadded tensors.
        max_seqlen (`int`, *optional*):
            Maximum sequence length in the batch excluding padding tokens. Used to unpad input_ids and pad output tensors.
        batch_size (`int`, *optional*):
            Batch size of the input sequences. Used to pad the output tensors.
        seq_len (`int`, *optional*):
            Sequence length of the input sequences including padding tokens. Used to pad the output tensors.
        """
        # noinspection PyProtectedMember
        if self.config._attn_implementation == "flash_attention_2":
            if indices is None and cu_seqlens is None and max_seqlen is None:
                if batch_size is None and seq_len is None:
                    if inputs_embeds is not None:
                        batch_size, seq_len = inputs_embeds.shape[:2]
                    else:
                        batch_size, seq_len = input_ids.shape[:2]
                device = input_ids.device if input_ids is not None else inputs_embeds.device

                if attention_mask is None:
                    attention_mask = torch.ones((batch_size, seq_len), device=device, dtype=torch.bool)

                if inputs_embeds is None:
                    with torch.no_grad():
                        input_ids, indices, cu_seqlens, max_seqlen, position_ids, labels = _unpad_cm3p_input(
                            inputs=input_ids, attention_mask=attention_mask, position_ids=position_ids, labels=labels
                        )
                else:
                    inputs_embeds, indices, cu_seqlens, max_seqlen, position_ids, labels = _unpad_cm3p_input(
                        inputs=inputs_embeds, attention_mask=attention_mask, position_ids=position_ids, labels=labels
                    )

        outputs = self.beatmap_model(
            input_ids=input_ids,
            input_features=input_features,
            attention_mask=attention_mask,
            sliding_window_mask=sliding_window_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            indices=indices,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            batch_size=batch_size,
            seq_len=seq_len,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_pooler=False,
        )
        last_hidden_state = outputs.last_hidden_state

        if self.sparse_prediction and labels is not None:
            # flatten labels and output first
            labels = labels.view(-1)
            last_hidden_state = last_hidden_state.view(labels.shape[0], -1)

            # then filter out the non-masked tokens
            mask_tokens = labels != self.sparse_pred_ignore_index
            last_hidden_state = last_hidden_state[mask_tokens]
            labels = labels[mask_tokens]

        logits = (
            self.compiled_head(last_hidden_state)
            if self.config.reference_compile
            else self.decoder(self.head(last_hidden_state))
        )

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, vocab_size=self.config.vocab_size, **kwargs)

        # noinspection PyProtectedMember
        if self.config._attn_implementation == "flash_attention_2":
            with nullcontext() if self.config.repad_logits_with_grad or labels is None else torch.no_grad():
                logits = _pad_cm3p_output(inputs=logits, indices=indices, batch=batch_size, seqlen=seq_len)

        return MaskedLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


AutoModel.register(CM3PMetadataConfig, CM3PMetadataModel)
AutoModel.register(CM3PBeatmapConfig, CM3PBeatmapModel)
AutoModel.register(CM3PConfig, CM3PModel)
AutoModelForSequenceClassification.register(CM3PBeatmapConfig, CM3PForBeatmapClassification)
AutoModelForMaskedLM.register(CM3PBeatmapConfig, CM3PForMaskedLM)

__all__ = [
    "CM3PModel",
    "CM3PPreTrainedModel",
    "CM3PMetadataModel",
    "CM3PMetadataModelWithProjection",
    "CM3PBeatmapModel",
    "CM3PBeatmapModelWithProjection",
    "CM3PForBeatmapClassification",
]
