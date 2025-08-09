"""PyTorch CM3P model."""

from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import ModernBertModel
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput, auto_docstring, can_return_tuple, logging

from .configuration_cm3p import CM3PConfig, CM3PMetadataConfig, CM3PBeatmapConfig, CM3PAudioConfig


logger = logging.get_logger(__name__)


# contrastive loss function, adapted from
# https://sachinruk.github.io/blog/2021-03-07-clip.html
def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))


# CM3P loss function, adapted from CM3P
def cm3p_loss(similarity: torch.Tensor) -> torch.Tensor:
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
    audio_model_output: BaseModelOutput = None


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
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
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

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> BaseModelOutputWithPooling:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        if input_ids is None:
            raise ValueError("You have to specify input_ids")

        encoder_outputs: BaseModelOutput = self.encoder(
            input_ids=input_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        last_hidden_state = encoder_outputs.last_hidden_state
        pooled_output = torch.mean(last_hidden_state, dim=1)

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
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> BaseModelOutputWithPooling:
        return self.metadata_model(
            input_ids=input_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )


class CM3PAudioEncoder(nn.Module):
    def __init__(self, config: CM3PAudioConfig):
        super().__init__()
        self.config = config
        self.conv1 = nn.Conv1d(config.n_mels, config.hidden_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(config.hidden_size, config.hidden_size, kernel_size=3, stride=2, padding=1)
        self.encoder = ModernBertModel(config)

    def forward(
            self,
            input_features: torch.FloatTensor,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
    ) -> torch.tensor:
        inputs_embeds = nn.functional.gelu(self.conv1(input_features))
        inputs_embeds = nn.functional.gelu(self.conv2(inputs_embeds))

        inputs_embeds = inputs_embeds.permute(0, 2, 1)

        position_ids = torch.arange(inputs_embeds.size(1), device=inputs_embeds.device).unsqueeze(0).repeat(
            inputs_embeds.size(0), 1)

        encoder_outputs: BaseModelOutput = self.encoder(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        return encoder_outputs


class CM3PBeatmapTransformer(nn.Module):
    def __init__(self, config: CM3PBeatmapConfig):
        super().__init__()
        self.config = config
        self.audio_encoder = CM3PAudioEncoder(config.audio_config)
        self.encoder = ModernBertModel(config)

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        input_features: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> CM3PBeatmapModelOutput:
        r"""
        input_features (`torch.FloatTensor` of shape `(batch_size, num_frames, num_mels)`, *optional*):
            The audio frames to be processed by the audio encoder. If provided, the model will use these frames to
            compute the beatmap embeddings.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        if inputs_embeds is None:
            inputs_embeds = self.encoder.embeddings.tok_embeddings(input_ids)

        audio_model_outputs = None
        if input_features is not None:
            audio_model_outputs = self.audio_encoder(
                input_features=input_features,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            # replace text-audio token placeholders with audio embeddings
            audio_token_mask = input_ids == self.config.audio_token_id
            inputs_embeds[audio_token_mask] = audio_model_outputs.last_hidden_state

        encoder_outputs: BaseModelOutput = self.encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        last_hidden_state = encoder_outputs.last_hidden_state
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
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> CM3PBeatmapModelOutput:
        r"""
        input_features (`torch.FloatTensor` of shape `(batch_size, num_frames, num_mels)`, *optional*):
            The audio frames to be processed by the audio encoder. If provided, the model will use these frames to
            compute the beatmap embeddings.
        """

        return self.beatmap_model(
            input_ids=input_ids,
            input_features=input_features,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
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

        metadata_model = CM3PMetadataModel._from_config(metadata_config)
        self.metadata_model = metadata_model.metadata_model

        beatmap_model = CM3PBeatmapModel._from_config(beatmap_config)
        self.beatmap_model = beatmap_model.beatmap_model

        self.beatmap_projection = nn.Linear(self.beatmap_embed_dim, self.projection_dim, bias=False)
        self.metadata_projection = nn.Linear(self.metadata_embed_dim, self.projection_dim, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(self.config.logit_scale_init_value))

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

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        input_features: Optional[torch.FloatTensor] = None,
        metadata_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        return_loss: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> CM3POutput:
        r"""
        input_features (`torch.FloatTensor` of shape `(batch_size, num_frames, num_mels)`, *optional*):
            The audio frames to be processed by the audio encoder. If provided, the model will use these frames to
            compute the beatmap embeddings.
        metadata_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            The input IDs for the metadata model. The model will use these IDs to compute the metadata embeddings.
        return_loss (`bool`, *optional*):
            Whether to return the contrastive loss.
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

        metadata_outputs: BaseModelOutputWithPooling = self.metadata_model(
            input_ids=metadata_ids,
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

        logits_per_beatmap = logits_per_metadata.t()

        loss = None
        if return_loss:
            loss = cm3p_loss(logits_per_metadata)

        return CM3POutput(
            loss=loss,
            logits_per_beatmap=logits_per_beatmap,
            logits_per_metadata=logits_per_metadata,
            metadata_embeds=metadata_embeds,
            beatmap_embeds=beatmap_embeds,
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
    def __init__(self, config: CM3PConfig) -> None:
        super().__init__(config)

        self.num_labels = config.num_labels
        beatmap_model = CM3PBeatmapModel._from_config(config.beatmap_config)
        self.beatmap_model = beatmap_model.beatmap_model

        # Classifier head
        self.classifier = (
            nn.Linear(config.beatmap_config.hidden_size, config.num_labels) if config.num_labels > 0 else nn.Identity()
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


__all__ = [
    "CM3PModel",
    "CM3PPreTrainedModel",
    "CM3PMetadataModel",
    "CM3PMetadataModelWithProjection",
    "CM3PBeatmapModel",
    "CM3PBeatmapModelWithProjection",
    "CM3PForBeatmapClassification",
]
