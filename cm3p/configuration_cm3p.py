"""CM3P model configuration"""
from typing import Literal

from transformers import AutoConfig
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)


class CM3PMetadataConfig(PretrainedConfig):
    model_type = "cm3p_metadata_model"
    base_config_key = "metadata_config"

    def __init__(
        self,
        projection_dim=512,
        hidden_act="quick_gelu",
        layer_norm_eps=1e-5,
        initializer_factor=1.0,
        # stuff
        vocab_size=1000,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=6,
        num_attention_heads=4,
        hidden_activation="gelu",
        max_position_embeddings=128,
        initializer_range=0.02,
        initializer_cutoff_factor=2.0,
        norm_eps=1e-5,
        norm_bias=False,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        cls_token_id=3,
        sep_token_id=4,
        global_rope_theta=10000.0,
        attention_bias=False,
        attention_dropout=0.0,
        global_attn_every_n_layers=1,
        local_attention=128,
        local_rope_theta=10000.0,
        embedding_dropout=0.0,
        mlp_bias=False,
        mlp_dropout=0.0,
        decoder_bias=True,
        classifier_pooling: Literal["cls", "mean"] = "cls",
        classifier_dropout=0.0,
        classifier_bias=False,
        classifier_activation="gelu",
        deterministic_flash_attn=False,
        sparse_prediction=False,
        sparse_pred_ignore_index=-100,
        reference_compile=None,
        repad_logits_with_grad=False,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            cls_token_id=cls_token_id,
            sep_token_id=sep_token_id,
            **kwargs,
        )

        self.projection_dim = projection_dim
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.initializer_factor = initializer_factor
        self.attention_dropout = attention_dropout

        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.initializer_range = initializer_range
        self.initializer_cutoff_factor = initializer_cutoff_factor
        self.norm_eps = norm_eps
        self.norm_bias = norm_bias
        self.global_rope_theta = global_rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.hidden_activation = hidden_activation
        self.global_attn_every_n_layers = global_attn_every_n_layers
        self.local_attention = local_attention
        self.local_rope_theta = local_rope_theta
        self.embedding_dropout = embedding_dropout
        self.mlp_bias = mlp_bias
        self.mlp_dropout = mlp_dropout
        self.decoder_bias = decoder_bias
        self.classifier_pooling = classifier_pooling
        self.classifier_dropout = classifier_dropout
        self.classifier_bias = classifier_bias
        self.classifier_activation = classifier_activation
        self.deterministic_flash_attn = deterministic_flash_attn
        self.sparse_prediction = sparse_prediction
        self.sparse_pred_ignore_index = sparse_pred_ignore_index
        self.reference_compile = reference_compile
        self.repad_logits_with_grad = repad_logits_with_grad

        if self.classifier_pooling not in ["cls", "mean"]:
            raise ValueError(
                f'Invalid value for `classifier_pooling`, should be either "cls" or "mean", but is {self.classifier_pooling}.'
            )

    def to_dict(self):
        output = super().to_dict()
        output.pop("reference_compile", None)
        return output


class CM3PAudioConfig(PretrainedConfig):
    model_type = "cm3p_audio_model"
    base_config_key = "audio_config"

    def __init__(
        self,
        hidden_size=512,
        intermediate_size=1024,
        num_hidden_layers=6,
        num_attention_heads=8,
        hidden_activation="gelu",
        max_position_embeddings=4096,
        initializer_range=0.02,
        initializer_cutoff_factor=2.0,
        norm_eps=1e-5,
        norm_bias=False,
        global_rope_theta=160000.0,
        attention_bias=False,
        attention_dropout=0.0,
        global_attn_every_n_layers=3,
        local_attention=128,
        local_rope_theta=10000.0,
        embedding_dropout=0.0,
        mlp_bias=False,
        mlp_dropout=0.0,
        decoder_bias=True,
        classifier_pooling: Literal["cls", "mean"] = "cls",
        classifier_dropout=0.0,
        classifier_bias=False,
        classifier_activation="gelu",
        deterministic_flash_attn=False,
        sparse_prediction=False,
        sparse_pred_ignore_index=-100,
        reference_compile=None,
        repad_logits_with_grad=False,

        projector_intermediate_size=2048,  # 4 * hidden_size for a 4x reduction in tokens
        projector_dim=768,
        projector_hidden_act="gelu",

        sample_rate: int = 16000,
        n_ftt: int = 2048,
        n_mels: int = 80,
        hop_length: int = 128,
        f_min: int = 0,
        f_max: int = 8000,
        pad_mode: str = "constant",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = 1
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.initializer_range = initializer_range
        self.initializer_cutoff_factor = initializer_cutoff_factor
        self.norm_eps = norm_eps
        self.norm_bias = norm_bias
        self.global_rope_theta = global_rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.hidden_activation = hidden_activation
        self.global_attn_every_n_layers = global_attn_every_n_layers
        self.local_attention = local_attention
        self.local_rope_theta = local_rope_theta
        self.embedding_dropout = embedding_dropout
        self.mlp_bias = mlp_bias
        self.mlp_dropout = mlp_dropout
        self.decoder_bias = decoder_bias
        self.classifier_pooling = classifier_pooling
        self.classifier_dropout = classifier_dropout
        self.classifier_bias = classifier_bias
        self.classifier_activation = classifier_activation
        self.deterministic_flash_attn = deterministic_flash_attn
        self.sparse_prediction = sparse_prediction
        self.sparse_pred_ignore_index = sparse_pred_ignore_index
        self.reference_compile = reference_compile
        self.repad_logits_with_grad = repad_logits_with_grad

        self.projector_intermediate_size = projector_intermediate_size
        self.projector_dim = projector_dim
        self.projector_hidden_act = projector_hidden_act

        self.sample_rate = sample_rate
        self.n_ftt = n_ftt
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.f_min = f_min
        self.f_max = f_max
        self.pad_mode = pad_mode

        if self.classifier_pooling not in ["cls", "mean"]:
            raise ValueError(
                f'Invalid value for `classifier_pooling`, should be either "cls" or "mean", but is {self.classifier_pooling}.'
            )

    def to_dict(self):
        output = super().to_dict()
        output.pop("reference_compile", None)
        return output


class CM3PBeatmapConfig(PretrainedConfig):
    model_type = "cm3p_beatmap_model"
    base_config_key = "beatmap_config"
    sub_configs = {"audio_config": CM3PAudioConfig}

    def __init__(
        self,
        audio_config: CM3PAudioConfig = None,
        audio_sos_token_id=3164,
        audio_eos_token_id=3165,
        audio_token_id=3166,
        # stuff
        projection_dim=512,
        num_channels=3,
        image_size=224,
        patch_size=32,
        hidden_act="quick_gelu",
        layer_norm_eps=1e-5,
        initializer_factor=1.0,
        # stuff
        vocab_size=3167,
        hidden_size=768,
        intermediate_size=1152,
        num_hidden_layers=22,
        num_attention_heads=12,
        hidden_activation="gelu",
        max_position_embeddings=8192,
        initializer_range=0.02,
        initializer_cutoff_factor=2.0,
        norm_eps=1e-5,
        norm_bias=False,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        cls_token_id=3,
        sep_token_id=4,
        global_rope_theta=160000.0,
        attention_bias=False,
        attention_dropout=0.0,
        global_attn_every_n_layers=3,
        local_attention=128,
        local_rope_theta=10000.0,
        embedding_dropout=0.0,
        mlp_bias=False,
        mlp_dropout=0.0,
        decoder_bias=True,
        classifier_pooling: Literal["cls", "mean"] = "cls",
        classifier_dropout=0.0,
        classifier_bias=False,
        classifier_activation="gelu",
        deterministic_flash_attn=False,
        sparse_prediction=False,
        sparse_pred_ignore_index=-100,
        reference_compile=None,
        repad_logits_with_grad=False,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            cls_token_id=cls_token_id,
            sep_token_id=sep_token_id,
            **kwargs,
        )

        if audio_config is None:
            audio_config = {}
            logger.info("`audio_config` is `None`. Initializing the `CM3PAudioConfig` with default values.")

        self.audio_config = CM3PAudioConfig(**audio_config)
        self.audio_sos_token_id = audio_sos_token_id
        self.audio_eos_token_id = audio_eos_token_id
        self.audio_token_id = audio_token_id

        self.projection_dim = projection_dim
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.initializer_factor = initializer_factor
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act

        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.initializer_range = initializer_range
        self.initializer_cutoff_factor = initializer_cutoff_factor
        self.norm_eps = norm_eps
        self.norm_bias = norm_bias
        self.global_rope_theta = global_rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.hidden_activation = hidden_activation
        self.global_attn_every_n_layers = global_attn_every_n_layers
        self.local_attention = local_attention
        self.local_rope_theta = local_rope_theta
        self.embedding_dropout = embedding_dropout
        self.mlp_bias = mlp_bias
        self.mlp_dropout = mlp_dropout
        self.decoder_bias = decoder_bias
        self.classifier_pooling = classifier_pooling
        self.classifier_dropout = classifier_dropout
        self.classifier_bias = classifier_bias
        self.classifier_activation = classifier_activation
        self.deterministic_flash_attn = deterministic_flash_attn
        self.sparse_prediction = sparse_prediction
        self.sparse_pred_ignore_index = sparse_pred_ignore_index
        self.reference_compile = reference_compile
        self.repad_logits_with_grad = repad_logits_with_grad

        if self.classifier_pooling not in ["cls", "mean"]:
            raise ValueError(
                f'Invalid value for `classifier_pooling`, should be either "cls" or "mean", but is {self.classifier_pooling}.'
            )

    def to_dict(self):
        output = super().to_dict()
        output.pop("reference_compile", None)
        return output


class CM3PConfig(PretrainedConfig):
    model_type = "cm3p"
    sub_configs = {"metadata_config": CM3PMetadataConfig, "beatmap_config": CM3PBeatmapConfig}

    def __init__(
        self,
        metadata_config=None,
        beatmap_config=None,
        projection_dim=512,
        logit_scale_init_value=2.6592,
        initializer_factor=1.0,
        initializer_range=0.02,
        **kwargs
    ):
        super().__init__(**kwargs)

        if metadata_config is None:
            metadata_config = {}
            logger.info("`text_config` is `None`. Initializing the `CM3PMetadataConfig` with default values.")

        if beatmap_config is None:
            beatmap_config = {}
            logger.info("`vision_config` is `None`. initializing the `CM3PBeatmapConfig` with default values.")

        self.metadata_config = CM3PMetadataConfig(**metadata_config)
        self.beatmap_config = CM3PBeatmapConfig(**beatmap_config)

        self.projection_dim = projection_dim
        self.logit_scale_init_value = logit_scale_init_value
        self.initializer_factor = initializer_factor
        self.initializer_range = initializer_range


AutoConfig.register("cm3p_metadata_model", CM3PMetadataConfig)
AutoConfig.register("cm3p_audio_model", CM3PAudioConfig)
AutoConfig.register("cm3p_beatmap_model", CM3PBeatmapConfig)
AutoConfig.register("cm3p", CM3PConfig)

__all__ = ["CM3PConfig", "CM3PMetadataConfig", "CM3PAudioConfig", "CM3PBeatmapConfig"]
