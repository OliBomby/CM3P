"""CM3P model configuration"""
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
        initializer_factor=1.0,

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
        deterministic_flash_attn=False,
        reference_compile=None,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )

        self.projection_dim = projection_dim
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
        self.deterministic_flash_attn = deterministic_flash_attn
        self.reference_compile = reference_compile

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
        deterministic_flash_attn=False,
        reference_compile=None,

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
        self.deterministic_flash_attn = deterministic_flash_attn
        self.reference_compile = reference_compile

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
        audio_config: dict = None,
        audio_sos_token_id=3164,
        audio_eos_token_id=3165,
        audio_token_id=3166,

        projection_dim=512,
        initializer_factor=1.0,

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
        self.initializer_factor = initializer_factor
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
        self.deterministic_flash_attn = deterministic_flash_attn
        self.sparse_prediction = sparse_prediction
        self.sparse_pred_ignore_index = sparse_pred_ignore_index
        self.reference_compile = reference_compile
        self.repad_logits_with_grad = repad_logits_with_grad

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
            logger.debug("`metadata_config` is `None`. Initializing the `CM3PMetadataConfig` with default values.")

        if beatmap_config is None:
            beatmap_config = {}
            logger.debug("`beatmap_config` is `None`. initializing the `CM3PBeatmapConfig` with default values.")

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
