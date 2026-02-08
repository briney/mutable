# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT

from typing import Optional, Literal

from transformers import PretrainedConfig


class MutableConfig(PretrainedConfig):
    """
    Configuration for the Mutable encoder-decoder model with Perceiver bottleneck.

    Parameters
    ----------
    vocab_size : int, default=32
        Vocabulary size.
    hidden_size : int, default=320
        Hidden dimension for encoder and decoder.
    num_encoder_layers : int, default=6
        Number of encoder transformer layers.
    num_decoder_layers : int, default=6
        Number of decoder transformer layers.
    num_attention_heads : int, default=20
        Number of attention heads.
    intermediate_size : int, default=None
        FFN intermediate dimension. Defaults to hidden_size * 4.
    activation : str, default="swiglu"
        Activation function for FFN layers.
    position_embedding_type : str, default="rotary"
        Type of positional embeddings ("rotary" or "absolute").
    num_latents : int, default=32
        Number of Perceiver bottleneck latent vectors.
    latent_dim : int, default=None
        Dimension of latent vectors. Defaults to hidden_size.
    dropout : float, default=0.1
        Default dropout rate.
    attention_dropout : float, default=None
        Dropout for attention layers. Defaults to dropout.
    hidden_dropout : float, default=None
        Dropout for hidden layers. Defaults to dropout.
    ffn_bias : bool, default=True
        Whether to use bias in FFN layers.
    max_position_embeddings : int, default=512
        Maximum sequence length (for absolute position embeddings).
    initializer_range : float, default=0.02
        Standard deviation for weight initialization.
    layer_norm_eps : float, default=1e-5
        Epsilon for layer normalization.
    pad_token_id : int, default=1
        Padding token ID.
    bos_token_id : int, default=0
        Beginning-of-sequence token ID.
    eos_token_id : int, default=2
        End-of-sequence token ID.
    sep_token_id : int, default=29
        Separator token ID (<sep>).
    mask_token_id : int, default=31
        Mask token ID.
    output_attentions : bool, default=False
        Whether to output attention weights.
    output_hidden_states : bool, default=False
        Whether to output all hidden states.
    return_dict : bool, default=True
        Whether to return dict outputs.
    use_cache : bool, default=True
        HuggingFace integration flag.
    """

    model_type = "mutable"

    def __init__(
        self,
        vocab_size: int = 32,
        hidden_size: int = 320,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        num_attention_heads: int = 20,
        intermediate_size: Optional[int] = None,
        activation: Literal[
            "gelu", "relu", "glu", "swiglu", "geglu", "reglu"
        ] = "swiglu",
        position_embedding_type: Literal["rotary", "absolute"] = "rotary",
        # bottleneck
        num_latents: int = 32,
        latent_dim: Optional[int] = None,
        # dropout
        dropout: float = 0.1,
        attention_dropout: Optional[float] = None,
        hidden_dropout: Optional[float] = None,
        # ffn
        ffn_bias: bool = True,
        # position
        max_position_embeddings: int = 512,
        # init
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-5,
        # special tokens
        pad_token_id: int = 1,
        bos_token_id: int = 0,
        eos_token_id: int = 2,
        sep_token_id: int = 29,
        mask_token_id: int = 31,
        # outputs
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        use_cache: bool = True,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )

        self.vocab_size = int(vocab_size)
        self.hidden_size = int(hidden_size)
        self.num_encoder_layers = int(num_encoder_layers)
        self.num_decoder_layers = int(num_decoder_layers)
        self.num_attention_heads = int(num_attention_heads)
        self.intermediate_size = int(intermediate_size or hidden_size * 4)
        self.activation = activation.lower()
        self.position_embedding_type = position_embedding_type.lower()

        # bottleneck
        self.num_latents = int(num_latents)
        self.latent_dim = int(latent_dim or hidden_size)

        # dropout
        self.dropout = float(dropout)
        self.attention_dropout = float(
            attention_dropout if attention_dropout is not None else dropout
        )
        self.hidden_dropout = float(
            hidden_dropout if hidden_dropout is not None else dropout
        )

        # ffn
        self.ffn_bias = bool(ffn_bias)

        # position
        self.max_position_embeddings = int(max_position_embeddings)

        # init
        self.initializer_range = float(initializer_range)
        self.layer_norm_eps = float(layer_norm_eps)

        # special tokens
        self.sep_token_id = int(sep_token_id)
        self.mask_token_id = int(mask_token_id)

        # outputs
        self.output_attentions = bool(output_attentions)
        self.output_hidden_states = bool(output_hidden_states)
        self.return_dict = bool(return_dict)
        self.use_cache = bool(use_cache)

        # validation
        if self.position_embedding_type not in ["rotary", "absolute"]:
            raise ValueError(
                f"Invalid position_embedding_type: {self.position_embedding_type}. "
                "Options are 'rotary' or 'absolute'."
            )
        valid_activations = ["gelu", "relu", "glu", "swiglu", "geglu", "reglu"]
        if self.activation not in valid_activations:
            raise ValueError(
                f"Invalid activation: {self.activation}. "
                f"Options are {valid_activations}."
            )
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_attention_heads ({self.num_attention_heads})."
            )
