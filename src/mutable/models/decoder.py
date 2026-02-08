# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT

from typing import Optional, Union

import torch
import torch.nn as nn

from ..config import MutableConfig
from ..modules import DecoderLayer
from ..outputs import DecoderOutput

__all__ = ["MutableDecoder"]


class MutableDecoder(nn.Module):
    """
    Autoregressive decoder that cross-attends to bottleneck latents.

    Parameters
    ----------
    config : MutableConfig
        Model configuration.
    """

    def __init__(self, config: MutableConfig):
        super().__init__()
        self.config = config

        # embeddings
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.position_embeddings = (
            nn.Embedding(config.max_position_embeddings, config.hidden_size)
            if config.position_embedding_type == "absolute"
            else None
        )

        # layers
        self.layers = nn.ModuleList(
            [DecoderLayer(config) for _ in range(config.num_decoder_layers)]
        )

        # final layer norm
        self.final_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def _make_causal_mask(
        self, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Create causal attention mask for autoregressive decoding."""
        mask = torch.ones(seq_len, seq_len, device=device, dtype=torch.bool)
        mask = torch.tril(mask)
        return mask

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[DecoderOutput, tuple]:
        """
        Parameters
        ----------
        input_ids : torch.LongTensor
            Decoder input token IDs.
        encoder_hidden_states : torch.FloatTensor
            Bottleneck latent states, shape (batch, num_latents, latent_dim).
        """
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.return_dict
        )

        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions else None

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot provide both input_ids and inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_len = input_shape[1]

        if position_ids is None and self.position_embeddings is not None:
            position_ids = torch.arange(seq_len, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(input_shape)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        x = inputs_embeds
        if self.position_embeddings is not None and position_ids is not None:
            x = x + self.position_embeddings(position_ids)

        # layers — causal masking is handled inside DecoderLayer via is_causal=True
        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (x,)

            result = layer(
                x,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                need_weights=output_attentions,
            )

            if output_attentions:
                x, self_attn, cross_attn = result
                all_self_attentions += (self_attn,)
                all_cross_attentions += (cross_attn,)
            else:
                x = result

        x = self.final_norm(x)

        if output_hidden_states:
            all_hidden_states += (x,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    x,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return DecoderOutput(
            last_hidden_state=x,
            hidden_states=all_hidden_states,
            self_attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )
