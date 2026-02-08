# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT

from typing import Optional, Union

import torch
import torch.nn as nn

from ..config import MutableConfig
from ..modules import EncoderLayer
from ..outputs import EncoderOutput

__all__ = ["MutableEncoder"]


class MutableEncoder(nn.Module):
    """
    Bidirectional transformer encoder.

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
            [EncoderLayer(config) for _ in range(config.num_encoder_layers)]
        )

        # final layer norm
        self.final_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[EncoderOutput, tuple]:
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

        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot provide both input_ids and inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        if position_ids is None and self.position_embeddings is not None:
            position_ids = torch.arange(input_shape[1], dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(input_shape)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        x = inputs_embeds
        if self.position_embeddings is not None and position_ids is not None:
            x = x + self.position_embeddings(position_ids)

        # layers
        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (x,)

            x = layer(
                x,
                attention_mask=attention_mask,
                need_weights=output_attentions,
            )
            if output_attentions:
                x, attn = x
                all_self_attentions += (attn,)

        x = self.final_norm(x)

        if output_hidden_states:
            all_hidden_states += (x,)

        if not return_dict:
            return tuple(
                v
                for v in [x, all_hidden_states, all_self_attentions]
                if v is not None
            )
        return EncoderOutput(
            last_hidden_state=x,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )
