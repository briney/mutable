# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT

from typing import Optional, Union

import torch

from ..config import MutableConfig
from ..outputs import MutableModelFullOutput, EncoderOutput, BottleneckOutput
from .base import MutablePreTrainedModel, ParameterCountMixin
from .encoder import MutableEncoder
from .bottleneck import PerceiverBottleneck
from .decoder import MutableDecoder

__all__ = ["MutableModel"]


class MutableModel(MutablePreTrainedModel, ParameterCountMixin):
    """
    Full Mutable model: encoder -> Perceiver bottleneck -> decoder.

    Parameters
    ----------
    config : MutableConfig
        Model configuration.
    """

    config_class = MutableConfig
    base_model_prefix = "mutable"

    def __init__(self, config: MutableConfig):
        super().__init__(config)
        self.config = config

        self.encoder = MutableEncoder(config)
        self.bottleneck = PerceiverBottleneck(config)
        self.decoder = MutableDecoder(config)

        self.post_init()

    def encode(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[BottleneckOutput, tuple]:
        """Encode input and compress through bottleneck. Returns latent states."""
        return_dict = (
            return_dict if return_dict is not None else self.config.return_dict
        )

        encoder_output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            return_dict=True,
        )

        bottleneck_output = self.bottleneck(
            encoder_output=encoder_output.last_hidden_state,
            encoder_mask=attention_mask,
            return_dict=return_dict,
        )

        return bottleneck_output

    def decode(
        self,
        decoder_input_ids: torch.LongTensor,
        encoder_hidden_states: torch.FloatTensor,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
    ):
        """Decode from latent states."""
        return self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            return_dict=return_dict,
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[MutableModelFullOutput, tuple]:
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

        # encoder
        encoder_output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        # bottleneck
        bottleneck_output = self.bottleneck(
            encoder_output=encoder_output.last_hidden_state,
            encoder_mask=attention_mask,
            need_weights=output_attentions,
            return_dict=True,
        )

        # decoder
        decoder_output = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=bottleneck_output.latent_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        if not return_dict:
            return tuple(
                v
                for v in [
                    decoder_output.last_hidden_state,
                    bottleneck_output.latent_states,
                    encoder_output.last_hidden_state,
                    encoder_output.hidden_states,
                    encoder_output.attentions,
                    decoder_output.hidden_states,
                    decoder_output.self_attentions,
                    decoder_output.cross_attentions,
                ]
                if v is not None
            )

        return MutableModelFullOutput(
            last_hidden_state=decoder_output.last_hidden_state,
            latent_states=bottleneck_output.latent_states,
            encoder_last_hidden_state=encoder_output.last_hidden_state,
            encoder_hidden_states=encoder_output.hidden_states,
            encoder_attentions=encoder_output.attentions,
            decoder_hidden_states=decoder_output.hidden_states,
            decoder_self_attentions=decoder_output.self_attentions,
            decoder_cross_attentions=decoder_output.cross_attentions,
        )
