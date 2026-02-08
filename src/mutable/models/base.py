# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT

from typing import Optional

import torch.nn as nn
from transformers import PreTrainedModel

__all__ = ["MutablePreTrainedModel", "FreezeBaseModelMixin", "ParameterCountMixin"]


class MutablePreTrainedModel(PreTrainedModel):
    """
    Base class for all Mutable models. Handles weight initialization and
    provides an interface for downloading and loading pretrained models.
    """

    supports_gradient_checkpointing = True

    # Copied from transformers.models.bert.modeling_bert.BertPreTrainedModel._init_weights
    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            if module.bias is not None:
                module.bias.data.zero_()
            if module.weight is not None:
                module.weight.data.fill_(1.0)


class FreezeBaseModelMixin:
    """
    Mixin class to freeze the base model's parameters during training.
    """

    def freeze_base_model(self, base_model: Optional[str] = None):
        if base_model is None:
            if not hasattr(self.__class__, "base_model_prefix"):
                raise ValueError(
                    f"{self.__class__.__name__} does not have a default base model prefix, "
                    "so you need to provide `base_model`."
                )
            base_model = getattr(self, self.__class__.base_model_prefix)
        else:
            if not hasattr(self, base_model):
                raise ValueError(
                    f"This model instance does not have the supplied base model ({base_model})."
                )
            base_model = getattr(self, base_model)
        for param in base_model.parameters():
            param.requires_grad = False

    def freeze_encoder_decoder(self):
        """Freeze encoder, bottleneck, and decoder (for Phase 2 flow training)."""
        if hasattr(self, "mutable"):
            for param in self.mutable.parameters():
                param.requires_grad = False


class ParameterCountMixin:
    """
    Mixin class to count model parameters.
    """

    def count_parameters(
        self,
        only_trainable: bool = True,
        exclude_embeddings: bool = False,
        human_readable: bool = False,
    ) -> int:
        if exclude_embeddings:
            exclude_param_names = [
                f"{name}.weight"
                for name, module_type in self.named_modules()
                if isinstance(module_type, nn.Embedding)
            ]
        else:
            exclude_param_names = []

        total_num_params = sum(
            p.numel()
            for name, p in self.named_parameters()
            if name not in exclude_param_names
            and (not only_trainable or p.requires_grad)
        )

        if human_readable:
            return self._human_readable(total_num_params)
        return total_num_params

    def _human_readable(self, total_num_params):
        units = ["T", "B", "M", "K"]
        thresholds = [1e12, 1e9, 1e6, 1e3]
        for unit, threshold in zip(units, thresholds):
            if total_num_params >= threshold:
                return f"{total_num_params / threshold:.2f}{unit}"
        return str(total_num_params)
