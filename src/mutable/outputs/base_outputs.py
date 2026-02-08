# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT

from dataclasses import dataclass
from typing import Union

import torch
from transformers.utils.generic import ModelOutput

__all__ = ["MutableModelOutput"]


@dataclass
class MutableModelOutput(ModelOutput):
    """
    Base class for Mutable model outputs, with potential hidden states and attentions.
    """

    def to(self, device: Union[torch.device, str]):
        for key, value in self.__dict__.items():
            if isinstance(value, torch.Tensor):
                self.__dict__[key] = value.to(device)
            elif isinstance(value, tuple) and all(
                isinstance(v, torch.Tensor) for v in value
            ):
                self.__dict__[key] = tuple(v.to(device) for v in value)
        if torch.cuda.is_available() and torch.device(device).type == "cpu":
            torch.cuda.empty_cache()
        return self
