# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT

"""
Setup for Mutable models and tokenizers to use Hugging Face's Auto classes.
"""

from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModel,
)

from .config.mutable_config import MutableConfig
from .config.flow_config import FlowMatchingConfig
from .models.mutable import MutableModel
from .models.mutable_denoising import MutableForDenoising
from .models.flow_matching import MutableFlowMatching
from .tokenizer import MutableTokenizer

# tokenizer
AutoTokenizer.register("mutable", fast_tokenizer_class=MutableTokenizer)

# Mutable base model
AutoConfig.register("mutable", MutableConfig)
AutoModel.register(MutableConfig, MutableModel)

# Flow matching config
AutoConfig.register("mutable_flow", FlowMatchingConfig)
