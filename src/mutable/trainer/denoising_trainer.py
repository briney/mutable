# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT

from transformers import Trainer

__all__ = ["DenoisingTrainer"]


class DenoisingTrainer(Trainer):
    """
    Trainer for Phase 1 denoising pre-training.

    Minimal customization over HuggingFace Trainer — the model's forward()
    handles loss computation when labels are provided.
    """

    pass
