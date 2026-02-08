# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT

from typing import Dict, Optional, Union

import torch
import torch.nn as nn
from transformers import Trainer

__all__ = ["FlowMatchingTrainer"]


class FlowMatchingTrainer(Trainer):
    """
    Trainer for Phase 2 flow matching training.

    Ensures the encoder/decoder backbone stays frozen and handles
    the flow matching forward pass signature.
    """

    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
        **kwargs,
    ) -> Union[torch.Tensor, tuple]:
        """
        Compute loss for flow matching.

        The inputs dict should contain:
        - germline_input_ids
        - germline_attention_mask
        - mutated_input_ids
        - mutated_attention_mask
        - mu (optional)
        """
        outputs = model(
            germline_input_ids=inputs["germline_input_ids"],
            germline_attention_mask=inputs.get("germline_attention_mask"),
            mutated_input_ids=inputs["mutated_input_ids"],
            mutated_attention_mask=inputs.get("mutated_attention_mask"),
            mu=inputs.get("mu"),
            return_dict=True,
        )

        loss = outputs.loss

        return (loss, outputs) if return_outputs else loss
