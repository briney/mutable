# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT

import os
import re
from typing import Optional

from tokenizers import Regex, Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Sequence, Split
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast

__all__ = [
    "MutableTokenizer",
    "MUTABLE_VOCAB",
]


class MutableTokenizer(PreTrainedTokenizerFast):
    """
    Tokenizer for Mutable models. Based on BALM's tokenizer with "." replaced by "<sep>"
    for heavy/light chain separation. Vocabulary size is 32 (multiple of 8 for tensor cores).

    Parameters
    ----------
    vocab_file : str, optional
        Path to the vocabulary file. If not provided, the default vocabulary is used.
    bos_token : str, default="<cls>"
        Beginning of sequence token.
    eos_token : str, default="<eos>"
        End of sequence token.
    unk_token : str, default="<unk>"
        Unknown token.
    pad_token : str, default="<pad>"
        Padding token.
    mask_token : str, default="<mask>"
        Mask token.
    sep_token : str, default="<sep>"
        Separator token for heavy/light chain separation.
    """

    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file: Optional[str] = None,
        tokenizer_file: Optional[str] = None,
        bos_token: str = "<cls>",
        eos_token: str = "<eos>",
        unk_token: str = "<unk>",
        pad_token: str = "<pad>",
        mask_token: str = "<mask>",
        sep_token: str = "<sep>",
        **kwargs,
    ):
        # load pretrained tokenizer (used by AutoTokenizer)
        if tokenizer_file is not None:
            super().__init__(
                tokenizer_file=tokenizer_file,
                bos_token=bos_token,
                eos_token=eos_token,
                unk_token=unk_token,
                pad_token=pad_token,
                mask_token=mask_token,
                additional_special_tokens=[sep_token],
                **kwargs,
            )
            return

        # parse vocab
        if vocab_file is not None and os.path.isfile(vocab_file):
            with open(vocab_file, "r", encoding="utf-8") as f:
                vocab = [line.strip() for line in f if line.strip()]
        else:
            vocab = MUTABLE_VOCAB

        vocab_dict = {token: i for i, token in enumerate(vocab)}

        # create tokenizer
        tokenizer = Tokenizer(
            WordLevel(
                vocab=vocab_dict,
                unk_token=unk_token,
            )
        )

        # special token regex
        special_start_char = Regex(r"[<\[]")
        special_end_char = Regex(r"[>\]]")

        # non-special token regex
        pattern = "|".join(re.escape(tok) for tok in vocab if len(tok) == 1)

        # pre-tokenization
        tokenizer.pre_tokenizer = Sequence([
            Split(special_start_char, behavior="merged_with_next"),
            Split(special_end_char, behavior="merged_with_previous"),
            Split(Regex(pattern), behavior="isolated"),
        ])

        # post-processing (add bos and eos tokens)
        tokenizer.post_processor = TemplateProcessing(
            single=f"{bos_token} $A {eos_token}",
            pair=f"{bos_token} $A $B {eos_token}",
            special_tokens=[
                (bos_token, vocab_dict[bos_token]),
                (eos_token, vocab_dict[eos_token]),
            ],
        )

        # initialize PreTrainedTokenizerFast
        super().__init__(
            tokenizer_object=tokenizer,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            mask_token=mask_token,
            additional_special_tokens=[sep_token],
            **kwargs,
        )


# Vocabulary: 32 tokens (BALM vocab with "." replaced by "<sep>")
# Index 29 = <sep>, Index 31 = <mask>
MUTABLE_VOCAB = [
    "<cls>",   # 0 - BOS
    "<pad>",   # 1
    "<eos>",   # 2
    "<unk>",   # 3
    "L",       # 4
    "A",       # 5
    "G",       # 6
    "V",       # 7
    "S",       # 8
    "E",       # 9
    "R",       # 10
    "T",       # 11
    "I",       # 12
    "D",       # 13
    "P",       # 14
    "K",       # 15
    "Q",       # 16
    "N",       # 17
    "F",       # 18
    "Y",       # 19
    "M",       # 20
    "H",       # 21
    "W",       # 22
    "C",       # 23
    "X",       # 24
    "B",       # 25
    "U",       # 26
    "O",       # 27
    "Z",       # 28
    "<sep>",   # 29 (replaces "." in BALM)
    "-",       # 30
    "<mask>",  # 31
]
