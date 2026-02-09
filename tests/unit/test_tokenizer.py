# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT

"""Tests for MutableTokenizer."""

import pytest
import torch

from mutable.tokenizer import MutableTokenizer, MUTABLE_VOCAB


class TestMutableTokenizer:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.tok = MutableTokenizer()

    def test_vocab_size(self):
        assert self.tok.vocab_size == 32

    def test_encode_single_aa_sequence(self):
        ids = self.tok.encode("ACDEFG")
        assert isinstance(ids, list)
        assert len(ids) > 0

    def test_encode_with_sep(self):
        ids = self.tok.encode("ACDE<sep>FGHI")
        # should contain sep token ID
        assert 29 in ids

    def test_decode_roundtrip(self):
        seq = "ACDEFGHIKL"
        ids = self.tok.encode(seq)
        decoded = self.tok.decode(ids, skip_special_tokens=True)
        # decoded should contain the original amino acids (spaces may be present)
        cleaned = decoded.replace(" ", "")
        assert cleaned == seq

    def test_special_token_ids(self):
        assert self.tok.bos_token_id == 0
        assert self.tok.pad_token_id == 1
        assert self.tok.eos_token_id == 2
        assert self.tok.convert_tokens_to_ids("<sep>") == 29
        assert self.tok.convert_tokens_to_ids("<mask>") == 31

    def test_bos_eos_auto_added(self):
        ids = self.tok.encode("ACD")
        assert ids[0] == 0  # BOS
        assert ids[-1] == 2  # EOS

    def test_padding(self):
        enc = self.tok(
            "ACD",
            padding="max_length",
            max_length=20,
            return_tensors="pt",
        )
        assert enc["input_ids"].shape[1] == 20
        assert enc["attention_mask"].sum() < 20  # some positions are padding

    def test_truncation(self):
        long_seq = "A" * 1000
        enc = self.tok(
            long_seq,
            truncation=True,
            max_length=50,
            return_tensors="pt",
        )
        assert enc["input_ids"].shape[1] == 50

    def test_all_20_amino_acids_in_vocab(self):
        standard_aa = set("LAGVSERTIDPKQNFYMHWC")
        vocab_chars = set(t for t in MUTABLE_VOCAB if len(t) == 1 and t.isalpha())
        assert standard_aa.issubset(vocab_chars)

    def test_return_tensors_pt(self):
        enc = self.tok("ACD", return_tensors="pt")
        assert isinstance(enc["input_ids"], torch.Tensor)
        assert enc["input_ids"].dim() == 2
