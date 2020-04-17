# -*- coding: utf-8 -*-
import unittest
from argparse import Namespace

import torch
from transformers import BertTokenizer

from caption.models.encoders import BERT


class TestBERTEncoder(unittest.TestCase):
    def setUp(self):
        # setup tests the from_pretrained function
        hparams = Namespace(pretrained_model="bert-base-cased")
        self.model_base = BERT.from_pretrained(hparams)
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

        hparams = Namespace(pretrained_model="bert-large-cased")
        self.model_large = BERT.from_pretrained(hparams)

    def test_num_layers(self):
        assert self.model_base.num_layers == 13
        assert self.model_large.num_layers == 25

    def test_output_units(self):
        assert self.model_base.output_units == 768
        assert self.model_large.output_units == 1024

    def test_prepare_sample(self):
        sample = ["hello world, welcome to COMET!", "This is a batch"]

        model_input = self.model_base.prepare_sample(sample)
        assert "tokens" in model_input
        assert "lengths" in model_input

        # Sanity Check: This is already checked when testing the tokenizer.
        expected = self.tokenizer.encode(sample[0])
        assert torch.equal(torch.tensor(expected), model_input["tokens"][0])
        assert len(expected) == model_input["lengths"][0]

        model_input = self.model_base.prepare_sample(sample, trackpos=True)
        assert "tokens" in model_input
        assert "lengths" in model_input
        assert "word_boundaries" in model_input
        assert "word_lengths" in model_input

    def test_forward(self):
        sample = ["hello world!", "This is a batch"]
        model_input = self.model_base.prepare_sample(sample)
        model_out = self.model_base(**model_input)

        assert "wordemb" in model_out
        assert "sentemb" in model_out
        assert "all_layers" in model_out
        assert "mask" in model_out
        assert "extra" in model_out

        assert len(model_out["all_layers"]) == self.model_base.num_layers
        assert self.model_base.output_units == model_out["sentemb"].size()[1]
        assert self.model_base.output_units == model_out["wordemb"].size()[2]
