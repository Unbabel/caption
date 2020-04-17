# -*- coding: utf-8 -*-
r"""
Encoder Model Base
==============
    Abstract base class used to build new pretrained Encoder models.
"""
import os

import torch
import torch.nn as nn

from caption.tokenizers import TextEncoderBase
from test_tube import HyperOptArgumentParser


class Encoder(nn.Module):
    """ Base class for an encoder model.
    
    :param output_units: Number of output features that will be passed to the Estimator.
    """

    def __init__(
        self, output_units: int, tokenizer: TextEncoderBase, lm_head: bool = False
    ) -> None:
        super().__init__()
        self.output_units = output_units
        self.tokenizer = tokenizer

    @property
    def num_layers(self):
        """ Number of model layers available. """
        return self._n_layers

    @classmethod
    def from_pretrained(cls, hparams: HyperOptArgumentParser, lm_head: bool = False):
        """ Function that loads a pretrained encoder and the respective tokenizer.
        
        Returns:
            - Encoder model
        """
        raise NotImplementedError

    def prepare_sample(
        self, sample: list, trackpos: bool = True
    ) -> (torch.tensor, torch.tensor):
        """ Receives a list of strings and applies model specific tokenization and vectorization."""
        if not trackpos:
            tokens, lengths = self.tokenizer.batch_encode(sample)
            return {"tokens": tokens, "lengths": lengths}

        (
            tokens,
            lengths,
            word_boundaries,
            word_lengths,
        ) = self.tokenizer.batch_encode_trackpos(sample)
        return {
            "tokens": tokens,
            "lengths": lengths,
            "word_boundaries": word_boundaries,
            "word_lengths": word_lengths,
        }

    def freeze(self) -> None:
        """ Frezees the entire encoder network. """
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self) -> None:
        """ Unfrezees the entire encoder network. """
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, tokens: torch.tensor, lengths: torch.tensor, **kwargs) -> dict:
        """
        Encodes a batch of sequences.

        :param tokens: Torch tensor with the input sequences [batch_size x seq_len].
        :param lengths: Torch tensor with the lenght of each sequence [seq_len].

        Returns: 
            - 'sentemb': tensor [batch_size x output_units] with the sentence encoding.
            - 'wordemb': tensor [batch_size x seq_len x output_units] with the word level embeddings.
            - 'mask': input mask.
            - 'all_layers': List with the word_embeddings returned by each layer.
            - 'extra': model specific outputs.
        """
        raise NotImplementedError
