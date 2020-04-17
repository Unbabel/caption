# -*- coding: utf-8 -*-
r"""
Masked Language Model
==============
    Model used to fine-tune encoder models such as RoBERTa and XLM-RoBERTa in a specific 
    domain.
"""
import torch
import torch.nn as nn

from caption.models.language_models.lm_base import LanguageModel
from caption.models.utils import mask_tokens
from caption.optimizers import build_optimizer
from caption.schedulers import build_scheduler
from test_tube import HyperOptArgumentParser
from torchnlp.utils import collate_tensors


class MaskedLanguageModel(LanguageModel):
    """
    Model used to pretrain encoder model such as BERT and XLM-R in with a 
    Masked Language Modeling objective.

    :param hparams: HyperOptArgumentParser containing the hyperparameters.
    """

    def __init__(self, hparams: HyperOptArgumentParser,) -> None:
        super().__init__(hparams)

    def _build_model(self) -> LanguageModel:
        """
        The Masked Language model head is already initialized by the encoder.
        """
        pass

    def prepare_sample(self, sample: list) -> (dict, dict):
        """
        Function that prepares a sample to input the model.
        :param sample: list of dictionaries.
        
        Returns:
            - dictionary with the expected model inputs.
            - dictionary with the expected target values (e.g. HTER score).
        """
        sample = collate_tensors(sample)
        sample = self.encoder.prepare_sample(sample["text"], trackpos=False)
        tokens, labels = mask_tokens(
            sample["tokens"], self.encoder.tokenizer, self.hparams.mlm_probability,
        )
        return {"tokens": tokens, "lengths": sample["lengths"]}, {"lm_labels": labels}

    def forward(self, tokens: torch.tensor, lengths: torch.tensor, **kwargs) -> dict:
        """
        :param tokens: sequences [batch_size x src_seq_len]
        :param lengths: sequence lengths [batch_size]
        Return: Dictionary with model outputs to be passed to the loss function.
        """
        # When using just one GPU this should not change behavior
        # but when splitting batches across GPU the tokens have padding
        # from the entire original batch
        if self.trainer and self.trainer.use_dp and self.trainer.num_gpus > 1:
            tokens = tokens[:, : lengths.max()]

        embeddings = self.encoder(tokens, lengths)["wordemb"]
        return {
            "scores": self.encoder.lm_head(embeddings),
        }
