# -*- coding: utf-8 -*-
r"""
Language Model Base
==============
    Abstract base class used to build new language models
    inside CAPTION.
"""
import json
import logging

import torch
import torch.nn as nn

from caption.datasets import load_mlm_dataset
from caption.models.caption_base_model import CaptionModelBase
from caption.models.encoders import Encoder, str2encoder
from test_tube import HyperOptArgumentParser

torch.set_printoptions(precision=6)
log = logging.getLogger("Shell")


class LanguageModel(CaptionModelBase):
    """
    Language Model base class used to fine-tune pretrained models such as RoBERTa.

    :param hparams: HyperOptArgumentParser containing the hyperparameters.
    """

    def __init__(self, hparams: HyperOptArgumentParser,) -> None:
        super().__init__(hparams)

    def _build_loss(self):
        """ Initializes the loss function/s. """
        if self.hparams.loss == "cross_entropy":
            self.loss = nn.CrossEntropyLoss()
        else:
            raise Exception(f"{loss_func} is not a valid loss option.")

    def _retrieve_dataset(self, hparams, train=True, val=True, test=True):
        """ Retrieves task specific dataset """
        return load_mlm_dataset(hparams, train, val, test)

    def _build_encoder(self, hparams: HyperOptArgumentParser) -> Encoder:
        """
        Initializes the encoder.
        """
        return str2encoder[self.hparams.encoder_model].from_pretrained(
            hparams, lm_head=True
        )

    def predict(self, sample: dict) -> dict:
        """ Function that runs a model prediction,
        :param sample: dictionary with 'src', 'mt' and 'ref' 
            or a list containing several dictionaries with those keys
        Return: Dictionary with model outputs
        """
        raise NotImplementedError

    def _compute_loss(self, model_out: dict, targets: dict) -> torch.tensor:
        """
        Computes Loss value according to a loss function.
        :param model_out: model specific output. Must contain a key 'score' with 
            a tensor [batch_size x 1] with model predictions
        :param targets: Target score values [batch_size]
        """
        word_logits = model_out["scores"].view(-1, model_out["scores"].size(-1))
        masked_lm_labels = targets["lm_labels"]
        return self.loss(word_logits, masked_lm_labels.view(-1))

    def validation_step(self, batch: tuple, batch_nb: int, dataloader_idx: int) -> dict:
        """ Overwrite validation step and to return only the loss."""
        batch_input, batch_target = batch
        batch_prediction = self.forward(**batch_input)
        loss_value = self._compute_loss(batch_prediction, batch_target)

        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss_value = loss_value.unsqueeze(0)

        return {"val_loss": loss_value}

    def validation_epoch_end(self, outputs: list) -> dict:
        """ Overwrite validation end to compute perplexity and skip _compute_metrics."""

        def perplexity(loss_value, nr_batches):
            return {"perplexity": torch.exp(loss_value / nr_batches).item()}

        train_batches, val_batches = outputs
        train_loss = torch.stack([x["val_loss"] for x in train_batches]).mean()
        val_loss = torch.stack([x["val_loss"] for x in val_batches]).mean()
        log.info(f"-- Avg Train loss {train_loss:.4}")
        log.info(
            "-- Train metrics:\n{}".format(
                json.dumps(perplexity(train_loss, len(train_batches)), indent=1)
            )
        )
        metrics = perplexity(val_loss, len(val_batches))
        log.info(f"-- Avg Dev loss {val_loss:.4}")
        log.info("-- Dev metrics:\n{}".format(json.dumps(metrics), indent=1))

        # Store internally the best pearson result achieved.
        if (
            metrics[self.hparams.monitor] < self._best[self.hparams.monitor]
            and self.hparams.metric_mode == "min"
        ):
            self._best = {
                self.hparams.monitor: metrics[self.hparams.monitor],
                "val_loss": val_loss.item(),
            }

        return {"log": {**metrics, "val_loss": val_loss, "train_loss": train_loss}}

    @staticmethod
    def add_model_specific_args(
        parser: HyperOptArgumentParser,
    ) -> HyperOptArgumentParser:
        """ Function that adds shared arguments/hyperparameters across all 
            estimators. 

        :param parser: HyperOptArgumentParser obj
        
        Returns:
            - updated parser
        """
        parser = super(LanguageModel, LanguageModel).add_model_specific_args(parser)
        parser.add_argument(
            "--loss",
            default="cross_entropy",
            type=str,
            help="Loss function to be used.",
            choices=["cross_entropy"],
        )
        # Parameters for the Encoder model
        parser.add_argument(
            "--encoder_model",
            default="RoBERTa",
            type=str,
            help="Encoder model to be used.",
            choices=["BERT", "XLM-RoBERTa", "RoBERTa"],
        )
        parser.add_argument(
            "--pretrained_model",
            default="roberta-base",
            type=str,
            help=(
                "Encoder pretrained model to be used. "
                "(e.g. roberta-base or roberta-large)"
            ),
        )
        parser.add_argument(
            "--mlm_probability",
            default=0.15,
            type=float,
            help="Ratio of tokens to mask for masked language modeling loss.",
        )
        # Data Arguments
        parser.add_argument(
            "--data_type",
            default="txt",
            type=str,
            help="The type of the file containing the training/val/test data.",
            choices=["txt"],
        )
        parser.add_argument(
            "--loader_workers",
            default=4,
            type=int,
            help="How many subprocesses to use for data loading. 0 means that \
                the data will be loaded in the main process.",
        )
        # Data Arguments
        parser.add_argument(
            "--train_path",
            default="data/WMT18/en-de/train.nmt.ref",
            type=str,
            help="Path to the file containing the train data.",
        )
        parser.add_argument(
            "--dev_path",
            default="data/WMT18/en-de/dev.nmt.ref",
            type=str,
            help="Path to the file containing the dev data.",
        )
        parser.add_argument(
            "--test_path",
            default="data/WMT18/en-de/dev.nmt.pe",
            type=str,
            help="Path to the file containing the test data.",
        )
        return parser
