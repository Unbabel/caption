# -*- coding: utf-8 -*-
r"""
Word Tagger Base
==============
    Abstract base class used to build new sequence tagging models 
    inside CAPTION.
"""
import sys
import pdb

import numpy as np
import torch
import torch.nn as nn

from caption.datasets import sequence_tagging_dataset
from caption.models.caption_base_model import CaptionModelBase
from caption.models.metrics import classification_report
from test_tube import HyperOptArgumentParser
from torchnlp.encoders import LabelEncoder
from torchnlp.encoders.text import stack_and_pad_tensors
from torchnlp.utils import collate_tensors


class Tagger(CaptionModelBase):
    """
    Tagger base class.

    :param hparams: HyperOptArgumentParser containing the hyperparameters.
    """

    def __init__(self, hparams: HyperOptArgumentParser,) -> None:
        super().__init__(hparams)

    def _build_model(self):
        self.label_encoder = LabelEncoder(
            self.hparams.tag_set.split(","), reserved_labels=[]
        )

    def _build_loss(self):
        """ Initializes the loss function/s. """
        weights = (
            np.array([float(x) for x in self.hparams.class_weights.split(",")])
            if self.hparams.class_weights != "ignore"
            else np.array([])
        )

        if self.hparams.loss == "cross_entropy":
            self.loss = nn.CrossEntropyLoss(
                reduction="sum",
                ignore_index=self.label_encoder.vocab_size,
                weight=torch.tensor(weights, dtype=torch.float32)
                if weights.any()
                else None,
            )
        else:
            raise Exception(f"{self.hparams.loss} is not a valid loss option.")

    def _retrieve_dataset(self, data_hparams, train=True, val=True, test=True):
        """ Retrieves task specific dataset """
        return sequence_tagging_dataset(data_hparams, train, val, test)

    @property
    def default_slot_index(self):
        """ Index of the default slot to be ignored. (e.g. 'O' in 'B-I-O' tags) """
        return 0

    def predict(self, sample: dict) -> list:
        """ Function that runs a model prediction,
        :param sample: a dictionary that must contain the the 'source' sequence.

        Return: list with predictions
        """
        if self.training:
            self.eval()

        return_dict = False
        if isinstance(sample, dict):
            sample = [sample]
            return_dict = True

        with torch.no_grad():
            model_input, _ = self.prepare_sample(sample, prepare_target=False)
            model_out = self.forward(**model_input)
            tag_logits = model_out["tags"]
            _, pred_labels = tag_logits.topk(1, dim=-1)

            for i in range(pred_labels.size(0)):
                sample_tags = pred_labels[i, :, :].view(-1)
                tags = [
                    self.label_encoder.index_to_token[sample_tags[j]]
                    for j in range(model_input["word_lengths"][i])
                ]
                sample[i]["predicted_tags"] = " ".join(tags)
                sample[i]["tagged_sequence"] = " ".join(
                    [
                        word + "/" + tag
                        for word, tag in zip(sample[i]["text"].split(), tags)
                    ]
                )

                sample[i][
                    "encoded_ground_truth_tags"
                ] = self.label_encoder.batch_encode(
                    [tag for tag in sample[i]["tags"].split()]
                )

                if self.hparams.ignore_last_tag:
                    if (
                        sample[i]["encoded_ground_truth_tags"][
                            model_input["word_lengths"][i] - 1
                        ]
                        == 1
                    ):
                        sample[i]["encoded_ground_truth_tags"][
                            model_input["word_lengths"][i] - 1
                        ] = self.label_encoder.vocab_size

        if return_dict:
            return sample[0]

        return sample

    def _compute_loss(self, model_out: dict, targets: dict) -> torch.tensor:
        """
        Computes Loss value according to a loss function.
        :param model_out: model specific output with predicted tag logits
            a tensor [batch_size x seq_length x num_tags]
        :param targets: Target tags [batch_size x seq_length]
        """
        logits = model_out["tags"].view(-1, model_out["tags"].size(-1))
        labels = targets["tags"].view(-1)
        return self.loss(logits, labels)

    def prepare_sample(self, sample: list, prepare_target: bool = True) -> (dict, dict):
        """
        Function that prepares a sample to input the model.
        :param sample: list of dictionaries.
        
        Returns:
            - dictionary with the expected model inputs.
            - dictionary with the expected target values.
        """
        sample = collate_tensors(sample)
        inputs = self.encoder.prepare_sample(sample["text"], trackpos=True)
        if not prepare_target:
            return inputs, {}

        tags, _ = stack_and_pad_tensors(
            [self.label_encoder.batch_encode(tags.split()) for tags in sample["tags"]],
            padding_index=self.label_encoder.vocab_size,
        )

        if self.hparams.ignore_first_title:
            first_tokens = tags[:, 0].clone()
            tags[:, 0] = first_tokens.masked_fill_(
                first_tokens == self._label_encoder.token_to_index["T"],
                self.label_encoder.vocab_size,
            )

        # TODO is this still needed ?
        if self.hparams.ignore_last_tag:
            lengths = [len(tags.split()) for tags in sample["tags"]]
            lengths = np.asarray(lengths)
            k = 0
            for length in lengths:
                if tags[k][length - 1] == 1:
                    tags[k][length - 1] = self.label_encoder.vocab_size
                k += 1

        targets = {"tags": tags}
        return inputs, targets

    def _compute_metrics(self, outputs: list) -> dict:
        """ 
        Private function that computes metrics of interest based on model predictions 
        and respective targets.
        """
        predictions = [batch_out["val_prediction"]["tags"] for batch_out in outputs]
        targets = [batch_out["val_target"]["tags"] for batch_out in outputs]

        predicted_tags, ground_truth = [], []
        for i in range(len(predictions)):
            # Get logits and reshape predictions
            batch_predictions = predictions[i]
            logits = batch_predictions.view(-1, batch_predictions.size(-1)).cpu()
            _, pred_labels = logits.topk(1, dim=-1)

            # Reshape targets
            batch_targets = targets[i].view(-1).cpu()

            assert batch_targets.size() == pred_labels.view(-1).size()
            ground_truth.append(batch_targets)
            predicted_tags.append(pred_labels.view(-1))

        return classification_report(
            torch.cat(predicted_tags).numpy(),
            torch.cat(ground_truth).numpy(),
            padding=self.label_encoder.vocab_size,
            labels=self.label_encoder.token_to_index,
            ignore=self.default_slot_index,
        )

    @classmethod
    def add_model_specific_args(
        cls, parser: HyperOptArgumentParser
    ) -> HyperOptArgumentParser:
        """ Parser for Estimator specific arguments/hyperparameters. 
        :param parser: HyperOptArgumentParser obj

        Returns:
            - updated parser
        """
        parser = super(Tagger, Tagger).add_model_specific_args(parser)
        parser.add_argument(
            "--tag_set",
            type=str,
            default="L,U,T",
            help="Task tags we want to use.\
                 Note that the 'default' label should appear first",
        )
        # Loss
        parser.add_argument(
            "--loss",
            default="cross_entropy",
            type=str,
            help="Loss function to be used.",
            choices=["cross_entropy"],
        )
        parser.add_argument(
            "--class_weights",
            default="ignore",
            type=str,
            help='Weights for each of the classes we want to tag (e.g: "1.0,7.0,8.0").',
        )
        ## Data args:
        parser.add_argument(
            "--data_type",
            default="csv",
            type=str,
            help="The type of the file containing the training/dev/test data.",
            choices=["csv"],
        )
        parser.add_argument(
            "--train_path",
            default="data/dummy_train.csv",
            type=str,
            help="Path to the file containing the train data.",
        )
        parser.add_argument(
            "--dev_path",
            default="data/dummy_test.csv",
            type=str,
            help="Path to the file containing the dev data.",
        )
        parser.add_argument(
            "--test_path",
            default="data/dummy_test.csv",
            type=str,
            help="Path to the file containing the test data.",
        )
        parser.add_argument(
            "--loader_workers",
            default=0,
            type=int,
            help=(
                "How many subprocesses to use for data loading. 0 means that"
                "the data will be loaded in the main process."
            ),
        )
        # Metric args:
        parser.add_argument(
            "--ignore_first_title",
            default=False,
            help="When used, this flag ignores T tags in the first position.",
            action="store_true",
        )
        parser.add_argument(
            "--ignore_last_tag",
            default=False,
            help="When used, this flag ignores S tags in the last position.",
            action="store_true",
        )
        return parser
