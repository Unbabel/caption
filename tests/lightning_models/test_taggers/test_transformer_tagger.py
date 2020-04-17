# -*- coding: utf-8 -*-
r"""
Test TransformerTagger integration with Lightning
==============
"""
import unittest
from unittest.mock import Mock

import torch

from caption.models.taggers import TransformerTagger
from test_tube import HyperOptArgumentParser


class TestTransformerTagger(unittest.TestCase):
    @property
    def hparams(self):
        parser = HyperOptArgumentParser()
        # metric mode and monitor are hparams required by COMET models
        # and lightning trainer.
        parser.add_argument("--monitor", default="slot_error_rate")
        parser.add_argument("--metric_mode", default="min")
        parser = TransformerTagger.add_model_specific_args(parser)
        hparams, _ = parser.parse_known_args([])
        return hparams

    @property
    def samples(self):
        """ Sample example """
        return [
            {"text": "hello world", "tags": "T L",},
            {"text": "how amazing is caption", "tags": "T L L U",},
        ]

    def setUp(self):
        """ Setup will test """
        self.model = TransformerTagger(self.hparams)

    def test_run_model(self):
        """ Function that tests the integration between: 
            prepare_sample, forward and _compute_loss
        """
        inputs, targets = self.model.prepare_sample(self.samples)
        predictions = self.model(**inputs)
        loss_value = self.model._compute_loss(predictions, targets)
        self.assertIsInstance(loss_value, torch.Tensor)

    def test_training_step(self):
        """ Function that tests the integration between: 
            prepare_sample, forward and _compute_loss.
        """
        # Single-GPU Distributed Parallel training step
        self.model.trainer = Mock(use_dp=True, num_gpus=1)
        batch = self.model.prepare_sample(self.samples)
        result = self.model.training_step(batch, 0)
        assert "loss" in result

        # MultiGPU Distributed Parallel training step
        self.model.trainer = Mock(use_dp=True, num_gpus=2)
        batch = self.model.prepare_sample(self.samples)
        result = self.model.training_step(batch, 0)
        assert "loss" in result

    def test_validation_step(self):
        """ Function that tests the integration between: 
            prepare_sample, forward and _compute_loss in validation.
        """
        self.model.eval()
        self.model.trainer = Mock(use_dp=True, num_gpus=1)
        batch = self.model.prepare_sample(self.samples)
        result = self.model.validation_step(batch, 0, 0)
        assert "val_loss" in result
        assert "val_prediction" in result
        assert "val_target" in result

    def test_validation_end(self):
        """ Function that tests the integration between: 
            validation_step and validation_epoch_end
        """
        self.model.eval()
        self.model.trainer = Mock(use_dp=True, num_gpus=1)

        with torch.no_grad():
            # Simulation of the first validation dataloader.
            outputs_first_dataloader = []
            for i in range(5):
                batch = self.model.prepare_sample(self.samples)
                outputs_first_dataloader.append(self.model.validation_step(batch, 0, 0))

            # Simulation of the second validation dataloader.
            outputs_second_dataloader = []
            for i in range(5):
                batch = self.model.prepare_sample(self.samples)
                outputs_second_dataloader.append(
                    self.model.validation_step(batch, 0, 1)
                )

            outputs = (outputs_first_dataloader, outputs_second_dataloader)
            result = self.model.validation_epoch_end(outputs)
            self.assertIsInstance(result["log"]["val_loss"], torch.Tensor)
            self.assertIsInstance(result["log"]["train_loss"], torch.Tensor)

            assert "slot_error_rate" in result["log"].keys()
            assert "L_f1_score" in result["log"].keys()
            assert "U_f1_score" in result["log"].keys()
            assert "T_f1_score" in result["log"].keys()
            assert "macro_fscore" in result["log"].keys()
            assert "micro_fscore" in result["log"].keys()

    def test_predict(self):
        sample = self.samples[0]
        result = self.model.predict(sample)
        assert result["text"] == sample["text"]
        assert result["tags"] == sample["tags"]
        assert "predicted_tags" in result.keys()
        assert "tagged_sequence" in result.keys()

        sample = self.samples
        result = self.model.predict(sample)
        assert len(result) == 2
        for i in range(2):
            assert result[i]["text"] == sample[i]["text"]
            assert result[i]["tags"] == sample[i]["tags"]
            assert "predicted_tags" in result[i].keys()
            assert "tagged_sequence" in result[i].keys()
