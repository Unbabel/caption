# -*- coding: utf-8 -*-
r"""
CAPTION Model Base
==============
    Abstract base class used to build new modules inside CAPTION.
"""
import json
import logging
import math

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, Subset
from tqdm import tqdm

import pytorch_lightning as ptl
from caption.models.encoders import Encoder
from caption.optimizers import build_optimizer
from caption.schedulers import build_scheduler
from caption.tokenizers import TextEncoderBase
from test_tube import HyperOptArgumentParser

torch.set_printoptions(precision=6)
log = logging.getLogger("Shell")


class CaptionModelBase(ptl.LightningModule):
    """
    Caption Modules extend PyTorch Lightning with a common structure and interface
    that will be shared across all modules (e.g. estimators, word_taggers, classifiers, etc..)
    in this project .

    :param hparams: HyperOptArgumentParser containing the hyperparameters.
    """

    def __init__(self, hparams: HyperOptArgumentParser,) -> None:
        super(CaptionModelBase, self).__init__()
        self.hparams = hparams
        self._encoder = self._build_encoder(hparams)

        # Model initialization
        self._build_model()

        # Loss criterion initialization.
        self._build_loss()

        # The encoder always starts in a frozen state.
        if hparams.nr_frozen_epochs > 0:
            self._frozen = True
            self.freeze_encoder()
        else:
            self._frozen = False

        self.nr_frozen_epochs = hparams.nr_frozen_epochs

        # training helpers.
        self._pbar = None  # used during training to produce a loading bar.

        # used during hyperparameter search only.
        self._best = {"val_loss": math.inf}
        self._best[self.hparams.monitor] = (
            -math.inf if self.hparams.metric_mode == "max" else math.inf
        )

    def _build_loss(self):
        """ Initializes the loss function/s. """
        raise NotImplementedError

    def _build_model(self) -> ptl.LightningModule:
        """
        Initializes the estimator architecture.
        """
        raise NotImplementedError

    def _build_encoder(
        self, hparams: HyperOptArgumentParser
    ) -> (Encoder, TextEncoderBase):
        """
        Initializes the encoder.
        """
        raise NotImplementedError

    def _retrieve_dataset(self, data_hparams, train=True, val=True, test=True):
        """ Retrieves task specific dataset """
        raise NotImplementedError

    @property
    def encoder(self):
        """ Model encoding layer. """
        return self._encoder

    def freeze_encoder(self) -> None:
        """ Freezes the encoder layer. """
        self.encoder.freeze()

    def unfreeze_encoder(self) -> None:
        """ un-freezes the encoder layer. """
        if self._frozen:
            log.info(f"\n-- Encoder model fine-tuning")
            self.encoder.unfreeze()
            self._frozen = False

    def predict(self, sample: dict) -> dict:
        """ Function that runs a model prediction,
        :param sample: dictionary with expected model sequences. 
            You can also pass a list of dictionaries to predict an entire batch.
        
        Return: Dictionary with model outputs
        """
        raise NotImplementedError

    def forward(self, *args, **kwargs) -> dict:
        """
        PyTorch Forward.
        Return: Dictionary with model outputs to be passed to the loss function.
        """
        raise NotImplementedError

    def _compute_loss(self, model_out: dict, targets: dict) -> torch.tensor:
        """
        Computes Loss value according to a loss function.
        :param model_out: model specific output.
        :param targets: Target score values [batch_size]
        """
        raise NotImplementedError

    def prepare_sample(self, sample: list, prepare_target: bool = True) -> (dict, dict):
        """
        Function that prepares a sample to input the model.
        :param sample: list of dictionaries.
        
        Returns:
            - dictionary with the expected model inputs.
            - dictionary with the expected target values (e.g. HTER score).
        """
        raise NotImplementedError

    def configure_optimizers(self):
        """ Function for setting up the optimizers and the schedulers to be used during training.
        
        Returns:
            - List with as many optimizers as we need
            - List with the respective schedulers.
        """
        optimizer = build_optimizer(self.parameters(), self.hparams)
        scheduler = build_scheduler(optimizer, self.hparams)
        return [optimizer], [scheduler]

    def _compute_metrics(self, outputs: list) -> dict:
        """ 
        Private function that computes metrics of interest based on the list of outputs 
        you defined in validation_step.
        """
        raise NotImplementedError

    def training_step(self, batch: tuple, batch_nb: int, *args, **kwargs) -> dict:
        """ Runs one training step. This usually consists in the forward function followed
        by the loss function.
        :param batch: The output of your dataloader. 
        :param batch_nb: Integer displaying which batch this is

        Returns:
            - dictionary containing the loss and the metrics to be added to the lightning logger.
        """
        batch_input, batch_target = batch
        batch_prediction = self.forward(**batch_input)
        loss_value = self._compute_loss(batch_prediction, batch_target)

        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss_value = loss_value.unsqueeze(0)

        return {"loss": loss_value}

    def validation_step(self, batch: tuple, batch_nb: int, dataloader_idx: int) -> dict:
        """ Similar to the training step but with the model in eval mode.

        Returns:
            - dictionary passed to the validation_end function.
        """
        batch_input, batch_target = batch
        batch_prediction = self.forward(**batch_input)
        loss_value = self._compute_loss(batch_prediction, batch_target)

        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss_value = loss_value.unsqueeze(0)

        return {
            "val_loss": loss_value,
            "val_prediction": batch_prediction,
            "val_target": batch_target,
        }

    def validation_epoch_end(self, outputs: list) -> dict:
        """ Function that takes as input a list of dictionaries returned by the validation_step
        function and measures the model performance accross the entire validation set.
        
        Returns:
            - Dictionary with metrics to be added to the lightning logger.  
        """
        train_batches, val_batches = outputs
        avg_train_loss = torch.stack([x["val_loss"] for x in train_batches]).mean()
        avg_val_loss = torch.stack([x["val_loss"] for x in val_batches]).mean()

        train_metrics = self._compute_metrics(train_batches)
        metrics = self._compute_metrics(val_batches)

        log.info(f"-- Avg Train loss {avg_train_loss:.4}")
        log.info("-- Train metrics:\n{}".format(json.dumps(train_metrics, indent=1)))

        log.info(f"-- Avg Dev loss {avg_val_loss:.4}")
        log.info("-- Dev metrics:\n{}".format(json.dumps(metrics, indent=1)))

        # Store internally the best pearson result achieved.
        if (
            metrics[self.hparams.monitor] > self._best[self.hparams.monitor]
            and self.hparams.metric_mode == "max"
        ):

            self._best = {
                self.hparams.monitor: metrics[self.hparams.monitor],
                "val_loss": avg_val_loss.item(),
            }
        elif (
            metrics[self.hparams.monitor] < self._best[self.hparams.monitor]
            and self.hparams.metric_mode == "min"
        ):

            self._best = {
                self.hparams.monitor: metrics[self.hparams.monitor],
                "val_loss": avg_val_loss.item(),
            }

        return {
            "log": {**metrics, "val_loss": avg_val_loss, "train_loss": avg_train_loss}
        }

    def test_step(self, batch: list, batch_nb: int, *args, **kwargs) -> dict:
        """ Redirects to validation step. """
        pass

    def test_epoch_end(self, outputs: list) -> dict:
        """ Redirects to validation end. """
        pass

    def prepare_data(self) -> None:
        """Data preparation function called before training by Lightning"""
        (
            self._train_dataset,
            self._val_dataset,
            self._test_dataset,
        ) = self._retrieve_dataset(self.hparams)

        train_subset = np.random.choice(
            a=len(self._train_dataset),
            size=int(len(self._train_dataset) * self.hparams.train_val_percent_check),
        )
        self._train_subset = Subset(self._train_dataset, train_subset)

    def train_dataloader(self) -> DataLoader:
        """ Function that loads the train set. """
        return DataLoader(
            dataset=self._train_dataset,
            # sampler=RandomSampler(self._train_dataset),
            batch_size=self.hparams.batch_size,
            collate_fn=self.prepare_sample,
            num_workers=self.hparams.loader_workers,
        )

    def val_dataloader(self) -> DataLoader:
        """ Function that loads the validation set. """
        return [
            DataLoader(
                dataset=self._train_subset,
                batch_size=self.hparams.batch_size,
                collate_fn=self.prepare_sample,
                num_workers=self.hparams.loader_workers,
            ),
            DataLoader(
                dataset=self._val_dataset,
                batch_size=self.hparams.batch_size,
                collate_fn=self.prepare_sample,
                num_workers=self.hparams.loader_workers,
            ),
        ]

    def test_dataloader(self) -> DataLoader:
        """ Function that loads the validation set. """
        return DataLoader(
            dataset=self._test_dataset,
            batch_size=self.hparams.batch_size,
            collate_fn=self.prepare_sample,
            num_workers=self.hparams.loader_workers,
        )

    def on_epoch_end(self):
        """ Pytorch lightning hook """
        if self.current_epoch + 1 >= self.nr_frozen_epochs and self._frozen:
            self.unfreeze_encoder()
            self._frozen = False

    def on_epoch_start(self):
        """ Pytorch lightning hook """
        if self.current_epoch == 0 and not self._frozen:
            log.info(f"\n-- Encoder model fine-tuning.")

        if not self.hparams.disable_progress_bar:
            nr_batches = math.ceil(
                (len(self._train_dataset) / self.hparams.batch_size)
                * self.hparams.train_percent_check
            )
            self._pbar = tqdm(total=nr_batches, unit="batch")

    def on_batch_start(self, batch):
        """ Pytorch lightning hook """
        if not self.hparams.disable_progress_bar:
            self._pbar.update(1)

    def on_pre_performance_check(self):
        """ Pytorch lightning hook """
        if (
            not self.hparams.disable_progress_bar
            and self.hparams.val_check_interval >= 1
        ):
            # closes tqdm progress bar before updating the shell
            self._pbar.close() if self._pbar else None

    def load_weights(self, checkpoint: str) -> None:
        """ Function that loads the weights from a given checkpoint file. 
        Note:
            If the checkpoint model architecture is different then `self`, only
            the common parts will be loaded.

        :param checkpoint: Path to the checkpoint containing the weights to be loaded.
        """
        log.info(f"loading model weights from {checkpoint}.")
        checkpoint = torch.load(checkpoint, map_location=lambda storage, loc: storage,)
        pretrained_dict = checkpoint["state_dict"]
        model_dict = self.state_dict()

        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        self.load_state_dict(model_dict)

    # ------------------------------------ Arg parsing ------------------------------------------
    @staticmethod
    def add_encoder_args(parser: HyperOptArgumentParser) -> HyperOptArgumentParser:
        """
        Functions that parses Encoder specific arguments/hyperparameters.
        :param hparams: HyperOptArgumentParser obj.

        Returns:
            - updated parser
        """
        raise NotImplementedError

    @staticmethod
    def add_model_specific_args(
        parser: HyperOptArgumentParser,
    ) -> HyperOptArgumentParser:
        """ Parser for Estimator specific arguments/hyperparameters. 
        :param parser: HyperOptArgumentParser obj

        Returns:
            - updated parser
        """
        parser.opt_list(
            "--nr_frozen_epochs",
            default=0,
            type=int,
            help="Number of epochs we want to keep the encoder model frozen.",
            tunable=True,
            options=[0, 1, 2, 3, 4, 5],
        )
        parser.add_argument(
            "--disable_progress_bar",
            default=False,
            help=(
                "By default the estimator class creates a progress bar during"
                "training. Using this flag you can desable this behavior."
            ),
            action="store_true",
        )
        return parser
