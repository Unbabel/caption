# -*- coding: utf-8 -*-
r""" 
Punctuation Predictor
=============================
    Punctuation Predictor based on XLMR implementing the PyTorch Lightning interface that can be used to train a punctuation predictor.
"""
import os
from argparse import Namespace
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import click
import pytorch_lightning as pl
import torch
import torch.nn as nn
import yaml
from pytorch_lightning.metrics import F1, MetricCollection
from pytorch_lightning.metrics.functional import accuracy
from transformers import AdamW, AutoModel, AutoTokenizer
from transformers.file_utils import ModelOutput
from utils import Config

from models.data_module import LABEL_ENCODER
from models.scalar_mix import ScalarMixWithDropout
from models.ser_metric import SlotErrorRate


@dataclass
class PunctModelOutput(ModelOutput):
    binary_loss: torch.FloatTensor = None
    binary_logits: torch.FloatTensor = None
    punct_loss: torch.FloatTensor = None
    punct_logits: torch.FloatTensor = None
    loss: torch.FloatTensor = None
    embeddings: torch.FloatTensor = None


ATTR_TO_SPECIAL_TOKEN = {
    "bos_token": "<bos>",
    "eos_token": "<eos>",
    "pad_token": "<pad>",
    # "additional_special_tokens": ["<en>", "<de>", "<it>", "<fr>"],
}
LANGUAGE_PAIRS = ["<en>", "<de>", "<it>", "<fr>"]

os.environ["TOKENIZERS_PARALLELISM"] = "1"


class PunctuationPredictor(pl.LightningModule):
    """PyTorch Lightning DataModule.

    :param hparams: Namespace with data specific arguments.
    :param tokenizer: Model Tokenizer.
    """

    class ModelConfig(Config):
        """The ModelConfig class is used to define Model settings."""

        pretrained_model: str = "xlm-roberta-base"
        # Training details
        batch_size: int = 2
        dropout: float = 0.1
        language_factors: bool = False
        nr_frozen_epochs: float = 0.4
        keep_embeddings_frozen: bool = False
        layerwise_decay: float = 0.95
        encoder_learning_rate: float = 3.0e-5
        learning_rate: float = 6.25e-5
        binary_loss: int = 1
        punct_loss: int = 1

    def __init__(self, hparams: Namespace):
        super().__init__()
        self.hparams = hparams
        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.pretrained_model)
        orig_vocab = len(self.tokenizer.get_vocab())
        num_added_tokens = self.tokenizer.add_special_tokens(ATTR_TO_SPECIAL_TOKEN)
        self.encoder = AutoModel.from_pretrained(self.hparams.pretrained_model)
        self.encoder.resize_token_embeddings(orig_vocab + num_added_tokens)

        if self.hparams.language_factors:
            self.language_embeddings = nn.Embedding(
                len(LANGUAGE_PAIRS), self.encoder.config.hidden_size
            )

        # The encoder always starts in a frozen state.
        if self.hparams.nr_frozen_epochs > 0:
            self._frozen = True
            self.freeze_encoder()
        else:
            self._frozen = False

        if self.hparams.keep_embeddings_frozen:
            self.freeze_embeddings()

        self.scalar_mix = ScalarMixWithDropout(
            mixture_size=self.encoder.config.num_hidden_layers + 1,
            do_layer_norm=True,
            dropout=self.hparams.dropout,
        )
        self.head_dropout = nn.Dropout(self.hparams.dropout)
        self.binary_head = nn.Linear(2 * self.encoder.config.hidden_size, 2)
        self.punct_head = nn.Linear(
            2 * self.encoder.config.hidden_size, len(LABEL_ENCODER)
        )
        self.binary_f1 = F1(num_classes=1)
        self.micro_f1 = F1(num_classes=len(LABEL_ENCODER), average="micro")
        self.macro_f1 = F1(num_classes=len(LABEL_ENCODER), average="macro")
        self.ser = SlotErrorRate(padding=-100, ignore=0)

    def freeze_embeddings(self) -> None:
        """ Freezes the encoder layer. """
        for param in self.encoder.embeddings.parameters():
            param.requires_grad = False

    def freeze_encoder(self) -> None:
        """ Freezes the encoder layer. """
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self) -> None:
        """ un-freezes the encoder layer. """
        if self._frozen:
            click.secho(f"Encoder model fine-tuning", fg="red")
            for param in self.encoder.parameters():
                param.requires_grad = True
            if self.hparams.keep_embeddings_frozen:
                self.freeze_embeddings()
            self._frozen = False

    def layerwise_lr(self, lr: float, decay: float):
        """
        :return: List with grouped model parameters with layer-wise decaying learning rate
        """
        # Embedding Layer
        opt_parameters = [
            {
                "params": self.encoder.embeddings.parameters(),
                "lr": lr * decay ** (self.encoder.config.num_hidden_layers),
            }
        ]
        # All layers
        opt_parameters += [
            {
                "params": self.encoder.encoder.layer[l].parameters(),
                "lr": lr * decay ** l,
            }
            for l in range(self.encoder.config.num_hidden_layers - 2, 0, -1)
        ]
        return opt_parameters

    def configure_optimizers(self):
        layer_parameters = self.layerwise_lr(
            self.hparams.encoder_learning_rate, self.hparams.layerwise_decay
        )
        top_layers_parameters = [
            {"params": self.binary_head.parameters(), "lr": self.hparams.learning_rate},
            {"params": self.punct_head.parameters(), "lr": self.hparams.learning_rate},
            {"params": self.scalar_mix.parameters(), "lr": self.hparams.learning_rate},
        ]
        optimizer = AdamW(
            layer_parameters + top_layers_parameters,
            lr=self.hparams.learning_rate,
            correct_bias=True,
        )
        # print (self.total_steps)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1)
        return [optimizer], [scheduler]

    def forward(
        self,
        input_ids,
        word_pointer,
        attention_mask,
        token_type_ids=None,
        binary_labels=None,
        punct_labels=None,
    ) -> PunctModelOutput:
        if self.hparams.language_factors:
            word_embeddings = self.encoder.embeddings.word_embeddings(input_ids)
            language_embeddings = self.language_embeddings(token_type_ids)
            inputs_embeds = word_embeddings + language_embeddings
            _, _, all_layers = self.encoder(
                None,
                attention_mask,
                inputs_embeds=inputs_embeds,
                output_hidden_states=True,
                return_dict=False,
            )
        else:
            _, _, all_layers = self.encoder(
                input_ids, attention_mask, output_hidden_states=True, return_dict=False
            )

        embeddings = self.scalar_mix(all_layers, attention_mask)
        embeddings = torch.cat(
            [
                w.index_select(0, i).unsqueeze(0)
                for w, i in zip(embeddings, word_pointer)
            ]
        )
        concat = [
            torch.cat([embeddings[i, j, :], embeddings[i, j + 1, :]], dim=0)
            if j < embeddings.shape[1] - 1
            else torch.cat([embeddings[i, j, :], embeddings[i, j, :]], dim=0)
            for i in range(embeddings.shape[0])
            for j in range(embeddings.shape[1])
        ]
        adjacent_embeddings = torch.stack(concat).view(
            embeddings.shape[0],
            embeddings.shape[1],
            2 * self.encoder.config.hidden_size,
        )
        binary_logits = self.binary_head(self.head_dropout(adjacent_embeddings))
        punct_logits = self.punct_head(self.head_dropout(adjacent_embeddings))

        loss_fct = nn.CrossEntropyLoss()

        if (binary_labels is not None) and (punct_labels is not None):
            binary_loss = loss_fct(
                binary_logits.view(-1, binary_logits.size(-1)), binary_labels.view(-1)
            )
            punct_loss = loss_fct(
                punct_logits.view(-1, punct_logits.size(-1)), punct_labels.view(-1)
            )
            loss = (
                self.hparams.binary_loss * binary_loss
                + self.hparams.punct_loss * punct_loss
            )
            return PunctModelOutput(
                binary_loss,
                binary_logits,
                punct_loss,
                punct_logits,
                loss,
                adjacent_embeddings,
            )
        else:
            return PunctModelOutput(
                None, binary_logits, None, punct_logits, None, adjacent_embeddings
            )

    def training_step(
        self, batch: Tuple[torch.Tensor], batch_nb: int, *args, **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Runs one training step. This usually consists in the forward function followed
            by the loss function.
        :param batch: The output of your dataloader.
        :param batch_nb: Integer displaying which batch this is
        Returns:
            - dictionary containing the loss and the metrics to be added to the lightning logger.
        """
        output = self.forward(*batch)
        binary_pred, binary_target = (
            torch.topk(output.binary_logits, 1)[1].view(-1),
            batch[-2].view(-1),
        )
        mask = (binary_target != -100).bool()
        binary_pred = torch.masked_select(binary_pred, mask)
        binary_target = torch.masked_select(binary_target, mask)

        punct_pred, punct_target = (
            torch.topk(output.punct_logits, 1)[1].view(-1),
            batch[-1].view(-1),
        )
        mask = (punct_target != -100).bool()
        punct_pred = torch.masked_select(punct_pred, mask)
        punct_target = torch.masked_select(punct_target, mask)

        # ser = slot_error_rate(punct_pred, punct_target, ignore=0)
        self.log(
            "train_binary_f1",
            self.binary_f1(binary_pred, binary_target),
            on_epoch=False,
            on_step=True,
            logger=True,
        )
        self.log(
            "train_micro_f1",
            self.micro_f1(punct_pred, punct_target),
            on_epoch=False,
            on_step=True,
            logger=True,
        )
        self.log(
            "train_macro_f1",
            self.macro_f1(punct_pred, punct_target),
            on_epoch=False,
            on_step=True,
            logger=True,
        )
        self.log(
            "train_ser",
            self.ser(punct_pred, punct_target),
            on_epoch=False,
            on_step=True,
            logger=True,
        )

        self.log("loss", output.loss, logger=True)
        self.log("binary_loss", output.binary_loss, logger=True)
        self.log("punct_loss", output.punct_loss, logger=True)

        if (
            self.hparams.nr_frozen_epochs < 1.0
            and self.hparams.nr_frozen_epochs > 0.0
            and batch_nb
            > (
                len(self.trainer.datamodule.train_dataloader())
                * self.hparams.nr_frozen_epochs
            )
        ):
            self.unfreeze_encoder()
            self._frozen = False

        return output.loss

    def training_epoch_end(self, outs):
        # log epoch metric
        self.log(
            "train_binary_f1", self.binary_f1.compute(), on_epoch=True, logger=True
        )
        self.log("train_micro_f1", self.micro_f1.compute(), on_epoch=True, logger=True)
        self.log("train_macro_f1", self.macro_f1.compute(), on_epoch=True, logger=True)
        self.log("train_ser", self.ser.compute(), on_epoch=True, logger=True)
        self.binary_f1.reset()
        self.micro_f1.reset()
        self.macro_f1.reset()

    def validation_step(
        self, batch: Tuple[torch.Tensor], batch_nb: int, *args, **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Similar to the training step but with the model in eval mode.
        :returns: dictionary passed to the validation_end function.
        """
        output = self.forward(*batch)

        binary_pred, binary_target = (
            torch.topk(output.binary_logits, 1)[1].view(-1),
            batch[-2].view(-1),
        )
        mask = (binary_target != -100).bool()
        binary_pred = torch.masked_select(binary_pred, mask)
        binary_target = torch.masked_select(binary_target, mask)

        punct_pred, punct_target = (
            torch.topk(output.punct_logits, 1)[1].view(-1),
            batch[-1].view(-1),
        )
        mask = (punct_target != -100).bool()
        punct_pred = torch.masked_select(punct_pred, mask)
        punct_target = torch.masked_select(punct_target, mask)
        self.log("binary_f1", self.binary_f1(binary_pred, binary_target), prog_bar=True)
        self.log("micro_f1", self.micro_f1(punct_pred, punct_target), prog_bar=True)
        self.log("macro_f1", self.macro_f1(punct_pred, punct_target), prog_bar=True)
        self.log("ser", self.ser(punct_pred, punct_target), prog_bar=True)
        return output.loss

    def validation_epoch_end(
        self, outputs: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        # log epoch metric
        self.log(
            "binary_f1",
            self.binary_f1.compute(),
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "micro_f1",
            self.micro_f1.compute(),
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "macro_f1",
            self.macro_f1.compute(),
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log("ser", self.ser.compute(), on_epoch=True, prog_bar=True, logger=True)
        self.binary_f1.reset()
        self.micro_f1.reset()
        self.macro_f1.reset()

    @classmethod
    def from_experiment(cls, experiment_folder: str):
        """Function that loads the model from an experiment folder.
        :param experiment_folder: Path to the experiment folder.
        :return:Pretrained model.
        """
        hparams_file = os.path.join(experiment_folder, "hparams.yaml")
        hparams = yaml.load(open(hparams_file).read(), Loader=yaml.FullLoader)

        checkpoints = [
            file for file in os.listdir(experiment_folder) if file.endswith(".ckpt")
        ]
        checkpoint_path = os.path.join(experiment_folder, checkpoints[-1])
        model = cls.load_from_checkpoint(
            checkpoint_path, hparams=Namespace(**hparams), strict=True
        )
        # Make sure model is in prediction mode
        model.eval()
        model.freeze()
        return model
