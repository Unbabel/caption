# -*- coding: utf-8 -*-
r""" 
Punctuation Predictor
=============================
    Punctuation Predictor based on XLMR implementing the PyTorch Lightning interface that can be used to train a punctuation predictor.
"""
import os
from argparse import Namespace
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import click
from click.termui import secho
import pytorch_lightning as pl
import torch
import torch.nn as nn
import yaml
from pytorch_lightning.metrics import F1
from transformers import AdamW, AutoModel, AutoTokenizer
from transformers.file_utils import ModelOutput
from utils import Config

from models.data_module import PUNCTUATION_LABEL_ENCODER, CAPITALIZATION_LABEL_ENCODER
from models.scalar_mix import ScalarMixWithDropout
from models.ser_metric import SlotErrorRate


@dataclass
class PunctModelOutput(ModelOutput):
    cap_loss: torch.FloatTensor = None
    cap_logits: torch.FloatTensor = None
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
        # language_factors: bool = False # Deprecated!
        nr_frozen_epochs: float = 0.4
        keep_embeddings_frozen: bool = False
        layerwise_decay: float = 0.95
        encoder_learning_rate: float = 3.0e-5
        learning_rate: float = 6.25e-5
        cap_loss: int = 1
        punct_loss: int = 1

    def __init__(self, hparams: Namespace, load_weights_from_checkpoint: Optional[str] = None):
        super().__init__()
        self.hparams = hparams
        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.pretrained_model)
        orig_vocab = len(self.tokenizer.get_vocab())
        num_added_tokens = self.tokenizer.add_special_tokens(ATTR_TO_SPECIAL_TOKEN)
        self.encoder = AutoModel.from_pretrained(self.hparams.pretrained_model)
        self.encoder.resize_token_embeddings(orig_vocab + num_added_tokens)

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
        self.cap_head = nn.Linear(
            2 * self.encoder.config.hidden_size, len(CAPITALIZATION_LABEL_ENCODER)
        )
        self.punctuation_head = nn.Linear(
            2 * self.encoder.config.hidden_size, len(PUNCTUATION_LABEL_ENCODER)
        )
        self.cap_micro_f1 = F1(num_classes=len(CAPITALIZATION_LABEL_ENCODER), average="micro")
        self.cap_macro_f1 = F1(num_classes=len(CAPITALIZATION_LABEL_ENCODER), average="macro")
        self.punct_micro_f1 = F1(num_classes=len(PUNCTUATION_LABEL_ENCODER), average="micro")
        self.punct_macro_f1 = F1(num_classes=len(PUNCTUATION_LABEL_ENCODER), average="macro")
        self.punct_ser = SlotErrorRate(padding=-100, ignore=0)
        self.cap_ser = SlotErrorRate(padding=-100) 
        self.punct_loss_fct = nn.CrossEntropyLoss(ignore_index=-100) # Using the ignore_index prevents the model from learning that PAD is followed by PAD
        self.cap_loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

        if load_weights_from_checkpoint is not None:
            if os.path.exists(load_weights_from_checkpoint):
                self.load_weights_from_punctuation_model(load_weights_from_checkpoint)
            else:
                secho(f"Path {load_weights_from_checkpoint} does not exist!")

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
            {"params": self.cap_head.parameters(), "lr": self.hparams.learning_rate},
            {"params": self.punctuation_head.parameters(), "lr": self.hparams.learning_rate},
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
        # token_type_ids=None,
        cap_labels=None,
        punct_labels=None,
    ) -> PunctModelOutput:

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
        cap_logits = self.cap_head(self.head_dropout(adjacent_embeddings))
        punct_logits = self.punctuation_head(self.head_dropout(adjacent_embeddings))


        if (cap_labels is not None) and (punct_labels is not None):
            cap_loss = self.cap_loss_fct(
                cap_logits.view(-1, cap_logits.size(-1))[1:, :], cap_labels.view(-1)[1:] # We ignore the first position, so it doesn't learn to capitalise based on position
            )
            punct_loss = self.punct_loss_fct(
                punct_logits.view(-1, punct_logits.size(-1)), punct_labels.view(-1)
            )
            loss = (
                self.hparams.cap_loss * cap_loss
                + self.hparams.punct_loss * punct_loss
            )
            return PunctModelOutput(
                cap_loss,
                cap_logits,
                punct_loss,
                punct_logits,
                loss,
                adjacent_embeddings,
            )
        else:
            return PunctModelOutput(
                None, cap_logits, None, punct_logits, None, adjacent_embeddings
            )

    def predict(self, batch, encode: bool):
        # Creating a mask to ignore pad, bos and eos tokens
        input_ids = batch[1].clone()
        input_ids[
            input_ids == self.tokenizer.eos_token_id
        ] = self.tokenizer.pad_token_id
        mask = (input_ids != self.tokenizer.pad_token_id).bool()

        try:
            mask[:, 0] = 1
        except:
            # If the file was empty
            return None, None

        punct_label_decoder = {v: k for k, v in PUNCTUATION_LABEL_ENCODER.items()}
        cap_label_decoder = {v: k for k, v in CAPITALIZATION_LABEL_ENCODER.items()}
        with torch.no_grad():
            batch = [model_input.cuda() for model_input in batch]
            output = self.forward(*batch)
            cap_pred = torch.topk(output.cap_logits, 1)[1].view(-1)
            punct_pred = torch.topk(output.punct_logits, 1)[1].view(-1)
            if encode:
                cap_pred = torch.masked_select(cap_pred, mask.view(-1).cuda())
                punct_pred = torch.masked_select(punct_pred, mask.view(-1).cuda())
                punct_pred = punct_pred.cpu().tolist()
                cap_pred = cap_pred.cpu().tolist()
                # TODO(joao.janeiro@): This will probably error out, since the match is no longer 1-to-1, same key for multiple values
                punct_pred = [punct_label_decoder[label] for label in punct_pred]
                cap_pred = [cap_label_decoder[label] for label in cap_pred]
            return cap_pred, punct_pred

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
        cap_pred, cap_target = (
            torch.topk(output.cap_logits, 1)[1].view(-1),
            batch[-2].view(-1),
        )
        mask = (cap_target != -100).bool() # Let's remove padding
        cap_pred = torch.masked_select(cap_pred, mask)
        cap_target = torch.masked_select(cap_target, mask)

        punct_pred, punct_target = (
            torch.topk(output.punct_logits, 1)[1].view(-1),
            batch[-1].view(-1),
        )
        mask = (punct_target != -100).bool()
        punct_pred = torch.masked_select(punct_pred, mask)
        punct_target = torch.masked_select(punct_target, mask)

        # ser = slot_error_rate(punct_pred, punct_target, ignore=0)
        self.log(
            "train_cap_micro_f1",
            self.cap_micro_f1(cap_pred, cap_target),
            on_epoch=False,
            on_step=True,
            logger=True,
        )
        self.log(
            "train_cap_macro_f1",
            self.cap_macro_f1(cap_pred, cap_target),
            on_epoch=False,
            on_step=True,
            logger=True,
        )
        self.log(
            "train_punct_micro_f1",
            self.punct_micro_f1(punct_pred, punct_target),
            on_epoch=False,
            on_step=True,
            logger=True,
        )
        self.log(
            "train_punct_macro_f1",
            self.punct_macro_f1(punct_pred, punct_target),
            on_epoch=False,
            on_step=True,
            logger=True,
        )
        self.log(
            "train_punct_ser",
            self.punct_ser(punct_pred, punct_target),
            on_epoch=False,
            on_step=True,
            logger=True,
        )
        self.log(
            "train_cap_ser",
            self.cap_ser(cap_pred, cap_target),
            on_epoch=False,
            on_step=True,
            logger=True,
        )

        self.log("loss", output.loss, logger=True)
        self.log("cap_loss", output.cap_loss, logger=True)
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
            "train_cap_micro_f1", self.cap_micro_f1.compute(), on_epoch=True, logger=True
        )
        self.log(
            "train_cap_macro_f1", self.cap_macro_f1.compute(), on_epoch=True, logger=True
        )
        self.log("train_punct_micro_f1", self.punct_micro_f1.compute(), on_epoch=True, logger=True)
        self.log("train_punct_macro_f1", self.punct_macro_f1.compute(), on_epoch=True, logger=True)
        self.log("train_punct_ser", self.punct_ser.compute(), on_epoch=True, logger=True)
        self.log("train_cap_ser", self.cap_ser.compute(), on_epoch=True, logger=True)
        self.log(
            "train_avg_ser",
            (self.cap_ser.compute()+self.punct_ser.compute())/2,
            on_epoch=True,
            logger=True,
        )
        self.cap_micro_f1.reset()
        self.cap_macro_f1.reset()
        self.punct_micro_f1.reset()
        self.punct_macro_f1.reset()

    def validation_step(
        self, batch: Tuple[torch.Tensor], batch_nb: int, *args, **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Similar to the training step but with the model in eval mode.
        :returns: dictionary passed to the validation_end function.
        """
        output = self.forward(*batch)

        cap_pred, cap_target = (
            torch.topk(output.cap_logits, 1)[1].view(-1),
            batch[-2].view(-1),
        )
        mask = (cap_target != -100).bool()
        cap_pred = torch.masked_select(cap_pred, mask)
        cap_target = torch.masked_select(cap_target, mask)

        punct_pred, punct_target = (
            torch.topk(output.punct_logits, 1)[1].view(-1),
            batch[-1].view(-1),
        )
        mask = (punct_target != -100).bool()
        punct_pred = torch.masked_select(punct_pred, mask)
        punct_target = torch.masked_select(punct_target, mask)


        self.log("cap_micro_f1", self.cap_micro_f1(cap_pred, cap_target), prog_bar=True)
        self.log("cap_macro_f1", self.cap_macro_f1(cap_pred, cap_target), prog_bar=True)
        self.log("punct_micro_f1", self.punct_micro_f1(punct_pred, punct_target), prog_bar=True)
        self.log("punct_macro_f1", self.punct_macro_f1(punct_pred, punct_target), prog_bar=True)
        self.log("punct_ser", self.punct_ser(punct_pred, punct_target), prog_bar=True)
        self.log("cap_ser", self.cap_ser(cap_pred, cap_target), prog_bar=True)
        return output.loss

    def validation_epoch_end(
        self, outputs: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        # log epoch metric
        self.log(
            "cap_micro_f1",
            self.cap_micro_f1.compute(),
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "cap_macro_f1",
            self.cap_macro_f1.compute(),
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "punct_micro_f1",
            self.punct_micro_f1.compute(),
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "punct_macro_f1",
            self.punct_macro_f1.compute(),
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log("punct_ser", self.punct_ser.compute(), on_epoch=True, prog_bar=True, logger=True)
        self.log("cap_ser", self.cap_ser.compute(), on_epoch=True, logger=True)
        self.log(
            "avg_ser",
            (self.cap_ser.compute()+self.punct_ser.compute())/2,
            on_epoch=True,
            logger=True,
        )
        self.cap_micro_f1.reset()
        self.cap_macro_f1.reset()
        self.punct_micro_f1.reset()
        self.punct_macro_f1.reset()

    def load_weights_from_punctuation_model(self, checkpoint: str) -> None:
        """Function that loads the weights from a given checkpoint file.
        Note:
            If the checkpoint model architecture is different then `self`, only
            the common parts will be loaded.
        :param checkpoint: Path to the checkpoint containing the weights to be loaded.
        """
        checkpoint = checkpoint if checkpoint.endswith("/") else checkpoint + "/"
        checkpoints = [
            file
            for file in os.listdir(checkpoint + "checkpoints/")
            if file.endswith(".ckpt")
        ]
        checkpoint_path = os.path.join(
            checkpoint + "checkpoints/", checkpoints[-1]
        )

        secho(f"Loading weights from {checkpoint_path}.")
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        pretrained_dict = checkpoint["state_dict"]
        model_dict = self.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        self.load_state_dict(model_dict, strict=False)

    @classmethod
    def from_experiment(cls, experiment_folder: str):
        """Function that loads the model from an experiment folder.
        :param experiment_folder: Path to the experiment folder.
        :return:Pretrained model.
        """
        hparams_file = os.path.join(experiment_folder, "hparams.yaml")
        hparams = yaml.load(open(hparams_file).read(), Loader=yaml.FullLoader)

        checkpoints = [
            file
            for file in os.listdir(experiment_folder + "checkpoints/")
            if file.endswith(".ckpt")
        ]
        checkpoint_path = os.path.join(
            experiment_folder + "checkpoints/", checkpoints[-1]
        )
        model = cls.load_from_checkpoint(
            checkpoint_path, hparams=Namespace(**hparams), strict=True
        )
        # Make sure model is in prediction mode
        model.eval()
        model.freeze()
        return model

