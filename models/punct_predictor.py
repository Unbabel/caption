# -*- coding: utf-8 -*-
r""" 
Punctuation Predictor
=============================
    Punctuation Predictor based on XLMR implementing the PyTorch Lightning interface that can be used to train a punctuation predictor.
"""
import os
from argparse import Namespace
from typing import Dict, List, Optional, Tuple

import torch
import yaml
from transformers import AdamW, AutoModel, AutoTokenizer
from transformers.file_utils import ModelOutput
from dataclasses import dataclass

import pytorch_lightning as pl

from pytorch_lightning.metrics.functional import accuracy
from utils import Config
from models.scalar_mix import ScalarMixWithDropout
from models.data_module import LABEL_ENCODER
from pytorch_lightning.metrics import MetricCollection, F1


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
    "additional_special_tokens": ["<en>", "<de>", "<it>", "<fr>"],
}

class PunctuationPredictor(pl.LightningModule):
    """PyTorch Lightning DataModule.

    :param hparams: Namespace with data specific arguments.
    :param tokenizer: Model Tokenizer.
    """

    class ModelConfig(Config):
        """The ModelConfig class is used to define Model settings.
        :param pretrained_model: Pretrained GPT2 model to be used.
        :param learning_rate: Learning Rate used during training.
        :param lm_coef: Weight assigned to the LM loss.
        :param mc_coef: Weight assigned to the Multiple-Choice loss.
        :param train_data: Path to a json file containing the train data.
        :param valid_data: Path to a json file containing the validation data.
        :param batch_size: Batch Size used during training.
        :param max_history: Max number of context sentences.
        :param num_candidates: Number of distractors used during training.
        """
        pretrained_model: str = "xlm-roberta-base"

        # Optimizer
        learning_rate: float = 6.25e-5
        
        # Training details
        batch_size: int = 2

    def __init__(self, hparams: Namespace):
        super().__init__()
        self.hparams = hparams
        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.pretrained_model)
        orig_vocab = len(self.tokenizer.get_vocab())
        num_added_tokens = self.tokenizer.add_special_tokens(ATTR_TO_SPECIAL_TOKEN)
        self.encoder = AutoModel.from_pretrained(self.hparams.pretrained_model)
        self.encoder.resize_token_embeddings(orig_vocab + num_added_tokens)
        self.scalar_mix = ScalarMixWithDropout(
            mixture_size=self.encoder.config.num_hidden_layers +1,
            do_layer_norm=True,
            dropout=self.hparams.scalar_mix_dropout,
        )
        self.binary_head = nn.Linear(2 * self.encoder.config.hidden_size, 2)
        self.punct_head  = nn.Linear(2 * self.encoder.config.hidden_size, len(LABEL_ENCODER))
        self.binary_f1 = F1(num_classes=1)
        self.micro_f1 =F1(num_classes=len(LABEL_ENCODER), average="micro")
        self.macro_f1 =F1(num_classes=len(LABEL_ENCODER), average="macro")
        

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(), lr=self.hparams.learning_rate, correct_bias=True
        )
        #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], []

    def forward(self, input_ids, attention_mask, word_pointer, token_type_ids=None, binary_labels=None, punct_labels=None) -> PunctModelOutput:
        _, _, all_layers = self.model(
            input_ids, attention_mask, output_hidden_states=True, return_dict=False
        )
        embeddings = self.scalar_mix(all_layers, attention_mask)
        embeddings = torch.cat([
            w.index_select(0, i).unsqueeze(0)
            for w, i in zip(embeddings, word_pointer)
        ])
        concat = [
            torch.cat(
                [embeddings[i, j, :], embeddings[i, j + 1, :]], dim=0
            )
            if j < embeddings.shape[1] - 1
            else torch.cat(
                [embeddings[i, j, :], embeddings[i, j, :]], dim=0
            )
            for i in range(embeddings.shape[0])
            for j in range(embeddings.shape[1])
        ]
        adjacent_embeddings = torch.stack(concat).view(
            embeddings.shape[0],
            embeddings.shape[1],
            2 * self.encoder.config.hidden_size,
        )
        binary_logits = self.binary_head(self.dropout(adjacent_embeddings))
        punct_logits = self.punct_head(self.dropout(adjacent_embeddings))
        
        loss_fct = CrossEntropyLoss()
        if binary_labels:
            binary_loss = loss_fct(binary_logits.view(-1, binary_logits.size(-1)), binary_labels.view(-1))
        else:
            binary_loss = None

        if punct_labels:
            punct_loss = loss_fct(punct_logits.view(-1, punct_logits.size(-1)), punct_labels.view(-1))
        else:
            punct_loss = None
        
        if punct_loss and binary_loss:
            loss = punct_loss + binary_loss
        else:
            loss = None

        return PunctModelOutput(binary_logits, binary_loss, punct_logits, punct_loss, loss, adjacent_embeddings)

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
        self.log('train_binary_f1_step', self.binary_f1(output.binary_logits, batch[-2]))
        self.log('train_micro_f1_step', self.micro_f1(output.punct_logits, batch[-1]))
        self.log('train_macro_f1_step', self.macro_f1(output.punct_logits, batch[-1]))
        return {
            "loss": output.loss,
            "log": {
                "loss": output.loss,
            },
        }
    
    def validation_step(
        self, batch: Tuple[torch.Tensor], batch_nb: int, *args, **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Similar to the training step but with the model in eval mode.
        :returns: dictionary passed to the validation_end function.
        """
        output = self.forward(*batch)
        return {
            "val_loss": output.loss,
            "val_binary_f1": self.binary_f1(output.binary_logits, batch[-2]),
            "val_micro_f1": self.micro_f1(output.punct_logits, batch[-1]),
            "val_macro_f1": self.micro_f1(output.punct_logits, batch[-1]),
        }

    def validation_epoch_end(
        self, outputs: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """Function that takes as input a list of dictionaries returned by the validation_step
        function and measures the model performance accross the entire validation set.
        Returns:
            - Dictionary with metrics to be added to the lightning logger.
        """
        # Average all metrics
        metrics = {
            "val_loss": torch.stack(
                [x["val_loss"] for x in outputs]
            ).mean(),
            "val_binary_f1": torch.stack([x["val_binary_f1"] for x in outputs]).mean(),
            "val_micro_f1": torch.stack([x["val_micro_f1"] for x in outputs]).mean(),
            "val_macro_f1": torch.stack([x["val_macro_f1"] for x in outputs]).mean(),
        }
        return {
            "progress_bar": metrics,
            "log": metrics,
        }

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