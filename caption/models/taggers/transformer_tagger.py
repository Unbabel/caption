# -*- coding: utf-8 -*-
r"""
Transformer Tagger
==============
    Model that uses a pretrained transformer model to tag sequences of words.
"""
import torch
from torch import nn

from caption.models.encoders import Encoder, str2encoder
from caption.models.scalar_mix import ScalarMixWithDropout
from caption.models.taggers.tagger_base import Tagger
from caption.optimizers import build_optimizer
from caption.schedulers import build_scheduler
from test_tube import HyperOptArgumentParser


class TransformerTagger(Tagger):
    """
    Word tagger that uses a pretrained Transformer model to extract features from text.

    :param hparams: HyperOptArgumentParser containing the hyperparameters.
    """

    def __init__(self, hparams: HyperOptArgumentParser,) -> None:
        super().__init__(hparams)

    def _build_model(self) -> Tagger:
        """
        Initializes the estimator architecture.
        """
        super()._build_model()
        self.layer = (
            int(self.hparams.layer)
            if self.hparams.layer != "mix"
            else self.hparams.layer
        )

        if self.hparams.concat_tokens:
            self.tagging_head = nn.Linear(
                2 * self.encoder.output_units, self.label_encoder.vocab_size
            )
        else:
            self.tagging_head = nn.Linear(
                self.encoder.output_units, self.label_encoder.vocab_size
            )

        self.dropout = nn.Dropout(self.hparams.dropout)
        self.scalar_mix = (
            ScalarMixWithDropout(
                mixture_size=self.encoder.num_layers,
                do_layer_norm=True,
                dropout=self.hparams.scalar_mix_dropout,
            )
            if self.layer == "mix"
            else None
        )

    def _build_encoder(self, hparams: HyperOptArgumentParser) -> Encoder:
        """
        Initializes the encoder.
        """
        return str2encoder[self.hparams.encoder_model].from_pretrained(hparams)

    def configure_optimizers(self):
        """ Sets different Learning rates for different parameter groups. """
        parameters = [
            {"params": self.tagging_head.parameters()},
            {
                "params": self.encoder.parameters(),
                "lr": self.hparams.encoder_learning_rate,
            },
        ]
        if self.scalar_mix:
            parameters.append(
                {
                    "params": self.scalar_mix.parameters(),
                    "lr": self.hparams.encoder_learning_rate,
                }
            )
        optimizer = build_optimizer(parameters, self.hparams)
        scheduler = build_scheduler(optimizer, self.hparams)
        return [optimizer], [scheduler]

    def forward(
        self,
        tokens: torch.tensor,
        lengths: torch.tensor,
        word_boundaries: torch.tensor,
        word_lengths: torch.tensor,
    ) -> dict:
        """
        Function that encodes a sequence and returns the punkt and cap tags.

        :param tokens: wordpiece tokens [batch_size x wordpiece_length]
        :param lengths: wordpiece sequences lengths [batch_size]
        :param word_boundaries: wordpiece token positions [batch_size x word_length]
        :param word_lengths: word sequences lengths [batch_size]

        Return: Dictionary with model outputs to be passed to the loss function.
        """
        # When using just one GPU this should not change behavior
        # but when splitting batches across GPU the tokens have padding
        # from the entire original batch
        if self.trainer and self.trainer.use_dp and self.trainer.num_gpus > 1:
            tokens = tokens[:, : lengths.max()]

        encoder_out = self.encoder(tokens, lengths)

        if self.scalar_mix:
            embeddings = self.scalar_mix(encoder_out["all_layers"], encoder_out["mask"])
        else:
            try:
                embeddings = encoder_out["all_layers"][self.layer]
            except IndexError:
                raise Exception(
                    "Invalid model layer {}. Only {} layers available".format(
                        self.hparams.layer, self.encoder.num_layers
                    )
                )

        word_embeddings = torch.cat(
            [
                w.index_select(0, i).unsqueeze(0)
                for w, i in zip(embeddings, word_boundaries)
            ]
        )
        if self.hparams.concat_tokens:
            concat = [
                torch.cat(
                    [word_embeddings[i, j, :], word_embeddings[i, j + 1, :]], dim=0
                )
                if j < word_embeddings.shape[1] - 1
                else torch.cat(
                    [word_embeddings[i, j, :], word_embeddings[i, j, :]], dim=0
                )
                for i in range(word_embeddings.shape[0])
                for j in range(word_embeddings.shape[1])
            ]
            new_embedds = torch.stack(concat).view(
                word_embeddings.shape[0],
                word_embeddings.shape[1],
                2 * self.encoder.output_units,
            )
            tag_predictions = self.tagging_head(self.dropout(new_embedds))
        else:
            tag_predictions = self.tagging_head(self.dropout(word_embeddings))

        return {
            "tags": tag_predictions,
        }

    @staticmethod
    def add_model_specific_args(
        parser: HyperOptArgumentParser,
    ) -> HyperOptArgumentParser:
        parser = super(TransformerTagger, TransformerTagger).add_model_specific_args(
            parser
        )
        # Parameters for the Encoder model
        parser.add_argument(
            "--encoder_model",
            default="RoBERTa",
            type=str,
            help="Encoder model to be used.",
            choices=["BERT", "RoBERTa", "XLM-RoBERTa"],
        )
        parser.add_argument(
            "--pretrained_model",
            default="roberta.base",
            type=str,
            help=(
                "Encoder pretrained model to be used. "
                "(e.g. roberta.base or roberta.large"
            ),
        )
        parser.add_argument(
            "--encoder_learning_rate",
            default=1e-05,
            type=float,
            help="Encoder specific learning rate.",
        )
        parser.opt_list(
            "--dropout",
            default=0.1,
            type=float,
            help="Dropout to be applied in feed forward net on top.",
            tunable=True,
            options=[0.1, 0.2, 0.3, 0.4, 0.5],
        )
        parser.add_argument(
            "--layer",
            default="-1",
            type=str,
            help=(
                "Encoder model layer to be used. Last one is the default. "
                "If 'mix' all the encoder layer's will be combined with layer-wise attention"
            ),
        )
        parser.opt_list(
            "--scalar_mix_dropout",
            default=0.0,
            type=float,
            tunable=False,
            options=[0.0, 0.05, 0.1, 0.15, 0.2],
            help=(
                "The ammount of layer wise dropout when using scalar_mix option for layer pooling. "
                "Only applicable if the 'layer' parameters is set to mix"
            ),
        )
        parser.add_argument(
            "--concat_tokens",
            default=False,
            help="Apply concatenation of consecutive words to feed to the linear projection",
            action="store_true",
        )
        return parser
