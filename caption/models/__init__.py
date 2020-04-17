# -*- coding: utf-8 -*-
import logging

import pytorch_lightning as ptl
import pandas as pd
import os

from .taggers import TransformerTagger
from .language_models import MaskedLanguageModel

str2model = {
    "TransformerTagger": TransformerTagger,
    "MaskedLanguageModel": MaskedLanguageModel,
}


def build_model(hparams) -> ptl.LightningModule:
    """
    Function that builds an estimator model from the HyperOptArgumentParser
    :param hparams: HyperOptArgumentParser
    """
    return str2model[hparams.model](hparams)


def add_model_args(parser, model: str):
    return str2model[model].add_model_specific_args(parser)
    try:
        return str2model[model].add_model_specific_args(parser)
    except KeyError:
        raise Exception(f"{model} is not a valid model type!")
