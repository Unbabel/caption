# -*- coding: utf-8 -*-
from test_tube import HyperOptArgumentParser

from .lazy_dataset import LineByLineTextDataset


def load_mlm_dataset(hparams: HyperOptArgumentParser, train=True, val=True, test=True):
    """
    This dataset loader is used for loading data for language modeling.

    :param hparams: HyperOptArgumentParser obj containg the path to the data files.
    :param train: flag to return the train set.
    :param val: flag to return the validation set.
    :param test: flag to return the test set.
    
    Returns:
        - Training Dataset, Development Dataset, Testing Dataset
    """
    func_out = []
    if train:
        func_out.append(LineByLineTextDataset(hparams.train_path))
    if val:
        func_out.append(LineByLineTextDataset(hparams.dev_path))
    if test:
        func_out.append(LineByLineTextDataset(hparams.test_path))

    return tuple(func_out)
