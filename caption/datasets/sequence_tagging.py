# -*- coding: utf-8 -*-
import pandas as pd

from test_tube import HyperOptArgumentParser


def collate_lists(text: list, tags: list) -> dict:
    """ For each line of the text and tags creates a dictionary. """
    collated_dataset = []
    for i in range(len(text)):
        collated_dataset.append(
            {"text": str(text[i]), "tags": str(tags[i]),}
        )
    return collated_dataset


def load_from_csv(hparams: HyperOptArgumentParser, train=True, val=True, test=True):
    """
    Dataset loader function used for loading:
        text and tags for training, development and testing.

    :param hparams: HyperOptArgumentParser obj containg the path to the data files.
    :param train: flag to return the train set.
    :param val: flag to return the validation set.
    :param test: flag to return the test set.
    
    Returns:
        - Training Dataset, Development Dataset, Testing Dataset
    """

    def load_dataset(path):
        df = pd.read_csv(path)
        text = list(df.text)
        tags = list(df.tags)
        assert len(text) == len(tags)
        return collate_lists(text, tags)

    func_out = []
    if train:
        func_out.append(load_dataset(hparams.train_path))
    if val:
        func_out.append(load_dataset(hparams.dev_path))
    if test:
        func_out.append(load_dataset(hparams.test_path))
    return tuple(func_out)


def sequence_tagging_dataset(
    hparams: HyperOptArgumentParser, train=True, val=True, test=True
):
    """
    Function that loads a tagging dataset for automatic capitalization and punctuation recovery.

    :param hparams: HyperOptArgumentParser obj containg the path to the data files.

    Returns:
        - Training Dataset, Development Dataset, Testing Dataset
    """
    if hparams.data_type == "csv":
        return load_from_csv(hparams, train, val, test)
    else:
        raise Exception(
            "Invalid configs data_type. Only csv and txt files are supported."
        )
