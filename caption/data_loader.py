# -*- coding: utf-8 -*-
import pandas as pd

from test_tube import HyperOptArgumentParser
from torchnlp.datasets.dataset import Dataset


def add_data_args(parser: HyperOptArgumentParser) -> HyperOptArgumentParser:
    """
    Functions that parses dataset specific arguments/hyperparameters.
    :param hparams: HyperOptArgumentParser obj.

    Returns:
        - updated parser
    """
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
        help="How many subprocesses to use for data loading. 0 means that \
            the data will be loaded in the main process.",
    )
    return parser


def file_to_list(filename: str) -> list:
    """ Reads a file and returns a list with the cotent of each line. """
    with open(filename, "r") as filehandler:
        content = [line.strip() for line in filehandler.readlines()]
    return content


def collate_lists(source: list, target: list, tags: list) -> dict:
    """ For each line of the source, target and tags tags creates a dictionary. """
    collated_dataset = []
    for i in range(len(source)):
        collated_dataset.append(
            {"source": str(source[i]), "target": str(target[i]), "tags": str(tags[i]),}
        )
    return collated_dataset


def load_from_csv(hparams: HyperOptArgumentParser, train=True, val=True, test=True):
    """
    This dataset loader is used for loading:
        Source text, target text and tags for training, development and testing.

    :param hparams: HyperOptArgumentParser obj containg the path to the data files.
    :param train: flag to return the train set.
    :param val: flag to return the validation set.
    :param test: flag to return the test set.
    
    Returns:
        - Training Dataset, Development Dataset, Testing Dataset
    """

    def load_dataset(path):
        df = pd.read_csv(path)
        source = list(df.source)
        target = list(df.target)
        tags = list(df.tags)
        assert len(source) == len(target) == len(tags)
        return Dataset(collate_lists(source, target, tags))

    func_out = []
    if train:
        func_out.append(load_dataset(hparams.train_path))
    if val:
        func_out.append(load_dataset(hparams.dev_path))
    if test:
        func_out.append(load_dataset(hparams.test_path))
    return tuple(func_out)


def text_recovery_dataset(
    hparams: HyperOptArgumentParser, train=True, val=True, test=True
):
    """
    This dataset loader is used for loading:
        Source text, arget text and tags for training, development and testing.

    This task consists in automatic capitalization and punctuation recovery.

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
