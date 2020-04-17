""" Module defining Optimizers. """
from .adamw import AdamW
from .radam import RAdam
from .optim_args import OptimArgs
from .adam import Adam
from .adamax import Adamax

str2optimizer = {"AdamW": AdamW, "RAdam": RAdam, "Adam": Adam, "Adamax": Adamax}


def build_optimizer(params, hparams):
    """
    Function that builds an optimizer from the HyperOptArgumentParser
    :param params: Model parameters
    :param hparams: HyperOptArgumentParser
    """
    return str2optimizer[hparams.optimizer].from_hparams(params, hparams)


def add_optimizer_args(parser, optimizer: str):
    try:
        return str2optimizer[optimizer].add_optim_specific_args(parser)
    except KeyError:
        raise Exception(f"{optimizer} is not a valid optimizer option!")


__all__ = ["AdamW", "RAdam", "Adam", "Adamax"]
