""" Module defining Schedulers. """
from .linear_warmup import LinearWarmup
from .constant_lr import ConstantLR
from .warmup_constant import WarmupConstant


str2scheduler = {
    "linear_warmup": LinearWarmup,
    "constant": ConstantLR,
    "warmup_constant": WarmupConstant,
}


def build_scheduler(optimizer, hparams):
    """
    Function that builds a scheduler from the HyperOptArgumentParser
    :param hparams: HyperOptArgumentParser
    """
    return str2scheduler[hparams.scheduler].from_hparams(optimizer, hparams)


def add_scheduler_args(parser, scheduler: str):
    try:
        return str2scheduler[scheduler].add_scheduler_specific_args(parser)
    except KeyError:
        raise Exception(f"{scheduler} is not a valid scheduler option!")


__all__ = ["LinearWarmup", "ConstantLR", "WarmupConstant"]
