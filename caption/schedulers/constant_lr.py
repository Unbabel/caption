# -*- coding: utf-8 -*-
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from test_tube import HyperOptArgumentParser

from .scheduler_args import SchedulerArgs


class ConstantLR(LambdaLR, SchedulerArgs):
    """
    Constant learning rate schedule
    
    Wrapper for the huggingface Constant LR Scheduler.
        https://huggingface.co/transformers/v2.1.1/main_classes/optimizer_schedules.html

    :param optimizer: torch.optim.Optimizer
    :param last_epoch:
    """

    def __init__(self, optimizer: Optimizer, last_epoch: int = -1) -> None:
        super(ConstantLR, self).__init__(optimizer, lambda _: 1, last_epoch)

    @classmethod
    def from_hparams(cls, optimizer, hparams):
        """
        Initializes the scheduler from the parameters in the HyperOptArgumentParser
        """
        return ConstantLR(optimizer, hparams.last_epoch)

    @staticmethod
    def add_scheduler_specific_args(
        parser: HyperOptArgumentParser,
    ) -> HyperOptArgumentParser:
        """
        Functions that parses Optimizer specific arguments and adds 
            them to the Namespace
        :param parser: 
        """
        return super(ConstantLR, ConstantLR).add_scheduler_specific_args(parser)
