# -*- coding: utf-8 -*-
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from test_tube import HyperOptArgumentParser

from .scheduler_args import SchedulerArgs


class WarmupConstant(LambdaLR, SchedulerArgs):
    """
    Warmup Linear scheduler. 
    1) Linearly increases learning rate from 0 to 1 over warmup_steps
        training steps. 
    2) Keeps the learning rate constant afterwards.
    
    :param optimizer: torch.optim.Optimizer
    :param warmup_steps: Linearly increases learning rate from 0 to 1 over warmup_steps.
    :param last_epoch: 
    """

    def __init__(
        self, optimizer: Optimizer, warmup_steps: int, last_epoch: int = -1
    ) -> None:
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1.0, warmup_steps))
            return 1.0

        super(WarmupConstant, self).__init__(optimizer, lr_lambda, last_epoch)

    @classmethod
    def from_hparams(cls, optimizer, hparams):
        """
        Initializes the scheduler from the parameters in the HyperOptArgumentParser
        """
        return WarmupConstant(optimizer, hparams.warmup_steps, hparams.last_epoch)

    @staticmethod
    def add_scheduler_specific_args(
        parser: HyperOptArgumentParser,
    ) -> HyperOptArgumentParser:
        """
        Functions that parses Optimizer specific arguments and adds 
            them to the Namespace
        :param parent_parser: 
        """
        parser = super(WarmupConstant, WarmupConstant).add_scheduler_specific_args(
            parser
        )
        parser.add_argument(
            "--warmup_steps",
            type=int,
            default=1,
            help="Linearly increases learning rate from 0*learning_rate to 1*learning_rate over warmup_steps.",
        )
        return parser
