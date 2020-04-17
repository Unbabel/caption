# -*- coding: utf-8 -*-
import sys

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from test_tube import HyperOptArgumentParser

from .scheduler_args import SchedulerArgs


class LinearWarmup(LambdaLR, SchedulerArgs):
    """
    Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.
    
    :param optimizer: torch.optim.Optimizer
    :param warmup_steps: Linearly increases learning rate from 0 to 1*learning_rate over warmup_steps.
    :param num_training_steps: Linearly decreases learning rate from 1*learning_rate to 0. over remaining 
        t_total - warmup_steps steps.
    :param last_epoch: 
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        num_training_steps: int,
        last_epoch: int = -1,
    ) -> None:
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return max(
                0.0,
                float(num_training_steps - current_step)
                / float(max(1, num_training_steps - warmup_steps)),
            )

        super(LinearWarmup, self).__init__(optimizer, lr_lambda, last_epoch)

    @classmethod
    def from_hparams(cls, optimizer, hparams):
        """
        Initializes the scheduler from the parameters in the HyperOptArgumentParser
        """
        return LinearWarmup(
            optimizer,
            hparams.warmup_steps,
            hparams.num_training_steps,
            hparams.last_epoch,
        )

    @staticmethod
    def add_scheduler_specific_args(
        parser: HyperOptArgumentParser,
    ) -> HyperOptArgumentParser:
        """
        Functions that parses Optimizer specific arguments and adds 
            them to the Namespace
        :param parent_parser: 
        """
        parser = super(LinearWarmup, LinearWarmup).add_scheduler_specific_args(parser)
        parser.add_argument(
            "--warmup_steps",
            type=int,
            default=1,
            help="Linearly increases learning rate from 0 to 1 over warmup_steps.",
        )
        parser.add_argument(
            "--num_training_steps",
            type=int,
            default=sys.maxsize,
            help="Linearly decreases learning rate from 1*learning_rate to 0*learning_rate over \
                remaining t_total - warmup_steps steps.",
        )
        return parser
