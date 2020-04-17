# -*- coding: utf-8 -*-
from test_tube import HyperOptArgumentParser


class SchedulerArgs(object):
    """
    The Schedulers can Inheritance directly from the Pytorch lr schedulers
        but we want to extend the normal lr scheduler class behavior 
        with the add_scheduler_specific_args function.
    
    This class defines an Interface for adding Scheduler specific arguments
        to the Namespace and a method to build from the HyperOptArgumentParser
    """

    @classmethod
    def from_hparams(cls, optimizer, hparams):
        """
        Initializes the scheduler from the parameters in the HyperOptArgumentParser
        """
        raise NotImplementedError

    @staticmethod
    def add_scheduler_specific_args(
        parser: HyperOptArgumentParser,
    ) -> HyperOptArgumentParser:
        """
        Functions that parses scheduler specific arguments and adds 
            them to the Namespace
        :param parser: 
        """
        parser.add_argument(
            "--last_epoch", default=-1, type=int, help="Scheduler last epoch step"
        )
        return parser
