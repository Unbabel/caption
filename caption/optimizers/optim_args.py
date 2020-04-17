# -*- coding: utf-8 -*-
from test_tube import HyperOptArgumentParser


class OptimArgs(object):
    """
    Optimizer classes can Inheritance directly from the Pytorch Optimizer
        class but we want to extend the normal Optimizer class behavior 
        with the add_optim_specific_args function.

    This class defines an Interface for adding Optimizer specific arguments
        to the Namespace
    """

    @classmethod
    def from_hparams(cls, params, hparams):
        """
        Initializes the optimizer from the parameters in the HyperOptArgumentParser
        """
        raise NotImplementedError

    @staticmethod
    def add_optim_specific_args(
        parser: HyperOptArgumentParser,
    ) -> HyperOptArgumentParser:
        """
        Functions that parses Optimizer specific arguments and adds 
            them to the Namespace
        :param parser: 
        """
        parser.opt_list(
            "--learning_rate",
            default=5e-5,
            type=float,
            tunable=True,
            options=[1e-05, 3e-05, 5e-05, 8e-05, 1e-04],
            help="Optimizer learning rate.",
        )
        return parser
