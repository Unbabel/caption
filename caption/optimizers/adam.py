# -*- coding: utf-8 -*-
from test_tube import HyperOptArgumentParser

from torch.optim import Adam as RegularAdam

from .optim_args import OptimArgs


class Adam(RegularAdam, OptimArgs):
    """
    Wrapper for the pytorch Adam optimizer.
        https://pytorch.org/docs/stable/_modules/torch/optim/adam.html

    :param params: Model parameters
    :param lr: learning rate.
    :param betas: Adams beta parameters (b1, b2).
    :param eps: Adams epsilon. 
    :param weight_decay: Weight decay.
    :param correct_bias: Can be set to False to avoid correcting bias in Adam.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: list = [0.9, 0.999],
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        amsgrad: bool = False,
    ) -> None:
        super(Adam, self).__init__(
            params=params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
        )

    @classmethod
    def from_hparams(cls, params, hparams):
        """
        Initializes the scheduler from the parameters in the HyperOptArgumentParser
        """
        return Adam(
            params,
            hparams.learning_rate,
            (hparams.b1, hparams.b2),
            hparams.eps,
            hparams.weight_decay,
            hparams.amsgrad,
        )

    @staticmethod
    def add_optim_specific_args(
        parser: HyperOptArgumentParser,
    ) -> HyperOptArgumentParser:
        """
        Functions that parses Optimizer specific arguments and adds 
            them to the Namespace
        :param parser: 
        """
        parser = super(Adam, Adam).add_optim_specific_args(parser)
        parser.add_argument(
            "--b1", default=0.9, type=float, help="Adams beta parameters (b1, b2)."
        )
        parser.add_argument(
            "--b2", default=0.999, type=float, help="Adams beta parameters (b1, b2)."
        )
        parser.add_argument("--eps", default=1e-6, type=float, help="Adams epsilon.")
        parser.add_argument(
            "--weight_decay", default=0.0, type=float, help="Weight decay."
        )
        parser.add_argument(
            "--amsgrad",
            default=False,
            help="Whether to use the AMSGrad variant of this algorithm from the paper:\
                'On the Convergence of Adam and Beyond'",
            action="store_true",
        )
        return parser
