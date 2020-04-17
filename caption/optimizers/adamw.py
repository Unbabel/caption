# -*- coding: utf-8 -*-
from test_tube import HyperOptArgumentParser

from transformers import AdamW as HuggingFaceAdamW

from .optim_args import OptimArgs


class AdamW(HuggingFaceAdamW, OptimArgs):
    """
    Wrapper for the huggingface AdamW optimizer.
        https://huggingface.co/transformers/v2.1.1/main_classes/optimizer_schedules.html#adamw

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
        correct_bias: bool = True,
    ) -> None:
        super(AdamW, self).__init__(
            params=params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            correct_bias=correct_bias,
        )

    @classmethod
    def from_hparams(cls, params, hparams):
        """
        Initializes the scheduler from the parameters in the HyperOptArgumentParser
        """
        return AdamW(
            params,
            hparams.learning_rate,
            (hparams.b1, hparams.b2),
            hparams.eps,
            hparams.weight_decay,
            hparams.correct_bias,
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
        parser = super(AdamW, AdamW).add_optim_specific_args(parser)
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
            "--correct_bias",
            default=False,
            help="If this flag is on the correct_bias AdamW parameter is set to True.",
            action="store_true",
        )
        return parser
