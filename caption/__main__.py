# -*- coding: utf-8 -*-
import logging
from data_loader import add_data_args

from models import add_model_args, build_model
from optimizers import add_optimizer_args
from schedulers import add_scheduler_args
from test_tube import HyperOptArgumentParser
from testing import run_testing, setup_testing
from torchnlp.random import set_seed
from training import add_trainer_specific_args, setup_training
from utils import get_main_args_from_yaml, load_yaml_args

log = logging.getLogger("Shell")
logging.basicConfig(level=logging.INFO)


def run_training_pipeline(parser):
    parser.add_argument(
        "-f", "--config", default=False, type=str, help="Path to a YAML config file."
    )
    parser.add_argument(
        "--optimizer",
        default=False,
        type=str,
        help="Optimizer to be used during training.",
    )
    parser.add_argument(
        "--scheduler",
        default=False,
        type=str,
        help="LR scheduler to be used during training.",
    )
    parser.add_argument(
        "--model",
        default=False,
        type=str,
        help="The estimator architecture we we wish to use.",
    )
    args, _ = parser.parse_known_args()

    if not args.optimizer and not args.scheduler and not args.model:
        optimizer, scheduler, model = get_main_args_from_yaml(args)
    else:
        optimizer = args.optimizer
        scheduler = args.scheduler
        model = args.model

    parser = add_optimizer_args(parser, optimizer)
    parser = add_scheduler_args(parser, scheduler)
    parser = add_model_args(parser, model)
    parser = add_trainer_specific_args(parser)
    hparams = load_yaml_args(parser=parser, log=log)

    set_seed(hparams.seed)
    model = build_model(hparams)
    trainer = setup_training(hparams)

    if hparams.load_weights:
        model.load_weights(hparams.load_weights)

    log.info(f"{model.__class__.__name__} train starting:")
    trainer.fit(model)


def run_testing_pipeline(parser):
    parser = add_data_args(parser)
    parser.add_argument(
        "--checkpoint", default=None, help="Checkpoint file path.",
    )
    hparams = parser.parse_args()
    run_testing(hparams)


if __name__ == "__main__":
    parser = HyperOptArgumentParser(
        strategy="random_search", description="CAPTION project", add_help=True
    )
    parser.add_argument(
        "pipeline", choices=["train", "test"], help="train a model or test.",
    )
    args, _ = parser.parse_known_args()
    if args.pipeline == "test":
        run_testing_pipeline(parser)
    else:
        run_training_pipeline(parser)
