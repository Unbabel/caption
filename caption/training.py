# -*- coding: utf-8 -*-
import logging
import os

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from test_tube import HyperOptArgumentParser

from caption.utils import setup_testube_logger


log = logging.getLogger("Shell")


def setup_training(hparams: HyperOptArgumentParser) -> tuple:
    """
    Setup for the training loop.
    :param hparams: HyperOptArgumentParser

    Returns:
        - pytorch_lightning Trainer
    """
    if hparams.verbose:
        log.info(hparams)

    if hparams.early_stopping:
        # Enable Early stopping
        early_stop_callback = EarlyStopping(
            monitor=hparams.monitor,
            min_delta=hparams.min_delta,
            patience=hparams.patience,
            verbose=hparams.verbose,
            mode=hparams.metric_mode,
        )
    else:
        early_stop_callback = None

    # configure trainer
    if hparams.epochs > 0.0:
        hparams.min_epochs = hparams.epochs
        hparams.max_epochs = hparams.epochs

    trainer = Trainer(
        logger=setup_testube_logger(),
        checkpoint_callback=True,
        early_stop_callback=early_stop_callback,
        default_save_path="experiments/",
        gradient_clip_val=hparams.gradient_clip_val,
        gpus=hparams.gpus,
        show_progress_bar=False,
        overfit_pct=hparams.overfit_pct,
        check_val_every_n_epoch=hparams.check_val_every_n_epoch,
        fast_dev_run=False,
        accumulate_grad_batches=hparams.accumulate_grad_batches,
        max_epochs=hparams.max_epochs,
        min_epochs=hparams.min_epochs,
        train_percent_check=hparams.train_percent_check,
        val_percent_check=hparams.val_percent_check,
        val_check_interval=hparams.val_check_interval,
        log_save_interval=hparams.log_save_interval,
        row_log_interval=hparams.row_log_interval,
        distributed_backend=hparams.distributed_backend,
        precision=hparams.precision,
        weights_summary=hparams.weights_summary,
        resume_from_checkpoint=hparams.resume_from_checkpoint,
        profiler=hparams.profiler,
        log_gpu_memory="all",
    )

    ckpt_path = os.path.join(
        trainer.default_save_path,
        trainer.logger.name,
        f"version_{trainer.logger.version}",
        "checkpoints",
    )

    # initialize Model Checkpoint Saver
    checkpoint_callback = ModelCheckpoint(
        filepath=ckpt_path,
        save_top_k=hparams.save_top_k,
        verbose=hparams.verbose,
        monitor=hparams.monitor,
        save_weights_only=hparams.save_weights_only,
        period=hparams.period,
        mode=hparams.metric_mode,
    )
    trainer.checkpoint_callback = checkpoint_callback
    return trainer


def add_trainer_specific_args(parser: HyperOptArgumentParser) -> HyperOptArgumentParser:
    parser.add_argument("--seed", default=3, type=int, help="Training seed.")
    parser.add_argument(
        "--batch_size", default=32, type=int, help="Batch size to be used."
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        default=None,
        type=str,
        help=(
            "To resume training from a specific checkpoint pass in the path here."
            "(e.g. 'some/path/to/my_checkpoint.ckpt')"
        ),
    )
    parser.add_argument(
        "--load_weights",
        default=None,
        type=str,
        help=(
            "Loads the model weights from a given checkpoint.  "
            "This flag differs from resume_from_checkpoint beacuse it only loads the"
            "weights that match between the checkpoint model and the model we want to train. "
            "It does not resume the entire training session (model/optimizer/scheduler etc..)."
        ),
    )
    parser.add_argument(
        "--save_top_k",
        default=1,
        type=int,
        help="The best k models according to the quantity monitored will be saved.",
    )
    parser.add_argument(
        "--monitor", default="pearson", type=str, help="Quantity to monitor."
    )
    parser.add_argument(
        "--metric_mode",
        default="max",
        type=str,
        help=(
            "One of {min, max}. If `--save_best_only`, the decision to "
            "overwrite the current checkpoint is based on either the maximization "
            "or the minimization of the monitored quantity."
        ),
        choices=["min", "max"],
    )
    parser.add_argument(
        "--period",
        default=1,
        type=int,
        help="Interval (number of epochs) between checkpoints.",
    )
    parser.add_argument(
        "--save_weights_only",
        default=False,
        help="If True, then only the model's weights will be saved.",
        action="store_true",
    )
    parser.add_argument(
        "--verbose",
        default=False,
        help="Verbosity mode, True or False",
        action="store_true",
    )
    # Early Stopping
    parser.add_argument(
        "--early_stopping",
        default=False,
        help="If set to True Early Stopping is enabled.",
        action="store_true",
    )
    parser.add_argument(
        "--patience",
        default=3,
        type=int,
        help=(
            "Number of epochs with no improvement "
            "after which training will be stopped."
        ),
    )
    parser.add_argument(
        "--min_delta",
        default=0.0,
        type=float,
        help="Minimum change in the monitored quantity.",
    )
    # pytorch-lightning trainer class specific args
    parser.add_argument(
        "--epochs",
        default=-1,
        type=int,
        help=(
            "Number of epochs to run. By default the number of epochs "
            "is controlled but the min_nb_epochs and max_nb_epochs parameters."
        ),
    )
    parser.add_argument(
        "--min_epochs",
        default=1,
        type=int,
        help="Limits training to a minimum number of epochs",
    )
    parser.add_argument(
        "--max_epochs",
        default=3,
        type=int,
        help="Limits training to a max number number of epochs",
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        default=1,
        type=int,
        help=(
            "Accumulated gradients runs K small batches of size N before "
            "doing a backwards pass. The effect is a large effective batch "
            "size of size KxN."
        ),
    )
    parser.add_argument(
        "--gradient_clip_val",
        default=1.0,
        type=float,
        help="Max norm of the gradients.",
    )
    parser.add_argument(
        "--gpus",
        default=1,
        type=int,
        help="ie: 2 gpus OR -1 to use all available gpus",
    )
    parser.add_argument(
        "--distributed_backend",
        default="dp",
        type=str,
        help="Options: 'dp' (lightning ddp and ddp2  not working!)",
        choices=["dp"],
    )
    parser.add_argument(
        "--precision",
        default=32,
        type=int,
        help="Full precision (32), half precision (16).",
        choices=[16, 32],
    )
    parser.add_argument(
        "--log_save_interval",
        default=100,
        type=int,
        help="Writes logs to disk this often",
    )
    parser.add_argument(
        "--row_log_interval",
        default=10,
        type=int,
        help="How often to add logging rows (does not write to disk)",
    )
    parser.add_argument(
        "--check_val_every_n_epoch",
        default=1,
        type=int,
        help="Check val every n train epochs.",
    )
    parser.add_argument(
        "--train_percent_check",
        default=1.0,
        type=float,
        help=(
            "If you don't want to use the entire training set, "
            "set how much of the train set you want to use with this flag."
        ),
    )
    parser.add_argument(
        "--val_percent_check",
        default=1.0,
        type=float,
        help=(
            "If you don't want to use the entire dev set, set how much of the dev "
            "set you want to use with this flag."
        ),
    )
    parser.add_argument(
        "--val_check_interval",
        default=1.0,
        type=float,
        help=(
            "For large datasets it's often desirable to check validation multiple "
            "times within a training loop. Pass in a float to check that often "
            "within 1 training epoch."
        ),
    )
    parser.add_argument(
        "--train_val_percent_check",
        default=0.01,
        type=float,
        help=(
            "In the end of each epoch a subset of the training data will be selected "
            "to measure performance against training. Pass a float to set how much of "
            "the training data you want to use"
        ),
    )
    # Debugging
    parser.add_argument(
        "--overfit_pct",
        default=0.0,
        type=float,
        help=(
            "A useful debugging trick is to make your model overfit a tiny fraction "
            "of the data. Default: don't overfit (ie: normal training)"
        ),
    )
    parser.add_argument(
        "--weights_summary",
        default="full",
        type=str,
        help="Prints a summary of the weights when training begins.",
        choices=["full", "top"],
    )
    parser.add_argument(
        "--profiler",
        default=False,
        help="If you only wish to profile the standard actions during training.",
        action="store_true",
    )
    parser.add_argument(
        "--log_gpu_memory",
        default=None,
        type=str,
        help="Logs (to a logger) the GPU usage for each GPU on the master machine.",
        choices=["min_max", "full"],
    )
    return parser
