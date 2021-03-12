# -*- coding: utf-8 -*-
r"""
Command Line Interface
=======================
   Commands:
   - train: for Training a new model.
   - interact: Model interactive mode where we can "talk" with a trained model.
   - test: Tests the model ability to rank candidate answers and generate text.
"""
import json
import logging
import os

import click
import optuna
import torch
import yaml
from pytorch_lightning import seed_everything
from tqdm import tqdm

from models.data_module import DataModule
from models.punct_predictor import PunctuationPredictor
from trainer import TrainerConfig, build_trainer


@click.group()
def cli():
    pass


@cli.command(name="train")
@click.option(
    "--config",
    "-f",
    type=click.Path(exists=True),
    required=True,
    help="Path to the configure YAML file",
)
def train(config: str) -> None:
    yaml_file = yaml.load(open(config).read(), Loader=yaml.FullLoader)
    # Build Trainer
    train_configs = TrainerConfig(yaml_file)
    seed_everything(train_configs.seed)
    trainer = build_trainer(train_configs.namespace())

    # Print Trainer parameters into terminal
    result = "Hyperparameters:\n"
    for k, v in train_configs.namespace().__dict__.items():
        result += "{0:30}| {1}\n".format(k, v)
    click.secho(f"{result}", fg="blue", nl=False)

    model_config = PunctuationPredictor.ModelConfig(yaml_file)
    # Print Model parameters into terminal
    for k, v in model_config.namespace().__dict__.items():
        result += "{0:30}| {1}\n".format(k, v)
    click.secho(f"{result}", fg="cyan")

    model = PunctuationPredictor(model_config.namespace())
    data = DataModule(model.hparams, model.tokenizer)
    trainer.fit(model, data)


@cli.command(name="search")
@click.option(
    "--config",
    "-f",
    type=click.Path(exists=True),
    required=True,
    help="Path to the configure YAML file",
)
@click.option(
    "--n_trials",
    type=int,
    default=50,
    help="Number of search trials",
)
def search(config: str, n_trials: int) -> None:
    def objective(trial, train_config, model_config):
        train_config.accumulate_grad_batches = trial.suggest_int(
            "accumulate_grad_batches", 1, 32
        )
        model_config.nr_frozen_epochs = trial.suggest_uniform(
            "nr_frozen_epochs", 0.02, 0.4
        )
        model_config.dropout = trial.suggest_uniform("dropout", 0.1, 0.5)
        model_config.layerwise_decay = trial.suggest_uniform(
            "layerwise_decay", 0.75, 1.0
        )
        model_config.encoder_learning_rate = trial.suggest_loguniform(
            "encoder_learning_rate", 1e-5, 1e-4
        )
        model_config.learning_rate = trial.suggest_loguniform(
            "learning_rate", 1e-5, 3e-4
        )
        model_config.binary_loss.suggest_categorical("binary_loss", [1, 2])
        model_config.punct_loss.suggest_categorical("punct_loss", [1, 2, 3])

        trainer = build_trainer(train_configs.namespace())
        model = PunctuationPredictor(model_config.namespace())

        try:
            trainer.fit(model)
        except RuntimeError:
            click.secho("CUDA OUT OF MEMORY, SKIPPING TRIAL", fg="red")
            return -1

        best_score = trainer.callbacks[0].best_score.item()
        return -1 if math.isnan(best_score) else best_score

    yaml_file = yaml.load(open(config).read(), Loader=yaml.FullLoader)

    # Build Trainer
    train_configs = TrainerConfig(yaml_file)
    model_config = PunctuationPredictor.ModelConfig(yaml_file)

    pruner = optuna.pruners.MedianPruner()
    study = optuna.create_study(direction="maximize", pruner=pruner)

    try:
        study.optimize(
            partial(objective, train_config=train_config, model_config=model_config),
            n_trials=n_trials,
        )

    except KeyboardInterrupt:
        click.secho("Early stopping search caused by ctrl-C", fg="red")

    except Exception as e:
        click.secho(
            f"Error occured during search: {e}; current best params are {study.best_params}",
            fg="red",
        )

    try:
        click.secho(
            "Number of finished trials: {}".format(len(study.trials)), fg="yellow"
        )
        click.secho("Best trial:", fg="yellow")
        trial = study.best_trial
        click.secho("  Value: {}".format(trial.value), fg="yellow")
        click.secho("  Params: ", fg="yellow")
        for key, value in trial.params.items():
            click.secho("    {}: {}".format(key, value), fg="blue")

    except Exception as e:
        click.secho(f"Logging at end of search failed: {e}", fg="red")

    click.secho(
        f"Saving Optuna plots for this search to experiments/optuna/", fg="yellow"
    )
    try:
        fig = optuna.visualization.plot_optimization_history(study)
        fig.write_html("experiments/optuna/optimization_history.html")

    except Exception as e:
        click.secho(f"Failed to create plot: {e}", fg="red")

    try:
        fig = optuna.visualization.plot_parallel_coordinate(
            study, params=list(trial.params.keys())
        )
        fig.write_html("experiments/optuna/parallel_coordinate.html")

    except Exception as e:
        click.secho(f"Failed to create plot: {e}", fg="red")

    try:
        fig = optuna.visualization.plot_param_importances(study)
        fig.write_html("experiments/optuna/param_importances.html")

    except Exception as e:
        click.secho(f"Failed to create plot: {e}", fg="red")


@cli.command(name="test")
@click.option(
    "--experiment",
    type=click.Path(exists=True),
    required=True,
    help="Path to the experiment folder containing the checkpoint we want to interact with.",
)
@click.option(
    "--test_set",
    type=click.Path(exists=True),
    required=True,
    help="Path to the json file containing the testset.",
)
@click.option(
    "--cuda/--cpu",
    default=True,
    help="Flag that either runs inference on cuda or in cpu.",
    show_default=True,
)
@click.option(
    "--seed",
    default=12,
    help="Seed value used during inference. This influences results only when using sampling.",
    type=int,
)
def predict(
    experiment: str,
    test_set: str,
    cuda: bool,
    seed: int,
) -> None:
    """Testing function where a trained model is tested in its ability to rank candidate
    answers and produce replies.
    """
    pass


if __name__ == "__main__":
    cli()
