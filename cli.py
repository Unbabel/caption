# -*- coding: utf-8 -*-
r"""
Command Line Interface
=======================
   Commands:
   - train: for Training a new model.
   - interact: Model interactive mode where we can "talk" with a trained model.
   - test: Tests the model ability to rank candidate answers and generate text.
"""
import math
import multiprocessing
import os
from functools import partial

import click
import optuna
import pandas as pd
import torch
import yaml
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from models.data_module import MODEL_INPUTS, DataModule
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


@cli.command(name="test")
@click.option(
    "--model",
    type=click.Path(exists=True),
    required=True,
    help="Folder containing the config files and model checkpoints.",
)
@click.option(
    "--language",
    type=click.Choice(["en", "de", "fr", "it"], case_sensitive=False),
    help="Language pair",
    required=True,
)
@click.option(
    "--dev/--test",
    default=True,
    help="Flag that either runs devset or testset.",
    show_default=True,
)
@click.option(
    "--dataset",
    type=click.Path(exists=True),
    default="data/sepp_nlg_2021_data/",
    help="Path to the folder containing the dataset.",
)
@click.option(
    "--prediction_dir",
    type=click.Path(exists=True),
    default="data/pred/",
    help="Folder used to save predictions.",
)
@click.option(
    "--batch_size",
    default=32,
    help="Batch size used during inference.",
    type=int,
)
def predict(
    model: str,
    language: str,
    dev: bool,
    dataset: str,
    prediction_dir: str,
    batch_size: int,
) -> None:
    """Testing function where a trained model is tested in its ability to rank candidate
    answers and produce replies.
    """
    # Fix paths
    model = model if model.endswith("/") else model + "/"
    dataset = dataset if dataset.endswith("/") else dataset + "/"
    prediction_dir = (
        prediction_dir if prediction_dir.endswith("/") else prediction_dir + "/"
    )
    test_folder = dataset + f"{language}/" + ("dev/" if dev else "test/")
    output_folder = prediction_dir + f"{language}/" + ("dev/" if dev else "test/")

    click.secho(f"Loading model from folder: {model}", fg="yellow")
    model = PunctuationPredictor.from_experiment(model)
    data_module = DataModule(model.hparams, model.tokenizer)

    model.to("cuda")
    for file in tqdm(
        os.listdir(test_folder),
        desc="Processing {} data...".format(test_folder),
    ):
        if file.endswith(".tsv"):
            model_inputs = data_module.preprocess_file(test_folder + file, language)
            model_inputs = data_module.pad_dataset(
                model_inputs, padding=model.tokenizer.pad_token_id
            )
            file_data = []
            for input_name in MODEL_INPUTS:
                tensor = torch.tensor(model_inputs[input_name])
                file_data.append(tensor)

            dataloader = DataLoader(
                TensorDataset(*file_data),
                batch_size=batch_size,
                shuffle=False,
                num_workers=multiprocessing.cpu_count(),
            )
            binary_labels, punct_labels = [], []
            for batch in dataloader:
                bin_y_hat, punct_y_hat = model.predict(batch)
                binary_labels += bin_y_hat
                punct_labels += punct_y_hat
            words = [
                line.strip().split("\t")[0]
                for line in open(test_folder + file).readlines()
            ]
            output = pd.DataFrame(
                {"word": words, "binary": binary_labels, "punct": punct_labels}
            )
            output.to_csv(output_folder + file, sep="\t", header=None, index=False)


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
            "accumulate_grad_batches", 1, 8
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
        model_config.binary_loss = trial.suggest_int("binary_loss", 1, 2, step=1)
        model_config.punct_loss = trial.suggest_int("punct_loss", 1, 3, step=1)

        trainer = build_trainer(train_config.namespace())
        model = PunctuationPredictor(model_config.namespace())
        data = DataModule(model.hparams, model.tokenizer)

        try:
            trainer.fit(model, data)
        except RuntimeError:
            click.secho("CUDA OUT OF MEMORY, SKIPPING TRIAL", fg="red")
            return -1

        best_score = trainer.callbacks[0].best_score.item()
        return -1 if math.isnan(best_score) else best_score

    yaml_file = yaml.load(open(config).read(), Loader=yaml.FullLoader)

    # Build Trainer
    train_config = TrainerConfig(yaml_file)
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

    click.secho(f"Saving Optuna plots for this search to experiments/", fg="yellow")
    try:
        fig = optuna.visualization.plot_optimization_history(study)
        fig.write_html("experiments/optimization_history.html")

    except Exception as e:
        click.secho(f"Failed to create plot: {e}", fg="red")

    try:
        fig = optuna.visualization.plot_parallel_coordinate(
            study, params=list(trial.params.keys())
        )
        fig.write_html("experiments/parallel_coordinate.html")

    except Exception as e:
        click.secho(f"Failed to create plot: {e}", fg="red")

    try:
        fig = optuna.visualization.plot_param_importances(study)
        fig.write_html("experiments/param_importances.html")

    except Exception as e:
        click.secho(f"Failed to create plot: {e}", fg="red")



if __name__ == "__main__":
    cli()
