# -*- coding: utf-8 -*-
import json
import logging
import pdb

import numpy as np
import pandas as pd
import torch
from torch.utils.data import SequentialSampler, BatchSampler
from tqdm import tqdm

from caption.models.metrics import classification_report
from caption.models import str2model
from pytorch_lightning import Trainer
from test_tube import HyperOptArgumentParser
from torchnlp.encoders.text import stack_and_pad_tensors
from torchnlp.utils import collate_tensors, sampler_to_iterator

log = logging.getLogger("Shell")


def setup_testing(hparams: HyperOptArgumentParser):
    """
    Setup for the testing loop.
    :param hparams: HyperOptArgumentParser

    Returns:
        - CAPTION model
        - Test Set to be used.
    """
    tags_csv_file = "/".join(hparams.checkpoint.split("/")[:-1] + ["meta_tags.csv"])
    tags = pd.read_csv(tags_csv_file, header=None, index_col=0, squeeze=True).to_dict()
    model = str2model[tags["model"]].load_from_metrics(
        weights_path=hparams.checkpoint, tags_csv=tags_csv_file
    )
    log.info(model.hparams)

    # Make sure model is in prediction mode
    model.eval()
    model.freeze()
    log.info(f"{hparams.checkpoint} reloaded for testing.")
    log.info(f"Testing with {hparams.test_path} testset.")
    model._test_dataset = model._retrieve_dataset(hparams, train=False, val=False)[0]
    return model, model._test_dataset


def run_testing(hparams: HyperOptArgumentParser):
    model, testset = setup_testing(hparams)
    log.info("Testing model in CPU. This might take a while.")

    sampler = SequentialSampler(testset)
    iterator = sampler_to_iterator(testset, sampler)
    predictions = [
        model.predict(sample)
        for i, sample in tqdm(enumerate(iterator), total=len(testset))
    ]

    predicted_tags = model.label_encoder.batch_encode(
        [tag for pred in predictions for tag in pred["predicted_tags"].split()]
    )

    ground_truth_tags = torch.stack(
        [tag for pred in predictions for tag in pred["encoded_ground_truth_tags"]]
    )

    metrics = classification_report(
        np.array(predicted_tags),
        np.array(ground_truth_tags),
        padding=model.label_encoder.vocab_size,
        labels=model.label_encoder.token_to_index,
        ignore=model.default_slot_index,
    )
    log.info("-- Test metrics:\n{}".format(json.dumps(metrics, indent=1)))

    testing_output_file = "/".join(
        hparams.checkpoint.split("/")[:-1]
        + [hparams.checkpoint.split(".")[0].split("/")[-1] + "_predictions.json"]
    )

    predictions = [
        {k: v for k, v in d.items() if k != "encoded_ground_truth_tags"}
        for d in predictions
    ]

    with open(testing_output_file, "w") as outfile:
        json.dump({"results": metrics, "predictions": predictions}, outfile)
