# -*- coding: utf-8 -*-
import os
import random
import shutil
from datetime import datetime

import numpy as np
import torch
import yaml

from pytorch_lightning.logging import TestTubeLogger
from test_tube import HyperOptArgumentParser
from test_tube.argparse_hopt import TTNamespace


def load_yaml_args(parser: HyperOptArgumentParser, log):
    """ Function that load the args defined in a YAML file and replaces the values
        parsed by the HyperOptArgumentParser """
    old_args = vars(parser.parse_args())
    configs = old_args.get("config")
    if configs:
        yaml_file = yaml.load(open(configs).read(), Loader=yaml.FullLoader)
        for key, value in yaml_file.items():
            if key in old_args:
                old_args[key] = value
            else:
                raise Exception(
                    "{} argument defined in {} is not valid!".format(key, configs)
                )
    else:
        log.warning(
            "We recommend the usage of YAML files to keep track \
            of the hyperparameter during testing and training."
        )
    return TTNamespace(**old_args)


def get_main_args_from_yaml(args):
    """ Function for loading the __main__ arguments directly from the YAML """
    if not args.config:
        raise Exception("You must pass a YAML file if not using the command line.")
    try:
        yaml_file = yaml.load(open(args.config).read(), Loader=yaml.FullLoader)
        return yaml_file["optimizer"], yaml_file["scheduler"], yaml_file["model"]
    except KeyError as e:
        raise Exception("YAML file is missing the {} parameter.".format(e.args[0]))


def setup_testube_logger():
    """ Function that sets the TestTubeLogger to be used. """
    try:
        job_id = os.environ["SLURM_JOB_ID"]
    except Exception:
        job_id = None

    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y--%H-%M-%S")
    return TestTubeLogger(
        save_dir="experiments/",
        version=job_id if job_id else dt_string,
        name="lightning_logs",
    )
