r""" 
DataModule
==========
    The DataModule encapsulates all the steps needed to process data.
"""
import hashlib
import multiprocessing
import os
from argparse import Namespace
from collections import defaultdict
from os import path

import click
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchnlp.download import download_file_maybe_extract
from tqdm import tqdm
import pandas as pd

SEPP_NLG_URL = "https://unbabel-experimental-data-sets.s3-eu-west-1.amazonaws.com/video-pt2020/sepp_nlg_2021_train_dev_data.zip"


PUNCTUATION_LABEL_ENCODER = {
    '0': 0,
    'P': 1,
    'C': 2,
    'Q': 3
}
PUNCTUATION_GROUP = {
    '0': '0',
    '.': 'P',
    '!': 'P',
    ';': 'P',
    ',': 'C',
    ':': 'C',
    '-': 'C',
    '?': 'Q'
}
CAPITALIZATION_LABEL_ENCODER = {
    "L": 0,
    "U": 1,
    "T": 2
}
MODEL_INPUTS = [
    "input_ids",
    "word_pointer",
    "attention_mask",
    # "token_type_ids",
    "cap_label",
    "punct_label",
]

ATTR_TO_SPECIAL_TOKEN = {
    "bos_token": "<bos>",
    "eos_token": "<eos>",
    "pad_token": "<pad>",
    # "additional_special_tokens": ["<en>", "<de>", "<it>", "<fr>"],
}


class DataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule.
    :param hparams: Namespace with data specific arguments.
    :param tokenizer: Model Tokenizer.
    """

    def __init__(self, hparams: Namespace, tokenizer, multiple_files_process: bool):
        super().__init__()
        self.hparams = hparams
        self.tokenizer = tokenizer
        self.multiple_files_process = multiple_files_process
        self.language_pairs = {
            "en": 0,
            "ar": 1,
            "bg": 2,
            "cs": 3,
            "da": 4,
            "de": 5,
            "el": 6,
            "es": 7,
            "fi": 8,
            "fr": 9,
            "hu": 10,
            "id": 11,
            "it": 12,
            "ja": 13,
            "ko": 14,
            "nl": 15,
            "no": 16,
            "pl": 17,
            "pt": 18,
            "ro": 19,
            "ru": 20,
            "sv": 21,
            "th": 22,
            "vi": 23,
            "zh": 24
        }
    

    def update_model_inputs(self, model_inputs, subwords, punctuation, capitalisation, testing=False):
        if (
            len(model_inputs["input_ids"][-1]) + len(subwords)
        ) < self.tokenizer.model_max_length - 1:
            model_inputs["word_pointer"][-1].append(
                len(model_inputs["input_ids"][-1])
            )
            model_inputs["input_ids"][-1] += subwords
            if not testing:
                if capitalisation in CAPITALIZATION_LABEL_ENCODER.keys():
                    model_inputs["cap_label"][-1].append(CAPITALIZATION_LABEL_ENCODER[capitalisation])
                else:
                    model_inputs["cap_label"][-1].append(0)
                if punctuation in PUNCTUATION_GROUP.keys():
                    model_inputs["punct_label"][-1].append(PUNCTUATION_LABEL_ENCODER[PUNCTUATION_GROUP[punctuation]])
                else:
                    model_inputs["punct_label"][-1].append(0)

        else:
            model_inputs["input_ids"][-1].append(self.tokenizer.eos_token_id)
            model_inputs["input_ids"].append(
                [
                    self.tokenizer.bos_token_id,
                ]
            )
            model_inputs["word_pointer"].append([])
            if not testing:
                model_inputs["cap_label"].append([])
                model_inputs["punct_label"].append([])

            model_inputs["word_pointer"][-1].append(
                len(model_inputs["input_ids"][-1])
            )
            model_inputs["input_ids"][-1] += subwords
            if not testing:
                if capitalisation in CAPITALIZATION_LABEL_ENCODER.keys():
                    model_inputs["cap_label"][-1].append(CAPITALIZATION_LABEL_ENCODER[capitalisation])
                else:
                    model_inputs["cap_label"][-1].append(0)
                if punctuation in PUNCTUATION_GROUP.keys():
                    model_inputs["punct_label"][-1].append(PUNCTUATION_LABEL_ENCODER[PUNCTUATION_GROUP[punctuation]])
                else:
                    model_inputs["punct_label"][-1].append(0)
        return model_inputs

    def load_data_from_csv(self, path: str, testing: bool, language = None):
        """
        If the data is contained in a single .csv file for each language, this method should be used.
        This dataset loader is used for loading:
            Source text and tags for training, development and testing.
        
        :param path: path to the data's root folder.
        :param testing: flag to return the test set.
        """
        def load_dataset(path):
            df = pd.read_csv(path)
            df['words'] = df['words'].apply(eval)
            df['capitalisation'] = df['capitalisation'].apply(eval)
            df['punctuation'] = df['punctuation'].apply(eval)
            words = list(df.words)
            capitalisation = list(df.capitalisation)
            punctuation = list(df.punctuation)
            return words, capitalisation, punctuation

        if testing:
            datasets = {"test": defaultdict(list)}
        else:
            datasets = {"train": defaultdict(list), "dev": defaultdict(list)}

        languages = []
        if language is None:
            languages = self.language_pairs.keys()
        else:
            languages.append(language)

        for lp in languages:
            for dataset_name in datasets.keys():
                click.secho(f"Preparing {dataset_name} data:", fg="yellow")
                for file in tqdm(
                    os.listdir(f"{path}/{lp}/{dataset_name}/"),
                    desc=f"Preparing {lp}/{dataset_name} data...",
                ):
                    if file.endswith(".csv"):
                        sentences_list, capitalisation, punctuation = load_dataset(f"{path}/{lp}/{dataset_name}/{file}")
                        for sentence_index, sentence in enumerate(sentences_list):

                            model_inputs = {
                                "input_ids": [],
                                "attention_mask": [],
                                "word_pointer": [],
                                "cap_label": [],
                                "punct_label": []
                            }
                            model_inputs["input_ids"].append(
                                [
                                    self.tokenizer.bos_token_id,
                                ]
                            )
                            model_inputs["word_pointer"].append([])
                            model_inputs["cap_label"].append([])
                            model_inputs["punct_label"].append([])

                            for word_index, word in enumerate(sentence):
                                subwords = self.tokenizer(word, add_special_tokens=False)["input_ids"]
                                model_inputs = self.update_model_inputs(model_inputs, subwords, punctuation[sentence_index][word_index], capitalisation[sentence_index][word_index])

                            if len(model_inputs["input_ids"][-1]) != 1:
                                model_inputs["input_ids"][-1].append(self.tokenizer.eos_token_id)

                            for _input in model_inputs["input_ids"]:
                                model_inputs["attention_mask"].append([1 for _ in _input])
                            
                            for model_input in model_inputs.keys():
                                if model_input in datasets[dataset_name]:
                                    datasets[dataset_name][model_input] += model_inputs[
                                        model_input
                                    ]
                                else:
                                    datasets[dataset_name][model_input] = model_inputs[
                                        model_input
                                    ]
        return datasets



    def preprocess_file(self, filename, language, testing=False):
        model_inputs = {
            "input_ids": [],
            "attention_mask": [],
            "word_pointer": [],
        }
        if not testing:
            model_inputs["cap_label"] = []
            model_inputs["punct_label"] = []

        model_inputs["input_ids"].append(
            [
                self.tokenizer.bos_token_id,
            ]
        )
        model_inputs["word_pointer"].append([])
        if not testing:
            model_inputs["cap_label"].append([])
            model_inputs["punct_label"].append([])

        for i, line in enumerate(open(filename).readlines()):
            line = line.strip().split("\t")
            word = line[0] if i == 0 else " " + line[0]
            subwords = self.tokenizer(word, add_special_tokens=False)["input_ids"]
            model_inputs = self.update_model_inputs(model_inputs, subwords, line[2], line[1], testing)

        if len(model_inputs["input_ids"][-1]) != 1:
            model_inputs["input_ids"][-1].append(self.tokenizer.eos_token_id)

        for _input in model_inputs["input_ids"]:
            model_inputs["attention_mask"].append([1 for _ in _input])

        return model_inputs
    
    def process_multiple_separate_files(self, dataset_path: str):
        """
            If  the data is split in multiple .tsv files, this method should be used.
        """
        datasets = {"train": defaultdict(list), "dev": defaultdict(list)}
        for lp in self.language_pairs.keys():
            for dataset_name in datasets.keys():
                click.secho(f"Preparing {dataset_name} data:", fg="yellow")
                for file in tqdm(
                    os.listdir(dataset_path + lp + "/" + dataset_name + "/"),
                    desc=f"Preparing {lp} data...",
                ):
                    if file.endswith(".tsv"):
                        data = self.preprocess_file(
                            dataset_path + lp + "/" + dataset_name + "/" + file, lp
                        )
                        for model_input in data.keys():
                            if model_input in datasets[dataset_name]:
                                datasets[dataset_name][model_input] += data[
                                    model_input
                                ]
                            else:
                                datasets[dataset_name][model_input] = data[
                                    model_input
                                ]
        return datasets

    def prepare_data(self):
        if self.multiple_files_process:
            if not path.isdir("../data_pre_processing/comet-data/processed_data/"):
                click.secho(f"../data_pre_processing/comet-data/processed_data/ not found.")

            dataset_path = "../data_pre_processing/comet-data/processed_data/"
        else:
            dataset_path = "../data_pre_processing/comet-data/single_files_aggregated/"

        dataset_hash = (
            int(hashlib.sha256(dataset_path.encode("utf-8")).hexdigest(), 16) % 10 ** 8
        )
        # To avoid using cache for different models
        # split(/) for microsoft/DialoGPT-small
        pretrained_model = (
            self.hparams.pretrained_model.split("/")[1]
            if "/" in self.hparams.pretrained_model
            else self.hparams.pretrained_model
        )
        dataset_cache = (
            dataset_path + ".dataset_" + str(dataset_hash) + pretrained_model
        )

        if os.path.isfile(dataset_cache):
            click.secho(f"Loading datasets from cache: {dataset_cache}.")
            tensor_datasets = torch.load(dataset_cache)
        else:
            if self.multiple_files_process:
                datasets = self.process_multiple_separate_files(dataset_path)
            else:
                datasets = self.load_data_from_csv(dataset_path, testing=False)

            click.secho("Padding inputs and building tensors.", fg="yellow")
            tensor_datasets = {"train": [], "dev": []}
            for dataset_name, dataset in datasets.items():
                dataset = self.pad_dataset(dataset, padding=self.tokenizer.pad_token_id)
                for input_name in MODEL_INPUTS:
                    tensor = torch.tensor(dataset[input_name])
                    tensor_datasets[dataset_name].append(tensor)

            tensor_datasets["train"] = TensorDataset(*tensor_datasets["train"])
            tensor_datasets["dev"] = TensorDataset(*tensor_datasets["dev"])
            torch.save(tensor_datasets, dataset_cache)

        self.train_dataset = tensor_datasets["train"]
        self.valid_dataset = tensor_datasets["dev"]
        click.secho(
            "Train dataset (Batch, Candidates, Seq length): {}".format(
                self.train_dataset.tensors[0].shape
            ),
            fg="yellow",
        )
        click.secho(
            "Dev dataset (Batch, Candidates, Seq length): {}".format(
                self.valid_dataset.tensors[0].shape
            ),
            fg="yellow",
        )

    def pad_dataset(self, dataset: dict, padding: int = 0):
        for input_name in dataset.keys():
            max_l = (
                self.tokenizer.model_max_length
                if "input_ids" in input_name
                else max(len(x) for x in dataset[input_name])
            )
            if input_name == "attention_mask":
                dataset[input_name] = [
                    x + [0] * (self.tokenizer.model_max_length - len(x))
                    for x in dataset[input_name]
                ]
            else:
                dataset[input_name] = [
                    x + [-100 if "label" in input_name else padding] * (max_l - len(x))
                    for x in dataset[input_name]
                ]
        return dataset

    def train_dataloader(self) -> DataLoader:
        """ Function that loads the train set. """
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=multiprocessing.cpu_count(),
        )

    def val_dataloader(self) -> DataLoader:
        """ Function that loads the validation set. """
        return DataLoader(
            self.valid_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=multiprocessing.cpu_count(),
        )
