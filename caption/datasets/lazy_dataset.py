# -*- coding: utf-8 -*-
import linecache
import os
import torch


class LineByLineTextDataset(torch.utils.data.Dataset):
    """
    Dataset object that reads a txt file line by line.

    :param path: Path to the txt file containing our sentences.
    """

    def __init__(self, file_path: str):
        assert os.path.isfile(file_path)

        with open(file_path, "r") as fp:
            self.num_lines = sum(1 for line in fp) - 1

        self.file_path = file_path

    def __len__(self):
        return self.num_lines + 1

    def __getitem__(self, idx):
        if idx > self.num_lines:
            raise ValueError(
                "Trying to access index {} in a dataset with size={}".format(
                    idx, self.num_lines
                )
            )
        line = linecache.getline(self.file_path, idx + 1)
        return {"text": line.strip()}
