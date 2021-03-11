# -*- coding: utf-8 -*-
import torch
from pytorch_lightning.metrics import Metric


class SlotErrorRate(Metric):
    def __init__(self, dist_sync_on_step=False, padding=None, ignore=None):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("slots_ref", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("errors", default=torch.tensor(0), dist_reduce_fx="sum")
        self.padding = padding
        self.ignore = ignore

    def update(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        assert y_pred.shape == y_true.shape
        pad_mask = 1
        ref_mask = 1
        if self.padding is not None:
            pad_mask = y_true != self.padding
        if self.ignore is not None:
            ref_mask = y_true != self.ignore

        self.slots_ref += torch.sum(ref_mask * pad_mask)
        self.errors += torch.sum((y_true != y_pred) * pad_mask)

    def compute(self):
        return self.errors / torch.maximum(self.slots_ref, torch.tensor(1))
