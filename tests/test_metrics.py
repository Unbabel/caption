# -*- coding: utf-8 -*-
import unittest

import numpy as np

from caption.models.metrics import (
    classification_report,
    confusion_matrix,
    fscore,
    precision,
    recall,
)


class TestMetrics(unittest.TestCase):
    @property
    def y_pred(self):
        return np.array([0, 0, 1, 2, 2, 1, 1, 0, 1, 0])

    @property
    def y(self):
        return np.array([0, 1, 1, 2, 1, 2, 0, 2, 3, 3])

    def test_confusion_matrix(self):
        cm = confusion_matrix(self.y_pred, self.y, padding=3)
        expected = np.array([[1, 1, 1, 0], [1, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 2]])
        assert np.array_equal(cm, expected)

    def test_recall(self):
        tp = 3
        fp = 1
        fn = 1
        assert recall(tp, fp, fn) == 3 / 4

    def test_precision(self):
        tp = 3
        fp = 1
        fn = 1
        assert precision(tp, fp, fn) == 3 / 4

    def test_fscore(self):
        tp = 3
        fp = 1
        fn = 1
        assert fscore(tp, fp, fn) == 2 * (
            precision(tp, fp, fn) * recall(tp, fp, fn)
        ) / (recall(tp, fp, fn) + precision(tp, fp, fn))

    def test_classification_report(self):
        out = classification_report(
            self.y_pred, self.y, padding=3, labels={"L": 0, "U": 1, "T": 2}, ignore=0
        )
        assert out["L_f1_score"] == 2 * ((1 / 3) * (1 / 2)) / ((1 / 3) + (1 / 2))
        assert out["U_f1_score"] == 2 * ((1 / 3) * (1 / 3)) / ((1 / 3) + (1 / 3))
        assert out["T_f1_score"] == 2 * ((1 / 3) * (1 / 2)) / ((1 / 3) + (1 / 2))

        assert out["macro_fscore"] == (out["U_f1_score"] + out["T_f1_score"]) / 2
        assert out["micro_fscore"] != 0


if __name__ == "__main__":
    unittest.main()
