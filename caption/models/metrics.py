# -*- coding: utf-8 -*-
import numpy as np
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix


def classification_report(
    y_pred: np.array, y: np.array, padding: int, labels: dict, ignore: int = -1
) -> dict:
    """
    Function that computes the F Score for all labels, the Macro-average F Score \
        and the slot error rate.
        
    :param y: Ground-truth labels.
    :param y_pred: Model label predictions.
    :param padding: Label padding value.
    :param labels: dictionary with the name of each label (e.g. {'BAD': 0, 'OK': 1})
    :param ignore: By setting this value the F Score of this label will not be taken
        into consideration when computing the Macro-average F Score.
    """
    report = {
        "slot_error_rate": slot_error_rate(
            y, y_pred, padding, ignore=ignore if ignore >= 0 else None
        )
    }

    cm = confusion_matrix(y_pred, y, padding)
    index2label = {v: k for k, v in labels.items()}
    fscore_avg = []
    tpos, fpos, fneg = [], [], []
    for i in range(len(labels)):
        if i == padding:
            continue
        tp = cm[i][i]
        fp = np.sum(cm[i, :]) - tp
        fn = np.sum(cm[:, i]) - tp
        f_score = fscore(tp, fp, fn)
        report["{}_f1_score".format(index2label[i])] = f_score

        if i != ignore:
            fscore_avg.append(f_score)
            tpos.append(tp)
            fpos.append(fp)
            fneg.append(fn)

    report["macro_fscore"] = sum(fscore_avg) / len(fscore_avg)
    report["micro_fscore"] = fscore(sum(tpos), sum(fpos), sum(fneg))
    return report


def precision(tp: int, fp: int, fn: int) -> float:
    if tp + fp > 0:
        return tp / (tp + fp)
    return 0


def recall(tp: int, fp: int, fn: int) -> float:
    if tp + fn > 0:
        return tp / (tp + fn)
    return 0


def fscore(tp: int, fp: int, fn: int) -> float:
    p = precision(tp, fp, fn)
    r = recall(tp, fp, fn)
    if p + r > 0:
        return 2 * (p * r) / (p + r)
    return 0


def confusion_matrix(y_pred: np.array, y: np.array, padding: int) -> np.array:
    """ Function that creates a confusion matrix using the Wikipedia convention for the axis. 
    :param y_pred: predicted tags.
    :param y: the ground-truth tags.
    :param padding: padding index to be ignored.
    
    Returns:
        - Confusion matrix for all the labels + padding label."""
    y_pred = np.ma.masked_array(data=y_pred, mask=(y == padding)).filled(padding)
    return sklearn_confusion_matrix(y_pred, y)


def slot_error_rate(
    y_true: np.ndarray, y_pred: np.ndarray, padding=None, ignore=None
) -> np.float64:
    """ All classes associated with padding will be ignored in the evaluation.
    :param y_true: Ground-truth labels.
    :param y_pred: Model label predictions.
    :param padding: Label padding value.
    :param ignore: Label value corresponding to the majority label. (e.g. B in BIO tags)
    Returns:
        - np.float64 with slot error rate.
    """
    pad_mask = 1
    ref_mask = 1
    if padding is not None:
        pad_mask = y_true != padding
    if ignore is not None:
        ref_mask = y_true != ignore

    slots_ref = np.sum(ref_mask * pad_mask)
    errors = np.sum((y_true != y_pred) * pad_mask)
    return errors / np.maximum(slots_ref, 1)
