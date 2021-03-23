"""
Evaluate subtask 1 of SEPP-NLG 2021: https://sites.google.com/view/sentence-segmentation/
"""

__author__ = "don.tuggener@zhaw.ch"

import io
import os
from zipfile import ZipFile
from sklearn.metrics import classification_report, precision_recall_fscore_support


def evaluate_subtask1(
    data_zip: str, data_set: str, lang: str, prediction_dir: str, subtask: str
) -> None:
    relevant_dir = os.path.join("sepp_nlg_2021_data", lang, data_set)

    all_gt_labels, all_predicted_labels = (
        list(),
        list(),
    )  # aggregate all labels over all files
    with ZipFile(data_zip, "r") as zf:  # load ground truth
        fnames = zf.namelist()
        gt_tsv_files = [
            fname
            for fname in fnames
            if fname.startswith(relevant_dir) and fname.endswith(".tsv")
        ]

        for i, gt_tsv_file in enumerate(gt_tsv_files, 1):
            print(i, gt_tsv_file)
            basename = os.path.basename(gt_tsv_file)

            # get ground truth labels
            with io.TextIOWrapper(zf.open(gt_tsv_file), encoding="utf-8") as f:
                lines = f.read().strip().split("\n")
                rows = [line.split("\t") for line in lines]
                gt_labels = [
                    (row[1] if subtask == "binary" else row[2]) for row in rows
                ]

            # get predicted labels
            with open(
                os.path.join(prediction_dir, lang, data_set, basename),
                "r",
                encoding="utf8",
            ) as f:
                lines = f.read().strip().split("\n")
                rows = [line.split("\t") for line in lines]
                pred_labels = [
                    (row[1] if subtask == "binary" else row[2]) for row in rows
                ]

            assert len(gt_labels) == len(
                pred_labels
            ), f"unequal no. of labels for files {gt_tsv_file} and {os.path.join(prediction_dir,  lang, data_set, basename)}"
            all_gt_labels.extend(gt_labels)
            all_predicted_labels.extend(pred_labels)

        eval_result = classification_report(all_gt_labels, all_predicted_labels)
        print(eval_result)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate subtask 1 of SEPP-NLG 2021")
    parser.add_argument(
        "--data_zip",
        help="path to data zip file, e.g. 'data/sepp_nlg_2021_train_dev_data.zip'",
    )
    parser.add_argument(
        "--language",
        help="target language ('en', 'de', 'fr', 'it'; i.e. one of the subfolders in the zip file's main folder)",
    )
    parser.add_argument(
        "--set",
        help="dataset to be evaluated (usually 'dev', 'test'), subfolder of 'lang'",
    )
    parser.add_argument(
        "--prediction_dir",
        help="path to folder containing the prediction TSVs (language and test set folder names are appended automatically)",
    )
    parser.add_argument("--subtask", help="binary or punct", default="binary")
    args = parser.parse_args()
    evaluate_subtask1(
        args.data_zip, args.set, args.language, args.prediction_dir, args.subtask
    )
