#!/usr/bin/env python3
"""Evaluate a trained classifier on the untouched test split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mammography.data import adapt_dataset, load_image_dataset  # noqa: E402
from mammography.labels import CLASS_NAMES  # noqa: E402
from mammography.models import BACKBONES, image_size_for  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", choices=tuple(BACKBONES), required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def specificity_per_class(matrix: np.ndarray) -> list[float]:
    values = []
    for index in range(matrix.shape[0]):
        tp = matrix[index, index]
        fn = matrix[index, :].sum() - tp
        fp = matrix[:, index].sum() - tp
        tn = matrix.sum() - tp - fn - fp
        values.append(float(tn / (tn + fp)) if tn + fp else 0.0)
    return values


def main() -> None:
    import tensorflow as tf

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    image_size = image_size_for(args.model)
    raw_test = load_image_dataset(
        args.data_dir / "test",
        image_size=image_size,
        batch_size=args.batch_size,
        shuffle=False,
    )
    test_ds = adapt_dataset(raw_test, args.model, training=False)
    model = tf.keras.models.load_model(args.model_path)

    probabilities = model.predict(test_ds, verbose=1)
    y_pred = probabilities.argmax(axis=1)
    y_true = np.concatenate([labels.numpy().argmax(axis=1) for _, labels in test_ds])
    matrix = confusion_matrix(y_true, y_pred, labels=range(len(CLASS_NAMES)))
    y_binary = label_binarize(y_true, classes=range(len(CLASS_NAMES)))

    summary = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_auc_ovr": float(roc_auc_score(y_binary, probabilities, average="macro", multi_class="ovr")),
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=range(len(CLASS_NAMES)),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    report_frame = pd.DataFrame(report).transpose()
    report_frame.loc[list(CLASS_NAMES), "specificity"] = specificity_per_class(matrix)
    report_frame.to_csv(args.output_dir / "classification_report.csv")

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"{args.model} confusion matrix")
    plt.tight_layout()
    plt.savefig(args.output_dir / "confusion_matrix.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 6))
    for index, class_name in enumerate(CLASS_NAMES):
        fpr, tpr, _ = roc_curve(y_binary[:, index], probabilities[:, index])
        auc_value = np.trapz(tpr, fpr)
        plt.plot(fpr, tpr, label=f"{class_name} (AUC={auc_value:.3f})")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title(f"{args.model} one-vs-rest ROC curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output_dir / "roc_curves.png", dpi=180)
    plt.close()

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

