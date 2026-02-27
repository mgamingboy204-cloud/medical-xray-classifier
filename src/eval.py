import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from src.data import build_datasets
from src.losses import sparse_focal_loss
from src.utils import json_save


def _collect_labels(dataset):
    y_true = []
    for _, labels in dataset:
        y_true.extend(labels.numpy().tolist())
    return np.array(y_true)


def _predict_with_tta(model, dataset, tta_passes: int):
    prob_list = []
    for _ in range(tta_passes):
        probs = model.predict(dataset, verbose=0)
        prob_list.append(probs)
    return np.mean(prob_list, axis=0)


def evaluate_checkpoint(checkpoint, data_dir, img_size, batch_size, tta_passes=1):
    _, _, test_ds, class_names = build_datasets(
        data_dir=data_dir,
        img_size=img_size,
        batch_size=batch_size,
        aug_cfg={"aug_mode": "light", "allow_horizontal_flip": False},
        cache=False,
        seed=42,
    )

    model = tf.keras.models.load_model(
        checkpoint,
        custom_objects={"loss": sparse_focal_loss(), "SparseCategoricalCrossentropy": tf.keras.losses.SparseCategoricalCrossentropy},
        compile=False,
    )
    y_true = _collect_labels(test_ds)
    probs = _predict_with_tta(model, test_ds, max(1, tta_passes))
    y_pred = np.argmax(probs, axis=1)

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)

    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    run_dir = Path(checkpoint).resolve().parent
    with open(run_dir / "report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    fig.savefig(run_dir / "confusion_matrix.png")
    plt.close(fig)

    metrics = {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "per_class_f1": {name: float(score) for name, score in zip(class_names, per_class_f1)},
        "tta_passes": int(max(1, tta_passes)),
    }
    json_save(run_dir / "metrics.json", metrics)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained checkpoint on test split")
    parser.add_argument("--checkpoint", required=True, help="Path to best.keras")
    parser.add_argument("--data_dir", required=True, help="Path to Classification dataset root")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--tta_passes", type=int, default=1)
    args = parser.parse_args()

    metrics = evaluate_checkpoint(
        checkpoint=args.checkpoint,
        data_dir=args.data_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        tta_passes=args.tta_passes,
    )
    print(metrics)


if __name__ == "__main__":
    main()
