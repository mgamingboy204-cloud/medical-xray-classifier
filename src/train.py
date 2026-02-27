import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from sklearn.metrics import f1_score

from src.data import build_datasets, check_for_leakage
from src.losses import sparse_categorical_crossentropy, sparse_focal_loss
from src.model import build_model, set_trainable_for_phase
from src.utils import get_env_info, json_save, set_global_determinism, timestamp_run_name


class MacroF1Callback(tf.keras.callbacks.Callback):
    def __init__(self, val_ds):
        super().__init__()
        self.val_ds = val_ds

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        y_true = []
        for _, labels in self.val_ds:
            y_true.extend(labels.numpy().tolist())
        probs = self.model.predict(self.val_ds, verbose=0)
        y_pred = np.argmax(probs, axis=1)
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        logs["val_macro_f1"] = float(macro_f1)
        print(f" - val_macro_f1: {macro_f1:.4f}")


def _get_loss(config):
    if config["loss_type"] == "focal":
        return sparse_focal_loss(gamma=float(config["focal_gamma"]), alpha=float(config["focal_alpha"]))
    return sparse_categorical_crossentropy()


def _make_callbacks(run_dir: Path, val_ds, monitor_metric: str, scheduler: str):
    callbacks = [
        MacroF1Callback(val_ds),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(run_dir / "best.keras"),
            monitor=monitor_metric,
            mode="max",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor_metric,
            mode="max",
            patience=6,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    if scheduler == "cosine_decay":
        callbacks.append(
            tf.keras.callbacks.LearningRateScheduler(
                lambda epoch, lr: float(tf.keras.optimizers.schedules.CosineDecay(
                    initial_learning_rate=lr,
                    decay_steps=max(10, epoch + 1),
                    alpha=0.1,
                )(epoch))
            )
        )
    else:
        callbacks.append(
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor=monitor_metric,
                mode="max",
                factor=0.5,
                patience=2,
                min_lr=1e-7,
                verbose=1,
            )
        )
    return callbacks


def _plot_history(history_df: pd.DataFrame, output_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    if "loss" in history_df:
        axes[0].plot(history_df["loss"], label="train_loss")
    if "val_loss" in history_df:
        axes[0].plot(history_df["val_loss"], label="val_loss")
    axes[0].set_title("Loss")
    axes[0].legend()

    if "sparse_categorical_accuracy" in history_df:
        axes[1].plot(history_df["sparse_categorical_accuracy"], label="train_acc")
    if "val_sparse_categorical_accuracy" in history_df:
        axes[1].plot(history_df["val_sparse_categorical_accuracy"], label="val_acc")
    if "val_macro_f1" in history_df:
        axes[1].plot(history_df["val_macro_f1"], label="val_macro_f1")
    axes[1].set_title("Metrics")
    axes[1].legend()

    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def run_training(config: dict, data_dir: str):
    set_global_determinism(int(config["seed"]))

    leak = check_for_leakage(data_dir)
    if leak["has_leakage"]:
        raise RuntimeError(
            f"Data leakage detected. See {leak['report_path']} for duplicate file hashes across Train/Valid/Test."
        )

    use_mixed = bool(config.get("mixed_precision", True)) and bool(tf.config.list_physical_devices("GPU"))
    if use_mixed:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")

    run_name = timestamp_run_name(config["model_name"], int(config["img_size"]), config["loss_type"], config.get("run_tag", ""))
    run_dir = Path("outputs") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, _, class_names = build_datasets(
        data_dir=data_dir,
        img_size=int(config["img_size"]),
        batch_size=int(config["batch_size"]),
        aug_cfg=config,
        cache=bool(config.get("cache", False)),
        seed=int(config["seed"]),
    )

    model, backbone = build_model(config, num_classes=len(class_names))
    callbacks = _make_callbacks(run_dir, val_ds, config.get("monitor_metric", "val_macro_f1"), config.get("scheduler", "reduce_on_plateau"))

    # Phase 1: train classifier head
    set_trainable_for_phase(backbone, freeze=True)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=float(config["lr_head"])),
        loss=_get_loss(config),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )
    hist1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=int(config["epochs_head"]),
        callbacks=callbacks,
        verbose=1,
    )

    # Phase 2: fine-tune
    set_trainable_for_phase(backbone, freeze=False, unfreeze_layers=int(config.get("unfreeze_layers", 0)))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=float(config["lr_ft"])),
        loss=_get_loss(config),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )
    hist2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=int(config["epochs_head"]) + int(config["epochs_ft"]),
        initial_epoch=int(config["epochs_head"]),
        callbacks=callbacks,
        verbose=1,
    )

    history = {}
    for k, v in hist1.history.items():
        history[k] = list(v)
    for k, v in hist2.history.items():
        history.setdefault(k, [])
        history[k].extend(v)

    history_df = pd.DataFrame(history)
    history_df.to_csv(run_dir / "history.csv", index=False)
    _plot_history(history_df, run_dir / "training_curve.png")

    config_to_save = dict(config)
    config_to_save["class_names"] = class_names
    config_to_save["mixed_precision_active"] = use_mixed
    json_save(run_dir / "run_config.json", config_to_save)
    json_save(run_dir / "env.json", get_env_info())

    return run_dir


def run_with_oom_retry(config: dict, data_dir: str):
    tried_retry = False
    try:
        return run_training(config, data_dir)
    except tf.errors.ResourceExhaustedError:
        if int(config.get("batch_size", 16)) > 8 and not tried_retry:
            tried_retry = True
            retry_cfg = dict(config)
            retry_cfg["batch_size"] = 8
            retry_cfg["run_tag"] = f"{config.get('run_tag', 'run')}_oom_retry"
            print("OOM encountered. Retrying once with batch_size=8.")
            return run_training(retry_cfg, data_dir)
        raise


def main():
    parser = argparse.ArgumentParser(description="Train dental OPG classifier")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--data_dir", required=True, help="Path to Classification dataset root")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    run_dir = run_with_oom_retry(config, args.data_dir)
    print(json.dumps({"run_dir": str(run_dir)}, indent=2))


if __name__ == "__main__":
    main()
