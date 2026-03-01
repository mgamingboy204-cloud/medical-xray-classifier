import os
import json
import random
import hashlib
import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

BASE_DIR = "/kaggle/input/datasets/mohithreddy12345/dental-opg-classification/Classification"
EXPECTED_CLASSES = ["Cavities", "Damage/Broken", "Infection", "Wisdom teeth"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def section(title: str):
    print("\n" + "=" * 53)
    print(title)
    print("=" * 53)


def set_reproducibility(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def print_gpu_info_and_mixed_precision():
    gpus = tf.config.list_physical_devices("GPU")
    print(f"Detected GPUs: {gpus}")
    if gpus:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        print("Mixed precision enabled: mixed_float16")
    else:
        print("No GPU found, mixed precision disabled.")


def print_tree(root: Path, max_depth: int = 3):
    print(f"Directory tree under: {root}")
    if not root.exists():
        print("Path does not exist in current environment.")
        return
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root)
        depth = len(rel.parts)
        if depth > max_depth:
            continue
        indent = "  " * (depth - 1)
        marker = "/" if p.is_dir() else ""
        print(f"{indent}- {p.name}{marker}")


def find_split_dirs(base: Path) -> Dict[str, Path]:
    split_aliases = {
        "train": ["train", "training", "tr"],
        "val": ["val", "valid", "validation", "dev"],
        "test": ["test", "testing", "te"],
    }
    found = {}
    for child in base.iterdir():
        if not child.is_dir():
            continue
        name = child.name.lower()
        for canonical, aliases in split_aliases.items():
            if name in aliases:
                found[canonical] = child
    if set(found.keys()) != {"train", "val", "test"}:
        raise RuntimeError(f"Could not infer train/val/test from {base}. Found: {found}")
    return found


def _list_images(path: Path) -> List[Path]:
    return [p for p in sorted(path.rglob("*")) if p.is_file() and p.suffix.lower() in IMG_EXTS]


def discover_structure(split_dirs: Dict[str, Path]):
    has_json = any(list(sd.rglob("*.json")) for sd in split_dirs.values())
    folder_mode = all(any(c.is_dir() for c in sd.iterdir()) for sd in split_dirs.values())
    return "json" if has_json and not folder_mode else "folders"


def parse_json_labels(json_paths: List[Path]):
    mapping = {}
    sample = None
    for jp in json_paths:
        obj = json.loads(jp.read_text())
        if sample is None:
            sample = obj
        if isinstance(obj, dict):
            if "labels" in obj and isinstance(obj["labels"], dict):
                mapping.update(obj["labels"])
            else:
                for k, v in obj.items():
                    if isinstance(v, str):
                        mapping[k] = v
    return mapping, sample


def load_split_records(split_path: Path, mode: str):
    records = []
    if mode == "folders":
        for class_dir in sorted([d for d in split_path.iterdir() if d.is_dir()]):
            for img in _list_images(class_dir):
                records.append((img, class_dir.name))
    else:
        jsons = sorted(split_path.rglob("*.json"))
        label_map, sample = parse_json_labels(jsons)
        print(f"Sample JSON for {split_path.name}: {sample}")
        for img in _list_images(split_path):
            key = img.name
            if key not in label_map:
                key = str(img.relative_to(split_path))
            if key not in label_map:
                raise ValueError(f"Missing label for image: {img}")
            records.append((img, label_map[key]))
    return records


def assert_label_consistency(records_by_split):
    split_classes = {s: sorted(set(lbl for _, lbl in recs)) for s, recs in records_by_split.items()}
    print("Class names by split:", split_classes)
    first = split_classes["train"]
    for s, cls in split_classes.items():
        assert cls == first, f"Label mismatch in {s}: {cls} vs {first}"
    print("Label mapping consistency check passed.")
    return first


def display_random_grid(records_by_split, n=16):
    all_samples = []
    for split, recs in records_by_split.items():
        all_samples.extend([(split, p, y) for p, y in recs])
    picks = random.sample(all_samples, min(n, len(all_samples)))
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    for ax, item in zip(axes.flat, picks):
        _, path, label = item
        img = tf.io.decode_image(tf.io.read_file(str(path)), channels=3, expand_animations=False).numpy()
        ax.imshow(img, cmap="gray")
        ax.set_title(label)
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def decoded_sha1(path: Path) -> str:
    img = tf.io.decode_image(tf.io.read_file(str(path)), channels=3, expand_animations=False)
    arr = tf.cast(img, tf.uint8).numpy()
    return hashlib.sha1(arr.tobytes()).hexdigest()


def phash(path: Path):
    from PIL import Image
    import imagehash

    with Image.open(path) as im:
        return str(imagehash.phash(im.convert("L")))


def dedupe_against_train(records_by_split):
    train_records = records_by_split["train"]
    train_sha = {decoded_sha1(p) for p, _ in train_records}
    train_ph = {phash(p) for p, _ in train_records}

    removed = {"val": 0, "test": 0}
    for split in ["val", "test"]:
        cleaned = []
        for p, y in records_by_split[split]:
            s = decoded_sha1(p)
            h = phash(p)
            if s in train_sha or h in train_ph:
                removed[split] += 1
                continue
            cleaned.append((p, y))
        records_by_split[split] = cleaned
    print(f"Duplicate removal counts: {removed}")
    return removed


def print_counts(records_by_split):
    for split, recs in records_by_split.items():
        print(f"\n{split.upper()} count={len(recs)}")
        labels = [y for _, y in recs]
        for cls in sorted(set(labels)):
            print(f"  {cls}: {sum(1 for z in labels if z == cls)}")


def build_class_mapping(class_names):
    return {c: i for i, c in enumerate(class_names)}


def clahe_py(image):
    import cv2

    image = image.astype(np.uint8)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    cl = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    out = np.stack([cl, cl, cl], axis=-1)
    return out.astype(np.float32) / 255.0


def parse_record(path, label, img_size=384, training=False):
    img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
    img = tf.image.resize(img, [img_size, img_size], antialias=True)
    img = tf.cast(img, tf.uint8)
    img = tf.py_function(func=clahe_py, inp=[img], Tout=tf.float32)
    img.set_shape((img_size, img_size, 3))

    if training:
        aug = tf.keras.Sequential([
            tf.keras.layers.RandomRotation(0.02),
            tf.keras.layers.RandomZoom(0.08),
            tf.keras.layers.RandomTranslation(0.05, 0.05),
            tf.keras.layers.RandomContrast(0.08),
        ])
        img = aug(img, training=True)

    return img, tf.one_hot(label, depth=4)


def make_tfdata(records, class_to_idx, training, batch=16, img_size=384):
    paths = [str(p) for p, _ in records]
    labels = [class_to_idx[y] for _, y in records]
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(len(paths), reshuffle_each_iteration=True)
    ds = ds.map(lambda p, y: parse_record(p, y, img_size, training), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.cache().batch(batch).prefetch(tf.data.AUTOTUNE)
    return ds


class MacroF1(tf.keras.callbacks.Callback):
    def __init__(self, val_ds):
        self.val_ds = val_ds
        super().__init__()

    def on_epoch_end(self, epoch, logs=None):
        y_true = np.concatenate([np.argmax(y.numpy(), axis=1) for _, y in self.val_ds])
        probs = self.model.predict(self.val_ds, verbose=0)
        y_pred = np.argmax(probs, axis=1)
        f1 = f1_score(y_true, y_pred, average="macro")
        logs = logs or {}
        logs["val_macro_f1"] = f1
        print(f" - val_macro_f1: {f1:.4f}")


def build_model(input_size=384):
    inp = tf.keras.Input((input_size, input_size, 3))
    backbone = tf.keras.applications.EfficientNetV2S(include_top=False, weights="imagenet", input_tensor=inp)
    x = tf.keras.layers.GlobalAveragePooling2D()(backbone.output)
    x = tf.keras.layers.Dropout(0.4)(x)
    out = tf.keras.layers.Dense(4, activation="softmax", dtype="float32")(x)
    model = tf.keras.Model(inp, out)
    return model, backbone


def evaluate_with_reports(model, ds, class_names, title="VAL"):
    y_true = np.concatenate([np.argmax(y.numpy(), axis=1) for _, y in ds])
    probs = model.predict(ds, verbose=0)
    y_pred = np.argmax(probs, axis=1)
    acc = accuracy_score(y_true, y_pred)
    mf1 = f1_score(y_true, y_pred, average="macro")
    print(f"{title} ACC={acc:.4f} MACRO_F1={mf1:.4f}")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
    print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))
    return {"acc": acc, "macro_f1": mf1}, probs


def tta_predict(model, ds, passes=8):
    all_probs = []
    for _ in range(passes):
        probs = model.predict(ds, verbose=0)
        all_probs.append(probs)
    return np.mean(np.stack(all_probs, axis=0), axis=0)


def run_pipeline(base_dir=BASE_DIR):
    section("0) ENV + REPRODUCIBILITY")
    set_reproducibility(42)
    print_gpu_info_and_mixed_precision()

    section("1) DATA DISCOVERY + AUDIT")
    base = Path(base_dir)
    print_tree(base, max_depth=3)
    split_dirs = find_split_dirs(base)
    mode = discover_structure(split_dirs)
    print(f"Detected structure mode: {mode}")

    records_by_split = {k: load_split_records(v, mode) for k, v in split_dirs.items()}
    print_counts(records_by_split)

    class_names = assert_label_consistency(records_by_split)
    assert len(class_names) == 4, f"Expected 4 classes, got {class_names}"
    display_random_grid(records_by_split)

    removed = dedupe_against_train(records_by_split)
    print("Counts after duplicate cleanup")
    print_counts(records_by_split)

    section("2) TF.DATA PIPELINE")
    class_to_idx = build_class_mapping(class_names)
    train_ds = make_tfdata(records_by_split["train"], class_to_idx, training=True, batch=16, img_size=384)
    val_ds = make_tfdata(records_by_split["val"], class_to_idx, training=False, batch=16, img_size=384)
    test_ds = make_tfdata(records_by_split["test"], class_to_idx, training=False, batch=16, img_size=384)

    section("3) MODEL")
    model, backbone = build_model(384)

    section("4) TRAINING PLAN")
    macro_cb = MacroF1(val_ds)
    ckpt = tf.keras.callbacks.ModelCheckpoint("best_model.keras", monitor="val_macro_f1", mode="max", save_best_only=True)
    es = tf.keras.callbacks.EarlyStopping(monitor="val_macro_f1", mode="max", patience=6, restore_best_weights=True)

    backbone.trainable = False
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=["accuracy"],
    )
    model.fit(train_ds, validation_data=val_ds, epochs=6, callbacks=[macro_cb, ckpt, es], verbose=1)

    n_unfreeze = int(0.4 * len(backbone.layers))
    for lyr in backbone.layers[:-n_unfreeze]:
        lyr.trainable = False
    for lyr in backbone.layers[-n_unfreeze:]:
        lyr.trainable = True

    total_steps = 40 * max(len(records_by_split["train"]) // 16, 1)
    sched = tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate=2e-5, decay_steps=total_steps, alpha=0.1)
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=sched, weight_decay=1e-4),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=["accuracy"],
    )
    model.fit(train_ds, validation_data=val_ds, epochs=40, callbacks=[macro_cb, ckpt, es], verbose=1)

    section("5) EVALUATION")
    best = tf.keras.models.load_model("best_model.keras")
    val_metrics, _ = evaluate_with_reports(best, val_ds, class_names, title="VAL")
    test_metrics, _ = evaluate_with_reports(best, test_ds, class_names, title="TEST")

    y_true_test = np.concatenate([np.argmax(y.numpy(), axis=1) for _, y in test_ds])
    tta_probs = tta_predict(best, test_ds, passes=8)
    y_tta = np.argmax(tta_probs, axis=1)
    test_acc_tta = accuracy_score(y_true_test, y_tta)
    test_f1_tta = f1_score(y_true_test, y_tta, average="macro")
    print(f"TEST WITH TTA ACC={test_acc_tta:.4f} MACRO_F1={test_f1_tta:.4f}")

    section("7) FINAL REPORT")
    report = {
        "dataset_path": str(base),
        "detected_structure": mode,
        "duplicates_removed": removed,
        "model": "EfficientNetV2-S",
        "input_size": 384,
        "best_val_metrics": val_metrics,
        "test_metrics_no_tta": test_metrics,
        "test_metrics_tta": {"acc": float(test_acc_tta), "macro_f1": float(test_f1_tta)},
        "timestamp": dt.datetime.utcnow().isoformat() + "Z",
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    run_pipeline(BASE_DIR)
