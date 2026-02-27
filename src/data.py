import argparse
import hashlib
import sys
from collections import defaultdict
from pathlib import Path

import tensorflow as tf

from src.augment import augment_image, get_augmenter

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def detect_classes(data_dir: str):
    train_dir = Path(data_dir) / "Train"
    if not train_dir.exists():
        raise FileNotFoundError(f"Missing Train directory: {train_dir}")
    class_names = sorted([p.name for p in train_dir.iterdir() if p.is_dir()])
    if not class_names:
        raise ValueError("No class subfolders found under Train/")
    return class_names


def _iter_images(folder: Path):
    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def collect_split_files(data_dir: str, class_names: list[str]):
    splits = {}
    counts = defaultdict(lambda: defaultdict(int))
    data_root = Path(data_dir)

    for split in ["Train", "Valid", "Test"]:
        paths, labels = [], []
        for idx, class_name in enumerate(class_names):
            class_dir = data_root / split / class_name
            if not class_dir.exists():
                raise FileNotFoundError(f"Missing class directory: {class_dir}")
            class_images = list(_iter_images(class_dir))
            counts[split][class_name] = len(class_images)
            paths.extend([str(p) for p in class_images])
            labels.extend([idx] * len(class_images))
        splits[split] = {"paths": paths, "labels": labels}
    return splits, counts


def _sha256_file(path: str):
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def check_for_leakage(data_dir: str, report_path: str = "outputs/leakage_report.txt"):
    class_names = detect_classes(data_dir)
    split_files, counts = collect_split_files(data_dir, class_names)

    split_hashes = {}
    hash_to_paths = {}
    for split_name, payload in split_files.items():
        hashes = set()
        local_map = defaultdict(list)
        for path in payload["paths"]:
            digest = _sha256_file(path)
            hashes.add(digest)
            local_map[digest].append(path)
        split_hashes[split_name] = hashes
        hash_to_paths[split_name] = local_map

    overlaps = {
        "Train_Valid": split_hashes["Train"] & split_hashes["Valid"],
        "Train_Test": split_hashes["Train"] & split_hashes["Test"],
        "Valid_Test": split_hashes["Valid"] & split_hashes["Test"],
    }

    report_file = Path(report_path)
    report_file.parent.mkdir(parents=True, exist_ok=True)
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("=== Dataset Split Summary ===\n")
        for split in ["Train", "Valid", "Test"]:
            total = sum(counts[split].values())
            f.write(f"{split}: total={total}\n")
            for cls in class_names:
                f.write(f"  - {cls}: {counts[split][cls]}\n")

        f.write("\n=== Leakage Check (SHA256 overlap across splits) ===\n")
        total_overlap = 0
        for pair, hashes in overlaps.items():
            total_overlap += len(hashes)
            f.write(f"{pair}: {len(hashes)} duplicate hashes\n")
            for digest in list(hashes)[:5]:
                left, right = pair.split("_")
                left_samples = hash_to_paths[left][digest][:2]
                right_samples = hash_to_paths[right][digest][:2]
                f.write(f"  hash={digest}\n")
                for p in left_samples:
                    f.write(f"    {left}: {p}\n")
                for p in right_samples:
                    f.write(f"    {right}: {p}\n")
        f.write(f"\nTotal overlapping hashes: {total_overlap}\n")

    return {
        "class_names": class_names,
        "counts": counts,
        "overlaps": overlaps,
        "has_leakage": any(len(v) > 0 for v in overlaps.values()),
        "report_path": str(report_file),
    }


def _decode_resize(path, label, img_size):
    image_bytes = tf.io.read_file(path)
    image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    image = tf.image.resize(image, [img_size, img_size], antialias=True)
    image = tf.cast(image, tf.float32)
    return image, label


def _preprocess(image, label):
    return tf.cast(image, tf.float32), label


def build_datasets(data_dir, img_size, batch_size, aug_cfg, cache, seed):
    class_names = detect_classes(data_dir)
    splits, _ = collect_split_files(data_dir, class_names)
    augmenter = get_augmenter(aug_cfg)

    def make_dataset(split_name, training=False):
        payload = splits[split_name]
        ds = tf.data.Dataset.from_tensor_slices((payload["paths"], payload["labels"]))
        ds = ds.map(lambda p, y: _decode_resize(p, y, img_size), num_parallel_calls=tf.data.AUTOTUNE)

        if training and cache:
            ds = ds.cache()

        if training:
            ds = ds.shuffle(buffer_size=max(len(payload["paths"]), 1), seed=seed, reshuffle_each_iteration=True)
            ds = ds.map(lambda x, y: (augment_image(x, augmenter, aug_cfg), y), num_parallel_calls=tf.data.AUTOTUNE)

        ds = ds.map(_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(batch_size)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    train_ds = make_dataset("Train", training=True)
    val_ds = make_dataset("Valid", training=False)
    test_ds = make_dataset("Test", training=False)
    return train_ds, val_ds, test_ds, class_names


def _print_counts(counts, class_names):
    print("\nDataset class distribution")
    for split in ["Train", "Valid", "Test"]:
        total = sum(counts[split].values())
        print(f"{split} (total={total})")
        for cls in class_names:
            print(f"  {cls:<15} {counts[split][cls]}")


def main():
    parser = argparse.ArgumentParser(description="Data utilities and leakage check")
    parser.add_argument("--check-only", action="store_true", help="Run leakage check and exit")
    parser.add_argument("--data_dir", required=True, help="Path to Classification dataset directory")
    args = parser.parse_args()

    result = check_for_leakage(args.data_dir)
    _print_counts(result["counts"], result["class_names"])
    print(f"Leakage report written to: {result['report_path']}")

    if result["has_leakage"]:
        print("ERROR: Duplicate images were found across dataset splits. Fix split leakage before training.")
        sys.exit(1)

    if args.check_only:
        print("No leakage detected.")


if __name__ == "__main__":
    main()
