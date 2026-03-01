import hashlib
import os
import platform
from collections import defaultdict
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

_clahe_train = A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.8)
_clahe_eval = A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0)


def _resolve_workers(num_workers: int | None) -> int:
    if num_workers is not None and num_workers >= 0:
        return num_workers
    cpu = os.cpu_count() or 2
    return max(1, min(8, cpu - 1))


def _alb_to_tensor(image: np.ndarray) -> torch.Tensor:
    x = image.astype(np.float32) / 255.0
    x = (x - np.array(IMAGENET_MEAN, dtype=np.float32)) / np.array(IMAGENET_STD, dtype=np.float32)
    x = np.transpose(x, (2, 0, 1))
    return torch.from_numpy(x)


def _load_rgb(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _build_train_aug(img_size: int, aug_mode: str, hflip: bool) -> A.Compose:
    base = [
        A.Resize(img_size, img_size),
        _clahe_train,
        A.Sharpen(alpha=(0.05, 0.2), lightness=(0.9, 1.1), p=0.2),
    ]

    if aug_mode == "medium":
        base.extend(
            [
                A.Affine(
                    translate_percent={"x": (-0.06, 0.06), "y": (-0.06, 0.06)},
                    scale=(0.92, 1.08),
                    rotate=(-10, 10),
                    shear=0,
                    mode=cv2.BORDER_REFLECT_101,
                    p=0.85,
                ),
                A.RandomBrightnessContrast(brightness_limit=0.18, contrast_limit=0.18, p=0.55),
                A.GaussNoise(std_range=(0.01, 0.04), p=0.25),
            ]
        )
    else:
        base.extend(
            [
                A.Affine(
                    translate_percent={"x": (-0.03, 0.03), "y": (-0.03, 0.03)},
                    scale=(0.95, 1.05),
                    rotate=(-7, 7),
                    shear=0,
                    mode=cv2.BORDER_REFLECT_101,
                    p=0.75,
                ),
                A.RandomBrightnessContrast(brightness_limit=0.12, contrast_limit=0.12, p=0.4),
                A.GaussNoise(std_range=(0.005, 0.02), p=0.15),
            ]
        )

    if hflip:
        base.append(A.HorizontalFlip(p=0.5))

    base.append(
        A.Cutout(
            num_holes=8,
            max_h_size=max(1, int(img_size * 0.12)),
            max_w_size=max(1, int(img_size * 0.12)),
            fill_value=0,
            p=0.25,
        )
    )
    return A.Compose(base)


def _build_eval_aug(img_size: int) -> A.Compose:
    return A.Compose([A.Resize(img_size, img_size), _clahe_eval])


def _iter_images(folder: Path):
    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def _detect_classes(data_dir: str | Path) -> list[str]:
    train_dir = Path(data_dir) / "Train"
    if not train_dir.exists():
        raise FileNotFoundError(f"Missing Train directory: {train_dir}")
    class_names = sorted([p.name for p in train_dir.iterdir() if p.is_dir()])
    if not class_names:
        raise ValueError("No class subfolders found under Train/")
    return class_names


def _collect_split_files(data_dir: str | Path, class_names: list[str]):
    splits = {}
    counts = defaultdict(lambda: defaultdict(int))
    data_root = Path(data_dir)

    for split in ["Train", "Valid", "Test"]:
        paths = []
        for class_name in class_names:
            class_dir = data_root / split / class_name
            if not class_dir.exists():
                raise FileNotFoundError(f"Missing class directory: {class_dir}")
            class_images = [str(p) for p in _iter_images(class_dir)]
            counts[split][class_name] = len(class_images)
            paths.extend(class_images)
        splits[split] = {"paths": paths}
    return splits, counts


def _sha256_file(path: str) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def check_for_leakage(data_dir: str, report_path: str = "outputs/leakage_report.txt"):
    class_names = _detect_classes(data_dir)
    split_files, counts = _collect_split_files(data_dir, class_names)

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
        "has_leakage": any(len(v) > 0 for v in overlaps.values()),
        "report_path": str(report_file),
    }


class AlbumentationsImageFolder(Dataset):
    def __init__(self, root: str | Path, transform: A.Compose):
        self.inner = ImageFolder(str(root))
        self.samples = self.inner.samples
        self.targets = self.inner.targets
        self.classes = self.inner.classes
        self.class_to_idx = self.inner.class_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = _load_rgb(path)
        image = self.transform(image=image)["image"]
        return _alb_to_tensor(image), label


class MultiCropInferenceDataset(Dataset):
    def __init__(self, root: str | Path, img_size: int, eval_crops: int = 5):
        if eval_crops not in {5, 9}:
            raise ValueError("eval_crops must be one of {5, 9}.")
        self.inner = ImageFolder(str(root))
        self.samples = self.inner.samples
        self.targets = self.inner.targets
        self.classes = self.inner.classes
        self.eval_crops = eval_crops
        self.img_size = img_size
        self.resize_size = int(round(img_size * 1.15))

    def __len__(self):
        return len(self.samples)

    def _crop_positions(self):
        max_off = self.resize_size - self.img_size
        if self.eval_crops == 5:
            mid = max_off // 2
            return [(0, 0), (0, max_off), (max_off, 0), (max_off, max_off), (mid, mid)]
        steps = [0, max_off // 2, max_off]
        return [(y, x) for y in steps for x in steps]

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = _load_rgb(path)
        image = cv2.resize(image, (self.resize_size, self.resize_size), interpolation=cv2.INTER_AREA)
        image = _clahe_eval(image=image)["image"]
        crops = []
        for y, x in self._crop_positions():
            crop = image[y : y + self.img_size, x : x + self.img_size]
            crops.append(_alb_to_tensor(crop))
        return torch.stack(crops, dim=0), label


def build_loaders(data_dir, img_size, batch_size, aug_mode, hflip, seed, num_workers=None, val_tta_crops: int = 1):
    leak = check_for_leakage(data_dir)
    if leak["has_leakage"]:
        raise RuntimeError(
            f"Data leakage detected. Resolve split overlap before training. See {leak['report_path']}."
        )

    data_root = Path(data_dir)
    workers = _resolve_workers(num_workers)
    persistent = workers > 0 and platform.system() != "Windows"
    gen = torch.Generator()
    gen.manual_seed(seed)

    train_ds = AlbumentationsImageFolder(data_root / "Train", _build_train_aug(img_size, aug_mode, hflip))
    if val_tta_crops > 1:
        val_ds = MultiCropInferenceDataset(data_root / "Valid", img_size=img_size, eval_crops=val_tta_crops)
    else:
        val_ds = AlbumentationsImageFolder(data_root / "Valid", _build_eval_aug(img_size))
    test_ds = AlbumentationsImageFolder(data_root / "Test", _build_eval_aug(img_size))

    class_names = train_ds.classes
    if class_names != val_ds.classes or class_names != test_ds.classes:
        raise RuntimeError("Class order mismatch across Train/Valid/Test folders.")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=persistent,
        generator=gen,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=persistent,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=persistent,
    )
    return train_loader, val_loader, test_loader, class_names


def build_test_loader(data_dir, img_size, batch_size, tta_crops=1, num_workers=None):
    leak = check_for_leakage(data_dir)
    if leak["has_leakage"]:
        raise RuntimeError(f"Data leakage detected. See {leak['report_path']}.")

    data_root = Path(data_dir)
    workers = _resolve_workers(num_workers)
    persistent = workers > 0 and platform.system() != "Windows"

    if tta_crops > 1:
        ds = MultiCropInferenceDataset(data_root / "Test", img_size=img_size, eval_crops=tta_crops)
    else:
        ds = AlbumentationsImageFolder(data_root / "Test", _build_eval_aug(img_size))

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=persistent,
    )
    return loader, ds.classes
