import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from src_pt.data_pt import AlbumentationsImageFolder, MultiCropInferenceDataset, _build_eval_aug
from src_pt.model_pt import create_model
from src_pt.utils_pt import json_save


def _forward_logits(model, images, device):
    images = images.to(device, non_blocking=True)
    with torch.no_grad(), torch.autocast(device_type="cuda", enabled=(device.type == "cuda")):
        logits = model(images)
    return logits


def _build_eval_loader(data_dir: str, split: str, img_size: int, batch_size: int, tta_crops: int, num_workers=None):
    data_root = Path(data_dir)
    split_root = data_root / split
    workers = int(num_workers) if num_workers is not None else 4
    workers = max(0, workers)
    persistent = workers > 0

    if tta_crops > 1:
        ds = MultiCropInferenceDataset(split_root, img_size=img_size, eval_crops=tta_crops)
    else:
        ds = AlbumentationsImageFolder(split_root, _build_eval_aug(img_size))

    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=persistent,
    )
    return loader, ds.classes


def predict_probabilities(
    checkpoint_path: str,
    data_dir: str,
    img_size: int,
    batch_size: int,
    tta_crops: int = 1,
    split: str = "Test",
):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model_cfg = {"model_name": ckpt["model_name"], **ckpt.get("config", {})}
    bundle = create_model(model_cfg, num_classes=int(ckpt["num_classes"]))
    model = bundle.model
    model.load_state_dict(ckpt["state_dict"], strict=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    loader, loader_classes = _build_eval_loader(
        data_dir=data_dir,
        split=split,
        img_size=img_size,
        batch_size=batch_size,
        tta_crops=tta_crops,
        num_workers=model_cfg.get("num_workers"),
    )
    ckpt_classes = ckpt.get("class_names", loader_classes)
    if list(ckpt_classes) != list(loader_classes):
        raise RuntimeError(f"Class order mismatch between checkpoint and {split} dataset.")

    probs_all, y_true = [], []
    for images, labels in loader:
        if images.ndim == 5:
            b, ncrops, c, h, w = images.shape
            images = images.view(b * ncrops, c, h, w)
            logits = _forward_logits(model, images, device)
            logits = logits.view(b, ncrops, -1).mean(dim=1)
        else:
            logits = _forward_logits(model, images, device)
        probs = torch.softmax(logits, dim=1)
        probs_all.append(probs.cpu().numpy())
        y_true.extend(labels.numpy().tolist())

    probs_np = np.concatenate(probs_all, axis=0)
    return probs_np, np.array(y_true), list(loader_classes)


def evaluate_checkpoint(checkpoint, data_dir, img_size, batch_size, tta_crops=1, split="Test"):
    probs, y_true, class_names = predict_probabilities(
        checkpoint_path=checkpoint,
        data_dir=data_dir,
        img_size=img_size,
        batch_size=batch_size,
        tta_crops=tta_crops,
        split=split,
    )
    y_pred = probs.argmax(axis=1)

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
    ax.set_title(f"Confusion Matrix ({split})")
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    fig.savefig(run_dir / "confusion_matrix.png")
    plt.close(fig)

    metrics = {
        "split": split,
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "per_class_f1": {name: float(v) for name, v in zip(class_names, per_class_f1)},
        "tta_crops": int(tta_crops),
    }
    json_save(run_dir / "metrics.json", metrics)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate PyTorch checkpoint")
    parser.add_argument("--checkpoint", required=True, help="Path to best.pt")
    parser.add_argument("--data_dir", required=True, help="Classification dataset root")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--tta_crops", type=int, default=1)
    parser.add_argument("--split", choices=["Valid", "Test"], default="Test")
    args = parser.parse_args()

    metrics = evaluate_checkpoint(
        checkpoint=args.checkpoint,
        data_dir=args.data_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        tta_crops=max(1, args.tta_crops),
        split=args.split,
    )
    print(metrics)


if __name__ == "__main__":
    main()
