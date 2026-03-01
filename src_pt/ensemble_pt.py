import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score

from src_pt.data_pt import AlbumentationsImageFolder, MultiCropInferenceDataset, _build_eval_aug
from src_pt.model_pt import create_model
from src_pt.utils_pt import json_save


def _build_loader(data_dir: str, split: str, img_size: int, batch_size: int, tta_crops: int, num_workers=4):
    root = Path(data_dir) / split
    workers = max(0, int(num_workers))
    if tta_crops > 1:
        ds = MultiCropInferenceDataset(root, img_size=img_size, eval_crops=tta_crops)
    else:
        ds = AlbumentationsImageFolder(root, _build_eval_aug(img_size))

    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )
    return loader, ds.classes


def _predict_probs(model, loader, device):
    probs_all = []
    y_true = []
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            if images.ndim == 5:
                b, ncrops, c, h, w = images.shape
                images = images.view(b * ncrops, c, h, w)
                with torch.autocast(device_type="cuda", enabled=(device.type == "cuda")):
                    logits = model(images)
                logits = logits.view(b, ncrops, -1).mean(dim=1)
            else:
                with torch.autocast(device_type="cuda", enabled=(device.type == "cuda")):
                    logits = model(images)
            probs = torch.softmax(logits, dim=1)
            probs_all.append(probs.cpu().numpy())
            y_true.extend(labels.numpy().tolist())
    return np.concatenate(probs_all, axis=0), np.array(y_true)


def main():
    parser = argparse.ArgumentParser(description="Ensemble model checkpoints")
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--img_size", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--tta_crops", type=int, default=1)
    parser.add_argument("--split", choices=["Valid", "Test"], default="Test")
    args = parser.parse_args()

    payloads = [torch.load(c, map_location="cpu") for c in args.checkpoints]
    for i, payload in enumerate(payloads):
        for key in ("state_dict", "model_name", "num_classes", "class_names", "img_size"):
            if key not in payload:
                raise KeyError(f"Checkpoint {args.checkpoints[i]} missing key: {key}")

    class_names = list(payloads[0]["class_names"])
    for i, payload in enumerate(payloads[1:], start=1):
        if list(payload["class_names"]) != class_names:
            raise RuntimeError(f"Class names mismatch: {args.checkpoints[0]} vs {args.checkpoints[i]}")

    loader, ds_classes = _build_loader(
        data_dir=args.data_dir,
        split=args.split,
        img_size=args.img_size,
        batch_size=args.batch_size,
        tta_crops=max(1, args.tta_crops),
        num_workers=payloads[0].get("config", {}).get("num_workers", 4),
    )
    if class_names != list(ds_classes):
        raise RuntimeError(f"Class order mismatch between checkpoints and {args.split} dataset.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    probs_models = []
    y_true_ref = None

    for payload in payloads:
        model_cfg = {"model_name": payload["model_name"], **payload.get("config", {})}
        bundle = create_model(model_cfg, num_classes=int(payload["num_classes"]))
        model = bundle.model
        model.load_state_dict(payload["state_dict"], strict=True)
        model.to(device)

        probs, y_true = _predict_probs(model, loader, device)
        probs_models.append(probs)
        if y_true_ref is None:
            y_true_ref = y_true

    probs_ens = np.mean(np.stack(probs_models, axis=0), axis=0)
    y_pred = probs_ens.argmax(axis=1)

    metrics = {
        "split": args.split,
        "num_models": len(payloads),
        "tta_crops": int(max(1, args.tta_crops)),
        "accuracy": float(accuracy_score(y_true_ref, y_pred)),
        "macro_f1": float(f1_score(y_true_ref, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true_ref, y_pred, average="weighted", zero_division=0)),
        "per_class_f1": {
            name: float(v)
            for name, v in zip(class_names, f1_score(y_true_ref, y_pred, average=None, zero_division=0))
        },
    }
    print(metrics)

    out_dir = Path("outputs_pt")
    out_dir.mkdir(parents=True, exist_ok=True)
    json_save(out_dir / "ensemble_metrics.json", metrics)
    report = classification_report(y_true_ref, y_pred, target_names=class_names, zero_division=0)
    (out_dir / "ensemble_report.txt").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
