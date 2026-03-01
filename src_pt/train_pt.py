import argparse
import copy
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.metrics import f1_score

from src_pt.data_pt import build_loaders
from src_pt.model_pt import create_model
from src_pt.utils_pt import get_env_info, json_save, set_global_seed, timestamp_run_name




def _cfg_get(config: dict, key: str, default=None):
    loss_cfg = config.get("loss", {}) if isinstance(config.get("loss"), dict) else {}
    if key in loss_cfg:
        return loss_cfg[key]
    return config.get(key, default)


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight=None, label_smoothing: float = 0.0):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        ce = F.cross_entropy(
            logits,
            targets,
            weight=self.weight,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )
        pt = torch.exp(-ce)
        loss = ((1.0 - pt) ** self.gamma) * ce
        return loss.mean()

def _compute_class_weights(targets, num_classes, device):
    counts = torch.bincount(torch.tensor(targets), minlength=num_classes).float()
    weights = counts.sum() / (counts.clamp(min=1.0) * num_classes)
    return weights.to(device)


def _warmup_cosine_lambda(total_steps, warmup_steps, min_lr_ratio=0.05):
    def fn(step):
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.1415926535))).item()
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return fn


def _run_epoch(model, loader, criterion, optimizer, scaler, device, scheduler=None, train=True):
    model.train(train)
    total_loss = 0.0
    total = 0
    correct = 0
    preds_all = []
    labels_all = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.set_grad_enabled(train):
            with torch.autocast(device_type="cuda", enabled=(device.type == "cuda")):
                if images.ndim == 5:
                    b, ncrops, c, h, w = images.shape
                    images = images.view(b * ncrops, c, h, w)
                    logits = model(images).view(b, ncrops, -1).mean(dim=1)
                else:
                    logits = model(images)
                loss = criterion(logits, labels)

            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                if scheduler is not None:
                    scheduler.step()

        total_loss += loss.item() * labels.size(0)
        total += labels.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        preds_all.extend(pred.detach().cpu().tolist())
        labels_all.extend(labels.detach().cpu().tolist())

    loss_avg = total_loss / max(1, total)
    acc = correct / max(1, total)
    macro_f1 = f1_score(labels_all, preds_all, average="macro", zero_division=0)
    return loss_avg, acc, macro_f1


def _save_training_plot(history_df: pd.DataFrame, run_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history_df["train_loss"], label="train_loss")
    axes[0].plot(history_df["val_loss"], label="val_loss")
    axes[0].set_title("Loss")
    axes[0].legend()

    axes[1].plot(history_df["train_acc"], label="train_acc")
    axes[1].plot(history_df["val_acc"], label="val_acc")
    axes[1].plot(history_df["val_macro_f1"], label="val_macro_f1")
    axes[1].set_title("Metrics")
    axes[1].legend()

    plt.tight_layout()
    fig.savefig(run_dir / "training_curve.png")
    fig.savefig(run_dir / "training_curves.png")
    plt.close(fig)


def run_training(config: dict, data_dir: str, run_suffix: str = "") -> Path:
    set_global_seed(int(config["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_name = timestamp_run_name(
        config["model_name"], int(config["img_size"]), _cfg_get(config, "loss_name", config.get("loss_type", "ce")), config.get("run_tag", "")
    )
    if run_suffix:
        run_name = f"{run_name}_{run_suffix}"
    run_dir = Path("outputs_pt") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, _, class_names = build_loaders(
        data_dir=data_dir,
        img_size=int(config["img_size"]),
        batch_size=int(config["batch_size"]),
        aug_mode=config.get("aug_mode", "light"),
        hflip=bool(config.get("allow_horizontal_flip", False)),
        seed=int(config["seed"]),
        num_workers=config.get("num_workers"),
        val_tta_crops=int(config.get("val_tta_crops", 1)),
    )

    bundle = create_model(config, num_classes=len(class_names))
    model = bundle.model.to(device)
    ema = bundle.ema

    use_class_weights = bool(_cfg_get(config, "use_class_weights", False))
    class_weights = None
    if use_class_weights:
        class_weights = _compute_class_weights(train_loader.dataset.targets, len(class_names), device)

    loss_name = str(_cfg_get(config, "loss_name", "ce")).lower()
    label_smoothing = float(_cfg_get(config, "label_smoothing", 0.0))
    if loss_name == "focal":
        gamma = float(_cfg_get(config, "focal_gamma", 2.0))
        criterion = FocalLoss(gamma=gamma, weight=class_weights, label_smoothing=label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=float(config["lr"]),
        weight_decay=float(config.get("weight_decay", 1e-4)),
    )

    epochs = int(config["epochs"])
    steps_per_epoch = len(train_loader)
    total_steps = max(1, epochs * steps_per_epoch)
    warmup_steps = int(config.get("warmup_epochs", 1) * steps_per_epoch)
    lr_lambda = _warmup_cosine_lambda(total_steps=total_steps, warmup_steps=warmup_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

    history = []
    best_f1 = -1.0
    wait = 0
    patience = int(config.get("patience", 6))

    for epoch in range(1, epochs + 1):
        train_loss, train_acc, _ = _run_epoch(model, train_loader, criterion, optimizer, scaler, device, scheduler, train=True)
        val_loss, val_acc, val_f1 = _run_epoch(model, val_loader, criterion, optimizer, scaler, device, train=False)
        if ema is not None:
            ema.update(model)

        lr_now = float(optimizer.param_groups[0]["lr"])
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_macro_f1": val_f1,
            "lr": lr_now,
        }
        history.append(row)
        print(
            f"Epoch {epoch}/{epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_macro_f1={val_f1:.4f} lr={lr_now:.6g}"
        )

        payload = {
            "state_dict": model.state_dict(),
            "model_name": config["model_name"],
            "num_classes": len(class_names),
            "class_names": class_names,
            "img_size": int(config["img_size"]),
            "config": copy.deepcopy(config),
        }
        torch.save(payload, run_dir / "last.pt")

        if val_f1 > best_f1:
            best_f1 = val_f1
            wait = 0
            if ema is not None:
                ema.copy_to(model)
                payload["state_dict"] = model.state_dict()
            torch.save(payload, run_dir / "best.pt")
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch} (patience={patience}).")
                break

    history_df = pd.DataFrame(history)
    history_df.to_csv(run_dir / "history.csv", index=False)
    _save_training_plot(history_df, run_dir)

    json_save(run_dir / "run_config.json", config)
    json_save(run_dir / "env.json", get_env_info())
    return run_dir


def run_with_oom_retry(config: dict, data_dir: str) -> Path:
    retried = False
    current = copy.deepcopy(config)
    while True:
        try:
            return run_training(current, data_dir)
        except RuntimeError as e:
            oom = "out of memory" in str(e).lower() and torch.cuda.is_available()
            if (not oom) or retried:
                raise
            retried = True
            new_bs = max(1, int(current["batch_size"]) // 2)
            current["batch_size"] = new_bs
            current["run_tag"] = f"{current.get('run_tag', 'run')}_bs{new_bs}"
            torch.cuda.empty_cache()
            print(f"CUDA OOM encountered. Retrying once from scratch with batch_size={new_bs}.")


def main():
    parser = argparse.ArgumentParser(description="Train PyTorch classifier")
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("--data_dir", required=True, help="Classification dataset root")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    run_dir = run_with_oom_retry(config, args.data_dir)
    print({"run_dir": str(run_dir)})


if __name__ == "__main__":
    main()
