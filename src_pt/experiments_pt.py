import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import accuracy_score, f1_score

from src_pt.eval_pt import evaluate_checkpoint, predict_probabilities
from src_pt.train_pt import run_with_oom_retry
from src_pt.utils_pt import json_save


def _load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _run_train_eval(cfg, data_dir, tta_crops=1):
    run_dir = run_with_oom_retry(cfg, data_dir)
    metrics = evaluate_checkpoint(
        checkpoint=str(Path(run_dir) / "best.pt"),
        data_dir=data_dir,
        img_size=int(cfg["img_size"]),
        batch_size=int(cfg["batch_size"]),
        tta_crops=tta_crops,
    )
    row = {
        "run_dir": str(run_dir),
        "model_name": cfg["model_name"],
        "run_tag": cfg.get("run_tag", ""),
        "img_size": int(cfg["img_size"]),
        "tta_crops": int(tta_crops),
        "test_macro_f1": metrics["macro_f1"],
        "test_accuracy": metrics["accuracy"],
        "test_weighted_f1": metrics["weighted_f1"],
    }
    return run_dir, row


def _ensemble_top2(data_dir, top_rows):
    probs_list = []
    y_true = None
    class_names = None
    for row in top_rows:
        ckpt = str(Path(row["run_dir"]) / "best.pt")
        probs, y, names = predict_probabilities(
            checkpoint_path=ckpt,
            data_dir=data_dir,
            img_size=int(row["img_size"]),
            batch_size=16,
            tta_crops=1,
        )
        probs_list.append(probs)
        if y_true is None:
            y_true = y
            class_names = names
    ens_probs = np.mean(probs_list, axis=0)
    y_pred = ens_probs.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "class_names": class_names,
    }


def main():
    parser = argparse.ArgumentParser(description="PyTorch experiment suite + leaderboard")
    parser.add_argument("--data_dir", required=True, help="Classification dataset root")
    args = parser.parse_args()

    results = []

    configs = [
        _load_yaml("configs_pt/convnext_tiny_224.yaml"),
        _load_yaml("configs_pt/effnetv2_b3_320.yaml"),
        _load_yaml("configs_pt/dinov2_vits14_224.yaml"),
    ]

    for cfg in configs:
        _, row = _run_train_eval(cfg, args.data_dir, tta_crops=1)
        results.append(row)

    dino_cfg = configs[2]
    if bool(dino_cfg.get("optional_finetune", True)) and torch.cuda.is_available():
        total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if total_gb >= float(dino_cfg.get("finetune_min_vram_gb", 14)):
            ft_cfg = dict(dino_cfg)
            ft_cfg["freeze_backbone"] = False
            ft_cfg["run_tag"] = f"{dino_cfg.get('run_tag', 'dinov2')}_finetune"
            ft_cfg["epochs"] = int(dino_cfg.get("finetune_epochs", max(4, int(dino_cfg["epochs"]) // 2)))
            _, row = _run_train_eval(ft_cfg, args.data_dir, tta_crops=1)
            results.append(row)

    leaderboard = pd.DataFrame(results).sort_values(["test_macro_f1", "test_accuracy"], ascending=[False, False])
    best_row = leaderboard.iloc[0].to_dict()

    tta_metrics = evaluate_checkpoint(
        checkpoint=str(Path(best_row["run_dir"]) / "best.pt"),
        data_dir=args.data_dir,
        img_size=int(best_row["img_size"]),
        batch_size=16,
        tta_crops=5,
    )
    results.append(
        {
            **best_row,
            "run_tag": f"{best_row.get('run_tag', '')}_tta5",
            "tta_crops": 5,
            "test_macro_f1": tta_metrics["macro_f1"],
            "test_accuracy": tta_metrics["accuracy"],
            "test_weighted_f1": tta_metrics["weighted_f1"],
        }
    )

    top2 = leaderboard.head(2).to_dict(orient="records")
    if len(top2) == 2:
        ens = _ensemble_top2(args.data_dir, top2)
        results.append(
            {
                "run_dir": "ensemble_top2",
                "model_name": "ensemble",
                "run_tag": "top2_prob_avg",
                "img_size": top2[0]["img_size"],
                "tta_crops": 1,
                "test_macro_f1": ens["macro_f1"],
                "test_accuracy": ens["accuracy"],
                "test_weighted_f1": ens["weighted_f1"],
            }
        )
        json_save("outputs_pt/ensemble_metrics.json", ens)

    final_lb = pd.DataFrame(results).sort_values(["test_macro_f1", "test_accuracy"], ascending=[False, False])
    Path("outputs_pt").mkdir(parents=True, exist_ok=True)
    final_lb.to_csv("outputs_pt/leaderboard.csv", index=False)
    print(final_lb)


if __name__ == "__main__":
    main()
