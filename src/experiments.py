import argparse
from pathlib import Path

import pandas as pd
import yaml

from src.eval import evaluate_checkpoint
from src.train import run_with_oom_retry


def _load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _run_experiment(config, data_dir, results):
    run_dir = run_with_oom_retry(config, data_dir)
    metrics = evaluate_checkpoint(
        checkpoint=str(Path(run_dir) / "best.keras"),
        data_dir=data_dir,
        img_size=int(config["img_size"]),
        batch_size=int(config["batch_size"]),
        tta_passes=1,
    )
    results.append(
        {
            "run_dir": str(run_dir),
            "run_tag": config.get("run_tag", ""),
            "model_name": config["model_name"],
            "img_size": config["img_size"],
            "loss_type": config["loss_type"],
            "test_macro_f1": metrics["macro_f1"],
            "test_accuracy": metrics["accuracy"],
            "test_weighted_f1": metrics["weighted_f1"],
        }
    )
    return run_dir, metrics


def main():
    parser = argparse.ArgumentParser(description="Run experiment suite and generate leaderboard")
    parser.add_argument("--data_dir", required=True, help="Path to Classification dataset root")
    args = parser.parse_args()

    results = []

    base_cfgs = [
        _load_config("configs/baseline.yaml"),
        _load_config("configs/finetune.yaml"),
        _load_config("configs/efficientnet.yaml"),
    ]

    best_base = None
    best_base_metric = -1.0
    for cfg in base_cfgs:
        run_dir, metrics = _run_experiment(cfg, args.data_dir, results)
        if metrics["macro_f1"] > best_base_metric:
            best_base_metric = metrics["macro_f1"]
            best_base = (cfg, run_dir)

    best_backbone = best_base[0]["model_name"] if best_base else "mobilenetv3"
    focal_template = _load_config("configs/focal_loss.yaml")
    focal_template["model_name"] = best_backbone
    for gamma in [1.5, 2.0, 2.5]:
        cfg = dict(focal_template)
        cfg["focal_gamma"] = gamma
        cfg["run_tag"] = f"focal_gamma_{gamma}_{best_backbone}"
        _run_experiment(cfg, args.data_dir, results)

    highres_cfg = _load_config("configs/highres_320.yaml")
    _run_experiment(highres_cfg, args.data_dir, results)

    leaderboard = pd.DataFrame(results).sort_values(["test_macro_f1", "test_accuracy"], ascending=[False, False])
    Path("outputs").mkdir(exist_ok=True)
    leaderboard.to_csv("outputs/leaderboard.csv", index=False)

    if not leaderboard.empty:
        top = leaderboard.iloc[0]
        tta_cfg = _load_config("configs/tta.yaml")
        evaluate_checkpoint(
            checkpoint=str(Path(top["run_dir"]) / "best.keras"),
            data_dir=args.data_dir,
            img_size=int(tta_cfg["img_size"]),
            batch_size=int(tta_cfg["batch_size"]),
            tta_passes=int(tta_cfg.get("tta_passes", 5)),
        )

    print(leaderboard)


if __name__ == "__main__":
    main()
