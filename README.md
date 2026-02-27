# Medical X-ray Classifier (Dental OPG, Multi-class)

Leak-proof, reproducible TensorFlow training/evaluation pipeline for dental OPG X-ray classification.

## Dataset (not included in Git)

This repository **does not include the dataset** (3.7GB). Keep data local and point scripts to the local path.

Expected structure:

```text
Classification/
  Train/
    Cavities/
    Damage/
    Infection/
    Wisdom/
  Valid/
    Cavities/
    Damage/
    Infection/
    Wisdom/
  Test/
    Cavities/
    Damage/
    Infection/
    Wisdom/
```

Windows dataset path used in examples:

```text
C:/Users/MOHITH REDDY/Documents/projects/medical-image/Dental OPG Image dataset/Classification/
```

## Setup

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## One-command leakage check (required before training)

```bash
python -m src.data --check-only --data_dir "C:/Users/MOHITH REDDY/Documents/projects/medical-image/Dental OPG Image dataset/Classification/"
```

- SHA256 hash overlap is checked across Train/Valid/Test.
- Report is written to `outputs/leakage_report.txt`.
- If overlap exists, command exits non-zero and training must be stopped.

## One-command training (baseline)

```bash
python -m src.train --config configs/baseline.yaml --data_dir "C:/Users/MOHITH REDDY/Documents/projects/medical-image/Dental OPG Image dataset/Classification/"
```

Outputs are written per run:

```text
outputs/<timestamp>_<model>_<img>_<loss>_<tag>/
  best.keras
  history.csv
  training_curve.png
  run_config.json
  env.json
```

## One-command evaluation

```bash
python -m src.eval --checkpoint outputs/<run_dir>/best.keras --data_dir "C:/Users/MOHITH REDDY/Documents/projects/medical-image/Dental OPG Image dataset/Classification/" --img_size 224 --batch_size 16
```

Evaluation artifacts:

- `report.txt` (per-class precision/recall/F1)
- `confusion_matrix.png`
- `metrics.json` (accuracy, macro F1, weighted F1, per-class F1)

## One-command experiment runner + leaderboard

```bash
python -m src.experiments --data_dir "C:/Users/MOHITH REDDY/Documents/projects/medical-image/Dental OPG Image dataset/Classification/"
```

Runs:
1. Baseline MobileNetV3Large (224)
2. Deeper fine-tune (unfreeze 120)
3. EfficientNetV2B0 (224)
4. Focal loss gamma sweep (1.5, 2.0, 2.5) on best backbone
5. High-res 320 run (batch-safe)
6. Optional TTA evaluation on best run

Leaderboard:

- `outputs/leaderboard.csv` ranked by `test_macro_f1`, then `test_accuracy`.

## Reproducibility

- Global seeds are set for Python, NumPy, TensorFlow.
- Runtime environment is exported in each run (`env.json`): Python, TensorFlow, GPU, CUDA/cuDNN when available.

## GTX 1650 (4GB VRAM) defaults + OOM handling

Default configs are 1650-safe:
- `img_size=224`
- `batch_size=16`
- mixed precision enabled when GPU is available

If OOM occurs during training:
- Pipeline retries **once** automatically with `batch_size=8`.
- If OOM persists, the run fails with a clear error.

## Evidence for faculty

For each run, include these artifacts as evidence:
- `run_config.json` and `env.json` (reproducibility context)
- `history.csv` and `training_curve.png` (training behavior)
- `metrics.json`, `report.txt`, `confusion_matrix.png` (honest evaluation)
- `outputs/leakage_report.txt` (split leakage proof)
- `outputs/leaderboard.csv` (experiment ranking)
