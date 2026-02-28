# Medical X-ray Classifier (Dental OPG, Multi-class)

This repository now provides **two pipelines**:

1. **PyTorch (recommended/default)**: production-ready Kaggle GPU training/evaluation in `src_pt/`.
2. TensorFlow (legacy, kept intact) in `src/`.

The dataset is not stored in Git and must be mounted locally (including Kaggle input mounts).

## Dataset layout (required)

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

## Recommended setup (Kaggle GPU)

Install dependencies in a Kaggle notebook terminal or cell:

```bash
pip install -r requirements.txt
# If Kaggle image does not already include the desired CUDA PyTorch build:
# pip install torch torchvision
# pip install timm albumentations opencv-python
```

> `timm` is the PyTorch image-model collection used for ConvNeXt/EfficientNetV2 backbones.

### Kaggle run workflow

```bash
# 1) clone repository
# 2) install deps
# 3) leakage gate (hard requirement before training)
python -m src.data --check-only --data_dir "/kaggle/input/<dataset>/Classification"

# 4) train PyTorch model
python -m src_pt.train_pt --config configs_pt/convnext_tiny_224.yaml --data_dir "/kaggle/input/<dataset>/Classification"

# 5) evaluate saved best checkpoint on Test split
python -m src_pt.eval_pt --checkpoint outputs_pt/<run_name>/best.pt --data_dir "/kaggle/input/<dataset>/Classification" --img_size 224 --batch_size 32 --tta_crops 5

# 6) full experiment suite + leaderboard
python -m src_pt.experiments_pt --data_dir "/kaggle/input/<dataset>/Classification"
```

## Local workflow

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

python -m src.data --check-only --data_dir "<PATH>/Classification"
python -m src_pt.train_pt --config configs_pt/convnext_tiny_224.yaml --data_dir "<PATH>/Classification"
python -m src_pt.eval_pt --checkpoint outputs_pt/<run_name>/best.pt --data_dir "<PATH>/Classification" --img_size 224 --batch_size 32 --tta_crops 5
```

## PyTorch output artifacts (per run)

```text
outputs_pt/<run_name>/
  best.pt
  last.pt
  history.csv
  training_curve.png
  training_curves.png
  run_config.json
  env.json
  report.txt
  confusion_matrix.png
  metrics.json
```

These artifacts provide faculty-ready evidence for reproducibility and performance reporting.

## Experiment suite in `src_pt.experiments_pt`

The runner executes:

1. `convnext_tiny_224`
2. `effnetv2_b3_320`
3. `dinov2_vits14_224` (freeze + linear head), optional fine-tune on larger GPUs
4. best single model with multi-crop (`tta_crops=5`)
5. top-2 ensemble by probability averaging

Leaderboard output:

- `outputs_pt/leaderboard.csv` ranked by `test_macro_f1`, then `test_accuracy`.

## Leakage policy

The existing SHA256 split-overlap check from `src.data.check_for_leakage` is reused as a hard gate in the PyTorch pipeline as well. Any overlap across Train/Valid/Test aborts training/evaluation.

## TensorFlow legacy pipeline

TensorFlow commands still work exactly as before via `src.train`, `src.eval`, and `src.experiments`.

> Note: native TensorFlow Windows GPU support is not available beyond TF 2.10. For current GPU training, use Kaggle or WSL2/Linux.
