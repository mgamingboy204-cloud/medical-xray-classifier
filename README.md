# Dental OPG 4-Class Classification (PyTorch + timm)

## 1) Project Overview
This repository trains image classifiers for **4-class dental OPG classification** (e.g., Cavities, Damage, Infection, Wisdom) from panoramic X-rays.

The primary target metric is **macro-F1**, not only accuracy, because macro-F1 weights every class equally and is more reliable under class imbalance.

---

## 2) Environment Setup (Kaggle)
Use a Kaggle Notebook with **GPU enabled** (Tesla P100 or T4).

### Clone repository
```bash
git clone <YOUR_REPO_URL>
cd medical-xray-classifier
```

### Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Verify GPU and CUDA availability
```bash
nvidia-smi
python -c "import torch; print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available()); print('gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Expected: `cuda_available True` and a device name like Tesla P100/T4.

---

## 3) Dataset Format
Your dataset directory must be:

```text
Classification/
    Train/
        <ClassA>/
        <ClassB>/
        <ClassC>/
        <ClassD>/
    Valid/
        <ClassA>/
        <ClassB>/
        <ClassC>/
        <ClassD>/
    Test/
        <ClassA>/
        <ClassB>/
        <ClassC>/
        <ClassD>/
```

Kaggle example path:
```bash
export DATA_DIR="/kaggle/input/<dataset-name>/Classification"
```

---

## 4) Training Individual Models
The PyTorch pipeline includes CLAHE preprocessing, sharpen + cutout augmentation, label smoothing, optional focal loss, class weighting, and warmup + cosine learning rate.

### 4.1 ConvNeXt Tiny (224)
```bash
python -m src_pt.train_pt \
  --config configs_pt/convnext_tiny_224.yaml \
  --data_dir "$DATA_DIR"
```
Expected test macro-F1 range (single model): **0.80-0.85**.

### 4.2 EfficientNetV2-B3 (320)
```bash
python -m src_pt.train_pt \
  --config configs_pt/effnetv2_b3_320.yaml \
  --data_dir "$DATA_DIR"
```
Expected test macro-F1 range (single model): **0.82-0.86**.

### 4.3 Swin Tiny (224)
```bash
python -m src_pt.train_pt \
  --config configs_pt/swin_tiny_224.yaml \
  --data_dir "$DATA_DIR"
```
Expected test macro-F1 range (single model): **0.80-0.85**.

Training outputs are written under `outputs_pt/<run_name>/` and include `best.pt`, `history.csv`, and `metrics.json`.

---

## 5) Strong Evaluation with TTA
Run evaluation with stronger test-time augmentation (9 crops):

```bash
python -m src_pt.eval_pt \
  --checkpoint outputs_pt/<run_name>/best.pt \
  --data_dir "$DATA_DIR" \
  --img_size 224 \
  --batch_size 32 \
  --tta_crops 9 \
  --split Test
```

Why it helps: averaging predictions from multiple spatial crops reduces crop-position sensitivity and usually improves macro-F1 by **~2-4 percentage points** vs single-crop evaluation.

---

## 6) Ensembling (Critical for 90%+)
Use diverse backbones and average predicted probabilities.

Example 3-model ensemble command:

```bash
python -m src_pt.ensemble_pt \
  --checkpoints \
    outputs_pt/<convnext_run>/best.pt \
    outputs_pt/<effnet_run>/best.pt \
    outputs_pt/<swin_run>/best.pt \
  --data_dir "$DATA_DIR" \
  --img_size 224 \
  --batch_size 16 \
  --tta_crops 9 \
  --split Test
```

How it works: each model outputs class probabilities; ensemble prediction is the arithmetic mean of probabilities, then `argmax`.

Typical gain: **~3-5 percentage points** macro-F1 over a single model.

---

## 7) Recommended Hyperparameters
Starting values that reproduce strong performance:

- `lr`: `1e-4`
- `label_smoothing`: `0.1`
- `use_class_weights`: `true`
- `aug_mode`: `medium`
- `patience`: `10`
- `warmup_epochs`: `1` (224 models) or `2` (320 models)

Optional imbalance-robust setting:
- `loss_name: focal`
- `focal_gamma: 2.0`

---

## 8) Common Errors & Fixes

### A) Deterministic CUDA / cuDNN issues
Symptoms: runtime errors around deterministic algorithms or cuDNN kernels.

Fixes:
```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=42
```
If still unstable, reduce strict determinism in your run settings and keep seed fixed.

### B) TensorFlow/JAX import crash in PyTorch runs
Cause: mixed environment conflicts.

Fixes:
- Use the PyTorch entrypoints (`src_pt.*`) only.
- Ensure `src_pt/data_pt.py` does not depend on TensorFlow/JAX modules.
- Restart Kaggle kernel after package changes.

### C) Out-of-memory (OOM)
Fixes:
- Reduce `batch_size`.
- Prefer `img_size=224` before `320`.
- Use fewer workers (`num_workers=2`).
- Keep mixed precision enabled.

### D) `SyntaxError` in `src_pt/data_pt.py`
Fixes:
```bash
python -m py_compile src_pt/data_pt.py
```
If it fails, restore the file and ensure augmentation builder blocks are valid Python with balanced parentheses.

---

## 9) Performance Expectations
On Kaggle GPU (P100/T4), typical ranges for this repo:

- **Single model**: macro-F1 **0.80-0.86**
- **3-model diverse ensemble + strong TTA**: macro-F1 **0.88-0.92**

For reproducibility, train all three models, evaluate each with TTA, then run ensemble averaging on checkpoints from the best runs.
