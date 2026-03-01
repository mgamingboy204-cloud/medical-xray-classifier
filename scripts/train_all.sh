#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${1:-}"
if [[ -z "$DATA_DIR" ]]; then
  echo "usage: scripts/train_all.sh /path/to/Classification"
  exit 1
fi

python -m src_pt.train_pt --config configs_pt/effnetv2_l_384.yaml --data_dir "$DATA_DIR"
python -m src_pt.train_pt --config configs_pt/convnext_base_384.yaml --data_dir "$DATA_DIR"
python -m src_pt.train_pt --config configs_pt/swin_base_224.yaml --data_dir "$DATA_DIR"
