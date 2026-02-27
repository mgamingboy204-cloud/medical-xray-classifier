import json
import os
import platform
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf


def set_global_determinism(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def get_env_info() -> dict:
    gpus = tf.config.list_physical_devices("GPU")
    gpu_names = []
    for gpu in gpus:
        try:
            details = tf.config.experimental.get_device_details(gpu)
            gpu_names.append(details.get("device_name", gpu.name))
        except Exception:
            gpu_names.append(gpu.name)

    build_info = tf.sysconfig.get_build_info()
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "tensorflow_version": tf.__version__,
        "gpu_count": len(gpus),
        "gpu_names": gpu_names,
        "cuda_version": build_info.get("cuda_version"),
        "cudnn_version": build_info.get("cudnn_version"),
    }


def json_save(path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def timestamp_run_name(model_name: str, img_size: int, loss_type: str, run_tag: str = "") -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"_{run_tag}" if run_tag else ""
    return f"{ts}_{model_name}_{img_size}_{loss_type}{tag}"
