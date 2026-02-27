import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data import build_datasets


def main():
    parser = argparse.ArgumentParser(description="Smoke test for tf.data augmentation graph stability")
    parser.add_argument("--data_dir", required=True, help="Path to Classification dataset root")
    parser.add_argument("--img_size", type=int, default=224, help="Image size")
    parser.add_argument("--batch", type=int, default=2, help="Batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    aug_cfg = {
        "aug_mode": "light",
        "allow_horizontal_flip": False,
        "gaussian_noise": True,
    }

    train_ds, _, _, class_names = build_datasets(
        data_dir=args.data_dir,
        img_size=args.img_size,
        batch_size=args.batch,
        aug_cfg=aug_cfg,
        cache=False,
        seed=args.seed,
    )

    it = iter(train_ds)
    for step in range(2):
        images, labels = next(it)
        print(f"batch={step} images={images.shape} labels={labels.shape} classes={len(class_names)}")

    print("Smoke test passed: no tf.Variable singleton creation crash during augmentation mapping.")


if __name__ == "__main__":
    main()
