"""Split Synthetic V2 YOLO data into train/val/test directories.

Uses the same source-image-level split as cnn_synthetic_v2/convert.py
(seed=42, 80/10/10) so the test sets stay aligned across CNN and YOLO experiments.
"""
from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True, help="Path to V2 Synthetic dir (contains images/, labels/, data.yaml)")
    p.add_argument("--output", default="./yolo_data", help="Where to write train/val/test/")
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--test-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--copy", action="store_true", help="Copy files instead of symlink (slower, more disk)")
    return p.parse_args()


def assign_split(source_ids: list[str], ratios: tuple[float, float, float], seed: int) -> dict[str, str]:
    rng = random.Random(seed)
    shuffled = source_ids[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(round(n * ratios[0]))
    n_val = int(round(n * ratios[1]))
    n_train = max(1, min(n_train, n - 2))
    n_val = max(1, min(n_val, n - n_train - 1))
    out: dict[str, str] = {}
    for idx, src in enumerate(shuffled):
        if idx < n_train:
            out[src] = "train"
        elif idx < n_train + n_val:
            out[src] = "val"
        else:
            out[src] = "test"
    return out


def main() -> None:
    args = parse_args()
    src_dir = Path(args.source).expanduser().resolve()
    out_dir = Path(args.output).expanduser().resolve()

    images_dir = src_dir / "images"
    labels_dir = src_dir / "labels"
    data_yaml = src_dir / "data.yaml"

    for p in (images_dir, labels_dir, data_yaml):
        if not p.exists():
            raise FileNotFoundError(f"Missing required path: {p}")

    with data_yaml.open("r", encoding="utf-8") as h:
        spec = yaml.safe_load(h)
    names = spec.get("names")
    if isinstance(names, dict):
        names = [names[i] for i in sorted(names.keys())]
    if not names:
        raise ValueError(f"data.yaml missing 'names'")
    nc = len(names)
    print(f"Classes ({nc}): {names}")

    label_files = sorted(labels_dir.glob("*.txt"))
    if not label_files:
        raise FileNotFoundError(f"No labels under {labels_dir}")
    source_ids = [p.stem for p in label_files]
    print(f"Found {len(source_ids)} source images")

    ratios = (args.train_ratio, args.val_ratio, args.test_ratio)
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {ratios}")
    split_map = assign_split(source_ids, ratios, args.seed)
    counts = {"train": 0, "val": 0, "test": 0}
    for s in split_map.values():
        counts[s] += 1
    print(f"Split counts: {counts}")

    for split in ("train", "val", "test"):
        (out_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (out_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    skipped = 0
    for label_path in label_files:
        stem = label_path.stem
        split = split_map[stem]

        image_path = images_dir / f"{stem}.jpg"
        if not image_path.exists():
            for ext in (".png", ".jpeg", ".JPG"):
                alt = images_dir / f"{stem}{ext}"
                if alt.exists():
                    image_path = alt
                    break
            else:
                skipped += 1
                continue

        dst_img = out_dir / split / "images" / image_path.name
        dst_lbl = out_dir / split / "labels" / label_path.name

        if args.copy:
            shutil.copy2(image_path, dst_img)
            shutil.copy2(label_path, dst_lbl)
        else:
            try:
                if dst_img.exists() or dst_img.is_symlink():
                    dst_img.unlink()
                if dst_lbl.exists() or dst_lbl.is_symlink():
                    dst_lbl.unlink()
                dst_img.symlink_to(image_path)
                dst_lbl.symlink_to(label_path)
            except OSError:
                shutil.copy2(image_path, dst_img)
                shutil.copy2(label_path, dst_lbl)

    if skipped:
        print(f"Skipped (no matching image): {skipped}")

    final_yaml = {
        "path": str(out_dir),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": nc,
        "names": names,
    }
    yaml_path = out_dir / "data.yaml"
    with yaml_path.open("w", encoding="utf-8") as h:
        yaml.safe_dump(final_yaml, h, sort_keys=False, allow_unicode=True)
    print(f"\nWrote {yaml_path}")
    print(f"\nDone. Splits at:")
    for split in ("train", "val", "test"):
        n_img = sum(1 for _ in (out_dir / split / "images").iterdir())
        n_lbl = sum(1 for _ in (out_dir / split / "labels").iterdir())
        print(f"  {split:5s}  images={n_img}  labels={n_lbl}")


if __name__ == "__main__":
    main()
