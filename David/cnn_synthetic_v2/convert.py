"""Convert Synthetic V2 YOLO dataset into a classification dataset of cropped pokemon.

Reads the YOLO-format data (images/*.jpg + labels/*.txt + data.yaml),
crops each bounding box (with optional padding), filters out tiny boxes,
and writes a manifest split by source image (to prevent leakage of multiple
crops from the same background across train/val/test).
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from PIL import Image
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Synthetic V2 YOLO data to classification crops.")
    parser.add_argument("--source", required=True, help="Path to Synthetic dir containing images/, labels/, data.yaml")
    parser.add_argument("--output-dir", default="./data", help="Output dir for crops/ and manifest.csv")
    parser.add_argument("--padding", type=float, default=0.1, help="Bbox padding ratio (0.1 = +10%% on each side)")
    parser.add_argument("--min-bbox", type=int, default=16, help="Drop bboxes with width or height below this many pixels")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    return parser.parse_args()


def load_class_names(data_yaml_path: Path) -> list[str]:
    with data_yaml_path.open("r", encoding="utf-8") as handle:
        spec = yaml.safe_load(handle)
    names = spec.get("names")
    if not names:
        raise ValueError(f"'names' field missing or empty in {data_yaml_path}")
    if isinstance(names, dict):
        names = [names[i] for i in sorted(names.keys())]
    return list(names)


def parse_label_file(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    boxes = []
    with label_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                continue
            cls = int(parts[0])
            cx, cy, w, h = (float(x) for x in parts[1:])
            boxes.append((cls, cx, cy, w, h))
    return boxes


def denormalize_bbox(cx: float, cy: float, w: float, h: float, img_w: int, img_h: int, padding: float) -> tuple[int, int, int, int]:
    bw = w * img_w
    bh = h * img_h
    pad_x = bw * padding
    pad_y = bh * padding
    left = (cx * img_w) - bw / 2.0 - pad_x
    top = (cy * img_h) - bh / 2.0 - pad_y
    right = (cx * img_w) + bw / 2.0 + pad_x
    bottom = (cy * img_h) + bh / 2.0 + pad_y
    left = max(0, int(round(left)))
    top = max(0, int(round(top)))
    right = min(img_w, int(round(right)))
    bottom = min(img_h, int(round(bottom)))
    return left, top, right, bottom


def assign_split(source_ids: list[str], ratios: tuple[float, float, float], seed: int) -> dict[str, str]:
    rng = random.Random(seed)
    shuffled = source_ids[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(round(n * ratios[0]))
    n_val = int(round(n * ratios[1]))
    n_train = max(1, min(n_train, n - 2))
    n_val = max(1, min(n_val, n - n_train - 1))
    assignments: dict[str, str] = {}
    for idx, src_id in enumerate(shuffled):
        if idx < n_train:
            assignments[src_id] = "train"
        elif idx < n_train + n_val:
            assignments[src_id] = "val"
        else:
            assignments[src_id] = "test"
    return assignments


def main() -> None:
    args = parse_args()
    source_dir = Path(args.source).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    images_dir = source_dir / "images"
    labels_dir = source_dir / "labels"
    data_yaml = source_dir / "data.yaml"

    for p in (images_dir, labels_dir, data_yaml):
        if not p.exists():
            raise FileNotFoundError(f"Required path missing: {p}")

    class_names = load_class_names(data_yaml)
    label_to_id = {name: idx for idx, name in enumerate(class_names)}

    crops_dir = output_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    for name in class_names:
        (crops_dir / name).mkdir(parents=True, exist_ok=True)

    label_files = sorted(labels_dir.glob("*.txt"))
    if not label_files:
        raise FileNotFoundError(f"No label files found under {labels_dir}")

    ratios = (args.train_ratio, args.val_ratio, args.test_ratio)
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {ratios}")

    source_ids = [p.stem for p in label_files]
    split_assignment = assign_split(source_ids, ratios, args.seed)

    records: list[dict] = []
    skipped_small = 0
    skipped_invalid = 0
    skipped_missing_image = 0
    skipped_empty = 0

    for label_path in tqdm(label_files, desc="Cropping"):
        src_stem = label_path.stem
        split = split_assignment[src_stem]
        image_path = images_dir / f"{src_stem}.jpg"
        if not image_path.exists():
            for ext in (".png", ".jpeg", ".JPG"):
                alt = images_dir / f"{src_stem}{ext}"
                if alt.exists():
                    image_path = alt
                    break
            else:
                skipped_missing_image += 1
                continue

        boxes = parse_label_file(label_path)
        if not boxes:
            skipped_empty += 1
            continue

        try:
            with Image.open(image_path) as img:
                img = img.convert("RGB")
                img_w, img_h = img.size
                for bbox_idx, (cls, cx, cy, w, h) in enumerate(boxes):
                    if cls < 0 or cls >= len(class_names):
                        skipped_invalid += 1
                        continue
                    left, top, right, bottom = denormalize_bbox(cx, cy, w, h, img_w, img_h, args.padding)
                    if right - left < args.min_bbox or bottom - top < args.min_bbox:
                        skipped_small += 1
                        continue
                    label_name = class_names[cls]
                    crop = img.crop((left, top, right, bottom))
                    out_name = f"{src_stem}_b{bbox_idx:02d}.jpg"
                    rel_out = Path("crops") / label_name / out_name
                    abs_out = output_dir / rel_out
                    crop.save(abs_out, quality=args.jpeg_quality)
                    records.append(
                        {
                            "image_path": rel_out.as_posix(),
                            "label": label_name,
                            "label_id": label_to_id[label_name],
                            "split": split,
                            "source_image": f"{src_stem}.jpg",
                        }
                    )
        except (OSError, ValueError) as exc:
            skipped_invalid += 1
            print(f"WARN: failed to process {image_path}: {exc}")
            continue

    manifest_path = output_dir / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["image_path", "label", "label_id", "split", "source_image"],
        )
        writer.writeheader()
        writer.writerows(records)

    label_map_path = output_dir / "label_to_id.json"
    with label_map_path.open("w", encoding="utf-8") as handle:
        json.dump(label_to_id, handle, ensure_ascii=False, indent=2)

    summary = summarize(records, class_names)
    summary_path = output_dir / "summary.json"
    summary["skipped"] = {
        "small_bboxes": skipped_small,
        "invalid": skipped_invalid,
        "missing_image": skipped_missing_image,
        "empty_label": skipped_empty,
    }
    summary["args"] = {
        "source": str(source_dir),
        "padding": args.padding,
        "min_bbox": args.min_bbox,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "seed": args.seed,
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print("\nDone.")
    print(f"Manifest:     {manifest_path}")
    print(f"Label map:    {label_map_path}")
    print(f"Summary:      {summary_path}")
    print(f"Total crops:  {len(records)}")
    print(f"Skipped (too small):   {skipped_small}")
    print(f"Skipped (invalid):     {skipped_invalid}")
    print(f"Skipped (no image):    {skipped_missing_image}")
    print(f"Skipped (empty label): {skipped_empty}")
    print("\nSplit counts:")
    for split in ("train", "val", "test"):
        print(f"  {split}: {summary['split_counts'][split]}")
    print("\nPer-class counts (total | train | val | test):")
    for name in class_names:
        tot = summary["label_counts"][name]
        tr = summary["split_label_counts"]["train"].get(name, 0)
        va = summary["split_label_counts"]["val"].get(name, 0)
        te = summary["split_label_counts"]["test"].get(name, 0)
        print(f"  {name:12s}  {tot:5d}  |  {tr:5d}  {va:4d}  {te:4d}")


def summarize(records: list[dict], class_names: list[str]) -> dict:
    split_counts = Counter(r["split"] for r in records)
    label_counts = Counter(r["label"] for r in records)
    split_label_counts: dict[str, Counter] = defaultdict(Counter)
    for r in records:
        split_label_counts[r["split"]][r["label"]] += 1
    return {
        "num_crops": len(records),
        "num_classes": len(class_names),
        "class_names": class_names,
        "split_counts": {s: split_counts.get(s, 0) for s in ("train", "val", "test")},
        "label_counts": {name: label_counts.get(name, 0) for name in class_names},
        "split_label_counts": {s: dict(split_label_counts.get(s, Counter())) for s in ("train", "val", "test")},
    }


if __name__ == "__main__":
    main()
