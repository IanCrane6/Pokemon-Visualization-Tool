"""Evaluate a YOLO + CNN two-stage pipeline on the V2 test split.

For each test image:
  1. Run YOLO -> get predicted boxes (with classes + scores)
  2. Match each pred box to the closest GT box via IoU >= --iou-match
  3. For TP boxes: crop (with padding) and run CNN classifier -> CNN class prediction
  4. Aggregate metrics:
     - YOLO standalone (mAP from Ultralytics val)
     - YOLO classification accuracy on TP boxes
     - CNN classification accuracy on TP boxes (same crops)
     - Agreement between YOLO and CNN class predictions

Outputs (under --out):
  pipeline_metrics.json
  per_box_predictions.csv
  confusion_matrix_yolo.png
  confusion_matrix_cnn.png
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn as nn
import yaml
from PIL import Image
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torchvision import transforms
from torchvision.models import resnet50
from tqdm import tqdm
from ultralytics import YOLO

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------- CNN model loading (inlined, self-contained) ----------------


def create_model_from_ckpt(ckpt_path: Path, device: torch.device):
    """Reconstruct a CNN model + transform + label map from a cnn_synthetic_v2 checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    label_to_id: dict[str, int] = ckpt["label_to_id"]
    cfg = ckpt["config"]
    model_name = cfg["model"]["name"]
    image_size = int(cfg["train"].get("image_size", 224))
    num_classes = len(label_to_id)

    if model_name.lower() == "resnet50":
        model = resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        data_config = None
    else:
        import timm
        model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
        data_config = timm.data.resolve_model_data_config(model)

    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()

    if data_config is not None:
        import timm
        cfg_t = dict(data_config)
        cfg_t["input_size"] = (cfg_t.get("input_size", (3, image_size, image_size))[0], image_size, image_size)
        transform = timm.data.create_transform(**cfg_t, is_training=False)
    else:
        normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        transform = transforms.Compose(
            [
                transforms.Resize(int(image_size * 1.15)),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                normalize,
            ]
        )

    id_to_label = {v: k for k, v in label_to_id.items()}
    return model, transform, label_to_id, id_to_label, model_name, image_size


# ---------------- IoU + crop utils ----------------


def iou_xyxy(box_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Single box vs N boxes, all in xyxy pixel coords."""
    if len(boxes_b) == 0:
        return np.zeros((0,), dtype=np.float32)
    x1 = np.maximum(box_a[0], boxes_b[:, 0])
    y1 = np.maximum(box_a[1], boxes_b[:, 1])
    x2 = np.minimum(box_a[2], boxes_b[:, 2])
    y2 = np.minimum(box_a[3], boxes_b[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = max(0.0, (box_a[2] - box_a[0])) * max(0.0, (box_a[3] - box_a[1]))
    area_b = np.clip(boxes_b[:, 2] - boxes_b[:, 0], 0, None) * np.clip(boxes_b[:, 3] - boxes_b[:, 1], 0, None)
    union = area_a + area_b - inter + 1e-9
    return inter / union


def parse_yolo_label(label_path: Path, img_w: int, img_h: int) -> tuple[np.ndarray, np.ndarray]:
    """Read a YOLO label file -> (boxes_xyxy, classes) in pixel coords."""
    boxes: list[list[float]] = []
    classes: list[int] = []
    if not label_path.exists():
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    with label_path.open("r", encoding="utf-8") as h:
        for line in h:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls = int(parts[0])
            cx, cy, w, h_n = (float(x) for x in parts[1:])
            x1 = (cx - w / 2) * img_w
            y1 = (cy - h_n / 2) * img_h
            x2 = (cx + w / 2) * img_w
            y2 = (cy + h_n / 2) * img_h
            boxes.append([x1, y1, x2, y2])
            classes.append(cls)
    return np.array(boxes, dtype=np.float32), np.array(classes, dtype=np.int64)


def crop_with_padding(img: Image.Image, box_xyxy: np.ndarray, padding: float = 0.1) -> Image.Image:
    x1, y1, x2, y2 = box_xyxy
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    pad_x = w * padding
    pad_y = h * padding
    x1p = max(0, int(round(x1 - pad_x)))
    y1p = max(0, int(round(y1 - pad_y)))
    x2p = min(img.width, int(round(x2 + pad_x)))
    y2p = min(img.height, int(round(y2 + pad_y)))
    if x2p <= x1p or y2p <= y1p:
        return None
    return img.crop((x1p, y1p, x2p, y2p))


# ---------------- main eval ----------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--yolo", required=True, help="Path to YOLO best.pt")
    p.add_argument("--cnn", required=True, help="Path to CNN best.pt (from cnn_synthetic_v2)")
    p.add_argument("--data", required=True, help="Path to YOLO data.yaml")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    p.add_argument("--iou-match", type=float, default=0.5, help="IoU threshold for TP matching")
    p.add_argument("--padding", type=float, default=0.1, help="Crop padding for CNN")
    p.add_argument("--out", required=True, help="Output dir")
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-images", type=int, default=0, help="0 = all; >0 = truncate (smoke)")
    return p.parse_args()


def plot_cm(cm: np.ndarray, labels: list[str], path: Path, title: str) -> None:
    size = max(8, min(20, len(labels) * 0.8))
    fig, ax = plt.subplots(figsize=(size, size))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    cm_max = cm.max() if cm.size > 0 else 1
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm_max / 2 else "black", fontsize=8)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def metrics_block(y_true: list[int], y_pred: list[int], num_classes: int, id_to_label: dict[int, str]) -> dict:
    if len(y_true) == 0:
        return {"accuracy": 0.0, "macro_precision": 0.0, "macro_recall": 0.0, "macro_f1": 0.0,
                "per_class": {}, "n": 0}
    labels_range = list(range(num_classes))
    acc = float(np.mean(np.array(y_true) == np.array(y_pred)))
    p_m, r_m, f1_m, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels_range, average="macro", zero_division=0)
    p_p, r_p, f1_p, sup = precision_recall_fscore_support(y_true, y_pred, labels=labels_range, average=None, zero_division=0)
    per = {id_to_label[i]: {"precision": float(p_p[i]), "recall": float(r_p[i]), "f1": float(f1_p[i]),
                             "support": int(sup[i])} for i in labels_range}
    return {"accuracy": acc, "macro_precision": float(p_m), "macro_recall": float(r_m),
            "macro_f1": float(f1_m), "per_class": per, "n": len(y_true)}


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    print(f"Device: {device}")

    data_path = Path(args.data).resolve()
    with data_path.open("r", encoding="utf-8") as h:
        data_spec = yaml.safe_load(h)
    class_names = data_spec.get("names", [])
    if isinstance(class_names, dict):
        class_names = [class_names[i] for i in sorted(class_names.keys())]
    num_classes = len(class_names)
    cls_id_to_name = {i: n for i, n in enumerate(class_names)}
    cls_name_to_id = {n: i for i, n in cls_id_to_name.items()}
    print(f"Classes ({num_classes}): {class_names}")

    print(f"\nLoading YOLO: {args.yolo}")
    yolo = YOLO(args.yolo)

    print(f"\nLoading CNN: {args.cnn}")
    cnn, cnn_transform, cnn_label_to_id, cnn_id_to_label, cnn_model_name, cnn_image_size = create_model_from_ckpt(
        Path(args.cnn), device
    )
    print(f"  CNN model: {cnn_model_name}  image_size={cnn_image_size}")
    print(f"  CNN labels: {list(cnn_label_to_id.keys())}")

    data_root = Path(data_spec.get("path", str(data_path.parent)))
    if not data_root.is_absolute():
        data_root = data_path.parent / data_root
    split_img_subdir = data_spec.get(args.split, f"{args.split}/images")
    split_img_dir = (data_root / split_img_subdir).resolve()
    split_lbl_dir = (split_img_dir.parent / "labels").resolve()
    print(f"\nSplit images: {split_img_dir}")
    print(f"Split labels: {split_lbl_dir}")

    image_files = sorted([p for p in split_img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    if args.max_images and args.max_images > 0:
        image_files = image_files[: args.max_images]
    print(f"Eval images: {len(image_files)}")

    print("\n=== Stage 1: YOLO standalone metrics (Ultralytics val) ===")
    yolo_val = yolo.val(data=str(data_path), split=args.split)
    yolo_only = {
        "map50": float(yolo_val.box.map50),
        "map50_95": float(yolo_val.box.map),
        "mp": float(yolo_val.box.mp),
        "mr": float(yolo_val.box.mr),
    }
    print(f"YOLO mAP50={yolo_only['map50']:.4f}  mAP50-95={yolo_only['map50_95']:.4f}  P={yolo_only['mp']:.4f}  R={yolo_only['mr']:.4f}")

    print(f"\n=== Stage 2: per-image YOLO predict + IoU match + CNN classify ===")
    rows: list[dict] = []
    yolo_true_on_tp: list[int] = []
    yolo_pred_on_tp: list[int] = []
    cnn_true_on_tp: list[int] = []
    cnn_pred_on_tp: list[int] = []
    n_pred_total = 0
    n_tp = 0
    n_fp = 0
    n_fn = 0

    for img_path in tqdm(image_files):
        try:
            img_pil = Image.open(img_path).convert("RGB")
        except Exception:
            continue
        W, H = img_pil.size
        lbl_path = split_lbl_dir / f"{img_path.stem}.txt"
        gt_boxes, gt_classes = parse_yolo_label(lbl_path, W, H)

        results = yolo.predict(source=str(img_path), conf=args.conf, verbose=False, device=device.type)
        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            n_fn += len(gt_classes)
            continue
        pred_boxes = result.boxes.xyxy.cpu().numpy()
        pred_classes = result.boxes.cls.cpu().numpy().astype(np.int64)
        pred_conf = result.boxes.conf.cpu().numpy()
        n_pred_total += len(pred_boxes)

        gt_used = np.zeros(len(gt_classes), dtype=bool)
        order = np.argsort(-pred_conf)
        crops_for_cnn: list[Image.Image] = []
        crop_meta: list[dict] = []

        for idx in order:
            box = pred_boxes[idx]
            yolo_cls = int(pred_classes[idx])
            score = float(pred_conf[idx])
            ious = iou_xyxy(box, gt_boxes) if len(gt_boxes) else np.zeros((0,), dtype=np.float32)
            if len(ious) == 0:
                best_iou = 0.0
                best_gt = -1
            else:
                masked_ious = ious.copy()
                masked_ious[gt_used] = -1.0
                best_gt = int(masked_ious.argmax())
                best_iou = float(masked_ious[best_gt])
            is_tp = best_iou >= args.iou_match and best_gt >= 0 and not gt_used[best_gt]
            gt_class = int(gt_classes[best_gt]) if (best_gt >= 0 and len(gt_classes) > 0) else -1

            row = {
                "image": img_path.name,
                "yolo_cls_id": yolo_cls,
                "yolo_cls_name": cls_id_to_name.get(yolo_cls, str(yolo_cls)),
                "yolo_conf": score,
                "iou_to_gt": best_iou if best_gt >= 0 else 0.0,
                "is_tp": int(is_tp),
                "gt_cls_id": gt_class if is_tp else -1,
                "gt_cls_name": cls_id_to_name.get(gt_class, "") if is_tp else "",
                "cnn_cls_id": -1,
                "cnn_cls_name": "",
                "cnn_conf": 0.0,
            }

            if is_tp:
                gt_used[best_gt] = True
                n_tp += 1
                crop = crop_with_padding(img_pil, box, padding=args.padding)
                if crop is None:
                    rows.append(row)
                    yolo_true_on_tp.append(gt_class)
                    yolo_pred_on_tp.append(yolo_cls)
                    cnn_true_on_tp.append(gt_class)
                    cnn_pred_on_tp.append(-1)
                    continue
                crops_for_cnn.append(crop)
                crop_meta.append((row, gt_class, yolo_cls))
            else:
                n_fp += 1
                rows.append(row)

        if crops_for_cnn:
            tensors = torch.stack([cnn_transform(c) for c in crops_for_cnn]).to(device)
            with torch.no_grad():
                logits = cnn(tensors)
                probs = torch.softmax(logits, dim=1)
                cnn_pred = logits.argmax(dim=1).cpu().tolist()
                cnn_top_p = probs.max(dim=1).values.cpu().tolist()
            for (row, gt_class, yolo_cls), pred_cnn_local, p in zip(crop_meta, cnn_pred, cnn_top_p):
                cnn_global_id = -1
                cnn_label = cnn_id_to_label.get(int(pred_cnn_local), "")
                if cnn_label in cls_name_to_id:
                    cnn_global_id = cls_name_to_id[cnn_label]
                row["cnn_cls_id"] = cnn_global_id
                row["cnn_cls_name"] = cnn_label
                row["cnn_conf"] = float(p)
                rows.append(row)
                yolo_true_on_tp.append(gt_class)
                yolo_pred_on_tp.append(yolo_cls)
                cnn_true_on_tp.append(gt_class)
                cnn_pred_on_tp.append(cnn_global_id if cnn_global_id >= 0 else -1)

        n_fn += int((~gt_used).sum())

    valid_pairs_cnn = [(t, p) for t, p in zip(cnn_true_on_tp, cnn_pred_on_tp) if p >= 0]
    cnn_true_clean = [t for t, _ in valid_pairs_cnn]
    cnn_pred_clean = [p for _, p in valid_pairs_cnn]

    yolo_cls_metrics = metrics_block(yolo_true_on_tp, yolo_pred_on_tp, num_classes, cls_id_to_name)
    cnn_cls_metrics = metrics_block(cnn_true_clean, cnn_pred_clean, num_classes, cls_id_to_name)

    if len(yolo_pred_on_tp) > 0:
        agree_pairs = [(y, c) for y, c in zip(yolo_pred_on_tp, cnn_pred_on_tp) if c >= 0]
        if agree_pairs:
            agreement = float(np.mean([y == c for y, c in agree_pairs]))
        else:
            agreement = 0.0
    else:
        agreement = 0.0

    summary = {
        "yolo_ckpt": str(Path(args.yolo).resolve()),
        "cnn_ckpt": str(Path(args.cnn).resolve()),
        "cnn_model": cnn_model_name,
        "cnn_image_size": cnn_image_size,
        "data": str(data_path),
        "split": args.split,
        "conf_threshold": args.conf,
        "iou_match": args.iou_match,
        "n_images": len(image_files),
        "n_pred_boxes": int(n_pred_total),
        "n_tp": int(n_tp),
        "n_fp": int(n_fp),
        "n_fn": int(n_fn),
        "yolo_only": yolo_only,
        "yolo_classification_on_tp": yolo_cls_metrics,
        "cnn_classification_on_tp": cnn_cls_metrics,
        "agreement_yolo_vs_cnn_on_tp": agreement,
    }

    with (out_dir / "pipeline_metrics.json").open("w", encoding="utf-8") as h:
        json.dump(summary, h, ensure_ascii=False, indent=2)

    csv_fields = ["image", "is_tp", "iou_to_gt", "gt_cls_name", "yolo_cls_name", "yolo_conf",
                   "cnn_cls_name", "cnn_conf", "gt_cls_id", "yolo_cls_id", "cnn_cls_id"]
    with (out_dir / "per_box_predictions.csv").open("w", encoding="utf-8", newline="") as h:
        w = csv.DictWriter(h, fieldnames=csv_fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in csv_fields})

    if yolo_true_on_tp:
        cm_yolo = confusion_matrix(yolo_true_on_tp, yolo_pred_on_tp, labels=list(range(num_classes)))
        plot_cm(cm_yolo, class_names, out_dir / "confusion_matrix_yolo.png", "YOLO classification on TP boxes")
    if cnn_true_clean:
        cm_cnn = confusion_matrix(cnn_true_clean, cnn_pred_clean, labels=list(range(num_classes)))
        plot_cm(cm_cnn, class_names, out_dir / "confusion_matrix_cnn.png", "CNN classification on TP boxes")

    print(f"\n=== Summary ===")
    print(f"  Images:          {len(image_files)}")
    print(f"  YOLO predictions: {n_pred_total}")
    print(f"  TP / FP / FN:    {n_tp} / {n_fp} / {n_fn}")
    print(f"  YOLO mAP50:      {yolo_only['map50']:.4f}")
    print(f"  YOLO cls-on-TP:  acc={yolo_cls_metrics['accuracy']:.4f}  macro_f1={yolo_cls_metrics['macro_f1']:.4f}")
    print(f"  CNN  cls-on-TP:  acc={cnn_cls_metrics['accuracy']:.4f}  macro_f1={cnn_cls_metrics['macro_f1']:.4f}")
    print(f"  Agreement:       {agreement:.4f}")
    print(f"\nOutputs in: {out_dir}")


if __name__ == "__main__":
    main()
