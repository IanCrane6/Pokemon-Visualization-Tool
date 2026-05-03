"""Cross-domain evaluation on real Pokemon Snap frames (Roboflow data).

Runs three families of evaluation in one go:
  1. CNN-only on GT crops  (4 CNNs)  - real-domain classification with perfect localization
  2. YOLO-only on full frames (2 YOLOs) - real-domain detection (Ultralytics val)
  3. YOLO+CNN pipeline (2 YOLO * N CNN) - end-to-end real-domain accuracy

Outputs (under --out):
  cross_domain_metrics.json
  cross_domain_summary.md           (paper-ready markdown table)
  cnn_only_<cnn>_predictions.csv
  pipeline_<yolo>_<cnn>_predictions.csv
  confusion_matrix_*.png
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from ultralytics import YOLO

from eval_pipeline import (
    create_model_from_ckpt,
    crop_with_padding,
    iou_xyxy,
    metrics_block,
    parse_yolo_label,
    plot_cm,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--roboflow-dir", required=True, help="Roboflow data dir (contains data.yaml + train/)")
    p.add_argument("--cnn-dir", required=True, help="Dir with cnn_<name>_best.pt files")
    p.add_argument("--yolo-runs-dir", required=True, help="Dir containing runs/<exp>/weights/best.pt")
    p.add_argument("--out", required=True, help="Output dir")
    p.add_argument("--cnn-names", nargs="+", default=["resnet50", "dinov3", "aimv2", "siglip2"],
                   help="CNN names to evaluate in CNN-only stage")
    p.add_argument("--yolo-names", nargs="+", default=["yolov8n_v2", "yolo11n_v2"])
    p.add_argument("--pipeline-cnns", nargs="+", default=["aimv2", "dinov3"],
                   help="Subset of cnn-names to use for pipeline stage")
    p.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    p.add_argument("--iou-match", type=float, default=0.5, help="IoU threshold for TP matching")
    p.add_argument("--padding", type=float, default=0.1, help="Crop padding ratio for CNN")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=32)
    return p.parse_args()


def build_temp_data_yaml(roboflow_dir: Path, out_path: Path) -> dict:
    """Write Ultralytics-compatible data.yaml that points val to roboflow's train/."""
    with (roboflow_dir / "data.yaml").open("r", encoding="utf-8") as h:
        spec = yaml.safe_load(h)
    names = spec.get("names")
    if isinstance(names, dict):
        names = [names[i] for i in sorted(names.keys())]

    new_spec = {
        "path": str(roboflow_dir.resolve()),
        "train": "train/images",
        "val": "train/images",
        "test": "train/images",
        "nc": len(names),
        "names": names,
    }
    with out_path.open("w", encoding="utf-8") as h:
        yaml.safe_dump(new_spec, h, sort_keys=False, allow_unicode=True)
    return new_spec


def cnn_only_eval(cnn_ckpt: Path, roboflow_dir: Path, class_names: list[str],
                  name_to_id: dict[str, int], padding: float, device: torch.device, batch_size: int):
    model, transform, _, id_to_label, model_name, _ = create_model_from_ckpt(cnn_ckpt, device)

    images_dir = roboflow_dir / "train" / "images"
    labels_dir = roboflow_dir / "train" / "labels"

    crops: list[Image.Image] = []
    true_classes: list[int] = []
    image_meta: list[tuple[str, int]] = []

    image_files = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    for img_path in tqdm(image_files, desc=f"CNN-only {model_name}"):
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            continue
        W, H = img.size
        lbl_path = labels_dir / f"{img_path.stem}.txt"
        gt_boxes, gt_classes = parse_yolo_label(lbl_path, W, H)
        for box, cls in zip(gt_boxes, gt_classes):
            crop = crop_with_padding(img, box, padding=padding)
            if crop is None:
                continue
            crops.append(crop)
            true_classes.append(int(cls))
            image_meta.append((img_path.name, int(cls)))

    if not crops:
        del model
        torch.cuda.empty_cache()
        return ({"error": "no GT boxes"}, [], [], [])

    pred_classes: list[int] = []
    pred_confs: list[float] = []
    for i in range(0, len(crops), batch_size):
        batch = crops[i: i + batch_size]
        tensors = torch.stack([transform(c) for c in batch]).to(device)
        with torch.no_grad():
            logits = model(tensors)
            probs = torch.softmax(logits, dim=1)
            preds_local = logits.argmax(dim=1).cpu().tolist()
            top_p = probs.max(dim=1).values.cpu().tolist()
        for p_local, conf in zip(preds_local, top_p):
            cnn_class_name = id_to_label.get(int(p_local), "")
            cnn_global_id = name_to_id.get(cnn_class_name, -1)
            pred_classes.append(cnn_global_id)
            pred_confs.append(float(conf))

    valid_pairs = [(t, p) for t, p in zip(true_classes, pred_classes) if p >= 0]
    y_true = [t for t, _ in valid_pairs]
    y_pred = [p for _, p in valid_pairs]
    metrics = metrics_block(y_true, y_pred, len(class_names), {i: n for i, n in enumerate(class_names)})
    metrics["model"] = model_name

    rows: list[dict] = []
    for (img, true_cls), pred_cls, conf in zip(image_meta, pred_classes, pred_confs):
        rows.append(
            {
                "image": img,
                "true_id": true_cls,
                "true_name": class_names[true_cls],
                "pred_id": pred_cls,
                "pred_name": class_names[pred_cls] if pred_cls >= 0 else "",
                "conf": conf,
                "correct": int(pred_cls == true_cls),
            }
        )

    del model
    torch.cuda.empty_cache()
    return metrics, rows, y_true, y_pred


def yolo_only_eval(yolo_ckpt: Path, temp_yaml: Path, class_names: list[str]) -> dict:
    yolo = YOLO(str(yolo_ckpt))
    metrics = yolo.val(data=str(temp_yaml), split="val", verbose=False)
    box = metrics.box
    out = {
        "map50": float(box.map50),
        "map50_95": float(box.map),
        "mp": float(box.mp),
        "mr": float(box.mr),
    }
    per_class: dict = {}
    try:
        ap50 = box.ap50
        ap50_95 = box.ap
        for i, name in enumerate(class_names):
            per_class[name] = {
                "ap50": float(ap50[i]) if i < len(ap50) else None,
                "ap50_95": float(ap50_95[i]) if i < len(ap50_95) else None,
            }
    except Exception as exc:
        per_class["_error"] = f"{type(exc).__name__}: {exc}"
    out["per_class"] = per_class
    return out


def pipeline_eval(yolo_ckpt: Path, cnn_ckpt: Path, roboflow_dir: Path, class_names: list[str],
                  conf_th: float, iou_match: float, padding: float, device: torch.device,
                  name_to_id: dict[str, int]):
    yolo = YOLO(str(yolo_ckpt))
    cnn, cnn_transform, _, cnn_id_to_label, cnn_model_name, _ = create_model_from_ckpt(cnn_ckpt, device)

    images_dir = roboflow_dir / "train" / "images"
    labels_dir = roboflow_dir / "train" / "labels"
    image_files = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])

    rows: list[dict] = []
    yolo_true_on_tp: list[int] = []
    yolo_pred_on_tp: list[int] = []
    cnn_true_on_tp: list[int] = []
    cnn_pred_on_tp: list[int] = []
    n_pred = n_tp = n_fp = n_fn = 0

    yolo_run_name = Path(yolo_ckpt).parent.parent.name
    desc = f"Pipeline {yolo_run_name}+{cnn_model_name}"

    for img_path in tqdm(image_files, desc=desc):
        try:
            img_pil = Image.open(img_path).convert("RGB")
        except Exception:
            continue
        W, H = img_pil.size
        lbl_path = labels_dir / f"{img_path.stem}.txt"
        gt_boxes, gt_classes = parse_yolo_label(lbl_path, W, H)

        results = yolo.predict(source=str(img_path), conf=conf_th, verbose=False,
                               device=device.type if hasattr(device, "type") else device)
        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            n_fn += len(gt_classes)
            continue
        pred_boxes = result.boxes.xyxy.cpu().numpy()
        pred_classes = result.boxes.cls.cpu().numpy().astype(np.int64)
        pred_conf = result.boxes.conf.cpu().numpy()
        n_pred += len(pred_boxes)

        gt_used = np.zeros(len(gt_classes), dtype=bool)
        order = np.argsort(-pred_conf)
        crops_for_cnn: list[Image.Image] = []
        crop_meta: list[tuple] = []

        for idx in order:
            box = pred_boxes[idx]
            yolo_cls = int(pred_classes[idx])
            score = float(pred_conf[idx])
            ious = iou_xyxy(box, gt_boxes) if len(gt_boxes) else np.zeros((0,), dtype=np.float32)
            if len(ious) == 0:
                best_iou = 0.0
                best_gt = -1
            else:
                masked = ious.copy()
                masked[gt_used] = -1.0
                best_gt = int(masked.argmax())
                best_iou = float(masked[best_gt])
            is_tp = best_iou >= iou_match and best_gt >= 0 and not gt_used[best_gt]
            gt_class = int(gt_classes[best_gt]) if (best_gt >= 0 and len(gt_classes) > 0) else -1

            row = {
                "image": img_path.name,
                "yolo_cls_id": yolo_cls,
                "yolo_cls_name": class_names[yolo_cls] if 0 <= yolo_cls < len(class_names) else str(yolo_cls),
                "yolo_conf": score,
                "iou_to_gt": best_iou if best_gt >= 0 else 0.0,
                "is_tp": int(is_tp),
                "gt_cls_id": gt_class if is_tp else -1,
                "gt_cls_name": class_names[gt_class] if (is_tp and 0 <= gt_class < len(class_names)) else "",
                "cnn_cls_id": -1,
                "cnn_cls_name": "",
                "cnn_conf": 0.0,
            }

            if is_tp:
                gt_used[best_gt] = True
                n_tp += 1
                crop = crop_with_padding(img_pil, box, padding=padding)
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
                cnn_label = cnn_id_to_label.get(int(pred_cnn_local), "")
                cnn_global_id = name_to_id.get(cnn_label, -1)
                row["cnn_cls_id"] = cnn_global_id
                row["cnn_cls_name"] = cnn_label
                row["cnn_conf"] = float(p)
                rows.append(row)
                yolo_true_on_tp.append(gt_class)
                yolo_pred_on_tp.append(yolo_cls)
                cnn_true_on_tp.append(gt_class)
                cnn_pred_on_tp.append(cnn_global_id if cnn_global_id >= 0 else -1)

        n_fn += int((~gt_used).sum())

    del cnn
    torch.cuda.empty_cache()

    valid_pairs_cnn = [(t, p) for t, p in zip(cnn_true_on_tp, cnn_pred_on_tp) if p >= 0]
    cnn_true_clean = [t for t, _ in valid_pairs_cnn]
    cnn_pred_clean = [p for _, p in valid_pairs_cnn]

    yolo_cls_metrics = metrics_block(yolo_true_on_tp, yolo_pred_on_tp, len(class_names),
                                      {i: n for i, n in enumerate(class_names)})
    cnn_cls_metrics = metrics_block(cnn_true_clean, cnn_pred_clean, len(class_names),
                                     {i: n for i, n in enumerate(class_names)})

    if yolo_pred_on_tp:
        agree_pairs = [(y, c) for y, c in zip(yolo_pred_on_tp, cnn_pred_on_tp) if c >= 0]
        agreement = float(np.mean([y == c for y, c in agree_pairs])) if agree_pairs else 0.0
    else:
        agreement = 0.0

    summary = {
        "n_pred": n_pred,
        "n_tp": n_tp,
        "n_fp": n_fp,
        "n_fn": n_fn,
        "yolo_classification_on_tp": yolo_cls_metrics,
        "cnn_classification_on_tp": cnn_cls_metrics,
        "agreement_yolo_vs_cnn_on_tp": agreement,
    }
    return summary, rows, yolo_true_on_tp, yolo_pred_on_tp, cnn_true_clean, cnn_pred_clean


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    print(f"Device: {device}")

    roboflow_dir = Path(args.roboflow_dir).resolve()
    cnn_dir = Path(args.cnn_dir).resolve()
    yolo_runs_dir = Path(args.yolo_runs_dir).resolve()

    temp_yaml = out_dir / "_cross_domain_data.yaml"
    spec = build_temp_data_yaml(roboflow_dir, temp_yaml)
    class_names = spec["names"]
    name_to_id = {n: i for i, n in enumerate(class_names)}
    print(f"Classes ({len(class_names)}): {class_names}")
    print(f"Roboflow images: {roboflow_dir / 'train' / 'images'}")
    print(f"CNN dir:         {cnn_dir}")
    print(f"YOLO runs dir:   {yolo_runs_dir}")
    print(f"Output dir:      {out_dir}")

    summary = {
        "roboflow_dir": str(roboflow_dir),
        "class_names": class_names,
        "config": {
            "conf": args.conf,
            "iou_match": args.iou_match,
            "padding": args.padding,
        },
        "cnn_only": {},
        "yolo_only": {},
        "pipeline": {},
    }

    print("\n" + "=" * 60)
    print("Stage 1: CNN-only on GT crops (real-domain classification)")
    print("=" * 60)
    for cnn_name in args.cnn_names:
        ckpt = cnn_dir / f"cnn_{cnn_name}_best.pt"
        if not ckpt.exists():
            print(f"\n  SKIP {cnn_name}: ckpt missing at {ckpt}")
            continue
        print(f"\n  -> {cnn_name}")
        metrics, rows, y_true, y_pred = cnn_only_eval(
            ckpt, roboflow_dir, class_names, name_to_id, args.padding, device, args.batch_size
        )
        summary["cnn_only"][cnn_name] = metrics

        csv_path = out_dir / f"cnn_only_{cnn_name}_predictions.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as h:
            w = csv.DictWriter(h, fieldnames=["image", "true_id", "true_name", "pred_id", "pred_name", "conf", "correct"])
            w.writeheader()
            w.writerows(rows)

        if y_true:
            cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
            plot_cm(cm, class_names, out_dir / f"confusion_matrix_cnn_{cnn_name}.png",
                    f"CNN-only ({cnn_name}) on real-domain GT crops")
        if "accuracy" in metrics:
            print(f"     acc={metrics['accuracy']:.4f}  macro_f1={metrics['macro_f1']:.4f}  n={metrics['n']}")

    print("\n" + "=" * 60)
    print("Stage 2: YOLO-only on full real-domain frames")
    print("=" * 60)
    for yolo_name in args.yolo_names:
        ckpt = yolo_runs_dir / yolo_name / "weights" / "best.pt"
        if not ckpt.exists():
            print(f"\n  SKIP {yolo_name}: ckpt missing at {ckpt}")
            continue
        print(f"\n  -> {yolo_name}")
        metrics = yolo_only_eval(ckpt, temp_yaml, class_names)
        summary["yolo_only"][yolo_name] = metrics
        print(f"     mAP50={metrics['map50']:.4f}  mAP50-95={metrics['map50_95']:.4f}  "
              f"P={metrics['mp']:.4f}  R={metrics['mr']:.4f}")

    print("\n" + "=" * 60)
    print("Stage 3: YOLO + CNN pipeline on real-domain frames")
    print("=" * 60)
    for yolo_name in args.yolo_names:
        yolo_ckpt = yolo_runs_dir / yolo_name / "weights" / "best.pt"
        if not yolo_ckpt.exists():
            continue
        for cnn_name in args.pipeline_cnns:
            cnn_ckpt = cnn_dir / f"cnn_{cnn_name}_best.pt"
            if not cnn_ckpt.exists():
                print(f"\n  SKIP pipeline {yolo_name}+{cnn_name}: cnn ckpt missing")
                continue
            combo = f"{yolo_name}__{cnn_name}"
            print(f"\n  -> {combo}")
            p_metrics, rows, yt, yp, ct, cp = pipeline_eval(
                yolo_ckpt, cnn_ckpt, roboflow_dir, class_names,
                args.conf, args.iou_match, args.padding, device, name_to_id,
            )
            summary["pipeline"][combo] = p_metrics

            csv_path = out_dir / f"pipeline_{combo}_predictions.csv"
            fields = ["image", "is_tp", "iou_to_gt", "gt_cls_name", "yolo_cls_name", "yolo_conf",
                      "cnn_cls_name", "cnn_conf", "gt_cls_id", "yolo_cls_id", "cnn_cls_id"]
            with csv_path.open("w", encoding="utf-8", newline="") as h:
                w = csv.DictWriter(h, fieldnames=fields)
                w.writeheader()
                for r in rows:
                    w.writerow({k: r.get(k, "") for k in fields})

            if yt:
                cm = confusion_matrix(yt, yp, labels=list(range(len(class_names))))
                plot_cm(cm, class_names, out_dir / f"confusion_matrix_pipeline_{combo}_yolo.png",
                        f"Pipeline {combo} - YOLO classification on TP")
            if ct:
                cm = confusion_matrix(ct, cp, labels=list(range(len(class_names))))
                plot_cm(cm, class_names, out_dir / f"confusion_matrix_pipeline_{combo}_cnn.png",
                        f"Pipeline {combo} - CNN classification on TP")

            print(f"     TP/FP/FN: {p_metrics['n_tp']}/{p_metrics['n_fp']}/{p_metrics['n_fn']}")
            print(f"     YOLO cls on TP: acc={p_metrics['yolo_classification_on_tp']['accuracy']:.4f}")
            print(f"     CNN  cls on TP: acc={p_metrics['cnn_classification_on_tp']['accuracy']:.4f}")
            print(f"     Agreement:      {p_metrics['agreement_yolo_vs_cnn_on_tp']:.4f}")

    with (out_dir / "cross_domain_metrics.json").open("w", encoding="utf-8") as h:
        json.dump(summary, h, ensure_ascii=False, indent=2)

    md = ["# Cross-Domain Evaluation Summary", ""]
    md.append(f"- Roboflow images: {roboflow_dir.name}")
    md.append(f"- Classes ({len(class_names)}): {', '.join(class_names)}")
    md.append(f"- Config: conf={args.conf}, iou_match={args.iou_match}, padding={args.padding}")
    md.append("")
    md.append("## 1. CNN-only on GT crops")
    md.append("")
    md.append("(Real-domain classification accuracy assuming perfect localization)")
    md.append("")
    md.append("| CNN | Accuracy | Macro F1 | Macro Precision | Macro Recall | n |")
    md.append("|---|---|---|---|---|---|")
    for cnn_name, m in summary["cnn_only"].items():
        if "accuracy" not in m:
            md.append(f"| {cnn_name} | error | - | - | - | - |")
            continue
        md.append(f"| {cnn_name} | {m['accuracy']:.4f} | {m['macro_f1']:.4f} | "
                  f"{m['macro_precision']:.4f} | {m['macro_recall']:.4f} | {m['n']} |")
    md.append("")
    md.append("## 2. YOLO-only on full frames")
    md.append("")
    md.append("| YOLO | mAP50 | mAP50-95 | Precision | Recall |")
    md.append("|---|---|---|---|---|")
    for yolo_name, m in summary["yolo_only"].items():
        md.append(f"| {yolo_name} | {m['map50']:.4f} | {m['map50_95']:.4f} | {m['mp']:.4f} | {m['mr']:.4f} |")
    md.append("")
    md.append("## 3. YOLO + CNN pipeline (end-to-end)")
    md.append("")
    md.append("| YOLO | CNN | TP | FP | FN | YOLO_cls_acc | CNN_cls_acc | agreement |")
    md.append("|---|---|---|---|---|---|---|---|")
    for combo, m in summary["pipeline"].items():
        yolo_name, cnn_name = combo.split("__", 1)
        md.append(
            f"| {yolo_name} | {cnn_name} | {m['n_tp']} | {m['n_fp']} | {m['n_fn']} | "
            f"{m['yolo_classification_on_tp']['accuracy']:.4f} | "
            f"{m['cnn_classification_on_tp']['accuracy']:.4f} | "
            f"{m['agreement_yolo_vs_cnn_on_tp']:.4f} |"
        )
    md.append("")

    with (out_dir / "cross_domain_summary.md").open("w", encoding="utf-8") as h:
        h.write("\n".join(md))

    print(f"\n{'=' * 60}")
    print(f"Done. Outputs in: {out_dir}")
    print(f"  - cross_domain_metrics.json    (full JSON)")
    print(f"  - cross_domain_summary.md      (paper-ready markdown)")
    print(f"  - confusion_matrix_*.png       ({len(summary['cnn_only']) + 2 * len(summary['pipeline'])} plots)")
    print(f"  - *_predictions.csv            (per-box diagnostic)")


if __name__ == "__main__":
    main()
