"""Train a YOLO model (YOLOv8n / YOLO11n) on the V2 split.

Wraps Ultralytics Python API + dumps a clean test_metrics.json next to the run dir.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import yaml
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--data", required=True, help="Path to YOLO data.yaml")
    p.add_argument("--project", default="runs", help="Where Ultralytics saves runs")
    return p.parse_args()


def metrics_to_dict(metrics, class_names):
    """Extract a JSON-serializable dict from Ultralytics DetMetrics."""
    out = {}
    box = metrics.box
    out["map50"] = float(box.map50)
    out["map50_95"] = float(box.map)
    out["mp"] = float(box.mp)
    out["mr"] = float(box.mr)

    per_class = {}
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


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config).resolve()
    data_path = Path(args.data).resolve()
    project_dir = Path(args.project).resolve()

    with cfg_path.open("r", encoding="utf-8") as h:
        cfg = yaml.safe_load(h)

    with data_path.open("r", encoding="utf-8") as h:
        data_spec = yaml.safe_load(h)
    class_names = data_spec.get("names", [])

    print(f"Config: {cfg_path}")
    print(f"Data:   {data_path}")
    print(f"Model:  {cfg['model_pt']}  (pretrained COCO weights)")
    print(f"Epochs: {cfg['epochs']} | imgsz: {cfg['imgsz']} | batch: {cfg['batch']}")

    model = YOLO(cfg["model_pt"])
    model.train(
        data=str(data_path),
        epochs=int(cfg["epochs"]),
        imgsz=int(cfg["imgsz"]),
        batch=int(cfg["batch"]),
        project=str(project_dir),
        name=cfg["experiment_name"],
        seed=int(cfg.get("seed", 42)),
        deterministic=True,
        patience=int(cfg.get("patience", 20)),
        device=cfg.get("device", 0),
        exist_ok=True,
    )

    run_dir = project_dir / cfg["experiment_name"]
    print(f"\nTraining done. Run dir: {run_dir}")

    print("\nEvaluating on val split (Ultralytics default)...")
    val_metrics = model.val(data=str(data_path), split="val")
    val_dict = metrics_to_dict(val_metrics, class_names)

    print("Evaluating on test split...")
    test_metrics = model.val(data=str(data_path), split="test")
    test_dict = metrics_to_dict(test_metrics, class_names)

    summary = {
        "experiment_name": cfg["experiment_name"],
        "model_pt": cfg["model_pt"],
        "epochs": cfg["epochs"],
        "imgsz": cfg["imgsz"],
        "batch": cfg["batch"],
        "data_yaml": str(data_path),
        "val": val_dict,
        "test": test_dict,
    }
    out_path = run_dir / "test_metrics.json"
    with out_path.open("w", encoding="utf-8") as h:
        json.dump(summary, h, ensure_ascii=False, indent=2)
    print(f"\nWrote {out_path}")

    snap = run_dir / "config.snapshot.yaml"
    shutil.copy2(cfg_path, snap)

    print(f"\nKey numbers:")
    print(f"  val   mAP50 = {val_dict['map50']:.4f}  mAP50-95 = {val_dict['map50_95']:.4f}")
    print(f"  test  mAP50 = {test_dict['map50']:.4f}  mAP50-95 = {test_dict['map50_95']:.4f}")


if __name__ == "__main__":
    main()
