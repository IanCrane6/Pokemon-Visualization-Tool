# YOLO Checkpoints

The two YOLO best.pt files (`yolov8n_v2_best.pt`, `yolo11n_v2_best.pt`,
~6 MB each) for this run were not pulled back from the cloud instance
before it was destroyed.

To regenerate them locally or on a fresh GPU instance:

```bash
cd ../../yolo_cnn_pipeline_v2
python make_yolo_split.py --source <V2 path>/Synthetic --output yolo_data
python train_yolo.py --config configs/yolov8n.yaml --data yolo_data/data.yaml
python train_yolo.py --config configs/yolo11n.yaml --data yolo_data/data.yaml
```

This takes ~30-40 minutes per YOLO on an RTX 4090. Trained weights will
appear at `runs/<exp>/weights/best.pt`. Copy them into this folder.

The training is fully reproducible (seed=42, deterministic=True) — the
expected test mAP@0.5 is 0.9437 (YOLOv8n) and 0.9442 (YOLO11n), matching
the values reported in `results/v2_internal/yolo/*/test_metrics.json`.

## Why these weren't pulled

The team workflow had a missing scp step before instance destruction.
This is a known one-time gap; the metrics and confusion matrices
(`results/v2_internal/yolo/`) are all preserved.
