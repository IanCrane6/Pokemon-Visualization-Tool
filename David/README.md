# David's Contribution — YOLO Ablation

## What's here

- `configs/` — all training/evaluation configs, including:
  - `configs/yolo_ablation/` — 8 ablation experiment configs (E0-E5 on all, E0/E5 on beach)
  - `configs/yolo_arch/` — custom YOLOv8n architecture YAMLs (+CBAM, +P2 head, combined)
  - `configs/yolo/` — original baseline configs (kept for reference)
  - `configs/classification/` — original ResNet configs (kept for reference)
- `runs/` — pilot ablation training artifacts (weights .pt excluded to keep repo light)
  - `runs/yolo_ablation/` — 8 experiment directories with results.csv, training curves, confusion matrices
  - `runs/ablation_summary.md` — cross-experiment comparison table

## Known caveats (pilot results on fake-bbox data)

- Current bboxes are full-image `0.5 0.5 1.0 1.0` placeholders (to be replaced with real bboxes from Dataset/My_yolo_dataset in next round)
- Under fake bboxes, P2 head and SIoU loss cannot show their real value and are marked `pilot (fake-bbox)` in the summary
- SIoU implementation has a known `sin -> cos` bug in angle_cost (Gevorgyan eq.6); to be fixed in next push

## Next steps

- Upload full code (src/, scripts/, tests/) in a follow-up push
- Rerun with real bboxes when Dataset/My_yolo_dataset is finalized
- Fix bbox_siou angle_cost formula and rerun E3 / E5

