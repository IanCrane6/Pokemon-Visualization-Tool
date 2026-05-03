# Cross-Domain Evaluation Summary

- Roboflow images: roboflow_data
- Classes (12): butterfree, chansey, doduo, eevee, kangaskhan, lapras, magikarp, meowth, pidgey, pikachu, scyther, snorlax
- Config: conf=0.25, iou_match=0.5, padding=0.1

## 1. CNN-only on GT crops

(Real-domain classification accuracy assuming perfect localization)

| CNN | Accuracy | Macro F1 | Macro Precision | Macro Recall | n |
|---|---|---|---|---|---|
| resnet50 | 0.1750 | 0.1771 | 0.3021 | 0.1964 | 160 |
| dinov3 | 0.3000 | 0.3383 | 0.4846 | 0.3711 | 160 |
| aimv2 | 0.2625 | 0.2527 | 0.3625 | 0.3697 | 160 |
| siglip2 | 0.2188 | 0.1850 | 0.1842 | 0.2590 | 160 |

## 2. YOLO-only on full frames

| YOLO | mAP50 | mAP50-95 | Precision | Recall |
|---|---|---|---|---|
| yolov8n_v2 | 0.0408 | 0.0088 | 0.5280 | 0.0439 |
| yolo11n_v2 | 0.0819 | 0.0205 | 0.6898 | 0.0750 |

## 3. YOLO + CNN pipeline (end-to-end)

| YOLO | CNN | TP | FP | FN | YOLO_cls_acc | CNN_cls_acc | agreement |
|---|---|---|---|---|---|---|---|
| yolov8n_v2 | aimv2 | 11 | 11 | 149 | 0.3636 | 0.4545 | 0.5455 |
| yolov8n_v2 | dinov3 | 11 | 11 | 149 | 0.3636 | 0.6364 | 0.4545 |
| yolo11n_v2 | aimv2 | 19 | 23 | 141 | 0.5263 | 0.4737 | 0.5263 |
| yolo11n_v2 | dinov3 | 19 | 23 | 141 | 0.5263 | 0.6316 | 0.5789 |
