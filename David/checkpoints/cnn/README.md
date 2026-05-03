# CNN / ViT Checkpoints

The four classifier best.pt files are too large to commit to Git
(combined ~5.8 GB):

| Backbone | Architecture | Size | Pretraining |
|---|---|---|---|
| ResNet50 | CNN | 270 MB | ImageNet (supervised) |
| DINOv3 ViT-B/16 | ViT | 980 MB | LVD-1689M (self-supervised) |
| AIMv2-Large/14 | ViT | 3.5 GB | apple_pt (autoregressive multimodal) |
| SigLIP 2-Base/16 | ViT | 1.0 GB | WebLI (sigmoid VL) |

## How to obtain

**Option 1 — External link (preferred)**: David has these on local disk.
Once the team agrees on a hosting solution (Google Drive / OneDrive / HF Hub),
checkpoints will be uploaded and a download link will be added here.

**Option 2 — Retrain**: All four are reproducible from V2 with seed=42.

```bash
cd ../../cnn_synthetic_v2
python convert.py --source <V2 path>/Synthetic --output-dir ./data
python train.py --config configs/resnet50.yaml
python train.py --config configs/dinov3.yaml
python train.py --config configs/aimv2.yaml
python train.py --config configs/siglip2.yaml
```

Each takes 20-90 minutes on RTX 4090 (AIMv2 longest).

## Expected test accuracy on V2 internal

| Model | Top-1 | Macro F1 |
|---|---|---|
| ResNet50 | 0.9596 | 0.9599 |
| DINOv3 | 0.9678 | 0.9678 |
| AIMv2 | 0.9601 | 0.9598 |
| SigLIP 2 | 0.9509 | 0.9507 |
