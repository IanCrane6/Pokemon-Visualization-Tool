"""Run YOLO + CNN pipeline frame-by-frame on a video and write annotated mp4.

For each frame:
  1. YOLO detect -> boxes
  2. For each box: crop with padding, run CNN -> top-1 class + softmax confidence
  3. Draw box + label "<cnn_class> cnn=<p>  yolo=<class>(<p>)"
  4. Write to output mp4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import yaml
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet50
from tqdm import tqdm
from ultralytics import YOLO


# ---------------- inlined CNN loader ----------------


def create_model_from_ckpt(ckpt_path: Path, device: torch.device):
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
    return model, transform, id_to_label, model_name, image_size


def crop_with_padding_pil(img_pil: Image.Image, x1: float, y1: float, x2: float, y2: float, padding: float):
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    pad_x = w * padding
    pad_y = h * padding
    x1p = max(0, int(round(x1 - pad_x)))
    y1p = max(0, int(round(y1 - pad_y)))
    x2p = min(img_pil.width, int(round(x2 + pad_x)))
    y2p = min(img_pil.height, int(round(y2 + pad_y)))
    if x2p <= x1p or y2p <= y1p:
        return None
    return img_pil.crop((x1p, y1p, x2p, y2p))


def color_for_class(idx: int) -> tuple[int, int, int]:
    """Stable BGR color per class id."""
    rng = np.random.default_rng(idx + 7)
    return tuple(int(c) for c in rng.integers(60, 230, size=3))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--yolo", required=True, help="Path to YOLO best.pt")
    p.add_argument("--cnn", required=True, help="Path to CNN best.pt")
    p.add_argument("--video", required=True, help="Input video path")
    p.add_argument("--out", required=True, help="Output mp4 path")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--padding", type=float, default=0.1)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-frames", type=int, default=0, help="0 = all frames")
    p.add_argument("--every-n", type=int, default=1, help="Process 1 frame every N (skip = same labels copied)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    print(f"Device: {device}")

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    yolo = YOLO(args.yolo)
    yolo_class_names = list(yolo.names.values()) if isinstance(yolo.names, dict) else list(yolo.names)
    print(f"YOLO classes ({len(yolo_class_names)}): {yolo_class_names}")

    cnn, cnn_transform, cnn_id_to_label, cnn_model_name, cnn_image_size = create_model_from_ckpt(
        Path(args.cnn), device
    )
    print(f"CNN model: {cnn_model_name}  image_size={cnn_image_size}")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    print(f"Video: {width}x{height} @ {fps:.2f} fps, ~{n_frames} frames")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open writer for {out_path}")

    target = n_frames if args.max_frames == 0 else min(n_frames or args.max_frames, args.max_frames)
    pbar = tqdm(total=target if target > 0 else None)

    last_annotated = None
    frame_idx = 0
    processed = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        if args.every_n > 1 and frame_idx % args.every_n != 0:
            if last_annotated is not None:
                writer.write(last_annotated)
            else:
                writer.write(frame_bgr)
            frame_idx += 1
            pbar.update(1)
            if args.max_frames and frame_idx >= args.max_frames:
                break
            continue

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(frame_rgb)

        results = yolo.predict(source=frame_rgb, conf=args.conf, verbose=False, device=device.type)
        result = results[0]
        annotated = frame_bgr.copy()

        if result.boxes is not None and len(result.boxes) > 0:
            boxes_xyxy = result.boxes.xyxy.cpu().numpy()
            yolo_classes = result.boxes.cls.cpu().numpy().astype(np.int64)
            yolo_confs = result.boxes.conf.cpu().numpy()

            crops: list[Image.Image] = []
            kept_indices: list[int] = []
            for i, box in enumerate(boxes_xyxy):
                crop = crop_with_padding_pil(img_pil, box[0], box[1], box[2], box[3], args.padding)
                if crop is None:
                    continue
                crops.append(crop)
                kept_indices.append(i)

            if crops:
                tensors = torch.stack([cnn_transform(c) for c in crops]).to(device)
                with torch.no_grad():
                    logits = cnn(tensors)
                    probs = torch.softmax(logits, dim=1)
                    cnn_pred = logits.argmax(dim=1).cpu().tolist()
                    cnn_top_p = probs.max(dim=1).values.cpu().tolist()
            else:
                cnn_pred = []
                cnn_top_p = []

            cnn_iter = iter(zip(cnn_pred, cnn_top_p))
            for i in range(len(boxes_xyxy)):
                box = boxes_xyxy[i]
                yolo_cls_id = int(yolo_classes[i])
                yolo_cls_name = yolo_class_names[yolo_cls_id] if 0 <= yolo_cls_id < len(yolo_class_names) else str(yolo_cls_id)
                yolo_conf = float(yolo_confs[i])
                if i in kept_indices:
                    cnn_local, cnn_p = next(cnn_iter)
                    cnn_label = cnn_id_to_label.get(int(cnn_local), str(cnn_local))
                else:
                    cnn_label = ""
                    cnn_p = 0.0

                x1, y1, x2, y2 = (int(round(v)) for v in box.tolist())
                color = color_for_class(yolo_cls_id)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                if cnn_label:
                    label_text = f"{cnn_label} cnn={cnn_p:.2f} | yolo={yolo_cls_name}({yolo_conf:.2f})"
                else:
                    label_text = f"yolo={yolo_cls_name}({yolo_conf:.2f})"

                (text_w, text_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                ty = max(text_h + 4, y1)
                cv2.rectangle(annotated, (x1, ty - text_h - 4), (x1 + text_w + 4, ty + baseline), color, -1)
                cv2.putText(annotated, label_text, (x1 + 2, ty - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        writer.write(annotated)
        last_annotated = annotated
        frame_idx += 1
        processed += 1
        pbar.update(1)
        if args.max_frames and frame_idx >= args.max_frames:
            break

    pbar.close()
    cap.release()
    writer.release()

    meta = {
        "video": str(Path(args.video).resolve()),
        "out": str(out_path),
        "yolo_ckpt": str(Path(args.yolo).resolve()),
        "cnn_ckpt": str(Path(args.cnn).resolve()),
        "cnn_model": cnn_model_name,
        "conf": args.conf,
        "padding": args.padding,
        "fps": fps,
        "width": width,
        "height": height,
        "n_frames_total": n_frames,
        "n_frames_written": frame_idx,
        "n_frames_processed": processed,
    }
    with out_path.with_suffix(".meta.json").open("w", encoding="utf-8") as h:
        json.dump(meta, h, ensure_ascii=False, indent=2)
    print(f"\nDone. {processed} frames processed, {frame_idx} frames written.")
    print(f"Output: {out_path}")
    print(f"Meta:   {out_path.with_suffix('.meta.json')}")


if __name__ == "__main__":
    main()
