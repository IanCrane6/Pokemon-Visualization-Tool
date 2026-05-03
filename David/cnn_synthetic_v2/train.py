"""Train a classification model on the synthetic pokemon crops produced by convert.py.

Supports torchvision ResNet50 (baseline) and any timm model (DINOv3 / AIMv2 / SigLIP 2 ...).
Reads a YAML config; writes logs, checkpoints, metrics, predictions, and confusion matrix
under output.root_dir / experiment_name.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import shutil
import time
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch
import torch.nn as nn
import yaml
from PIL import Image
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet50_Weights, resnet50

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------- utils ----------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_logging(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("cnn_syn_v2")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(formatter)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.propagate = False
    return logger


def resolve_device(name: str) -> torch.device:
    token = str(name).strip().lower()
    if token in {"cuda", "gpu"} and torch.cuda.is_available():
        return torch.device("cuda")
    if token.startswith("cuda") and torch.cuda.is_available():
        return torch.device(token)
    if token.isdigit() and torch.cuda.is_available():
        return torch.device(f"cuda:{token}")
    return torch.device("cpu")


def dump_json(data: Any, path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


# ---------------- dataset ----------------


class CropDataset(Dataset):
    def __init__(self, manifest_path: Path, split: str, transform):
        self.root_dir = manifest_path.resolve().parent
        self.transform = transform
        with manifest_path.open("r", encoding="utf-8", newline="") as handle:
            rows = [r for r in csv.DictReader(handle) if r["split"] == split]
        if not rows:
            raise ValueError(f"No rows for split={split} in {manifest_path}")
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        image_path = self.root_dir / row["image_path"]
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            if self.transform is not None:
                tensor = self.transform(img)
            else:
                tensor = transforms.ToTensor()(img)
        label_id = int(row["label_id"])
        return tensor, label_id, row["image_path"], row["label"]


# ---------------- model factory ----------------


def create_model(name: str, num_classes: int, pretrained: bool = True):
    """Return (model, data_config) where data_config is None for torchvision models."""
    lower = name.lower()
    if lower == "resnet50":
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        model = resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model, None

    try:
        import timm
    except ImportError as exc:
        raise ImportError(
            "timm is required for non-torchvision models. Install via `pip install timm>=1.0.11`."
        ) from exc

    model = timm.create_model(name, pretrained=pretrained, num_classes=num_classes)
    data_config = timm.data.resolve_model_data_config(model)
    return model, data_config


# ---------------- transforms ----------------


def build_transforms(data_config: dict | None, image_size: int, train: bool):
    if data_config is not None:
        import timm
        cfg = dict(data_config)
        if image_size:
            cfg["input_size"] = (cfg.get("input_size", (3, image_size, image_size))[0], image_size, image_size)
        return timm.data.create_transform(**cfg, is_training=train)

    normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    if train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.15)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ]
    )


# ---------------- training loops ----------------


def train_one_epoch(model, loader, criterion, optimizer, device, scaler, use_amp: bool) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0
    for images, targets, _, _ in loader:
        images = images.to(device, non_blocking=device.type == "cuda")
        targets = targets.to(device, non_blocking=device.type == "cuda")
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        bs = targets.size(0)
        total_loss += loss.item() * bs
        total_samples += bs
    return total_loss / max(total_samples, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device, use_amp: bool, collect_rows: bool = False):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    y_true: list[int] = []
    y_pred: list[int] = []
    rows: list[dict] = []
    for images, targets, image_paths, labels in loader:
        images = images.to(device, non_blocking=device.type == "cuda")
        targets = targets.to(device, non_blocking=device.type == "cuda")
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, targets)
        preds = logits.argmax(dim=1)
        bs = targets.size(0)
        total_loss += loss.item() * bs
        total_samples += bs
        t_cpu = targets.cpu().tolist()
        p_cpu = preds.cpu().tolist()
        y_true.extend(t_cpu)
        y_pred.extend(p_cpu)
        if collect_rows:
            for img_path, tid, pid, lbl in zip(image_paths, t_cpu, p_cpu, labels):
                rows.append(
                    {
                        "image_path": img_path,
                        "true_label": lbl,
                        "predicted_id": pid,
                        "true_id": tid,
                        "correct": bool(tid == pid),
                    }
                )
    loss_value = total_loss / max(total_samples, 1)
    return loss_value, y_true, y_pred, rows


def compute_metrics(y_true: list[int], y_pred: list[int], id_to_label: dict[int, str]) -> dict:
    num_classes = len(id_to_label)
    labels_range = list(range(num_classes))
    accuracy = float(np.mean(np.array(y_true) == np.array(y_pred))) if y_true else 0.0
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels_range, average="macro", zero_division=0
    )
    p_per, r_per, f1_per, sup_per = precision_recall_fscore_support(
        y_true, y_pred, labels=labels_range, average=None, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels_range).tolist()
    per_class = {}
    for idx in labels_range:
        name = id_to_label[idx]
        per_class[name] = {
            "precision": float(p_per[idx]),
            "recall": float(r_per[idx]),
            "f1": float(f1_per[idx]),
            "support": int(sup_per[idx]),
        }
    return {
        "accuracy": accuracy,
        "macro_precision": float(p_macro),
        "macro_recall": float(r_macro),
        "macro_f1": float(f1_macro),
        "per_class": per_class,
        "confusion_matrix": cm,
    }


def plot_confusion_matrix(cm: list[list[int]], labels: list[str], path: Path) -> None:
    size = max(8, min(20, len(labels) * 0.8))
    fig, ax = plt.subplots(figsize=(size, size))
    arr = np.array(cm)
    im = ax.imshow(arr, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            ax.text(j, i, str(arr[i, j]), ha="center", va="center",
                    color="white" if arr[i, j] > arr.max() / 2 else "black", fontsize=8)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_checkpoint(path: Path, epoch: int, model, optimizer, label_to_id: dict[str, int], config: dict, metric_value: float) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "label_to_id": label_to_id,
            "config": config,
            "metric_value": metric_value,
        },
        path,
    )


def write_predictions_csv(path: Path, rows: list[dict], id_to_label: dict[int, str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["image_path", "true_label", "predicted_label", "true_id", "predicted_id", "correct"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "image_path": row["image_path"],
                    "true_label": row["true_label"],
                    "predicted_label": id_to_label[row["predicted_id"]],
                    "true_id": row["true_id"],
                    "predicted_id": row["predicted_id"],
                    "correct": row["correct"],
                }
            )


# ---------------- main ----------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    project_root = Path(__file__).resolve().parent
    seed = int(config.get("seed", 42))
    set_seed(seed)

    experiment_name = config.get("experiment_name", config_path.stem)
    output_root = Path(config.get("output", {}).get("root_dir", "runs"))
    if not output_root.is_absolute():
        output_root = project_root / output_root
    run_dir = output_root / experiment_name
    run_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(config_path, run_dir / "config.snapshot.yaml")
    logger = configure_logging(run_dir / "train.log")
    logger.info("Experiment: %s", experiment_name)
    logger.info("Run dir: %s", run_dir)

    data_cfg = config.get("data", {})
    manifest_path = Path(data_cfg.get("manifest_path", "data/manifest.csv"))
    label_map_path = Path(data_cfg.get("label_map_path", "data/label_to_id.json"))
    if not manifest_path.is_absolute():
        manifest_path = project_root / manifest_path
    if not label_map_path.is_absolute():
        label_map_path = project_root / label_map_path
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}. Run convert.py first.")

    with label_map_path.open("r", encoding="utf-8") as handle:
        label_to_id: dict[str, int] = json.load(handle)
    id_to_label = {idx: name for name, idx in label_to_id.items()}
    num_classes = len(label_to_id)
    logger.info("Classes (%d): %s", num_classes, list(label_to_id.keys()))

    train_cfg = config.get("train", {})
    model_cfg = config.get("model", {})
    device = resolve_device(train_cfg.get("device", "cuda"))
    image_size = int(train_cfg.get("image_size", 224))
    batch_size = int(train_cfg.get("batch_size", 64))
    num_workers = int(train_cfg.get("num_workers", 8))
    epochs = int(train_cfg.get("epochs", 15))
    lr = float(train_cfg.get("learning_rate", 1e-4))
    weight_decay = float(train_cfg.get("weight_decay", 1e-4))
    selection_metric = train_cfg.get("selection_metric", "accuracy")
    use_amp = bool(train_cfg.get("amp", device.type == "cuda")) and device.type == "cuda"
    logger.info("Device: %s | AMP: %s", device, use_amp)

    model_name = model_cfg.get("name", "resnet50")
    pretrained = bool(model_cfg.get("pretrained", True))
    logger.info("Model: %s (pretrained=%s)", model_name, pretrained)
    model, data_config = create_model(model_name, num_classes, pretrained=pretrained)
    if data_config is not None:
        logger.info("timm data_config: %s", data_config)
    model = model.to(device)

    train_tf = build_transforms(data_config, image_size, train=True)
    eval_tf = build_transforms(data_config, image_size, train=False)

    train_set = CropDataset(manifest_path, "train", train_tf)
    val_set = CropDataset(manifest_path, "val", eval_tf)
    test_set = CropDataset(manifest_path, "test", eval_tf)
    logger.info("Dataset sizes -> train: %d | val: %d | test: %d", len(train_set), len(val_set), len(test_set))

    loader_kwargs = {"batch_size": batch_size, "num_workers": num_workers, "pin_memory": device.type == "cuda"}
    train_loader = DataLoader(train_set, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_set, shuffle=False, **loader_kwargs)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    history: list[dict] = []
    best_metric = float("-inf")
    best_ckpt_path = run_dir / "best.pt"
    last_ckpt_path = run_dir / "last.pt"

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler, use_amp)
        val_loss, y_t, y_p, _ = evaluate(model, val_loader, criterion, device, use_amp)
        val_metrics = compute_metrics(y_t, y_p, id_to_label)
        epoch_sec = time.time() - t0
        peak_mem_mb = (
            round(torch.cuda.max_memory_allocated(device) / (1024 ** 2), 2) if device.type == "cuda" else 0.0
        )

        metric_value = float(val_metrics.get(selection_metric, val_metrics["accuracy"]))
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
                "epoch_seconds": epoch_sec,
                "max_gpu_memory_mb": peak_mem_mb,
            }
        )
        dump_json(history, run_dir / "history.json")

        save_checkpoint(last_ckpt_path, epoch, model, optimizer, label_to_id, config, metric_value)
        if metric_value > best_metric:
            best_metric = metric_value
            save_checkpoint(best_ckpt_path, epoch, model, optimizer, label_to_id, config, metric_value)

        logger.info(
            "Epoch %d/%d | train_loss=%.4f | val_loss=%.4f | val_acc=%.4f | val_macro_f1=%.4f | %.2fs | peak=%.0fMB",
            epoch, epochs, train_loss, val_loss, val_metrics["accuracy"], val_metrics["macro_f1"], epoch_sec, peak_mem_mb,
        )

    best_state = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(best_state["model_state_dict"])
    logger.info("Loaded best checkpoint (epoch=%d, metric=%.4f)", best_state["epoch"], best_state["metric_value"])

    test_loss, y_t, y_p, rows = evaluate(model, test_loader, criterion, device, use_amp, collect_rows=True)
    test_metrics = compute_metrics(y_t, y_p, id_to_label)
    test_metrics["loss"] = test_loss
    dump_json(test_metrics, run_dir / "test_metrics.json")
    write_predictions_csv(run_dir / "test_predictions.csv", rows, id_to_label)
    plot_confusion_matrix(
        test_metrics["confusion_matrix"],
        [id_to_label[i] for i in range(num_classes)],
        run_dir / "confusion_matrix.png",
    )

    logger.info("Test accuracy: %.4f | macro_f1: %.4f | loss: %.4f",
                test_metrics["accuracy"], test_metrics["macro_f1"], test_loss)
    logger.info("Done. Outputs in %s", run_dir)


if __name__ == "__main__":
    main()
