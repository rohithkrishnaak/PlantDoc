"""Training script for PlantDoc EfficientNet-B0 transfer-learning classifier.

Fine-tunes an ImageNet-pretrained backbone on the 6-class PlantDoc field subset.
Includes:
- Field-specific data augmentations
- Class-balanced loss weighting
- 2-phase transfer learning (frozen head warmup -> fine-tuning with cosine decay)
- Best-checkpoint preservation to weights/plantdoc_effnet_b0.pth
- Training history logging to training_metrics.json
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from src import mlflow_utils as mlfu
from src.model import CLASSES, MODEL_ARCH, MODEL_WEIGHTS

TRAIN_DIR = Path("data/train")
VAL_FRACTION = 0.15
SEED = 42
WEIGHTS_PATH = Path(MODEL_WEIGHTS)
METRICS_PATH = Path("training_metrics.json")

BATCH_SIZE = 16
NUM_EPOCHS_WARMUP = 3
NUM_EPOCHS_FINETUNE = 7
LR_WARMUP = 1e-3
LR_FINETUNE = 1e-4

TRAIN_TRANSFORMS = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

VAL_TRANSFORMS = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    top3_correct = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()

        # Top-3 accuracy
        _, top3_preds = torch.topk(outputs, k=min(3, outputs.size(1)), dim=1)
        top3_correct += sum([labels[i] in top3_preds[i] for i in range(labels.size(0))])

        total += labels.size(0)
    return total_loss / total, correct / total, top3_correct / total


def main():
    parser = argparse.ArgumentParser(description="Train EfficientNet-B0 transfer learning classifier")
    parser.add_argument("--no-mlflow", action="store_true", help="Disable MLflow tracking for this run")
    parser.add_argument("--experiment-name", type=str, default=None, help="MLflow experiment name override")
    parser.add_argument("--run-name", type=str, default="effnet-b0-transfer-learning", help="MLflow run name")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | Architecture: {MODEL_ARCH}")

    torch.manual_seed(SEED)

    # Load datasets. The test split is never touched here: checkpoint selection
    # uses a stratified validation split carved out of data/train.
    full_train = datasets.ImageFolder(str(TRAIN_DIR), transform=TRAIN_TRANSFORMS)
    full_val = datasets.ImageFolder(str(TRAIN_DIR), transform=VAL_TRANSFORMS)

    print(f"Dataset classes detected: {full_train.classes}")
    assert full_train.classes == sorted(full_train.classes), "Classes must match sorted folder names"

    targets = np.array(full_train.targets)
    train_idx, val_idx = train_test_split(
        np.arange(len(targets)), test_size=VAL_FRACTION, stratify=targets, random_state=SEED
    )
    train_dataset = Subset(full_train, train_idx)
    val_dataset = Subset(full_val, val_idx)

    # Class weights for loss
    class_counts = np.bincount(targets[train_idx])
    class_weights = 1.0 / np.maximum(class_counts, 1)
    class_weights = torch.FloatTensor(class_weights / class_weights.sum()).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"Train samples: {len(train_dataset)} | Validation samples: {len(val_dataset)}")

    num_classes = len(full_train.classes)

    with mlfu.mlflow_run(
        run_name=args.run_name,
        experiment_name=args.experiment_name,
        enabled=not args.no_mlflow,
        tags={"model": MODEL_ARCH, "stage": "transfer_learning", "device": str(device)},
    ):
        mlfu.log_params({
            "model_architecture": MODEL_ARCH,
            "batch_size": BATCH_SIZE,
            "val_fraction": VAL_FRACTION,
            "seed": SEED,
            "lr_warmup": LR_WARMUP,
            "lr_finetune": LR_FINETUNE,
            "num_epochs_warmup": NUM_EPOCHS_WARMUP,
            "num_epochs_finetune": NUM_EPOCHS_FINETUNE,
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "num_classes": num_classes,
            "classes": ",".join(full_train.classes),
            "device": str(device),
        })

        # Create backbone
        model = timm.create_model(MODEL_ARCH, pretrained=True, num_classes=num_classes)
        model = model.to(device)

        # Phase 1: Warmup with frozen backbone
        print("\n--- Phase 1: Classification Head Warmup (Backbone Frozen) ---")
        for param in model.parameters():
            param.requires_grad = False
        for param in model.get_classifier().parameters():
            param.requires_grad = True

        optimizer = torch.optim.AdamW(model.get_classifier().parameters(), lr=LR_WARMUP, weight_decay=1e-4)

        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_top3": []}
        best_val_acc = 0.0

        for epoch in range(1, NUM_EPOCHS_WARMUP + 1):
            t0 = time.time()
            tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc, val_top3 = evaluate(model, val_loader, criterion, device)
            elapsed = time.time() - t0
            print(f"Epoch {epoch}/{NUM_EPOCHS_WARMUP} [{elapsed:.1f}s] - "
                  f"Tr Loss: {tr_loss:.4f}, Tr Acc: {tr_acc:.1%} | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.1%}, Val Top-3: {val_top3:.1%}")
            history["train_loss"].append(tr_loss)
            history["train_acc"].append(tr_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            history["val_top3"].append(val_top3)

            mlfu.log_metrics({
                "phase": 1,
                "train_loss": tr_loss,
                "train_acc": tr_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_top3": val_top3,
                "epoch_time_sec": elapsed,
            }, step=epoch)

        # Phase 2: Full Fine-Tuning
        print("\n--- Phase 2: End-to-End Fine-Tuning ---")
        for param in model.parameters():
            param.requires_grad = True

        optimizer = torch.optim.AdamW(model.parameters(), lr=LR_FINETUNE, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS_FINETUNE)

        for epoch in range(1, NUM_EPOCHS_FINETUNE + 1):
            t0 = time.time()
            tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc, val_top3 = evaluate(model, val_loader, criterion, device)
            scheduler.step()
            elapsed = time.time() - t0

            global_step = NUM_EPOCHS_WARMUP + epoch
            print(f"Epoch {epoch}/{NUM_EPOCHS_FINETUNE} [{elapsed:.1f}s] - "
                  f"Tr Loss: {tr_loss:.4f}, Tr Acc: {tr_acc:.1%} | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.1%}, Val Top-3: {val_top3:.1%}")

            history["train_loss"].append(tr_loss)
            history["train_acc"].append(tr_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            history["val_top3"].append(val_top3)

            mlfu.log_metrics({
                "phase": 2,
                "train_loss": tr_loss,
                "train_acc": tr_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_top3": val_top3,
                "epoch_time_sec": elapsed,
                "lr": scheduler.get_last_lr()[0],
            }, step=global_step)

            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), WEIGHTS_PATH)
                print(f"  --> Best checkpoint saved to {WEIGHTS_PATH} (Val Acc: {val_acc:.1%})")

        # Save training metrics
        with open(METRICS_PATH, "w") as f:
            json.dump(history, f, indent=2)

        mlfu.log_metric("best_val_acc", best_val_acc)
        mlfu.log_artifact(WEIGHTS_PATH, artifact_path="checkpoints")
        mlfu.log_artifact(METRICS_PATH, artifact_path="metrics")
        mlfu.log_dict(history, "training_history.json")

        print(f"\nTraining Complete! Best Validation Accuracy: {best_val_acc:.1%}")
        print(f"Weights saved at: {WEIGHTS_PATH}")


if __name__ == "__main__":
    main()
