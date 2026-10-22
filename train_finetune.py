"""Fine-tune a pretrained timm backbone end-to-end on data/train_merged.

    python train_finetune.py --arch convnext_tiny.fb_in22k_ft_in1k --epochs 15

Protocol (no test data is used for any decision):
1. Grouped 5-fold CV on the training directory (near-duplicates share a fold, see
   src/merge_plantwild.py). Each fold logs per-epoch train / validation loss and
   accuracy to a nested MLflow run, so over-fitting is visible as a widening gap.
2. The epoch count with the best mean validation accuracy across folds is chosen.
   Out-of-fold logits at that epoch give the CV accuracy, the softmax temperature
   and the trust threshold (same rules as train_hybrid.py).
3. A final model is trained on all training photos for that many epochs and saved
   with its temperature, trust threshold and an out-of-distribution feature bank.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from experiments.benchmark_backbones import cv_splits, groups_for
from src import mlflow_utils as mlfu
from src.model import CLASSES, PREPROCESS
from src.quality_gate import check_quality
from train_hybrid import _log_dataset, _log_oof_confusion, choose_trust_threshold, fit_temperature, photo_source

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
CACHE_SIDE = 320  # photos are decoded once and kept in RAM at this shorter side

TRAIN_TF = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.35, 1.0), ratio=(0.75, 1.33)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomApply([transforms.RandomRotation(20)], p=0.5),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.25, hue=0.03),
    transforms.RandomApply([transforms.GaussianBlur(5, sigma=(0.1, 1.5))], p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.12)),
])


class Photos(Dataset):
    def __init__(self, images: list[Image.Image], labels: np.ndarray, tf):
        self.images, self.labels, self.tf = images, labels, tf

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        return self.tf(self.images[i]), int(self.labels[i])


def load_images(samples) -> list[Image.Image]:
    out = []
    for p, _ in samples:
        img = Image.open(p).convert("RGB")
        img.thumbnail((CACHE_SIDE * 4, CACHE_SIDE * 4))
        w, h = img.size
        s = CACHE_SIDE / min(w, h)
        out.append(img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BICUBIC) if s < 1 else img)
    return out


def build_model(arch: str, drop_path: float) -> nn.Module:
    kw = {"img_size": 224} if "dinov2" in arch else {}
    return timm.create_model(arch, pretrained=True, num_classes=len(CLASSES), drop_path_rate=drop_path, **kw)


def param_groups(model: nn.Module, lr: float, head_lr_mult: float, wd: float):
    head = set(id(p) for p in model.get_classifier().parameters())
    decay, no_decay, head_p = [], [], []
    for n, p in model.named_parameters():
        if id(p) in head:
            head_p.append(p)
        elif p.ndim <= 1 or n.endswith(".bias") or "pos_embed" in n or "cls_token" in n:
            no_decay.append(p)
        else:
            decay.append(p)
    return [{"params": decay, "lr": lr, "weight_decay": wd},
            {"params": no_decay, "lr": lr, "weight_decay": 0.0},
            {"params": head_p, "lr": lr * head_lr_mult, "weight_decay": wd}]


@torch.inference_mode()
def predict(model: nn.Module, images: list[Image.Image], batch: int = 64):
    """Logits and pooled features, averaged over the original and h-flipped view
    (the app does the same at inference)."""
    model.eval()
    logits, feats = [], []
    for i in range(0, len(images), batch):
        x = torch.stack([PREPROCESS(im) for im in images[i:i + batch]]).to(DEVICE)
        with torch.autocast(DEVICE.type, dtype=torch.bfloat16, enabled=DEVICE.type == "cuda"):
            f = (model.forward_head(model.forward_features(x), pre_logits=True) +
                 model.forward_head(model.forward_features(torch.flip(x, dims=[3])), pre_logits=True)) / 2
        f = f.float()
        logits.append(model.get_classifier()(f).cpu().numpy())
        feats.append(f.cpu().numpy())
    return np.concatenate(logits), np.concatenate(feats)


def train(arch: str, images, y, args, val=None, log_prefix: str = ""):
    """Train for args.epochs; if val=(images, labels) is given, return per-epoch val logits."""
    torch.manual_seed(args.seed)
    model = build_model(arch, args.drop_path).to(DEVICE)
    loader = DataLoader(Photos(images, y, TRAIN_TF), batch_size=args.batch_size, shuffle=True,
                        num_workers=0, drop_last=True)
    opt = torch.optim.AdamW(param_groups(model, args.lr, args.head_lr_mult, args.weight_decay))
    steps = args.epochs * len(loader)
    warm = len(loader)  # one epoch of linear warm-up, then cosine
    base = [g["lr"] for g in opt.param_groups]
    counts = np.bincount(y, minlength=len(CLASSES))
    weight = torch.tensor(len(y) / (len(CLASSES) * np.maximum(counts, 1)), dtype=torch.float32, device=DEVICE)
    crit = nn.CrossEntropyLoss(weight=weight, label_smoothing=args.label_smoothing)
    val_logits, step = [], 0
    for epoch in range(args.epochs):
        model.train()
        t0, tot, correct, n = time.time(), 0.0, 0, 0
        for xb, yb in loader:
            f = step / warm if step < warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, steps - warm)))
            for g, b in zip(opt.param_groups, base):
                g["lr"] = b * f
            xb, yb = xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)
            with torch.autocast(DEVICE.type, dtype=torch.bfloat16, enabled=DEVICE.type == "cuda"):
                out = model(xb)
                loss = crit(out.float(), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            tot += loss.item() * len(yb)
            correct += (out.argmax(1) == yb).sum().item()
            n += len(yb)
        m = {f"{log_prefix}train_loss": tot / n, f"{log_prefix}train_acc": correct / n}
        if val is not None:
            lg, _ = predict(model, val[0])
            val_logits.append(lg)
            vl = nn.functional.cross_entropy(torch.tensor(lg), torch.tensor(val[1])).item()
            m.update({f"{log_prefix}val_loss": vl, f"{log_prefix}val_acc": float((lg.argmax(1) == val[1]).mean())})
        mlfu.log_metrics(m, step=epoch + 1)
        print(f"  epoch {epoch + 1:2d}/{args.epochs} " + " ".join(f"{k.split('_', 1)[-1] if log_prefix else k}={v:.3f}"
                                                         for k, v in m.items()) + f" ({time.time() - t0:.0f}s)", flush=True)
    return model, val_logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="convnext_tiny.fb_in22k_ft_in1k")
    ap.add_argument("--train-dir", default="data/train_merged")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--head-lr-mult", type=float, default=10.0)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--drop-path", type=float, default=0.1)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--folds", type=int, default=5, help="CV folds to run (<5 for a quicker estimate)")
    ap.add_argument("--target-acc", type=float, default=0.80)
    ap.add_argument("--out", default="weights/plantdoc_finetuned.pth")
    ap.add_argument("--metrics-out", default="training_metrics_finetuned.json")
    ap.add_argument("--no-final", action="store_true", help="CV only, skip the final fit")
    ap.add_argument("--no-mlflow", action="store_true")
    ap.add_argument("--experiment-name", default=None)
    ap.add_argument("--run-name", default=None)
    args = ap.parse_args()

    root = args.train_dir
    samples = datasets.ImageFolder(root).samples
    y = np.array([t for _, t in samples])
    source = np.array([photo_source(p) for p, _ in samples])
    groups = groups_for(root)
    print(f"loading {len(samples)} photos ...", flush=True)
    images = load_images(samples)
    run_name = args.run_name or f"finetune-{args.arch.split('.')[0]}"

    with mlfu.mlflow_run(run_name=run_name, experiment_name=args.experiment_name, enabled=not args.no_mlflow,
                         tags={"model_type": "finetuned", "stage": "train", "train_dir": root, "arch": args.arch}):
        hp = {k: v for k, v in vars(args).items() if k not in ("no_mlflow", "experiment_name", "run_name")}
        mlfu.log_params({**hp, "n_train": len(y), "n_groups": len(np.unique(groups)), "device": str(DEVICE),
                         **{f"n_train_{s}": int((source == s).sum()) for s in np.unique(source)}})
        _log_dataset(samples, root)

        # 1. Grouped CV with per-epoch validation curves
        per_epoch = np.zeros((args.epochs, len(y), len(CLASSES)))
        seen = np.zeros(len(y), bool)
        for k, (tr, te) in enumerate(cv_splits(y, groups)):
            if k >= args.folds:
                break
            print(f"fold {k + 1}/{args.folds}: train {len(tr)} / val {len(te)}", flush=True)
            with mlfu.mlflow_run(run_name=f"{run_name}-fold{k + 1}", nested=True, enabled=not args.no_mlflow,
                                 tags={"fold": k + 1}):
                _, vlog = train(args.arch, [images[i] for i in tr], y[tr], args,
                                val=([images[i] for i in te], y[te]))
            per_epoch[:, te] = np.stack(vlog)
            seen[te] = True
            torch.cuda.empty_cache()
        acc_by_epoch = (per_epoch[:, seen].argmax(2) == y[seen]).mean(1)
        for e, a in enumerate(acc_by_epoch):
            mlfu.log_metric("cv_val_acc_by_epoch", float(a), step=e + 1)
        best_ep = int(acc_by_epoch.argmax()) + 1
        oof = per_epoch[best_ep - 1]
        correct = oof.argmax(1) == y
        cv_acc = float(correct[seen].mean())
        per_source = {f"oof_cv_acc_{s}": float(correct[seen & (source == s)].mean()) for s in np.unique(source)}
        print(f"CV accuracy by epoch: {np.round(acc_by_epoch, 3).tolist()}")
        print(f"best epoch {best_ep}: OOF CV accuracy {cv_acc:.3f} {per_source}", flush=True)

        metrics = {"arch": args.arch, "train_dir": root, "hyperparameters": hp, "n_train": int(len(y)),
                   "folds_run": int(min(args.folds, 5)), "cv_acc_by_epoch": acc_by_epoch.tolist(),
                   "best_epoch": best_ep, "oof_cv_acc": cv_acc, **per_source}
        mlfu.log_metrics({"oof_cv_acc": cv_acc, "best_epoch": best_ep, **per_source})
        if seen.all():
            T = fit_temperature(oof, y)
            z = oof / T
            probs = np.exp(z - z.max(1, keepdims=True))
            probs /= probs.sum(1, keepdims=True)
            q = [check_quality(Image.open(p).convert("RGB")) for p, _ in samples]
            passed, quality = np.array([r[0] for r in q]), np.array([r[1] for r in q])
            thr, cov, acc, sweep = choose_trust_threshold(probs, y, quality, passed, args.target_acc)
            print(f"temperature {T:.2f} | trust threshold {thr:.2f}: OOF coverage {cov:.1%}, accuracy {acc:.1%}")
            per_class = {f"oof_recall_{c.lower().replace(' ', '_')}": float(correct[y == i].mean())
                         for i, c in enumerate(CLASSES)}
            metrics.update({"temperature": T, "trust_threshold": thr, "oof_coverage_at_threshold": cov,
                            "oof_accuracy_at_threshold": acc, "oof_per_class_recall": per_class,
                            "oof_trust_sweep": [{"threshold": t, "coverage": c, "accuracy": a} for t, c, a in sweep]})
            mlfu.log_metrics({"temperature": T, "trust_threshold": thr, "oof_coverage_at_threshold": cov,
                              "oof_accuracy_at_threshold": acc, **per_class})
            _log_oof_confusion(y, oof.argmax(1))

        # 2. Final fit on all training photos for the chosen number of epochs
        if not args.no_final and seen.all():
            args.epochs = best_ep
            print(f"final fit on all {len(y)} photos for {best_ep} epochs", flush=True)
            model, _ = train(args.arch, images, y, args, log_prefix="final_")
            logits, feats = predict(model, images)
            train_acc = float((logits.argmax(1) == y).mean())
            f = feats / np.linalg.norm(feats, axis=1, keepdims=True)
            sim = f @ f.T
            sim[groups[:, None] == groups[None, :]] = -1.0
            ood_thr = float(np.percentile(sim.max(1), 2.0))
            print(f"train accuracy {train_acc:.3f} vs OOF {cv_acc:.3f} | OOD threshold {ood_thr:.3f}")
            ckpt = {"type": "finetuned", "arch": args.arch, "classes": CLASSES, "clahe": False,
                    "state_dict": {k: v.half() if v.is_floating_point() else v
                                   for k, v in model.state_dict().items()},
                    "temperature": T, "cv_acc": cv_acc, "trust_threshold": thr,
                    "ood_bank": torch.tensor(f.astype(np.float16)), "ood_threshold": ood_thr, "train_dir": root}
            out = Path(args.out)
            out.parent.mkdir(parents=True, exist_ok=True)
            torch.save(ckpt, out)
            metrics.update({"train_acc": train_acc, "train_oof_gap": train_acc - cv_acc, "ood_threshold": ood_thr})
            mlfu.log_metrics({"train_acc": train_acc, "train_oof_gap": train_acc - cv_acc, "ood_threshold": ood_thr})
            mlfu.log_artifact(out, artifact_path="checkpoints")
            print(f"Saved {out}")

        Path(args.metrics_out).write_text(json.dumps(metrics, indent=2))
        mlfu.log_artifact(args.metrics_out, artifact_path="metrics")


if __name__ == "__main__":
    main()
