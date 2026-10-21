"""Fit the hybrid feature-fusion classifier (see src/hybrid.py).

    python train_hybrid.py --backbones convnext_tiny.fb_in22k_ft_in1k vit_base_patch14_dinov2.lvd142m --clahe --C 0.2

Uses data/train only. The temperature is fitted on 5-fold out-of-fold logits.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from experiments.benchmark_backbones import extract, l2
from src.model import CLASSES

OUT = Path("weights/plantdoc_hybrid.pth")


def fit_head(x2: np.ndarray, y: np.ndarray, C: float):
    """x2: [N, 2, D] (original + hflip). Train on both views."""
    scaler = StandardScaler().fit(x2.reshape(-1, x2.shape[-1]))
    clf = LogisticRegression(C=C, max_iter=5000, class_weight="balanced")
    clf.fit(scaler.transform(x2.reshape(-1, x2.shape[-1])), np.repeat(y, 2))
    return scaler, clf


def view_avg_logits(scaler, clf, x2):
    return (clf.decision_function(scaler.transform(x2[:, 0])) +
            clf.decision_function(scaler.transform(x2[:, 1]))) / 2


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    def nll(t):
        z = logits / t
        z = z - z.max(1, keepdims=True)
        return -(z[np.arange(len(y)), y] - np.log(np.exp(z).sum(1))).mean()
    grid = np.linspace(0.3, 5.0, 95)
    return float(grid[np.argmin([nll(t) for t in grid])])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", nargs="+", required=True)
    ap.add_argument("--clahe", action="store_true")
    ap.add_argument("--C", type=float, default=0.2)
    args = ap.parse_args()

    parts = [extract(b, "data/train", args.clahe) for b in args.backbones]
    y = parts[0][1]
    x2 = np.concatenate([l2(p[0]) for p in parts], axis=-1)

    oof = np.zeros((len(y), len(CLASSES)))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=42).split(x2[:, 0], y):
        s, c = fit_head(x2[tr], y[tr], args.C)
        oof[te] = view_avg_logits(s, c, x2[te])
    cv_acc = float((oof.argmax(1) == y).mean())
    T = fit_temperature(oof, y)
    print(f"OOF CV accuracy: {cv_acc:.3f} | temperature: {T:.2f}")

    scaler, clf = fit_head(x2, y, args.C)
    # The model averages logits over the two views at inference, matching training eval.
    ckpt = {
        "type": "hybrid",
        "backbones": args.backbones,
        "clahe": args.clahe,
        "classes": CLASSES,
        "feat_mean": torch.tensor(scaler.mean_, dtype=torch.float32),
        "feat_scale": torch.tensor(scaler.scale_, dtype=torch.float32),
        "head": {"weight": torch.tensor(clf.coef_, dtype=torch.float32),
                 "bias": torch.tensor(clf.intercept_, dtype=torch.float32)},
        "temperature": T,
        "cv_acc": cv_acc,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, OUT)
    Path("training_metrics_hybrid.json").write_text(json.dumps(
        {"backbones": args.backbones, "clahe": args.clahe, "C": args.C,
         "oof_cv_acc": cv_acc, "temperature": T, "n_train": int(len(y))}, indent=2))
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
