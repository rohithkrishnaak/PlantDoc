"""Fit the hybrid feature-fusion classifier (see src/hybrid.py).

    python train_hybrid.py --backbones convnext_tiny.fb_in22k_ft_in1k vit_base_patch14_dinov2.lvd142m --clahe --C 0.2

Uses data/train only. Grouped 5-fold CV (near-duplicates share a fold) gives
out-of-fold logits, which are used to fit the temperature and to choose the
trust threshold, so the test split is never used for any decision.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from PIL import Image
from torchvision import datasets

from experiments.benchmark_backbones import cv_splits, extract, groups_for, l2
from src.fuzzy_engine import evaluate_trust
from src.model import CLASSES
from src.quality_gate import check_quality

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


def choose_trust_threshold(probs: np.ndarray, y: np.ndarray, quality: np.ndarray,
                           passed: np.ndarray, target_acc: float = 0.80):
    """Lowest trust threshold whose accepted out-of-fold photos reach target accuracy.

    Lower threshold = more photos answered. Returns (threshold, coverage, accuracy, sweep).
    """
    trust = np.array([evaluate_trust(q, c, threshold=0.0).trust for q, c in zip(quality, probs.max(1))])
    correct = probs.argmax(1) == y
    sweep = []
    for t in np.round(np.arange(0.30, 0.86, 0.01), 2):
        acc_mask = passed & (trust >= t)
        if acc_mask.sum() >= 20:  # ignore thresholds resting on a handful of photos
            sweep.append((float(t), float(acc_mask.mean()), float(correct[acc_mask].mean())))
    ok = [s for s in sweep if s[2] >= target_acc]
    best = min(ok, key=lambda s: s[0]) if ok else max(sweep, key=lambda s: s[2])
    return best[0], best[1], best[2], sweep


def ood_bank(scaler, x2: np.ndarray, groups: np.ndarray, percentile: float = 2.0):
    """Feature bank + threshold for the out-of-distribution check.

    Each training photo's similarity to its nearest *other* training photo
    (excluding its own near-duplicate group) approximates how close a new
    in-distribution photo will be. The threshold is a low percentile of that.
    """
    f = scaler.transform(x2.mean(1))
    f /= np.linalg.norm(f, axis=1, keepdims=True)
    sim = f @ f.T
    sim[groups[:, None] == groups[None, :]] = -1.0
    nn_sim = sim.max(1)
    return f.astype(np.float16), float(np.percentile(nn_sim, percentile)), nn_sim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", nargs="+", required=True)
    ap.add_argument("--clahe", action="store_true")
    ap.add_argument("--C", type=float, default=0.2)
    ap.add_argument("--target-acc", type=float, default=0.80,
                    help="accuracy required on accepted photos when choosing the trust threshold")
    args = ap.parse_args()

    parts = [extract(b, "data/train", args.clahe) for b in args.backbones]
    y = parts[0][1]
    x2 = np.concatenate([l2(p[0]) for p in parts], axis=-1)

    groups = groups_for("data/train")
    oof = np.zeros((len(y), len(CLASSES)))
    for tr, te in cv_splits(y, groups):
        s, c = fit_head(x2[tr], y[tr], args.C)
        oof[te] = view_avg_logits(s, c, x2[te])
    cv_acc = float((oof.argmax(1) == y).mean())
    T = fit_temperature(oof, y)
    print(f"OOF CV accuracy: {cv_acc:.3f} | temperature: {T:.2f}")

    # Trust threshold from out-of-fold predictions + quality of the raw training photos
    z = oof / T
    probs = np.exp(z - z.max(1, keepdims=True))
    probs /= probs.sum(1, keepdims=True)
    q = [check_quality(Image.open(p).convert("RGB")) for p, _ in datasets.ImageFolder("data/train").samples]
    passed = np.array([r[0] for r in q])
    quality = np.array([r[1] for r in q])
    thr, cov, acc, sweep = choose_trust_threshold(probs, y, quality, passed, args.target_acc)
    print(f"Trust threshold {thr:.2f}: OOF coverage {cov:.1%}, accuracy on accepted {acc:.1%}")

    scaler, clf = fit_head(x2, y, args.C)
    bank, ood_thr, nn_sim = ood_bank(scaler, x2, groups)
    print(f"OOD threshold (2nd pct of leave-group-out NN similarity): {ood_thr:.3f}")
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
        "trust_threshold": thr,
        "ood_bank": torch.tensor(bank),
        "ood_threshold": ood_thr,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, OUT)
    Path("training_metrics_hybrid.json").write_text(json.dumps(
        {"backbones": args.backbones, "clahe": args.clahe, "C": args.C,
         "oof_cv_acc": cv_acc, "temperature": T, "n_train": int(len(y)),
         "trust_threshold": thr, "target_acc": args.target_acc, "ood_threshold": ood_thr,
         "oof_coverage_at_threshold": cov, "oof_accuracy_at_threshold": acc,
         "oof_trust_sweep": [{"threshold": t, "coverage": c, "accuracy": a} for t, c, a in sweep]}, indent=2))
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
