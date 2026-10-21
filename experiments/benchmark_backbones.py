"""Benchmark frozen pretrained backbones (± CLAHE) and feature-fusion hybrids.

Protocol: features are extracted once per (backbone, CLAHE) and cached. A
logistic-regression head is scored with stratified 5-fold CV on data/train
only. data/test is never used here.

    python -m experiments.benchmark_backbones
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import timm
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torchvision import datasets, transforms

from src.preprocess import apply_clahe

CACHE = Path("experiments/cache")
BACKBONES = [
    "efficientnet_b0.ra_in1k",
    "tf_efficientnetv2_s.in21k_ft_in1k",
    "convnext_tiny.fb_in22k_ft_in1k",
    "swin_tiny_patch4_window7_224.ms_in22k_ft_in1k",
    "vit_small_patch14_dinov2.lvd142m",
    "vit_base_patch14_dinov2.lvd142m",
]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build(name: str):
    kw = {"img_size": 224} if "dinov2" in name else {}
    m = timm.create_model(name, pretrained=True, num_classes=0, **kw).eval()
    cfg = timm.data.resolve_data_config({}, model=m)
    tf = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize(cfg["mean"], cfg["std"]),
    ])
    return m, tf


@torch.inference_mode()
def extract(name: str, root: str, clahe: bool) -> tuple[np.ndarray, np.ndarray]:
    """Returns features [N, 2, D] (original + hflip view) and labels."""
    tag = f"{Path(root).name}_{name}_{'clahe' if clahe else 'raw'}.npz"
    f = CACHE / tag
    if f.exists():
        d = np.load(f)
        return d["x"], d["y"]
    model, tf = build(name)
    model = model.to(DEVICE)
    ds = datasets.ImageFolder(root)
    feats = []
    batch, out = [], []
    for path, _ in ds.samples:
        img = Image.open(path).convert("RGB")
        if clahe:
            img = apply_clahe(img)
        t = tf(img)
        batch += [t, torch.flip(t, dims=[2])]
        if len(batch) >= 64:
            out.append(model(torch.stack(batch).to(DEVICE)).float().cpu().numpy()); batch = []
    if batch:
        out.append(model(torch.stack(batch).to(DEVICE)).float().cpu().numpy())
    x = np.concatenate(out).reshape(len(ds), 2, -1)
    y = np.array(ds.targets)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(f, x=x, y=y)
    return x, y


def l2(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def cv_score(x: np.ndarray, y: np.ndarray, C: float = 0.5, seed: int = 42) -> float:
    """5-fold CV. Train on both views; predict by averaging view probabilities."""
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    correct = 0
    for tr, te in skf.split(x[:, 0], y):
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000,
                                                                 class_weight="balanced"))
        clf.fit(x[tr].reshape(-1, x.shape[-1]), np.repeat(y[tr], 2))
        p = clf.predict_proba(x[te, 0]) + clf.predict_proba(x[te, 1])
        correct += (p.argmax(1) == y[te]).sum()
    return correct / len(y)


def main():
    names = sys.argv[1:] or BACKBONES
    results, feats = {}, {}
    for name in names:
        for clahe in (False, True):
            x, y = extract(name, "data/train", clahe)
            feats[(name, clahe)] = l2(x)
            best = max((cv_score(feats[(name, clahe)], y, C), C) for C in (0.05, 0.2, 1.0))
            key = f"{name} | {'CLAHE' if clahe else 'raw'}"
            results[key] = {"cv_acc": round(float(best[0]), 4), "C": best[1]}
            print(f"{key:60s} CV acc {best[0]:.3f} (C={best[1]})", flush=True)

    # Fusion: concatenate L2-normalised features of the best single configs
    singles = sorted(feats, key=lambda k: -results[f"{k[0]} | {'CLAHE' if k[1] else 'raw'}"]["cv_acc"])
    top = []
    for k in singles:  # best CLAHE setting per backbone
        if k[0] not in [t[0] for t in top]:
            top.append(k)
    top = top[:4]
    for r in (2, 3):
        for combo in itertools.combinations(top, r):
            x = np.concatenate([feats[k] for k in combo], axis=-1)
            best = max((cv_score(x, y, C), C) for C in (0.05, 0.2, 1.0))
            key = " + ".join(f"{k[0].split('.')[0]}{'(C)' if k[1] else ''}" for k in combo)
            results[f"HYBRID {key}"] = {"cv_acc": round(float(best[0]), 4), "C": best[1]}
            print(f"HYBRID {key:53s} CV acc {best[0]:.3f} (C={best[1]})", flush=True)

    Path("experiments/benchmark_results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
