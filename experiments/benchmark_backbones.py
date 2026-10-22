"""Benchmark frozen pretrained backbones (± CLAHE) and feature-fusion hybrids.

Protocol: features are extracted once per (backbone, CLAHE) and cached. A
logistic-regression head is scored with stratified 5-fold CV on data/train
only. data/test is never used here.

    python -m experiments.benchmark_backbones
"""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import timm
import torch
from PIL import Image
from src.gpu_logreg import TorchLogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torchvision import datasets, transforms

from src import mlflow_utils as mlfu
from src.dedup import train_groups
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
    ds = datasets.ImageFolder(root)
    # Key the cache on the exact file list so moving images (dedup) invalidates it.
    fp = hashlib.md5("|".join(p for p, _ in ds.samples).encode()).hexdigest()[:8]
    tag = f"{Path(root).name}_{name}_{'clahe' if clahe else 'raw'}_{fp}.npz"
    f = CACHE / tag
    if f.exists():
        d = np.load(f)
        return d["x"], d["y"]
    model, tf = build(name)
    model = model.to(DEVICE)
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


MERGE_MANIFEST = Path("experiments/merge_manifest.json")


def groups_for(root: str) -> np.ndarray:
    """Near-duplicate group ids in ImageFolder order, so copies never straddle a CV fold.

    For data/train_merged the groups written by src/merge_plantwild.py (MD5 +
    dHash + DINOv2 similarity) are used; otherwise dHash groups.
    """
    paths = [Path(p) for p, _ in datasets.ImageFolder(root).samples]
    if MERGE_MANIFEST.exists():
        known = json.loads(MERGE_MANIFEST.read_text())["train_groups"]
        if all(p.as_posix() in known for p in paths):
            _, ids = np.unique([known[p.as_posix()] for p in paths], return_inverse=True)
            return ids
    return train_groups(paths)


def cv_splits(y: np.ndarray, groups: np.ndarray, seed: int = 42):
    return StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(np.zeros(len(y)), y, groups)


def cv_score(x: np.ndarray, y: np.ndarray, C: float = 0.5, seed: int = 42, groups=None) -> float:
    """Grouped 5-fold CV. Train on both views; predict by averaging view probabilities."""
    groups = np.arange(len(y)) if groups is None else groups
    correct = 0
    for tr, te in cv_splits(y, groups, seed):
        clf = make_pipeline(StandardScaler(), TorchLogisticRegression(C=C))
        clf.fit(x[tr].reshape(-1, x.shape[-1]), np.repeat(y[tr], 2))
        p = clf.predict_proba(x[te, 0]) + clf.predict_proba(x[te, 1])
        correct += (p.argmax(1) == y[te]).sum()
    return correct / len(y)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("backbones", nargs="*")
    ap.add_argument("--train-dir", default="data/train")
    ap.add_argument("--no-fusion", action="store_true")
    args = ap.parse_args()
    names = args.backbones or BACKBONES
    root = args.train_dir
    results, feats = {}, {}
    groups = groups_for(root)
    Cs = (0.002, 0.005, 0.02, 0.05, 0.2)

    with mlfu.mlflow_run(
        run_name=f"backbone-benchmarks-{Path(root).name}",
        tags={"experiment": "backbone_exploration", "train_dir": root},
    ):
        mlfu.log_params({"train_dir": root, "n_train": len(groups), "n_groups": len(np.unique(groups))})
        for name in names:
            for clahe in (False, True):
                x, y = extract(name, root, clahe)
                feats[(name, clahe)] = l2(x)
                best = max((cv_score(feats[(name, clahe)], y, C, groups=groups), C) for C in Cs)
                key = f"{name} | {'CLAHE' if clahe else 'raw'}"
                results[key] = {"cv_acc": round(float(best[0]), 4), "C": best[1]}
                print(f"{key:60s} CV acc {best[0]:.3f} (C={best[1]})", flush=True)
                with mlfu.mlflow_run(run_name=key, nested=True):
                    mlfu.log_params({"backbone": name, "clahe": clahe, "C": best[1]})
                    mlfu.log_metric("cv_acc", float(best[0]))

        # Fusion: concatenate L2-normalised features of the best single configs
        singles = sorted(feats, key=lambda k: -results[f"{k[0]} | {'CLAHE' if k[1] else 'raw'}"]["cv_acc"])
        top = []
        for k in singles:  # best CLAHE setting per backbone
            if k[0] not in [t[0] for t in top]:
                top.append(k)
        top = top[:4]
        for r in (() if args.no_fusion else (2, 3)):
            for combo in itertools.combinations(top, r):
                x = np.concatenate([feats[k] for k in combo], axis=-1)
                best = max((cv_score(x, y, C, groups=groups), C) for C in Cs)
                key = " + ".join(f"{k[0].split('.')[0]}{'(C)' if k[1] else ''}" for k in combo)
                results[f"HYBRID {key}"] = {"cv_acc": round(float(best[0]), 4), "C": best[1]}
                print(f"HYBRID {key:53s} CV acc {best[0]:.3f} (C={best[1]})", flush=True)
                with mlfu.mlflow_run(run_name=f"HYBRID {key}", nested=True):
                    mlfu.log_params({"type": "hybrid", "combo": key, "C": best[1]})
                    mlfu.log_metric("cv_acc", float(best[0]))

        out_path = Path("experiments/benchmark_results.json" if root == "data/train"
                        else f"experiments/benchmark_results_{Path(root).name}.json")
        out_path.write_text(json.dumps(results, indent=2))
        mlfu.log_artifact(out_path, artifact_path="benchmarks")
        mlfu.log_dict(results, "benchmark_results.json")


if __name__ == "__main__":
    main()
