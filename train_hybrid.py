"""Fit the hybrid feature-fusion classifier (see src/hybrid.py).

    python train_hybrid.py --backbones convnext_tiny.fb_in22k_ft_in1k vit_base_patch14_dinov2.lvd142m --clahe --C 0.02 0.05 0.2

Uses the training directory only (default data/train_merged = de-duplicated
PlantDoc train + PlantWild, see src/merge_plantwild.py). Grouped 5-fold CV
(near-duplicates share a fold) gives out-of-fold logits, which are used to pick
C, fit the temperature and choose the trust threshold, so neither test split
(data/test, data/test_plantwild) is used for any decision.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from src.gpu_logreg import TorchLogisticRegression
from sklearn.preprocessing import StandardScaler

from PIL import Image
from torchvision import datasets

from experiments.benchmark_backbones import cv_splits, extract, groups_for, l2
from src import mlflow_utils as mlfu
from src.fuzzy_engine import evaluate_trust
from src.model import CLASSES
from src.quality_gate import check_quality

OUT = Path("weights/plantdoc_hybrid.pth")
METRICS_OUT = Path("training_metrics_hybrid.json")


def fit_head(x2: np.ndarray, y: np.ndarray, C: float):
    """x2: [N, 2, D] (original + hflip). Train on both views."""
    scaler = StandardScaler().fit(x2.reshape(-1, x2.shape[-1]))
    clf = TorchLogisticRegression(C=C)  # same objective as sklearn LogisticRegression(class_weight="balanced")
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


def photo_source(path: str) -> str:
    """'pd' (PlantDoc) or 'pw' (PlantWild); data/train_merged files carry a pd__ / pw__ prefix."""
    name = Path(path).name
    return name.split("__")[0] if "__" in name else "pd"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", nargs="+", required=True)
    ap.add_argument("--clahe", action="store_true")
    ap.add_argument("--C", type=float, nargs="+", default=[0.2],
                    help="one or more inverse regularisation strengths; the best by OOF accuracy is kept")
    ap.add_argument("--train-dir", default="data/train_merged")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--metrics-out", default=str(METRICS_OUT))
    ap.add_argument("--target-acc", type=float, default=0.80,
                    help="accuracy required on accepted photos when choosing the trust threshold")
    ap.add_argument("--no-mlflow", action="store_true", help="Disable MLflow tracking for this run")
    ap.add_argument("--experiment-name", type=str, default=None, help="MLflow experiment name override")
    ap.add_argument("--run-name", type=str, default=None, help="MLflow run name")
    args = ap.parse_args()

    run_name = args.run_name or f"hybrid-{'+'.join(b.split('.')[0] for b in args.backbones)}"
    out, metrics_out, root = Path(args.out), Path(args.metrics_out), args.train_dir

    with mlfu.mlflow_run(
        run_name=run_name,
        experiment_name=args.experiment_name,
        enabled=not args.no_mlflow,
        tags={"model_type": "hybrid_fusion", "clahe": str(args.clahe), "stage": "train", "train_dir": root},
    ):
        mlfu.log_params({
            "backbones": args.backbones,
            "clahe": args.clahe,
            "C_grid": args.C,
            "target_acc": args.target_acc,
            "num_classes": len(CLASSES),
            "classes": ",".join(CLASSES),
            "train_dir": root,
        })

        parts = [extract(b, root, args.clahe) for b in args.backbones]
        y = parts[0][1]
        x2 = np.concatenate([l2(p[0]) for p in parts], axis=-1)
        samples = datasets.ImageFolder(root).samples
        source = np.array([photo_source(p) for p, _ in samples])
        groups = groups_for(root)
        mlfu.log_params({"n_train": int(len(y)), "n_groups": int(len(np.unique(groups))),
                         "feat_dim": int(x2.shape[-1]),
                         **{f"n_train_{s}": int((source == s).sum()) for s in np.unique(source)}})
        _log_dataset(samples, root)

        # Model selection: grouped 5-fold CV over the C grid, training data only.
        by_C = {}
        for C in args.C:
            oof_c = np.zeros((len(y), len(CLASSES)))
            for tr, te in cv_splits(y, groups):
                s, c = fit_head(x2[tr], y[tr], C)
                oof_c[te] = view_avg_logits(s, c, x2[te])
            by_C[C] = (float((oof_c.argmax(1) == y).mean()), oof_c)
            print(f"C={C:<6} OOF CV accuracy {by_C[C][0]:.3f}", flush=True)
        for i, C in enumerate(args.C):
            mlfu.log_metric("cv_acc_by_C", by_C[C][0], step=i)
        best_C = max(args.C, key=lambda c: by_C[c][0])
        cv_acc, oof = by_C[best_C]
        correct = oof.argmax(1) == y
        per_source = {f"oof_cv_acc_{s}": float(correct[source == s].mean()) for s in np.unique(source)}
        T = fit_temperature(oof, y)
        print(f"best C={best_C} | OOF CV accuracy: {cv_acc:.3f} {per_source} | temperature: {T:.2f}")

        # Trust threshold from out-of-fold predictions + quality of the raw training photos
        z = oof / T
        probs = np.exp(z - z.max(1, keepdims=True))
        probs /= probs.sum(1, keepdims=True)
        q = [check_quality(Image.open(p).convert("RGB")) for p, _ in samples]
        passed = np.array([r[0] for r in q])
        quality = np.array([r[1] for r in q])
        thr, cov, acc, sweep = choose_trust_threshold(probs, y, quality, passed, args.target_acc)
        print(f"Trust threshold {thr:.2f}: OOF coverage {cov:.1%}, accuracy on accepted {acc:.1%}")

        scaler, clf = fit_head(x2, y, best_C)
        train_acc = float((view_avg_logits(scaler, clf, x2).argmax(1) == y).mean())
        print(f"Train accuracy {train_acc:.3f} vs OOF {cv_acc:.3f} (gap {train_acc - cv_acc:+.3f})")
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
            "train_dir": root,
        }
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(ckpt, out)

        per_class = {f"oof_recall_{c.lower().replace(' ', '_')}": float(correct[y == k].mean())
                     for k, c in enumerate(CLASSES)}
        metrics_data = {
            "backbones": args.backbones, "clahe": args.clahe, "C": best_C, "train_dir": root,
            "C_sweep": {str(c): v[0] for c, v in by_C.items()},
            "oof_cv_acc": cv_acc, **per_source, "train_acc": train_acc, "temperature": T,
            "n_train": int(len(y)), "trust_threshold": thr, "target_acc": args.target_acc,
            "ood_threshold": ood_thr,
            "oof_coverage_at_threshold": cov, "oof_accuracy_at_threshold": acc,
            "oof_per_class_recall": per_class,
            "oof_trust_sweep": [{"threshold": t, "coverage": c, "accuracy": a} for t, c, a in sweep]
        }
        metrics_out.write_text(json.dumps(metrics_data, indent=2))
        print(f"Saved {out}")

        mlfu.log_param("C", best_C)
        mlfu.log_metrics({
            "oof_cv_acc": cv_acc,
            **per_source,
            "train_acc": train_acc,
            "train_oof_gap": train_acc - cv_acc,
            "temperature": T,
            "trust_threshold": thr,
            "oof_coverage_at_threshold": cov,
            "oof_accuracy_at_threshold": acc,
            "ood_threshold": ood_thr,
            **per_class,
        })
        for t, c, a in sweep:  # step = threshold x 100, so MLflow plots the sweep as a curve
            mlfu.log_metrics({"oof_sweep_coverage": c, "oof_sweep_accuracy": a}, step=int(round(t * 100)))
        _log_oof_confusion(y, oof.argmax(1))
        mlfu.log_artifact(out, artifact_path="checkpoints")
        mlfu.log_artifact(metrics_out, artifact_path="metrics")


def _log_dataset(samples, root: str):
    """Record the exact training file list as an MLflow dataset input (lineage)."""
    if not mlfu.is_tracking_enabled() or mlfu.mlflow.active_run() is None:
        return
    try:
        import pandas as pd
        df = pd.DataFrame({"path": [Path(p).as_posix() for p, _ in samples],
                           "label": [CLASSES[t] for _, t in samples]})
        mlfu.mlflow.log_input(mlfu.mlflow.data.from_pandas(df, source=root, name=Path(root).name),
                              context="training")
    except Exception as e:
        print(f"[MLflow Warning] dataset logging failed: {e}")


def _log_oof_confusion(y, pred):
    if not mlfu.is_tracking_enabled() or mlfu.mlflow.active_run() is None:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay.from_predictions(
        y, pred, labels=range(len(CLASSES)), display_labels=[c.replace(" ", "\n", 1) for c in CLASSES],
        cmap="Blues", colorbar=False, ax=ax)
    ax.set_title("Out-of-fold confusion matrix (training CV)")
    plt.tight_layout()
    mlfu.log_figure(fig, "figures/oof_confusion_matrix.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
