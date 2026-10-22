"""Held-out evaluation for PlantDoc.

Evaluates the full pipeline (quality gate -> model -> fuzzy trust engine) on
data/test (PlantDoc), which has been de-duplicated against the training data
(src/dedup.py, src/merge_plantwild.py), or with --test-dir data/test_plantwild on
the held-out PlantWild split:
- Top-1 / top-3 accuracy with 95% bootstrap confidence intervals
- Per-class precision / recall / F1 on ALL test photos, plus a confusion matrix
- Quality-gate pass rate, trusted rate and accuracy on trusted photos
- Coverage curve (accuracy on accepted photos vs share accepted) with the
  operating threshold marked
Outputs: evaluation_report.json, coverage_curve.png, confusion_matrix.png,
experiments/test_predictions.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from torchvision import datasets

from src import mlflow_utils as mlfu
from src.fuzzy_engine import evaluate_trust
from src.model import CLASSES, PlantDocModel
from src.quality_gate import check_quality

TEST_DIR = Path("data/test")
OUTPUT_PLOT = Path("coverage_curve.png")
OUTPUT_CM = Path("confusion_matrix.png")
OUTPUT_REPORT = Path("evaluation_report.json")
OUTPUT_CSV = Path("experiments/test_predictions.csv")


def bootstrap_ci(values: np.ndarray, n: int = 5000, seed: int = 0) -> list[float]:
    """95% percentile bootstrap interval for the mean of a 0/1 array."""
    if len(values) == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(n, len(values)), replace=True).mean(1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def output_paths(test_dir: Path) -> tuple[Path, Path, Path, Path]:
    """Report, coverage plot, confusion matrix, predictions CSV; suffixed for non-default test sets."""
    if test_dir == TEST_DIR:
        return OUTPUT_REPORT, OUTPUT_PLOT, OUTPUT_CM, OUTPUT_CSV
    sfx = "_" + test_dir.name.removeprefix("test_")
    return (OUTPUT_REPORT.with_stem(OUTPUT_REPORT.stem + sfx), OUTPUT_PLOT.with_stem(OUTPUT_PLOT.stem + sfx),
            OUTPUT_CM.with_stem(OUTPUT_CM.stem + sfx), OUTPUT_CSV.with_stem(OUTPUT_CSV.stem + sfx))


def run_evaluation(
    use_mlflow: bool = True,
    experiment_name: str | None = None,
    run_name: str = "held-out-evaluation",
    test_dir: Path = TEST_DIR,
    weights: str | None = None,
):
    test_dir = Path(test_dir)
    out_report, out_plot, out_cm, out_csv = output_paths(test_dir)
    print(f"=== PlantDoc Held-Out Evaluation ({test_dir}) ===")
    model = PlantDocModel(checkpoint=weights)
    threshold = model.trust_threshold
    print(f"Model: {model.weights_source} | trust threshold {threshold:.2f}")
    if model.is_demo:
        print("Warning: Model is in demo mode! Results will reflect untrained weights.")

    test_dataset = datasets.ImageFolder(str(test_dir))
    print(f"Evaluating on {len(test_dataset)} held-out test samples across {len(CLASSES)} classes.")

    with mlfu.mlflow_run(
        run_name=run_name,
        experiment_name=experiment_name,
        enabled=use_mlflow,
        tags={"eval_type": "held_out_test", "model_source": str(model.weights_source),
              "test_set": test_dir.name, "stage": "evaluate"},
    ):
        mlfu.log_params({
            "model_source": str(model.weights_source),
            "is_demo": model.is_demo,
            "trust_threshold": threshold,
            "total_test_samples": len(test_dataset),
            "test_dir": test_dir.as_posix(),
            "weights_path": weights or "HYBRID_WEIGHTS default",
        })

        results = []
        for img_path, true_label_idx in test_dataset.samples:
            img = Image.open(img_path).convert("RGB")
            true_name = test_dataset.classes[true_label_idx]
            passed_q, q_score, _ = check_quality(img)
            preds, similarity, is_ood = model.diagnose(img, top_k=3)
            top1_name, top1_conf = preds[0]
            trust = evaluate_trust(q_score, top1_conf, threshold=threshold)
            # The app answers only if the photo passes the gate, is in-distribution, and is trusted.
            results.append({
                "path": img_path,
                "true_name": true_name,
                "pred_name": top1_name,
                "top3": "|".join(p[0] for p in preds),
                "confidence": top1_conf,
                "quality_score": q_score,
                "passed_quality": passed_q,
                "trust_score": trust.trust,
                "similarity": similarity,
                "ood": is_ood,
                "trusted": passed_q and not is_ood and trust.status == "TRUSTED",
                "top1_correct": top1_name == true_name,
                "top3_correct": true_name in [p[0] for p in preds],
            })

        total = len(results)
        top1 = np.array([r["top1_correct"] for r in results], dtype=float)
        top3 = np.array([r["top3_correct"] for r in results], dtype=float)
        trusted = np.array([r["trusted"] for r in results])
        passed = np.array([r["passed_quality"] for r in results])
        acc_trusted = top1[trusted]

        print(f"\nTop-1 Accuracy : {top1.mean():.2%}  95% CI {bootstrap_ci(top1)}")
        print(f"Top-3 Accuracy : {top3.mean():.2%}  95% CI {bootstrap_ci(top3)}")
        print(f"Quality Gate Pass Rate : {passed.mean():.1%} ({passed.sum()}/{total})")
        ood = np.array([r["ood"] for r in results])
        print(f"Flagged out-of-distribution: {ood.mean():.1%} ({ood.sum()}/{total}) (all are real leaves)")
        print(f"Trusted Diagnosis Rate : {trusted.mean():.1%} ({trusted.sum()}/{total})")
        if trusted.any():
            print(f"Accuracy on TRUSTED photos: {acc_trusted.mean():.2%}  95% CI {bootstrap_ci(acc_trusted)}")

        # Per-class results on ALL test photos (PS 2.2 asks for per-class reporting)
        y_true = [r["true_name"] for r in results]
        y_pred = [r["pred_name"] for r in results]
        per_class = classification_report(y_true, y_pred, labels=CLASSES, output_dict=True, zero_division=0)
        print("\n" + classification_report(y_true, y_pred, labels=CLASSES, zero_division=0))
        if trusted.any():
            per_class_trusted = classification_report(
                [r["true_name"] for r in results if r["trusted"]],
                [r["pred_name"] for r in results if r["trusted"]],
                labels=CLASSES, output_dict=True, zero_division=0)
        else:
            per_class_trusted = {}

        cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
        fig, ax = plt.subplots(figsize=(8, 7))
        ConfusionMatrixDisplay(cm, display_labels=[c.replace(" ", "\n", 1) for c in CLASSES]).plot(
            ax=ax, cmap="Greens", colorbar=False)
        ax.set_title(f"Confusion matrix, all {total} photos in {test_dir.name}", fontweight="bold")
        plt.tight_layout()
        plt.savefig(out_cm, dpi=200)
        plt.close()

        # Coverage curve: sweep the trust threshold
        thresholds = np.round(np.arange(0.30, 0.86, 0.025), 3)
        coverages, accuracies = [], []
        for t in thresholds:
            subset = [r for r in results if r["passed_quality"] and not r["ood"] and r["trust_score"] >= t]
            coverages.append(len(subset) / total)
            accuracies.append(float(np.mean([r["top1_correct"] for r in subset])) if subset else float("nan"))

        plt.figure(figsize=(8, 5))
        plt.plot(coverages, accuracies, "o-", color="#2e7d32", linewidth=2.5, markersize=5,
                 label="Quality gate + fuzzy trust engine")
        plt.plot([1.0], [top1.mean()], "s", color="#d32f2f", markersize=9,
                 label=f"No filtering: answer every photo ({top1.mean():.1%})")
        if trusted.any():
            plt.plot([trusted.mean()], [acc_trusted.mean()], "*", color="#1565c0", markersize=16,
                     label=f"Operating point, trust >= {threshold:.2f} "
                           f"({trusted.mean():.0%} answered, {acc_trusted.mean():.1%} correct)")
        plt.title("Coverage curve: accuracy on answered photos vs share answered", fontweight="bold")
        plt.xlabel("Coverage (share of photos answered with a diagnosis)")
        plt.ylabel("Accuracy on answered photos")
        plt.xlim(0, 1.05)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend(loc="lower left", fontsize=9)
        plt.tight_layout()
        plt.savefig(out_plot, dpi=200)
        plt.close()

        out_csv.parent.mkdir(exist_ok=True)
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(results[0]))
            w.writeheader()
            w.writerows(results)

        top1_ci = bootstrap_ci(top1)
        top3_ci = bootstrap_ci(top3)
        acc_trusted_ci = bootstrap_ci(acc_trusted) if trusted.any() else [None, None]

        report = {
            "model": model.weights_source,
            "test_dir": test_dir.as_posix(),
            "trust_threshold": threshold,
            "overall": {
                "total_test_samples": total,
                "top1_accuracy": float(top1.mean()),
                "top1_ci95": top1_ci,
                "top3_accuracy": float(top3.mean()),
                "top3_ci95": top3_ci,
                "quality_pass_rate": float(passed.mean()),
                "ood_false_reject_rate": float(ood.mean()),
                "trusted_rate": float(trusted.mean()),
                "accuracy_on_trusted": float(acc_trusted.mean()) if trusted.any() else None,
                "accuracy_on_trusted_ci95": acc_trusted_ci,
            },
            "coverage_curve": {"thresholds": thresholds.tolist(), "coverages": coverages,
                               "accuracies": accuracies},
            "per_class_report": per_class,
            "per_class_report_trusted": per_class_trusted,
            "confusion_matrix": {"labels": CLASSES, "matrix": cm.tolist()},
        }
        out_report.write_text(json.dumps(report, indent=2))
        print(f"Saved {out_report}, {out_plot}, {out_cm}, {out_csv}")

        # Log metrics to MLflow
        ml_metrics = {
            "top1_accuracy": float(top1.mean()),
            "top1_ci95_low": top1_ci[0],
            "top1_ci95_high": top1_ci[1],
            "top3_accuracy": float(top3.mean()),
            "top3_ci95_low": top3_ci[0],
            "top3_ci95_high": top3_ci[1],
            "quality_pass_rate": float(passed.mean()),
            "ood_false_reject_rate": float(ood.mean()),
            "trusted_rate": float(trusted.mean()),
        }
        if trusted.any():
            ml_metrics["accuracy_on_trusted"] = float(acc_trusted.mean())
            if acc_trusted_ci[0] is not None:
                ml_metrics["accuracy_on_trusted_ci95_low"] = acc_trusted_ci[0]
                ml_metrics["accuracy_on_trusted_ci95_high"] = acc_trusted_ci[1]

        for cls_name in CLASSES:
            if cls_name in per_class:
                safe_name = cls_name.lower().replace(" ", "_")
                ml_metrics[f"{safe_name}_precision"] = per_class[cls_name]["precision"]
                ml_metrics[f"{safe_name}_recall"] = per_class[cls_name]["recall"]
                ml_metrics[f"{safe_name}_f1"] = per_class[cls_name]["f1-score"]

        mlfu.log_metrics(ml_metrics)
        mlfu.log_artifact(out_report, artifact_path="evaluation")
        mlfu.log_artifact(out_plot, artifact_path="figures")
        mlfu.log_artifact(out_cm, artifact_path="figures")
        mlfu.log_artifact(out_csv, artifact_path="predictions")

        return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Held-out evaluation for PlantDoc")
    parser.add_argument("--no-mlflow", action="store_true", help="Disable MLflow tracking")
    parser.add_argument("--experiment-name", type=str, default=None, help="MLflow experiment name override")
    parser.add_argument("--run-name", type=str, default=None, help="MLflow run name")
    parser.add_argument("--test-dir", type=Path, default=TEST_DIR,
                        help="data/test (PlantDoc, default) or data/test_plantwild")
    parser.add_argument("--weights", default=None,
                        help="checkpoint to evaluate (default: HYBRID_WEIGHTS, i.e. the app's model)")
    args = parser.parse_args()
    run_evaluation(use_mlflow=not args.no_mlflow, experiment_name=args.experiment_name,
                   run_name=args.run_name or f"held-out-evaluation-{args.test_dir.name}", test_dir=args.test_dir,
                   weights=args.weights)
