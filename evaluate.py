"""Held-out evaluation for PlantDoc.

Evaluates the full pipeline (quality gate -> model -> fuzzy trust engine) on
data/test, which has been de-duplicated against data/train (src/dedup.py):
- Top-1 / top-3 accuracy with 95% bootstrap confidence intervals
- Per-class precision / recall / F1 on ALL test photos, plus a confusion matrix
- Quality-gate pass rate, trusted rate and accuracy on trusted photos
- Coverage curve (accuracy on accepted photos vs share accepted) with the
  operating threshold marked
Outputs: evaluation_report.json, coverage_curve.png, confusion_matrix.png,
experiments/test_predictions.csv
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from torchvision import datasets

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


def run_evaluation():
    print("=== PlantDoc Held-Out Evaluation ===")
    model = PlantDocModel()
    threshold = model.trust_threshold
    print(f"Model: {model.weights_source} | trust threshold {threshold:.2f}")
    if model.is_demo:
        print("Warning: Model is in demo mode! Results will reflect untrained weights.")

    test_dataset = datasets.ImageFolder(str(TEST_DIR))
    print(f"Evaluating on {len(test_dataset)} held-out test samples across {len(CLASSES)} classes.")

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
    ax.set_title(f"Confusion matrix, all {total} de-duplicated test photos", fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_CM, dpi=200)
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
    plt.savefig(OUTPUT_PLOT, dpi=200)
    plt.close()

    OUTPUT_CSV.parent.mkdir(exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0]))
        w.writeheader()
        w.writerows(results)

    report = {
        "model": model.weights_source,
        "trust_threshold": threshold,
        "overall": {
            "total_test_samples": total,
            "top1_accuracy": float(top1.mean()),
            "top1_ci95": bootstrap_ci(top1),
            "top3_accuracy": float(top3.mean()),
            "top3_ci95": bootstrap_ci(top3),
            "quality_pass_rate": float(passed.mean()),
            "ood_false_reject_rate": float(ood.mean()),
            "trusted_rate": float(trusted.mean()),
            "accuracy_on_trusted": float(acc_trusted.mean()) if trusted.any() else None,
            "accuracy_on_trusted_ci95": bootstrap_ci(acc_trusted),
        },
        "coverage_curve": {"thresholds": thresholds.tolist(), "coverages": coverages,
                           "accuracies": accuracies},
        "per_class_report": per_class,
        "per_class_report_trusted": per_class_trusted,
        "confusion_matrix": {"labels": CLASSES, "matrix": cm.tolist()},
    }
    OUTPUT_REPORT.write_text(json.dumps(report, indent=2))
    print(f"Saved {OUTPUT_REPORT}, {OUTPUT_PLOT}, {OUTPUT_CM}, {OUTPUT_CSV}")
    return report


if __name__ == "__main__":
    run_evaluation()
