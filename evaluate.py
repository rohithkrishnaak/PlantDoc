"""Comprehensive evaluation script for PlantDoc.

Evaluates the fine-tuned model and fuzzy trust engine on the held-out test split:
- Top-1 and Top-3 accuracy
- Per-class Precision, Recall, and F1-score
- Quality Gate rejection rate
- Coverage Curve (Selective Accuracy vs. Trust/Rejection threshold)
- Generates coverage_curve.png plot and evaluation_report.json
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from torchvision import datasets

from src.fuzzy_engine import evaluate_trust
from src.model import CLASSES, PREPROCESS, PlantDocModel
from src.quality_gate import check_quality

TEST_DIR = Path("data/test")
OUTPUT_PLOT = Path("coverage_curve.png")
OUTPUT_REPORT = Path("evaluation_report.json")


def run_evaluation():
    print("=== PlantDoc Held-Out Evaluation ===")
    model = PlantDocModel()
    if model.is_demo:
        print("Warning: Model is in demo mode! Results will reflect untrained weights.")

    test_dataset = datasets.ImageFolder(str(TEST_DIR))
    print(f"Evaluating on {len(test_dataset)} held-out test samples across {len(CLASSES)} classes.")

    results = []
    
    for img_path, true_label_idx in test_dataset.samples:
        img = Image.open(img_path).convert("RGB")
        true_name = test_dataset.classes[true_label_idx]

        # 1. Quality gate
        passed_q, q_score, q_msg = check_quality(img)

        # 2. Model prediction
        preds = model.predict(img, top_k=3)
        top1_name, top1_conf = preds[0]
        top3_names = [p[0] for p in preds]

        # 3. Trust engine
        trust = evaluate_trust(q_score, top1_conf)

        is_top1_correct = (top1_name == true_name)
        is_top3_correct = (true_name in top3_names)

        results.append({
            "path": img_path,
            "true_name": true_name,
            "pred_name": top1_name,
            "confidence": top1_conf,
            "quality_score": q_score,
            "passed_quality": passed_q,
            "trust_score": trust.trust,
            "trust_status": trust.status,
            "top1_correct": is_top1_correct,
            "top3_correct": is_top3_correct,
        })

    # Overall metrics
    total = len(results)
    top1_acc = sum(r["top1_correct"] for r in results) / total
    top3_acc = sum(r["top3_correct"] for r in results) / total
    passed_q_count = sum(r["passed_quality"] for r in results)
    trusted_count = sum(r["trust_status"] == "TRUSTED" for r in results)

    print(f"\n--- Baseline Test Metrics ---")
    print(f"Top-1 Accuracy : {top1_acc:.2%}")
    print(f"Top-3 Accuracy : {top3_acc:.2%}")
    print(f"Quality Gate Pass Rate : {passed_q_count / total:.1%} ({passed_q_count}/{total})")
    print(f"Trusted Diagnosis Rate : {trusted_count / total:.1%} ({trusted_count}/{total})")

    # Classification report on accepted samples
    accepted = [r for r in results if r["passed_quality"] and r["trust_status"] == "TRUSTED"]
    if accepted:
        y_true = [r["true_name"] for r in accepted]
        y_pred = [r["pred_name"] for r in accepted]
        clf_rep = classification_report(y_true, y_pred, labels=CLASSES, output_dict=True, zero_division=0)
        acc_on_trusted = sum(r["top1_correct"] for r in accepted) / len(accepted)
        print(f"\nAccuracy on TRUSTED samples only: {acc_on_trusted:.2%} (vs {top1_acc:.2%} overall)")
    else:
        clf_rep = {}
        acc_on_trusted = 0.0

    # Coverage Curve Calculation
    # Sweep threshold from 0.30 to 0.90
    thresholds = np.linspace(0.35, 0.85, 20)
    coverages = []
    accuracies = []

    for t in thresholds:
        subset = [r for r in results if r["passed_quality"] and r["trust_score"] >= t]
        if subset:
            cov = len(subset) / total
            acc = sum(r["top1_correct"] for r in subset) / len(subset)
        else:
            cov = 0.0
            acc = 1.0
        coverages.append(cov)
        accuracies.append(acc)

    # Plot Coverage Curve
    plt.figure(figsize=(8, 5))
    plt.plot(coverages, accuracies, "o-", color="#2e7d32", linewidth=2.5, markersize=6, label="PlantDoc Trust Engine")
    plt.axhline(y=top1_acc, color="#d32f2f", linestyle="--", label=f"Unfiltered Baseline ({top1_acc:.1%})")
    plt.title("Coverage Curve: Accuracy on Accepted Photos vs. System Coverage", fontsize=12, fontweight="bold")
    plt.xlabel("Coverage (Fraction of Photos Accepted)", fontsize=11)
    plt.ylabel("Selective Diagnostic Accuracy", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=200)
    plt.close()
    print(f"Coverage curve chart saved to {OUTPUT_PLOT}")

    # Save full report
    report = {
        "overall": {
            "total_test_samples": total,
            "top1_accuracy": top1_acc,
            "top3_accuracy": top3_acc,
            "quality_pass_rate": passed_q_count / total,
            "trusted_rate": trusted_count / total,
            "accuracy_on_trusted": acc_on_trusted,
        },
        "coverage_curve": {
            "thresholds": thresholds.tolist(),
            "coverages": coverages,
            "accuracies": accuracies,
        },
        "per_class_report": clf_rep,
    }

    with open(OUTPUT_REPORT, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Full evaluation report saved to {OUTPUT_REPORT}")
    return report


if __name__ == "__main__":
    run_evaluation()
