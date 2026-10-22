# MLflow Experiment Tracking & Model Management Guide 🧪

This project uses **MLflow** for end-to-end machine learning experiment tracking, hyperparameter logging, metric visualization, and artifact management across all model training, backbone benchmarking, and held-out evaluation workflows.

---

## 🚀 Quick Start

### 1. Launch the MLflow UI Dashboard
Start the local MLflow tracking server and UI:
```bash
python launch_mlflow.py
```
Or directly using MLflow CLI:
```bash
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```
Open **[http://127.0.0.1:5000](http://127.0.0.1:5000)** in your browser to inspect runs, compare parameters, analyze metrics, and inspect logged plots and model checkpoints.

---

## ⚙️ Configuration & Environment Variables

MLflow configuration can be customized via `.env` or system environment variables:

| Variable | Default | Description |
|---|---|---|
| `MLFLOW_TRACKING_URI` | `sqlite:///mlflow.db` | Tracking backend store (SQLite database or remote URL). |
| `MLFLOW_EXPERIMENT_NAME` | `PlantDoc-NexAi` | Default MLflow experiment name. |
| `MLFLOW_TRACKING_DISABLED` | `0` | Set to `1` to disable MLflow tracking globally if needed. |

---

## 🛠️ Integrated Workflows

### 1. Held-Out Model Evaluation (`evaluate.py`)
Runs held-out evaluation on `data/test`, logging all diagnostic and safety metrics:
```bash
python evaluate.py
```
**Logged to MLflow:**
- **Parameters:** model architecture, weights source, demo mode flag, trust threshold, test sample count.
- **Metrics:** Top-1 accuracy, Top-3 accuracy, 95% bootstrap confidence intervals (`top1_ci95_low`/`high`, `top3_ci95_low`/`high`), quality gate pass rate, OOD rejection rate, trusted diagnosis rate, accuracy on trusted samples with CIs, per-class precision, recall, and F1 scores.
- **Artifacts:**
  - `evaluation/evaluation_report.json`
  - `figures/coverage_curve.png`
  - `figures/confusion_matrix.png`
  - `predictions/test_predictions.csv`

To disable MLflow for an evaluation run:
```bash
python evaluate.py --no-mlflow
```

---

### 2. Hybrid Feature-Fusion Training (`train_hybrid.py`)
Trains the hybrid multi-backbone classifier with out-of-fold calibration and OOD feature bank:
```bash
python train_hybrid.py --backbones convnext_tiny.fb_in22k_ft_in1k vit_base_patch14_dinov2.lvd142m --clahe --C 0.2
```
**Logged to MLflow:**
- **Parameters:** backbones, CLAHE setting, regularization $C$, target accuracy, class list, training sample count.
- **Metrics:** Out-of-fold CV accuracy, optimal temperature $T$, chosen trust threshold, OOF coverage at threshold, OOF accuracy at threshold, OOD similarity threshold.
- **Artifacts:**
  - `checkpoints/plantdoc_hybrid.pth`
  - `metrics/training_metrics_hybrid.json`
  - `metrics_hybrid.json` (including threshold sweep details)

---

### 3. Backbone Transfer Learning (`train.py`)
Fine-tunes the single backbone classifier (warmup + fine-tuning):
```bash
python train.py
```
**Logged to MLflow:**
- **Parameters:** Backbone architecture, batch size, learning rates, epochs (warmup & finetune), loss weights.
- **Step-by-step Metrics:** Training loss, training accuracy, validation loss, validation accuracy, validation top-3 per epoch.
- **Artifacts:**
  - `checkpoints/plantdoc_effnet_b0.pth`
  - `metrics/training_metrics.json`
  - `training_history.json`

---

### 4. Backbone Benchmarking (`experiments/benchmark_backbones.py`)
Evaluates candidate backbones and feature fusion pairs with grouped 5-fold CV:
```bash
python -m experiments.benchmark_backbones
```
**Logged to MLflow:**
- Hierarchical runs: parent run `backbone-benchmarks` with nested runs for each candidate backbone and combination.
- CV accuracy and best regularization parameters logged per model.
- Complete benchmark summary JSON saved as artifact.

---

### 5. PlantWild Merge, Fine-Tuning, Registry & Leaderboard
```bash
python -m src.merge_plantwild                       # run "data-merge-plantwild": leak-check metrics + manifest
python train_finetune.py --epochs 20                # parent run + nested run per CV fold (per-epoch curves)
python evaluate.py --test-dir data/test             # tagged test_set=test
python evaluate.py --test-dir data/test_plantwild   # tagged test_set=test_plantwild
python register_model.py --weights weights/plantdoc_finetuned.pth --alias champion
python -m experiments.mlflow_leaderboard            # ranks training runs by OOF CV accuracy
```
- Training runs record the exact training file list as an MLflow **dataset input** (lineage).
- In the fold runs, compare `train_acc` vs `val_acc` to spot over-fitting. `cv_val_acc_by_epoch` on the parent run is the curve used to pick the epoch count.
- Load the registered model with `mlflow.pyfunc.load_model("models:/PlantDoc-LeafDoctor@champion")`. Input is a DataFrame with a `path` or `image_b64` column. Version tags hold its CV and held-out accuracy.

---

## 🧪 Testing the MLflow Setup

Run the automated MLflow test suite:
```bash
python tests/test_mlflow.py
```
This runs 5 comprehensive automated tests in an isolated temporary environment verifying:
1. MLflow availability and version compatibility.
2. Tracking URI and experiment initialization.
3. Parameter, multi-step metric, tag, dictionary, figure, and file artifact logging.
4. Nested parent/child run relationships.
5. Safe no-op behavior when tracking is disabled.
