"""Print (and save) a leaderboard of training runs from the MLflow store.

    python -m experiments.mlflow_leaderboard

Ranks every top-level training run (stage = train) by out-of-fold CV accuracy,
which is the only number used to choose a model, and shows the held-out test
accuracy of the matching evaluation runs next to it for reference.
"""
from __future__ import annotations

from pathlib import Path

from src import mlflow_utils as mlfu

OUT = Path("experiments/mlflow_leaderboard.csv")


def main():
    import mlflow

    exp_id = mlfu.setup_mlflow()
    train = mlflow.search_runs([exp_id], filter_string="tags.stage = 'train'", output_format="pandas")
    train = train[train["tags.mlflow.parentRunId"].isna()] if "tags.mlflow.parentRunId" in train else train
    cols = {"tags.mlflow.runName": "run", "params.train_dir": "train_dir", "metrics.oof_cv_acc": "oof_cv_acc",
            "metrics.oof_cv_acc_pd": "oof_pd", "metrics.oof_cv_acc_pw": "oof_pw", "metrics.train_acc": "train_acc",
            "metrics.train_oof_gap": "gap", "metrics.trust_threshold": "trust_thr",
            "metrics.oof_coverage_at_threshold": "oof_cov", "start_time": "start"}
    board = train[[c for c in cols if c in train]].rename(columns=cols).sort_values("oof_cv_acc", ascending=False)
    ev = mlflow.search_runs([exp_id], filter_string="tags.stage = 'evaluate'", output_format="pandas")
    if len(ev):
        ev = ev.sort_values("start_time").groupby(["tags.model_source", "tags.test_set"]).tail(1)
        piv = ev.pivot(index="tags.model_source", columns="tags.test_set", values="metrics.top1_accuracy")
        print("\nLatest held-out top-1 per model (reference only, never used for selection):")
        print(piv.round(3).to_string())
    print("\nTraining runs by OOF CV accuracy:")
    print(board.round(4).to_string(index=False))
    OUT.parent.mkdir(exist_ok=True)
    board.to_csv(OUT, index=False)
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
