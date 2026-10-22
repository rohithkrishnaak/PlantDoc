"""Automated test suite for MLflow integration in PlantDoc / NexAi."""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
from mlflow.tracking import MlflowClient

from src import mlflow_utils as mlfu


class TestMLflowIntegration(unittest.TestCase):
    def setUp(self):
        # Create an isolated temporary directory for testing MLflow tracking
        self.test_dir = tempfile.mkdtemp(prefix="plantdoc_mlflow_test_")
        db_path = Path(self.test_dir) / "test_mlflow.db"
        self.tracking_uri = f"sqlite:///{db_path.as_posix()}"
        os.environ["MLFLOW_TRACKING_URI"] = self.tracking_uri
        os.environ["MLFLOW_EXPERIMENT_NAME"] = "Test-PlantDoc-Experiment"
        os.environ["MLFLOW_TRACKING_DISABLED"] = "0"
        os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
        mlflow.set_tracking_uri(self.tracking_uri)
        self.client = MlflowClient(tracking_uri=self.tracking_uri)

    def tearDown(self):
        if mlflow.active_run() is not None:
            mlflow.end_run()
        # Clean up temp dir
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_mlflow_available(self):
        self.assertTrue(mlfu.is_mlflow_available())
        self.assertTrue(mlfu.is_tracking_enabled())

    def test_setup_mlflow(self):
        exp_id = mlfu.setup_mlflow("Test-Setup-Exp", tracking_uri=self.tracking_uri)
        self.assertIsNotNone(exp_id)
        exp = self.client.get_experiment_by_name("Test-Setup-Exp")
        self.assertIsNotNone(exp)
        self.assertEqual(exp.experiment_id, exp_id)

    def test_logging_params_metrics_artifacts(self):
        with mlfu.mlflow_run(
            run_name="unit-test-run",
            experiment_name="Test-Logging-Exp",
            tags={"framework": "pytorch", "model_type": "classifier"},
        ) as run:
            self.assertIsNotNone(run)
            run_id = run.info.run_id

            # Log parameters
            mlfu.log_params({
                "backbone": "efficientnet_b0",
                "batch_size": 16,
                "learning_rate": 0.001,
                "classes": ["healthy", "blight"],
            })

            # Log metrics across steps
            mlfu.log_metrics({"train_loss": 0.54, "train_acc": 0.82}, step=1)
            mlfu.log_metrics({"train_loss": 0.32, "train_acc": 0.91}, step=2)
            mlfu.log_metric("final_val_acc", 0.94)

            # Log dictionary artifact
            test_dict = {"accuracy": 0.94, "epochs": 2, "status": "completed"}
            mlfu.log_dict(test_dict, "summary.json")

            # Log figure artifact
            fig, ax = plt.subplots()
            ax.plot([1, 2], [0.82, 0.91])
            ax.set_title("Test Curve")
            mlfu.log_figure(fig, "figures/curve.png")
            plt.close(fig)

            # Log file artifact
            dummy_file = Path(self.test_dir) / "notes.txt"
            dummy_file.write_text("PlantDoc test log artifact")
            mlfu.log_artifact(dummy_file, artifact_path="logs")

        # Verify run in MlflowClient
        retrieved_run = self.client.get_run(run_id)
        params = retrieved_run.data.params
        metrics = retrieved_run.data.metrics
        tags = retrieved_run.data.tags

        self.assertEqual(params["backbone"], "efficientnet_b0")
        self.assertEqual(params["batch_size"], "16")
        self.assertAlmostEqual(metrics["final_val_acc"], 0.94, places=2)
        self.assertEqual(tags["framework"], "pytorch")

        # Check metric history for train_loss
        metric_history = self.client.get_metric_history(run_id, "train_loss")
        self.assertEqual(len(metric_history), 2)
        self.assertEqual(metric_history[0].step, 1)
        self.assertAlmostEqual(metric_history[0].value, 0.54, places=2)
        self.assertEqual(metric_history[1].step, 2)
        self.assertAlmostEqual(metric_history[1].value, 0.32, places=2)

        # Check artifacts
        artifacts = [a.path for a in self.client.list_artifacts(run_id)]
        self.assertIn("summary.json", artifacts)
        self.assertIn("logs", artifacts)
        self.assertIn("figures", artifacts)

    def test_nested_runs(self):
        with mlfu.mlflow_run(run_name="parent-run", experiment_name="Test-Nested") as parent:
            mlfu.log_param("level", "parent")
            parent_id = parent.info.run_id

            with mlfu.mlflow_run(run_name="child-run", nested=True) as child:
                mlfu.log_param("level", "child")
                child_id = child.info.run_id

        parent_run = self.client.get_run(parent_id)
        child_run = self.client.get_run(child_id)
        self.assertEqual(parent_run.data.params["level"], "parent")
        self.assertEqual(child_run.data.params["level"], "child")
        self.assertEqual(child_run.data.tags.get("mlflow.parentRunId"), parent_id)

    def test_disabled_safety(self):
        os.environ["MLFLOW_TRACKING_DISABLED"] = "1"
        self.assertFalse(mlfu.is_tracking_enabled())

        # None of these should throw exceptions when disabled
        with mlfu.mlflow_run(run_name="disabled-run") as run:
            self.assertIsNone(run)
            mlfu.log_param("dummy", "val")
            mlfu.log_metric("dummy", 1.0)
            mlfu.log_artifact(Path(__file__))
            mlfu.log_dict({"a": 1}, "dummy.json")


if __name__ == "__main__":
    unittest.main()
