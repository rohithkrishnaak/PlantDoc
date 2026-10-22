"""Register a trained checkpoint in the MLflow Model Registry.

    python register_model.py --weights weights/plantdoc_hybrid.pth --alias champion

Logs the checkpoint as an MLflow pyfunc model (so `mlflow models serve` or
`mlflow.pyfunc.load_model("models:/PlantDoc-LeafDoctor@champion")` returns the
same quality-gate-free diagnosis the app computes), registers a new version,
copies the held-out metrics of the latest evaluation run for each test set onto
the version as tags, and points the alias at it.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path

from src import mlflow_utils as mlfu

MODEL_NAME = "PlantDoc-LeafDoctor"


def _pyfunc_class():
    """Defined lazily so importing this module does not require mlflow."""
    import mlflow

    class _PlantDoc(mlflow.pyfunc.PythonModel):
        """Input: DataFrame with an `image_b64` (base64 JPEG/PNG) or `path` column.
        Output: top-1 class, its calibrated confidence, top-3, similarity to training data, OOD flag."""

        def load_context(self, context):
            from src.model import PlantDocModel
            self.model = PlantDocModel(checkpoint=context.artifacts["checkpoint"])

        def predict(self, context, model_input, params=None):
            import pandas as pd
            from PIL import Image
            rows = []
            for _, r in model_input.iterrows():
                if "image_b64" in r and isinstance(r["image_b64"], str):
                    img = Image.open(io.BytesIO(base64.b64decode(r["image_b64"])))
                else:
                    img = Image.open(r["path"])
                preds, sim, ood = self.model.diagnose(img.convert("RGB"), top_k=3)
                rows.append({"prediction": preds[0][0], "confidence": preds[0][1],
                             "top3": json.dumps(preds), "similarity": sim, "ood": bool(ood)})
            return pd.DataFrame(rows)

    return _PlantDoc


def latest_eval_metrics(client, experiment_id: str, test_set: str, model_source: str) -> dict:
    runs = client.search_runs([experiment_id],
                              filter_string=f"tags.stage = 'evaluate' and tags.test_set = '{test_set}'"
                                            f" and tags.model_source = '{model_source}'",
                              order_by=["attributes.start_time DESC"], max_results=1)
    if not runs:
        return {}
    m = runs[0].data.metrics
    return {"run_id": runs[0].info.run_id,
            **{k: round(m[k], 4) for k in ("top1_accuracy", "top3_accuracy", "trusted_rate", "accuracy_on_trusted")
               if k in m}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--alias", default="champion")
    ap.add_argument("--name", default=MODEL_NAME)
    args = ap.parse_args()

    import mlflow
    import torch
    from mlflow import MlflowClient

    from src.model import PlantDocModel

    exp_id = mlfu.setup_mlflow()
    source = PlantDocModel(checkpoint=args.weights).weights_source
    ckpt = torch.load(args.weights, map_location="cpu", weights_only=False)
    client = MlflowClient()
    with mlfu.mlflow_run(run_name=f"register-{Path(args.weights).stem}", tags={"stage": "register"}):
        mlfu.log_params({"weights": args.weights, "model_source": source, "cv_acc": ckpt.get("cv_acc"),
                         "trust_threshold": ckpt.get("trust_threshold"), "train_dir": ckpt.get("train_dir")})
        info = mlflow.pyfunc.log_model(
            name="plantdoc_model",
            python_model=_pyfunc_class()(),
            artifacts={"checkpoint": args.weights},
            code_paths=["src"],
            pip_requirements="requirements.txt",
        )
        mv = mlflow.register_model(info.model_uri, args.name)
    tags = {"model_source": source, "weights_file": Path(args.weights).name,
            "oof_cv_acc": f"{ckpt.get('cv_acc', float('nan')):.4f}",
            "trust_threshold": str(ckpt.get("trust_threshold")), "train_dir": str(ckpt.get("train_dir"))}
    for test_set in ("test", "test_plantwild"):
        for k, v in latest_eval_metrics(client, exp_id, test_set, source).items():
            tags[f"{test_set}.{k}"] = str(v)
    for k, v in tags.items():
        client.set_model_version_tag(args.name, mv.version, k, v)
    client.set_registered_model_alias(args.name, args.alias, mv.version)
    print(f"Registered {args.name} v{mv.version} as @{args.alias}: {json.dumps(tags, indent=2)}")


if __name__ == "__main__":
    main()
