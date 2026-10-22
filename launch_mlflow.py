"""Launch the local MLflow Tracking UI server for PlantDoc / NexAi.

Usage:
    python launch_mlflow.py
    python launch_mlflow.py --port 5000 --host 127.0.0.1
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from src.mlflow_utils import get_tracking_uri, is_mlflow_available


def main():
    if not is_mlflow_available():
        print("Error: MLflow is not installed. Please run: pip install mlflow", file=sys.stderr)
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Launch local MLflow UI")
    parser.add_argument("--port", type=int, default=5000, help="Port to run MLflow UI on (default: 5000)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface to bind (default: 127.0.0.1)")
    parser.add_argument("--backend-store-uri", type=str, default=None, help="MLflow backend store URI (default: project mlruns)")
    args = parser.parse_args()

    store_uri = args.backend_store_uri or get_tracking_uri()

    print("=" * 65)
    print("🌿 PlantDoc / NexAi - MLflow Experiment Tracking Dashboard")
    print("=" * 65)
    print(f"Tracking Store URI : {store_uri}")
    print(f"MLflow UI URL      : http://{args.host}:{args.port}")
    print("=" * 65)
    print("Press Ctrl+C to stop the MLflow server.\n")

    cmd = [
        sys.executable,
        "-m",
        "mlflow",
        "ui",
        "--backend-store-uri",
        store_uri,
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nMLflow UI server stopped.")


if __name__ == "__main__":
    main()
