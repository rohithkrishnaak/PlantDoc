"""MLflow experiment tracking and model management utilities for PlantDoc / NexAi.

Provides centralized tracking URI configuration, experiment management, run
context management, and safe logging helpers for parameters, metrics, artifacts,
and figures.
"""
from __future__ import annotations

import contextlib
import os
import sys
from pathlib import Path
from typing import Any, Dict, Generator, Optional, Union

# Silence MLflow assistant hints and allow filestore compatibility
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

try:
    import mlflow
    from mlflow.entities import Run
    _MLFLOW_AVAILABLE = True
except ImportError:
    mlflow = None  # type: ignore
    Run = None  # type: ignore
    _MLFLOW_AVAILABLE = False


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = PROJECT_ROOT / "mlflow.db"
DEFAULT_ARTIFACTS_DIR = PROJECT_ROOT / "mlartifacts"
DEFAULT_MLRUNS_DIR = PROJECT_ROOT / "mlruns"
DEFAULT_EXPERIMENT_NAME = "PlantDoc-NexAi"


def is_mlflow_available() -> bool:
    """Return True if mlflow is installed and can be imported."""
    return _MLFLOW_AVAILABLE


def is_tracking_enabled() -> bool:
    """Check if MLflow tracking is globally enabled."""
    if not _MLFLOW_AVAILABLE:
        return False
    disabled = os.getenv("MLFLOW_TRACKING_DISABLED", "0").strip().lower()
    return disabled not in ("1", "true", "yes")


def get_tracking_uri() -> str:
    """Resolve the tracking URI.
    
    Uses MLFLOW_TRACKING_URI from environment if set, otherwise defaults
    to local SQLite database at 'sqlite:///mlflow.db'.
    """
    env_uri = os.getenv("MLFLOW_TRACKING_URI", "").strip()
    if env_uri:
        return env_uri
    return f"sqlite:///{DEFAULT_DB_PATH.as_posix()}"


def setup_mlflow(
    experiment_name: Optional[str] = None,
    tracking_uri: Optional[str] = None,
) -> Optional[str]:
    """Configure MLflow tracking URI and experiment.
    
    Returns the experiment ID or None if MLflow is not available/enabled.
    """
    if not is_tracking_enabled():
        return None

    uri = tracking_uri or get_tracking_uri()
    mlflow.set_tracking_uri(uri)

    exp_name = experiment_name or os.getenv("MLFLOW_EXPERIMENT_NAME", DEFAULT_EXPERIMENT_NAME)
    experiment = mlflow.get_experiment_by_name(exp_name)
    if experiment is None:
        DEFAULT_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        try:
            exp_id = mlflow.create_experiment(
                exp_name,
                artifact_location=DEFAULT_ARTIFACTS_DIR.resolve().as_uri(),
            )
        except Exception:
            exp_id = mlflow.create_experiment(exp_name)
    else:
        exp_id = experiment.experiment_id

    mlflow.set_experiment(exp_name)
    return exp_id


@contextlib.contextmanager
def mlflow_run(
    run_name: Optional[str] = None,
    experiment_name: Optional[str] = None,
    tags: Optional[Dict[str, Any]] = None,
    nested: bool = False,
    description: Optional[str] = None,
    enabled: bool = True,
) -> Generator[Optional[Any], None, None]:
    """Context manager for an MLflow run.
    
    If MLflow is disabled or unavailable, yields None and gracefully does nothing.
    """
    if not (enabled and is_tracking_enabled()):
        yield None
        return

    setup_mlflow(experiment_name=experiment_name)
    with mlflow.start_run(run_name=run_name, nested=nested, description=description) as run:
        if tags:
            safe_tags = {str(k): str(v) for k, v in tags.items()}
            mlflow.set_tags(safe_tags)
        yield run


def log_param(key: str, value: Any) -> None:
    """Safely log a single parameter to current MLflow run."""
    if not is_tracking_enabled() or mlflow.active_run() is None:
        return
    try:
        if isinstance(value, (list, tuple, set)):
            val_str = ",".join(str(v) for v in value)
        else:
            val_str = str(value)
        # MLflow parameter values have a length limit (typically 500 chars)
        if len(val_str) > 500:
            val_str = val_str[:497] + "..."
        mlflow.log_param(key, val_str)
    except Exception as e:
        print(f"[MLflow Warning] Failed to log param {key}: {e}", file=sys.stderr)


def log_params(params: Dict[str, Any]) -> None:
    """Safely log a dictionary of parameters to current MLflow run."""
    if not is_tracking_enabled() or mlflow.active_run() is None:
        return
    for k, v in params.items():
        log_param(k, v)


def log_metric(key: str, value: Any, step: Optional[int] = None) -> None:
    """Safely log a numeric metric to current MLflow run."""
    if not is_tracking_enabled() or mlflow.active_run() is None:
        return
    if value is None:
        return
    try:
        val_float = float(value)
        if step is not None:
            mlflow.log_metric(key, val_float, step=int(step))
        else:
            mlflow.log_metric(key, val_float)
    except (ValueError, TypeError):
        pass
    except Exception as e:
        print(f"[MLflow Warning] Failed to log metric {key}: {e}", file=sys.stderr)


def log_metrics(metrics: Dict[str, Any], step: Optional[int] = None) -> None:
    """Safely log a dictionary of metrics to current MLflow run."""
    if not is_tracking_enabled() or mlflow.active_run() is None:
        return
    for k, v in metrics.items():
        log_metric(k, v, step=step)


def log_artifact(local_path: Union[str, Path], artifact_path: Optional[str] = None) -> None:
    """Safely log an artifact file or directory to current MLflow run."""
    if not is_tracking_enabled() or mlflow.active_run() is None:
        return
    path = Path(local_path)
    if not path.exists():
        return
    try:
        mlflow.log_artifact(str(path), artifact_path=artifact_path)
    except Exception as e:
        print(f"[MLflow Warning] Failed to log artifact {local_path}: {e}", file=sys.stderr)


def log_figure(fig: Any, artifact_file: str) -> None:
    """Safely log a matplotlib figure as an artifact."""
    if not is_tracking_enabled() or mlflow.active_run() is None:
        return
    try:
        mlflow.log_figure(fig, artifact_file)
    except Exception as e:
        print(f"[MLflow Warning] Failed to log figure {artifact_file}: {e}", file=sys.stderr)


def log_dict(dictionary: Dict[str, Any], artifact_file: str) -> None:
    """Safely log a dictionary as a JSON/YAML artifact."""
    if not is_tracking_enabled() or mlflow.active_run() is None:
        return
    try:
        mlflow.log_dict(dictionary, artifact_file)
    except Exception as e:
        print(f"[MLflow Warning] Failed to log dict to {artifact_file}: {e}", file=sys.stderr)


def set_tags(tags: Dict[str, Any]) -> None:
    """Safely set tags on current MLflow run."""
    if not is_tracking_enabled() or mlflow.active_run() is None:
        return
    try:
        safe_tags = {str(k): str(v) for k, v in tags.items()}
        mlflow.set_tags(safe_tags)
    except Exception as e:
        print(f"[MLflow Warning] Failed to set tags: {e}", file=sys.stderr)
