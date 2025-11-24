"""MLflow evaluation entry point for the Yelp helpfulness model."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import mlflow
import yaml

CONFIG_PATH = Path("src/config/config.yaml")


def load_config(config_path: Path = CONFIG_PATH) -> Dict[str, Any]:
    """Load the shared project configuration."""
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def evaluate(model_uri: Optional[str] = None) -> None:
    """Run a placeholder evaluation job and log metrics to MLflow."""
    config = load_config()
    default_model_uri = config.get("models", {}).get(
        "latest_registry_entry", "models/artifacts/dummy_model.txt"
    )
    target_model = model_uri or default_model_uri
    metrics = {"eval_accuracy": 0.48, "eval_f1": 0.47}
    eval_split_dir = (
        config.get("data", {})
        .get("datasets", {})
        .get("evaluation_dir", "data/datasets/evaluation")
    )

    with mlflow.start_run(run_name="yelp_helpfulness_evaluate"):
        mlflow.log_param("model_uri", target_model)
        mlflow.log_param("evaluation_split", eval_split_dir)
        for key, value in metrics.items():
            mlflow.log_metric(key, value)

    print("Evaluation metrics logged to MLflow (placeholder).")


if __name__ == "__main__":
    evaluate()
