"""MLflow-enabled training entry point for the Yelp helpfulness model."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple, Optional

import json
import time
from datetime import datetime

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from joblib import dump
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline

CONFIG_PATH = Path("src/config/config.yaml")


def load_config(config_path: Path = CONFIG_PATH) -> Dict[str, Any]:
    """Load the project configuration file."""
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _resolve_dataset_path(training_dir: Path) -> Path:
    """Resolve the training parquet path, preferring the canonical filename."""
    canonical = training_dir / "yelp_helpfulness_train.parquet"
    if canonical.exists():
        return canonical
    candidates = sorted(training_dir.glob("*.parquet"))
    if not candidates:
        raise FileNotFoundError(f"No parquet files found in {training_dir}")
    return candidates[0]


def _load_training_data(training_dir: Path) -> pd.DataFrame:
    path = _resolve_dataset_path(training_dir)
    df = pd.read_parquet(path)
    return df


def _build_pipeline(
    max_features: int = 5000, ngram_range: Tuple[int, int] = (1, 2), random_state: int = 42
) -> Pipeline:
    """Create a TF-IDF + RandomForestRegressor pipeline for text → usefulness ratio."""
    vectorizer = TfidfVectorizer(
        max_features=max_features, ngram_range=ngram_range, lowercase=True, stop_words="english"
    )
    regressor = RandomForestRegressor(
        n_estimators=300, max_depth=None, n_jobs=-1, random_state=random_state
    )
    pipe = Pipeline([("tfidf", vectorizer), ("regressor", regressor)])
    return pipe


def _prepare_xy(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """Filter rows and construct X (text) and y (ratio)."""
    needed_cols = {"text", "target_is_useful", "target_useful_votes"}
    missing = needed_cols - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns in training data: {sorted(missing)}")

    nonzero_mask = df["target_useful_votes"] != 0
    df_nz = df.loc[nonzero_mask].copy()
    if df_nz.empty:
        raise ValueError("No rows with non-zero target_useful_votes in training data.")

    y = (df_nz["target_is_useful"].astype(float)) / (df_nz["target_useful_votes"].astype(float))
    y = y.clip(lower=0.0, upper=1.0)
    X = df_nz["text"].fillna("")
    return X, y


def _save_registry_entry(
    registry_dir: Path,
    model_name: str,
    artifact_path: Path,
    run_id: str,
    metrics: Optional[Dict[str, float]] = None,
) -> Path:
    """Write or update the latest model registry entry."""
    registry_dir.mkdir(parents=True, exist_ok=True)
    entry = {
        "model_name": model_name,
        "version": int(time.time()),
        "source_run_id": run_id,
        "artifact_path": str(artifact_path),
        "metrics": metrics or {},
        "notes": f"Auto-registered at {datetime.utcnow().isoformat()}Z",
    }
    out_path = registry_dir / "latest_model.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(entry, f, indent=2)
    return out_path


def train() -> None:
    """Train a tree-based regressor to predict usefulness ratio from review text."""
    config = load_config()
    data_cfg = config.get("data", {})
    features_cfg = config.get("features", {}).get("text", {})
    training_cfg = config.get("training", {})
    models_cfg = config.get("models", {})

    training_dir = Path(data_cfg.get("datasets", {}).get("training_dir", "data/datasets/training"))
    artifacts_dir = Path(models_cfg.get("artifacts_dir", "models/artifacts"))
    registry_dir = Path(models_cfg.get("registry_dir", "models/registry"))
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    registry_dir.mkdir(parents=True, exist_ok=True)

    max_features = int(features_cfg.get("max_features", 5000))
    ngram_range_cfg = features_cfg.get("ngram_range", [1, 2])
    ngram_range = (int(ngram_range_cfg[0]), int(ngram_range_cfg[1]))
    random_state = int(training_cfg.get("random_state", 42))

    df = _load_training_data(training_dir)
    X, y = _prepare_xy(df)

    pipe = _build_pipeline(
        max_features=max_features, ngram_range=ngram_range, random_state=random_state
    )

    params = {
        "model_family": "tree_based_regressor",
        "model_type": "RandomForestRegressor",
        "random_state": random_state,
        "tfidf_max_features": max_features,
        "tfidf_ngram_range": str(ngram_range),
        "train_rows_after_filter": int(len(X)),
    }

    with mlflow.start_run(run_name="yelp_helpfulness_train"):
        for k, v in params.items():
            mlflow.log_param(k, v)

        pipe.fit(X, y)
        y_hat = pipe.predict(X)
        y_hat = np.clip(y_hat, 0.0, 1.0)
        train_r2 = float(r2_score(y, y_hat))
        mlflow.log_metric("train_r2_in_sample", train_r2)

        mlflow.sklearn.log_model(pipe, artifact_path="model")

        model_name = "yelp_helpfulness_text_rf"
        local_model_path = artifacts_dir / f"{model_name}.pkl"
        dump(pipe, local_model_path)
        mlflow.log_artifact(str(local_model_path))

        registry_entry = _save_registry_entry(
            registry_dir=registry_dir,
            model_name="yelp_helpfulness_regressor",
            artifact_path=local_model_path,
            run_id=mlflow.active_run().info.run_id,
            metrics={"train_r2_in_sample": train_r2},
        )

    print(
        f"Training complete. Saved model to: {local_model_path}\n"
        f"Updated registry: {registry_entry}"
    )


if __name__ == "__main__":
    train()
