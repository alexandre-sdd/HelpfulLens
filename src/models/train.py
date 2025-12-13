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
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import train_test_split

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


def _build_feature_transformer(
    recipe: str, max_features: int, ngram_range: Tuple[int, int]
) -> Pipeline:
    """Return a feature transformer by recipe name."""
    if recipe == "text_tfidf":
        return Pipeline([("tfidf", TfidfVectorizer(
            max_features=max_features, ngram_range=ngram_range, lowercase=True, stop_words="english"
        ))])
    if recipe == "text_tfidf_dense":
        return Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=max_features, ngram_range=ngram_range, lowercase=True, stop_words="english"
            )),
            ("densify", FunctionTransformer(_sparse_to_dense, accept_sparse=True)),
        ])
    raise ValueError(f"Unknown feature recipe: {recipe}")


def _sparse_to_dense(X):
    return X.toarray() if hasattr(X, "toarray") else X


def _build_estimator(model_type: str, random_state: int) -> Any:
    """Return an estimator by model_type."""
    mt = model_type.lower()
    if mt in ("logistic_regression", "logreg", "lr"):
        return LogisticRegression(max_iter=1000, random_state=random_state, class_weight="balanced", n_jobs=-1)
    if mt in ("random_forest_classifier", "rf", "random_forest"):
        return RandomForestClassifier(n_estimators=300, max_depth=None, n_jobs=-1, random_state=random_state, class_weight="balanced")
    if mt in ("linear_svc", "linearsvc", "svc"):
        return LinearSVC(random_state=random_state, class_weight="balanced")
    if mt in ("random_forest_regressor", "rf_regressor", "rf_reg"):
        return RandomForestRegressor(n_estimators=300, max_depth=None, n_jobs=-1, random_state=random_state)
    raise ValueError(f"Unknown model_type: {model_type}")


def _prepare_text_and_target(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """Construct X (text) and y (binary target_is_useful)."""
    needed_cols = {"text", "target_is_useful"}
    missing = needed_cols - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns for text baseline: {sorted(missing)}")
    X = df["text"].fillna("")
    y = df["target_is_useful"].astype(int)
    return X, y


def _prepare_text_and_target_by_name(df: pd.DataFrame, target_name: str) -> Tuple[pd.Series, pd.Series]:
    """Prepare X (text) and y based on a target column name."""
    if "text" not in df.columns:
        raise KeyError("Missing required column 'text'.")
    if target_name not in df.columns:
        raise KeyError(f"Missing required target column '{target_name}'.")
    X = df["text"].fillna("")
    if target_name == "target_is_useful":
        y = df[target_name].astype(int)
    else:
        y = pd.to_numeric(df[target_name], errors="coerce").fillna(0).astype(float)
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
    # Write to a model-specific latest file for clarity
    safe_name = model_name.replace(" ", "_").lower()
    out_path = registry_dir / f"latest_{safe_name}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(entry, f, indent=2)
    return out_path


def _sanitize_name(name: str) -> str:
    return name.replace(" ", "_").lower()


def _build_training_pipeline(
    feature_recipe: str,
    model_type: str,
    max_features: int,
    ngram_range: Tuple[int, int],
    random_state: int,
) -> Pipeline:
    """Build a unified pipeline: features -> estimator; densify added if needed."""
    features = _build_feature_transformer(feature_recipe, max_features, ngram_range)
    estimator = _build_estimator(model_type, random_state)
    steps = [("features", features)]
    needs_dense = isinstance(estimator, RandomForestClassifier)
    if feature_recipe == "text_tfidf" and needs_dense:
        steps.append(("densify", FunctionTransformer(_sparse_to_dense, accept_sparse=True)))
    steps.append(("classifier", estimator))
    return Pipeline(steps)


def train() -> None:
    """Train a configurable pipeline using shared features and selectable model/target."""
    config = load_config()
    data_cfg = config.get("data", {})
    features_root = config.get("features", {})
    features_text_cfg = features_root.get("text", {})
    feature_recipe = features_root.get("recipe", "text_tfidf")
    training_cfg = config.get("training", {})
    model_type = training_cfg.get("model_type", "logistic_regression")
    target_name = training_cfg.get("target", "target_is_useful")
    models_cfg = config.get("models", {})

    training_dir = Path(data_cfg.get("datasets", {}).get("training_dir", "data/datasets/training"))
    artifacts_dir = Path(models_cfg.get("artifacts_dir", "models/artifacts"))
    registry_dir = Path(models_cfg.get("registry_dir", "models/registry"))
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    registry_dir.mkdir(parents=True, exist_ok=True)

    max_features = int(features_text_cfg.get("max_features", 5000))
    ngram_range_cfg = features_text_cfg.get("ngram_range", [1, 2])
    ngram_range = (int(ngram_range_cfg[0]), int(ngram_range_cfg[1]))
    random_state = int(training_cfg.get("random_state", 42))
    sample_size = training_cfg.get("sample_size", None)

    df = _load_training_data(training_dir)
    # Optional sub-sampling from training data (e.g., 10000 rows) - random sampling
    if sample_size is not None:
        try:
            sample_n = int(sample_size)
        except Exception:
            sample_n = None
        if sample_n is not None and sample_n > 0 and len(df) > sample_n:
            rng = np.random.RandomState(random_state)
            idx = np.arange(len(df))
            chosen = rng.choice(idx, size=sample_n, replace=False)
            df = df.iloc[chosen].copy()
    X_text, y_vec = _prepare_text_and_target(df) if target_name == "target_is_useful" else _prepare_text_and_target_by_name(df, target_name)
    pipe = _build_training_pipeline(
        feature_recipe=feature_recipe,
        model_type=model_type,
        max_features=max_features,
        ngram_range=ngram_range,
        random_state=random_state,
    )
    model_name = _sanitize_name(f"{target_name}__{feature_recipe}__{model_type}")
    local_model_path = artifacts_dir / f"{model_name}.pkl"

    with mlflow.start_run(run_name="yelp_helpfulness_train"):
        # Params
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("tfidf_max_features", max_features)
        mlflow.log_param("tfidf_ngram_range", str(ngram_range))
        mlflow.log_param("train_rows_total", int(len(df)))
        mlflow.log_param("feature_recipe", feature_recipe)
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("target_name", target_name)

        # Fit
        pipe.fit(X_text, y_vec)
        # Classification vs Regression metrics
        if target_name == "target_is_useful":
            if hasattr(pipe, "predict_proba"):
                y_proba = pipe.predict_proba(X_text)[:, 1]
            elif hasattr(pipe, "decision_function"):
                scores = pipe.decision_function(X_text)
                y_proba = 1.0 / (1.0 + np.exp(-scores))
            else:
                y_proba = pipe.predict(X_text).astype(float)
            y_label = (y_proba >= 0.5).astype(int)
            acc = float(accuracy_score(y_vec, y_label))
            f1 = float(f1_score(y_vec, y_label, zero_division=0))
            try:
                rocauc = float(roc_auc_score(y_vec, y_proba))
            except Exception:
                rocauc = float("nan")
            mlflow.log_metric("acc_in_sample", acc)
            mlflow.log_metric("f1_in_sample", f1)
            if np.isfinite(rocauc):
                mlflow.log_metric("roc_auc_in_sample", rocauc)
        else:
            y_pred = pipe.predict(X_text).astype(float)
            mae = float(mean_absolute_error(y_vec, y_pred))
            rmse = float(np.sqrt(mean_squared_error(y_vec, y_pred)))
            r2 = float(r2_score(y_vec, y_pred))
            mlflow.log_metric("mae_in_sample", mae)
            mlflow.log_metric("rmse_in_sample", rmse)
            mlflow.log_metric("r2_in_sample", r2)

        # Persist
        dump(pipe, local_model_path)
        mlflow.log_artifact(str(local_model_path))
        _save_registry_entry(
            registry_dir=registry_dir,
            model_name=model_name,
            artifact_path=local_model_path,
            run_id=mlflow.active_run().info.run_id,
            metrics=None,
        )
        mlflow.sklearn.log_model(pipe, artifact_path=model_name)

    print(
        "Training complete.\n"
        f"- Model saved: {local_model_path}\n"
        f"- Registry updated: models/registry/latest_{model_name}.json"
    )


if __name__ == "__main__":
    train()
