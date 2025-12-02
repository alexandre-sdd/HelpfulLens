"""MLflow evaluation entry point for the Yelp helpfulness model."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import json
import numpy as np
import pandas as pd
import mlflow
import yaml
from joblib import load
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    median_absolute_error,
    explained_variance_score,
)
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns

CONFIG_PATH = Path("src/config/config.yaml")


def load_config(config_path: Path = CONFIG_PATH) -> Dict[str, Any]:
    """Load the shared project configuration."""
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _resolve_eval_dataset_path(evaluation_dir: Path) -> Path:
    canonical = evaluation_dir / "yelp_helpfulness_eval.parquet"
    if canonical.exists():
        return canonical
    candidates = sorted(evaluation_dir.glob("*.parquet"))
    if not candidates:
        raise FileNotFoundError(f"No parquet files found in {evaluation_dir}")
    return candidates[0]


def _load_eval_data(evaluation_dir: Path) -> pd.DataFrame:
    path = _resolve_eval_dataset_path(evaluation_dir)
    df = pd.read_parquet(path)
    return df


def _prepare_xy(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    needed_cols = {"text", "target_is_useful", "target_useful_votes"}
    missing = needed_cols - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns in evaluation data: {sorted(missing)}")
    nonzero_mask = df["target_useful_votes"] != 0
    df_nz = df.loc[nonzero_mask].copy()
    if df_nz.empty:
        raise ValueError("No rows with non-zero target_useful_votes in evaluation data.")
    y = (df_nz["target_is_useful"].astype(float)) / (df_nz["target_useful_votes"].astype(float))
    y = y.clip(lower=0.0, upper=1.0)
    X = df_nz["text"].fillna("")
    return X, y


def _find_model_path(model_uri: Optional[str], config: Dict[str, Any]) -> Path:
    """Resolve path to a serialized model artifact (joblib/pkl).

    Priority:
      1) model_uri argument if it points to a local file
      2) models/registry/latest_model.json (artifact_path inside)
      3) config['models']['latest_registry_entry'] JSON (artifact_path inside)
    """
    if model_uri:
        p = Path(model_uri)
        if p.exists():
            return p
        # allow passing a registry json
        if p.suffix.lower() == ".json" and p.exists():
            with p.open("r", encoding="utf-8") as f:
                entry = json.load(f)
            return Path(entry["artifact_path"])

    registry_latest = Path("models/registry/latest_model.json")
    if registry_latest.exists():
        with registry_latest.open("r", encoding="utf-8") as f:
            entry = json.load(f)
        candidate = Path(entry["artifact_path"])
        if candidate.exists():
            return candidate

    default_registry_json = Path(config.get("models", {}).get("latest_registry_entry", ""))
    if default_registry_json.exists():
        with default_registry_json.open("r", encoding="utf-8") as f:
            entry = json.load(f)
        candidate = Path(entry["artifact_path"])
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Could not resolve a trained model file. Pass model_uri or train a model first."
    )


def evaluate(model_uri: Optional[str] = None) -> None:
    """Evaluate the trained text→usefulness regressor on the evaluation split."""
    config = load_config()
    data_cfg = config.get("data", {})
    models_cfg = config.get("models", {})

    eval_split_dir = (
        config.get("data", {})
        .get("datasets", {})
        .get("evaluation_dir", "data/datasets/evaluation")
    )
    evaluation_dir = Path(eval_split_dir)
    df = _load_eval_data(evaluation_dir)
    X_eval, y_eval = _prepare_xy(df)

    model_path = _find_model_path(model_uri, config)
    pipe = load(model_path)

    y_pred = pipe.predict(X_eval)
    y_pred = np.clip(y_pred, 0.0, 1.0)

    # Core regression metrics
    mae = float(mean_absolute_error(y_eval, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_eval, y_pred)))
    r2 = float(r2_score(y_eval, y_pred))
    medae = float(median_absolute_error(y_eval, y_pred))
    evs = float(explained_variance_score(y_eval, y_pred))

    # Correlations (robust to constant arrays)
    try:
        pearson_corr, _ = pearsonr(y_eval, y_pred)
    except Exception:
        pearson_corr = np.nan
    try:
        spearman_corr, _ = spearmanr(y_eval, y_pred)
    except Exception:
        spearman_corr = np.nan

    # Error distribution
    abs_err = np.abs(y_eval - y_pred)
    p90_abs_err = float(np.quantile(abs_err, 0.90))

    # Simple aggregates
    y_true_mean = float(np.mean(y_eval))
    y_pred_mean = float(np.mean(y_pred))

    metrics = {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "medae": medae,
        "explained_variance": evs,
        "pearson_r": float(pearson_corr) if np.isfinite(pearson_corr) else None,
        "spearman_r": float(spearman_corr) if np.isfinite(spearman_corr) else None,
        "p90_abs_err": p90_abs_err,
        "y_true_mean": y_true_mean,
        "y_pred_mean": y_pred_mean,
        "eval_rows_after_filter": int(len(y_eval)),
    }

    # Save predictions for inspection
    preds_dir = Path(data_cfg.get("predictions", {}).get("dir", "data/predictions"))
    preds_dir.mkdir(parents=True, exist_ok=True)
    out_pred_path = preds_dir / "eval_predictions.parquet"
    pd.DataFrame({"y_true": y_eval.values, "y_pred": y_pred}).to_parquet(out_pred_path, index=False)

    # Visualization artifacts
    plots_dir = preds_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # 1) Scatter: y_true vs y_pred
    try:
        plt.figure(figsize=(6, 6))
        sns.scatterplot(x=y_eval, y=y_pred, alpha=0.4, edgecolor=None)
        plt.plot([0, 1], [0, 1], "r--", linewidth=1)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.xlabel("y_true (ratio)")
        plt.ylabel("y_pred (ratio)")
        plt.title("True vs Predicted (ratio)")
        plt.tight_layout()
        scatter_path = plots_dir / "eval_scatter_true_vs_pred.png"
        plt.savefig(scatter_path, dpi=150)
        plt.close()
    except Exception:
        scatter_path = None

    # 2) Histogram: y_pred distribution
    try:
        plt.figure(figsize=(7, 4))
        sns.histplot(y_pred, bins=50, kde=False)
        plt.xlim(0, 1)
        plt.xlabel("y_pred")
        plt.ylabel("count")
        plt.title("Distribution of y_pred")
        plt.tight_layout()
        hist_pred_path = plots_dir / "eval_hist_y_pred.png"
        plt.savefig(hist_pred_path, dpi=150)
        plt.close()
    except Exception:
        hist_pred_path = None

    # 3) Residuals histogram (y_true - y_pred)
    try:
        residuals = (y_eval - y_pred)
        plt.figure(figsize=(7, 4))
        sns.histplot(residuals, bins=50, kde=False)
        plt.xlabel("residual = y_true - y_pred")
        plt.ylabel("count")
        plt.title("Residuals distribution")
        plt.tight_layout()
        hist_resid_path = plots_dir / "eval_hist_residuals.png"
        plt.savefig(hist_resid_path, dpi=150)
        plt.close()
    except Exception:
        hist_resid_path = None

    # 4) Calibration curve via binning on y_pred
    try:
        df_cal = pd.DataFrame({"y_true": y_eval.values, "y_pred": y_pred})
        df_cal["bin"] = pd.qcut(df_cal["y_pred"], q=10, duplicates="drop")
        calib = df_cal.groupby("bin", observed=True).agg(
            mean_pred=("y_pred", "mean"), mean_true=("y_true", "mean")
        ).reset_index(drop=True)
        plt.figure(figsize=(6, 6))
        sns.lineplot(x="mean_pred", y="mean_true", data=calib, marker="o")
        plt.plot([0, 1], [0, 1], "r--", linewidth=1)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.xlabel("mean predicted (per bin)")
        plt.ylabel("mean true (per bin)")
        plt.title("Calibration plot (binned by y_pred)")
        plt.tight_layout()
        calib_path = plots_dir / "eval_calibration_plot.png"
        plt.savefig(calib_path, dpi=150)
        plt.close()
    except Exception:
        calib_path = None

    with mlflow.start_run(run_name="yelp_helpfulness_evaluate"):
        mlflow.log_param("model_path", str(model_path))
        mlflow.log_param("evaluation_split", eval_split_dir)
        for k, v in metrics.items():
            if isinstance(v, (int, float)) and np.isfinite(v):
                mlflow.log_metric(k, float(v))
        mlflow.log_artifact(str(out_pred_path))
        # Log plots if available
        for p in (scatter_path, hist_pred_path, hist_resid_path, calib_path):
            if p is not None:
                mlflow.log_artifact(str(p))

    print(
        f"Evaluation complete. Metrics: {metrics}\nSaved predictions to: {out_pred_path}"
    )


if __name__ == "__main__":
    evaluate()
