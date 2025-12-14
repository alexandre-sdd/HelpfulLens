"""Train and evaluate baseline + lightweight models for Yelp helpfulness."""

from __future__ import annotations

import argparse
import json
import math
import secrets
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.calibration import calibration_curve
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, PoissonRegressor
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    roc_auc_score,
)

from src.utils.logging_utils import get_logger
from src.utils.schema_checks import assert_aligned_columns

LOGGER = get_logger("train_and_evaluate")
CONFIG_PATH = Path("src/config/config.yaml")


def load_config(config_path: Path = CONFIG_PATH) -> Dict:
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return float("nan")
    a = pd.Series(y_true).rank()
    b = pd.Series(y_pred).rank()
    return float(a.corr(b))


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), a_min=0, a_max=None)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0:
        return {"mae_log1p": float("nan"), "rmse_log1p": float("nan"), "spearman": float("nan")}

    log_true = np.log1p(y_true[mask])
    log_pred = np.log1p(y_pred[mask])
    mae_log = mean_absolute_error(log_true, log_pred)
    rmse_log = math.sqrt(mean_squared_error(log_true, log_pred))
    spearman = _spearman(y_true[mask], y_pred[mask])
    return {"mae_log1p": mae_log, "rmse_log1p": rmse_log, "spearman": spearman}


def classification_metrics(y_true: np.ndarray, prob: np.ndarray) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    labels = np.asarray(y_true, dtype=int)
    probs = np.asarray(prob, dtype=float)
    if len(np.unique(labels)) < 2:
        metrics["roc_auc"] = float("nan")
        metrics["pr_auc"] = float("nan")
        return metrics
    metrics["roc_auc"] = roc_auc_score(labels, probs)
    metrics["pr_auc"] = average_precision_score(labels, probs)
    return metrics


def _ensure_array(series_or_array) -> np.ndarray:
    if isinstance(series_or_array, pd.Series):
        return series_or_array.to_numpy()
    if isinstance(series_or_array, pd.DataFrame) and series_or_array.shape[1] == 1:
        return series_or_array.iloc[:, 0].to_numpy()
    return np.asarray(series_or_array)


def _make_run_id() -> str:
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    suffix = secrets.token_hex(2)
    return f"{ts}_{suffix}"


def _plot_scatter(y_true: np.ndarray, y_pred: np.ndarray, path: Path) -> None:
    plt.figure(figsize=(6, 6))
    plt.scatter(np.log1p(y_true), np.log1p(np.clip(y_pred, 0, None)), alpha=0.25, s=10)
    plt.xlabel("log1p(true useful)")
    plt.ylabel("log1p(pred useful)")
    plt.title("y_true vs y_pred (log scale)")
    plt.grid(True, alpha=0.3)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, review_len: np.ndarray | None, path: Path) -> None:
    if review_len is None or review_len.size == 0:
        return
    residuals = np.log1p(y_true) - np.log1p(np.clip(y_pred, 0, None))
    plt.figure(figsize=(7, 4))
    plt.scatter(review_len, residuals, alpha=0.2, s=8)
    plt.xlabel("review length (words)")
    plt.ylabel("Residual (log1p true - log1p pred)")
    plt.title("Residuals vs review length")
    plt.grid(True, alpha=0.3)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _plot_hist(y_true: np.ndarray, y_pred: np.ndarray, path: Path) -> None:
    plt.figure(figsize=(7, 4))
    plt.hist(np.log1p(y_true), bins=60, alpha=0.6, label="true", density=True)
    plt.hist(np.log1p(np.clip(y_pred, 0, None)), bins=60, alpha=0.6, label="pred", density=True)
    plt.xlabel("log1p(useful)")
    plt.ylabel("Density")
    plt.title("True vs predicted distribution (log1p)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _plot_pr_curve(y_true: np.ndarray, prob: np.ndarray, path: Path) -> None:
    if len(np.unique(y_true)) < 2:
        return
    precision, recall, _ = precision_recall_curve(y_true, prob)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall (useful>0)")
    plt.grid(True, alpha=0.3)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _plot_calibration(y_true: np.ndarray, prob: np.ndarray, path: Path) -> None:
    if len(np.unique(y_true)) < 2:
        return
    frac_pos, mean_pred = calibration_curve(y_true, prob, n_bins=10, strategy="quantile")
    plt.figure(figsize=(6, 5))
    plt.plot(mean_pred, frac_pos, marker="o")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed fraction positive")
    plt.title("Calibration (useful>0)")
    plt.grid(True, alpha=0.3)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _load_feature_matrices(config: Dict) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    feature_cfg = config.get("data", {}).get("features", {})
    feature_dir = Path(feature_cfg.get("dir", "data/features"))
    X_train = pd.read_parquet(feature_dir / "X_train.parquet")
    y_train_df = pd.read_parquet(feature_dir / "y_train.parquet")
    X_eval = pd.read_parquet(feature_dir / "X_eval.parquet")
    y_eval_df = pd.read_parquet(feature_dir / "y_eval.parquet")
    target_col = config.get("modeling", {}).get("target_column", "target_useful_votes")

    if target_col not in y_train_df.columns:
        raise KeyError(f"Target column '{target_col}' missing from y_train parquet.")
    if target_col not in y_eval_df.columns:
        raise KeyError(f"Target column '{target_col}' missing from y_eval parquet.")

    assert_aligned_columns(X_train, X_eval)
    y_train = y_train_df[target_col]
    y_eval = y_eval_df[target_col]
    return X_train, y_train, X_eval, y_eval


def _filter_invalid_targets(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    target = y.to_numpy()
    mask = np.isfinite(target) & (target >= 0)
    if mask.sum() == 0:
        raise ValueError("No valid target values after filtering for finiteness and non-negativity.")
    if mask.all():
        return X, y
    return X.loc[mask].reset_index(drop=True), y.loc[mask].reset_index(drop=True)


def _filter_by_popularity(
    X: pd.DataFrame,
    y: pd.Series,
    min_total_votes: int,
    votes_column: str = "total_votes",
) -> Tuple[pd.DataFrame, pd.Series]:
    if min_total_votes is None or min_total_votes <= 0:
        return X, y
    if votes_column not in X.columns:
        LOGGER.warning(
            "Requested min_total_votes=%s but column '%s' not found in features; skipping popularity filter.",
            min_total_votes,
            votes_column,
        )
        return X, y
    mask = X[votes_column].fillna(0) >= min_total_votes
    if not mask.any():
        LOGGER.warning(
            "Popularity filter min_total_votes=%s removed all rows (column %s). Keeping full dataset.",
            min_total_votes,
            votes_column,
        )
        return X, y
    LOGGER.info(
        "Applying popularity filter: min_total_votes=%s (kept %d / %d rows)",
        min_total_votes,
        mask.sum(),
        len(mask),
    )
    return X.loc[mask].reset_index(drop=True), y.loc[mask].reset_index(drop=True)


def train_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_eval: pd.DataFrame,
    y_eval: pd.Series,
    config: Dict,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, np.ndarray], Dict[str, object]]:
    modeling_cfg = config.get("modeling", {})
    rng_seed = modeling_cfg.get("random_seed", 42)
    model_list: List[str] = modeling_cfg.get(
        "include_models",
        ["baseline_mean", "linear_log", "poisson", "hgb", "hurdle"],
    )
    small_feats_cfg = modeling_cfg.get("small_numeric_features", [])

    y_train_arr = _ensure_array(y_train)
    y_eval_arr = _ensure_array(y_eval)
    metrics: Dict[str, Dict[str, float]] = {}
    preds: Dict[str, np.ndarray] = {}
    models: Dict[str, object] = {}

    # Baseline 1: global mean
    if "baseline_mean" in model_list:
        mean_pred = float(np.mean(y_train_arr))
        pred_eval = np.full_like(y_eval_arr, mean_pred, dtype=float)
        metrics["baseline_mean"] = regression_metrics(y_eval_arr, pred_eval)
        preds["baseline_mean"] = pred_eval
        models["baseline_mean"] = {"type": "constant", "value": mean_pred}

    # Baseline 2: linear regression on log1p with small feature set
    if "linear_log" in model_list:
        small_feats = [c for c in small_feats_cfg if c in X_train.columns]
        if not small_feats:
            small_feats = X_train.columns.tolist()[: min(25, X_train.shape[1])]
        LOGGER.info("Training linear_log on %d features", len(small_feats))
        lr = LinearRegression()
        lr.fit(X_train[small_feats], np.log1p(y_train_arr))
        pred_log = lr.predict(X_eval[small_feats])
        pred_eval = np.expm1(pred_log).clip(min=0)
        metrics["linear_log"] = regression_metrics(y_eval_arr, pred_eval)
        preds["linear_log"] = pred_eval
        models["linear_log"] = {"estimator": lr, "features": small_feats}

    # Poisson regression
    if "poisson" in model_list:
        LOGGER.info("Training PoissonRegressor on %d features", X_train.shape[1])
        pr = PoissonRegressor(max_iter=300, alpha=1e-5)
        pr.fit(X_train, y_train_arr)
        pred_eval = pr.predict(X_eval).clip(min=0)
        metrics["poisson"] = regression_metrics(y_eval_arr, pred_eval)
        preds["poisson"] = pred_eval
        models["poisson"] = pr

    # Tree model: HistGradientBoostingRegressor on log1p target
    if "hgb" in model_list:
        LOGGER.info("Training HistGradientBoostingRegressor on %d features", X_train.shape[1])
        hgb = HistGradientBoostingRegressor(random_state=rng_seed)
        hgb.fit(X_train, np.log1p(y_train_arr))
        pred_eval = np.expm1(hgb.predict(X_eval)).clip(min=0)
        metrics["hgb"] = regression_metrics(y_eval_arr, pred_eval)
        preds["hgb"] = pred_eval
        models["hgb"] = hgb

    # Two-stage hurdle: classifier for useful>0 + regressor on positives
    if "hurdle" in model_list:
        threshold = modeling_cfg.get("useful_positive_threshold", 0)
        y_train_binary = (y_train_arr > threshold).astype(int)
        y_eval_binary = (y_eval_arr > threshold).astype(int)
        LOGGER.info("Training hurdle model (threshold=%s)", threshold)
        clf = LogisticRegression(
            max_iter=300,
            n_jobs=-1,
            class_weight="balanced",
        )
        clf.fit(X_train, y_train_binary)

        positive_mask = y_train_arr > threshold
        if positive_mask.any():
            reg = LinearRegression()
            reg.fit(X_train[positive_mask], np.log1p(y_train_arr[positive_mask]))
        else:
            reg = None

        prob_eval = clf.predict_proba(X_eval)[:, 1]
        if reg is not None:
            reg_pred = np.expm1(reg.predict(X_eval)).clip(min=0)
        else:
            reg_pred = np.zeros_like(prob_eval)
        pred_eval = prob_eval * reg_pred

        reg_metrics = regression_metrics(y_eval_arr, pred_eval)
        clf_metrics = classification_metrics(y_eval_binary, prob_eval)
        metrics["hurdle"] = {**reg_metrics, **clf_metrics}
        preds["hurdle"] = pred_eval
        preds["hurdle_prob"] = prob_eval
        models["hurdle"] = {"classifier": clf, "regressor": reg, "threshold": threshold}

    return metrics, preds, models


def _write_report(
    run_dir: Path,
    metrics: Dict[str, Dict[str, float]],
    best_model: str,
) -> None:
    lines = ["# Model Report", "", f"- Run id: {run_dir.name}", f"- Best model: **{best_model}**", ""]
    if metrics:
        header = "| model | mae_log1p | rmse_log1p | spearman | roc_auc | pr_auc |"
        lines.append(header)
        lines.append("|---|---:|---:|---:|---:|---:|")
        for name, vals in metrics.items():
            lines.append(
                "| {name} | {mae:.4f} | {rmse:.4f} | {spearman:.4f} | {roc:.4f} | {pr:.4f} |".format(
                    name=name,
                    mae=vals.get("mae_log1p", float("nan")),
                    rmse=vals.get("rmse_log1p", float("nan")),
                    spearman=vals.get("spearman", float("nan")),
                    roc=vals.get("roc_auc", float("nan")),
                    pr=vals.get("pr_auc", float("nan")),
                )
            )
    report_path = run_dir / "model_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    LOGGER.info("Report saved -> %s", report_path)


def save_artifacts(
    run_dir: Path,
    metrics: Dict[str, Dict[str, float]],
    preds: Dict[str, np.ndarray],
    y_eval: pd.Series,
    X_eval: pd.DataFrame,
    models: Dict[str, object],
    best_model: str,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = run_dir / "figures"

    preds_eval = pd.DataFrame({"y_true": y_eval.to_numpy()})
    for name, arr in preds.items():
        preds_eval[f"pred_{name}"] = arr
    preds_path = run_dir / "preds_eval.parquet"
    preds_eval.to_parquet(preds_path, index=False)

    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Plots for best model (regression)
    if best_model in preds and not best_model.endswith("_prob"):
        y_pred_best = preds[best_model]
        review_len = None
        for candidate in ("review_word_len", "review_char_len"):
            if candidate in X_eval.columns:
                review_len = X_eval[candidate].to_numpy()
                break
        _plot_scatter(y_eval.to_numpy(), y_pred_best, fig_dir / "scatter_true_vs_pred.png")
        _plot_residuals(y_eval.to_numpy(), y_pred_best, review_len, fig_dir / "residuals_vs_length.png")
        _plot_hist(y_eval.to_numpy(), y_pred_best, fig_dir / "hist_true_vs_pred.png")

    # Classification plots if hurdle probabilities are available
    if "hurdle_prob" in preds:
        prob = preds["hurdle_prob"]
        y_binary = (y_eval.to_numpy() > 0).astype(int)
        _plot_pr_curve(y_binary, prob, fig_dir / "pr_curve_hurdle.png")
        _plot_calibration(y_binary, prob, fig_dir / "calibration_hurdle.png")

    # Save model bundle (best model only)
    best_payload = {"model_name": best_model, "model": models.get(best_model), "feature_columns": X_eval.columns.tolist()}
    joblib.dump(best_payload, run_dir / "model.joblib")

    _write_report(run_dir, metrics, best_model)
    LOGGER.info("Artifacts saved -> %s", run_dir)


def select_best_model(metrics: Dict[str, Dict[str, float]]) -> str:
    if not metrics:
        raise ValueError("No metrics available to select best model.")
    scored: List[Tuple[str, float]] = []
    for name, vals in metrics.items():
        rmse = vals.get("rmse_log1p", math.inf)
        try:
            rmse_float = float(rmse)
        except (TypeError, ValueError):
            rmse_float = math.inf
        if math.isnan(rmse_float):
            rmse_float = math.inf
        scored.append((name, rmse_float))
    scored.sort(key=lambda t: t[1])
    return scored[0][0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate Yelp helpfulness models.")
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_PATH,
        help="Path to the shared YAML config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    run_root = Path(config.get("modeling", {}).get("run_output_dir", "artifacts"))
    run_id = _make_run_id()
    run_dir = run_root / run_id

    X_train, y_train, X_eval, y_eval = _load_feature_matrices(config)
    modeling_cfg = config.get("modeling", {})
    min_votes = modeling_cfg.get("min_total_votes", 0)

    # Sanitize targets and optionally filter to popular reviews
    X_train, y_train = _filter_invalid_targets(X_train, y_train)
    X_eval, y_eval = _filter_invalid_targets(X_eval, y_eval)
    X_train, y_train = _filter_by_popularity(X_train, y_train, min_votes)
    X_eval, y_eval = _filter_by_popularity(X_eval, y_eval, min_votes)

    metrics, preds, models = train_models(X_train, y_train, X_eval, y_eval, config)
    best_model = select_best_model(metrics)
    save_artifacts(run_dir, metrics, preds, y_eval, X_eval, models, best_model)


if __name__ == "__main__":
    main()
