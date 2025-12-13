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
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    balanced_accuracy_score,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    median_absolute_error,
    explained_variance_score,
)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

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


def _prepare_text_and_target_by_name(df: pd.DataFrame, target_name: str) -> Tuple[pd.Series, pd.Series]:
    """Prepare inputs: raw text and target (binary or numeric by column)."""
    needed_cols = {"text", target_name}
    missing = needed_cols - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns in evaluation data: {sorted(missing)}")
    X = df["text"].fillna("")
    if target_name == "target_is_useful":
        y = df[target_name].astype(int)
    else:
        y = pd.to_numeric(df[target_name], errors="coerce").fillna(0).astype(float)
    return X, y


def _find_model_path(model_uri: Optional[str], config: Dict[str, Any], model_name: str) -> Path:
    """Resolve path to a serialized model artifact (joblib/pkl) for classification.
    Priority:
      1) model_uri argument if it points to a local file or registry json
      2) models/registry/latest_{model_name}.json (artifact_path inside)
      3) fallback to baseline registry if available
    """
    if model_uri:
        p = Path(model_uri)
        if p.exists():
            return p
        if p.suffix.lower() == ".json":
            if p.exists():
                with p.open("r", encoding="utf-8") as f:
                    entry = json.load(f)
                return Path(entry["artifact_path"])

    # Model-specific registry
    ms_latest = Path(f"models/registry/latest_{model_name}.json")
    if ms_latest.exists():
        with ms_latest.open("r", encoding="utf-8") as f:
            entry = json.load(f)
        candidate = Path(entry["artifact_path"])
        if candidate.exists():
            return candidate

    # Fallback to baseline registry
    baseline_latest = Path("models/registry/latest_baseline_text_logreg.json")
    if baseline_latest.exists():
        with baseline_latest.open("r", encoding="utf-8") as f:
            entry = json.load(f)
        candidate = Path(entry.get("artifact_path", ""))
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Could not resolve a trained model. "
        "Pass model_uri or train models to populate models/registry."
    )


def _sanitize_alias(name: str) -> str:
    return name.replace(" ", "_").lower()


def evaluate(
    model_uri: Optional[str] = None,
    model_name: Optional[str] = None,
    target_name: Optional[str] = None,
) -> None:
    """Evaluate a text→usefulness model (classification or regression) on the evaluation split."""
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

    # If not provided, infer from config (mirrors train defaults)
    features_root = config.get("features", {})
    feature_recipe = features_root.get("recipe", "text_tfidf")
    training_cfg = config.get("training", {})
    inferred_target = target_name or training_cfg.get("target", "target_is_useful")
    inferred_model_type = training_cfg.get("model_type", "logistic_regression")

    # Optional evaluation sub-sampling
    eval_cfg = config.get("evaluation", {})
    sample_size = eval_cfg.get("sample_size", training_cfg.get("eval_sample_size", None))
    if sample_size is not None:
        try:
            sample_n = int(sample_size)
        except Exception:
            sample_n = None
        if sample_n is not None and sample_n > 0 and len(df) > sample_n:
            if inferred_target == "target_is_useful" and "target_is_useful" in df.columns:
                stratify = df["target_is_useful"].astype(int)
            else:
                stratify = None
            idx = np.arange(len(df))
            keep_idx, _ = train_test_split(
                idx,
                train_size=sample_n,
                random_state=training_cfg.get("random_state", 42),
                stratify=stratify,
                shuffle=True,
            )
            df = df.iloc[keep_idx].copy()

    # Default model name if not provided
    chosen_model_name = model_name or _sanitize_alias(f"{inferred_target}__{feature_recipe}__{inferred_model_type}")
    model_path = _find_model_path(model_uri, config, chosen_model_name)
    model_alias = _sanitize_alias(chosen_model_name)
    pipe = load(model_path)

    # Prepare X and y for the chosen target
    X_eval, y_eval = _prepare_text_and_target_by_name(df, inferred_target)

    # Switch by target type
    is_classification = inferred_target == "target_is_useful"
    preds_dir = Path(data_cfg.get("predictions", {}).get("dir", "data/predictions"))
    preds_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = preds_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    if is_classification:
        # Predict probability and label
        if hasattr(pipe, "predict_proba"):
            y_proba = pipe.predict_proba(X_eval)[:, 1]
        elif hasattr(pipe, "decision_function"):
            scores = pipe.decision_function(X_eval)
            y_proba = 1.0 / (1.0 + np.exp(-scores))
        else:
            y_proba = pipe.predict(X_eval).astype(float)
        y_pred = (y_proba >= 0.5).astype(int)

        # Classification metrics
        acc = float(accuracy_score(y_eval, y_pred))
        f1 = float(f1_score(y_eval, y_pred, zero_division=0))
        prec = float(precision_score(y_eval, y_pred, zero_division=0))
        rec = float(recall_score(y_eval, y_pred, zero_division=0))
        bal_acc = float(balanced_accuracy_score(y_eval, y_pred))
        try:
            rocauc = float(roc_auc_score(y_eval, y_proba))
        except Exception:
            rocauc = float("nan")
        try:
            ap = float(average_precision_score(y_eval, y_proba))
        except Exception:
            ap = float("nan")

        metrics = {
            "accuracy": acc,
            "f1": f1,
            "precision": prec,
            "recall": rec,
            "balanced_accuracy": bal_acc,
            "roc_auc": rocauc,
            "average_precision": ap,
            "eval_rows": int(len(y_eval)),
        }

        # Save predictions
        out_pred_path = preds_dir / f"{model_alias}_eval_cls_predictions.parquet"
        pd.DataFrame({"y_true": y_eval.values, "y_pred": y_pred, "y_proba": y_proba}).to_parquet(
            out_pred_path, index=False
        )

        # Plots: ROC, PR, conf mat, prob hist, calibration
        try:
            fpr, tpr, _ = roc_curve(y_eval, y_proba)
            plt.figure(figsize=(6, 5))
            plt.plot(fpr, tpr, label=f"ROC AUC = {metrics['roc_auc']:.3f}")
            plt.plot([0, 1], [0, 1], "k--")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC Curve")
            plt.legend(loc="lower right")
            plt.tight_layout()
            roc_path = plots_dir / f"{model_alias}_eval_cls_roc.png"
            plt.savefig(roc_path, dpi=150)
            plt.close()
        except Exception:
            roc_path = None

        try:
            precision, recall, _ = precision_recall_curve(y_eval, y_proba)
            plt.figure(figsize=(6, 5))
            plt.plot(recall, precision, label=f"AP = {metrics['average_precision']:.3f}")
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title("Precision-Recall Curve")
            plt.legend(loc="lower left")
            plt.tight_layout()
            pr_path = plots_dir / f"{model_alias}_eval_cls_pr.png"
            plt.savefig(pr_path, dpi=150)
            plt.close()
        except Exception:
            pr_path = None

        try:
            cm = confusion_matrix(y_eval, y_pred)
            plt.figure(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title("Confusion Matrix")
            plt.tight_layout()
            cm_path = plots_dir / f"{model_alias}_eval_cls_confusion_matrix.png"
            plt.savefig(cm_path, dpi=150)
            plt.close()
        except Exception:
            cm_path = None

        try:
            plt.figure(figsize=(7, 4))
            sns.histplot(y_proba, bins=50, kde=False)
            plt.xlabel("Predicted probability (positive)")
            plt.ylabel("count")
            plt.title("Predicted probability distribution")
            plt.tight_layout()
            prob_hist_path = plots_dir / f"{model_alias}_eval_cls_prob_hist.png"
            plt.savefig(prob_hist_path, dpi=150)
            plt.close()
        except Exception:
            prob_hist_path = None

        try:
            df_cal = pd.DataFrame({"y_true": y_eval.values, "y_proba": y_proba})
            df_cal["bin"] = pd.qcut(df_cal["y_proba"], q=10, duplicates="drop")
            calib = df_cal.groupby("bin", observed=True).agg(
                mean_pred=("y_proba", "mean"), mean_true=("y_true", "mean")
            ).reset_index(drop=True)
            plt.figure(figsize=(6, 6))
            sns.lineplot(x="mean_pred", y="mean_true", data=calib, marker="o")
            plt.plot([0, 1], [0, 1], "r--", linewidth=1)
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.xlabel("mean predicted (per bin)")
            plt.ylabel("mean true (per bin)")
            plt.title("Calibration plot (binned by predicted probability)")
            plt.tight_layout()
            calib_path = plots_dir / f"{model_alias}_eval_cls_calibration.png"
            plt.savefig(calib_path, dpi=150)
            plt.close()
        except Exception:
            calib_path = None
    else:
        # Regression predictions
        y_pred = pipe.predict(X_eval).astype(float)
        # Regression metrics
        mae = float(mean_absolute_error(y_eval, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_eval, y_pred)))
        r2 = float(r2_score(y_eval, y_pred))
        medae = float(median_absolute_error(y_eval, y_pred))
        evs = float(explained_variance_score(y_eval, y_pred))
        metrics = {
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "medae": medae,
            "explained_variance": evs,
            "eval_rows": int(len(y_eval)),
        }

        # Save predictions
        out_pred_path = preds_dir / f"{model_alias}_eval_reg_predictions.parquet"
        pd.DataFrame({"y_true": y_eval.values, "y_pred": y_pred}).to_parquet(out_pred_path, index=False)

        # Plots: true vs pred, residuals hist, pred hist, calibration-style bins
        try:
            plt.figure(figsize=(6, 6))
            sns.scatterplot(x=y_eval, y=y_pred, alpha=0.4, edgecolor=None)
            plt.plot([min(y_eval), max(y_eval)], [min(y_eval), max(y_eval)], "r--", linewidth=1)
            plt.xlabel("y_true")
            plt.ylabel("y_pred")
            plt.title("True vs Predicted")
            plt.tight_layout()
            scatter_path = plots_dir / f"{model_alias}_eval_reg_scatter.png"
            plt.savefig(scatter_path, dpi=150)
            plt.close()
        except Exception:
            scatter_path = None

        try:
            residuals = y_eval - y_pred
            plt.figure(figsize=(7, 4))
            sns.histplot(residuals, bins=50, kde=False)
            plt.xlabel("residual = y_true - y_pred")
            plt.ylabel("count")
            plt.title("Residuals distribution")
            plt.tight_layout()
            resid_path = plots_dir / f"{model_alias}_eval_reg_residuals.png"
            plt.savefig(resid_path, dpi=150)
            plt.close()
        except Exception:
            resid_path = None

        try:
            plt.figure(figsize=(7, 4))
            sns.histplot(y_pred, bins=50, kde=False)
            plt.xlabel("y_pred")
            plt.ylabel("count")
            plt.title("Predicted distribution")
            plt.tight_layout()
            pred_hist_path = plots_dir / f"{model_alias}_eval_reg_pred_hist.png"
            plt.savefig(pred_hist_path, dpi=150)
            plt.close()
        except Exception:
            pred_hist_path = None

        try:
            df_cal = pd.DataFrame({"y_true": y_eval.values, "y_pred": y_pred})
            df_cal["bin"] = pd.qcut(df_cal["y_pred"], q=10, duplicates="drop")
            calib = df_cal.groupby("bin", observed=True).agg(
                mean_pred=("y_pred", "mean"), mean_true=("y_true", "mean")
            ).reset_index(drop=True)
            plt.figure(figsize=(6, 6))
            sns.lineplot(x="mean_pred", y="mean_true", data=calib, marker="o")
            lo = float(np.nanmin([calib["mean_pred"].min(), calib["mean_true"].min(), 0]))
            hi = float(np.nanmax([calib["mean_pred"].max(), calib["mean_true"].max(), 1]))
            plt.plot([lo, hi], [lo, hi], "r--", linewidth=1)
            plt.xlabel("mean predicted (per bin)")
            plt.ylabel("mean true (per bin)")
            plt.title("Calibration plot (binned by y_pred)")
            plt.tight_layout()
            reg_calib_path = plots_dir / f"{model_alias}_eval_reg_calibration.png"
            plt.savefig(reg_calib_path, dpi=150)
            plt.close()
        except Exception:
            reg_calib_path = None

    with mlflow.start_run(run_name="yelp_helpfulness_evaluate"):
        mlflow.log_param("model_path", str(model_path))
        mlflow.log_param("model_name", chosen_model_name)
        mlflow.log_param("target_name", inferred_target)
        mlflow.log_param("evaluation_split", eval_split_dir)
        mlflow.log_param("evaluation_rows_total", int(len(_load_eval_data(evaluation_dir))))
        mlflow.log_param("evaluation_rows_used", int(len(df)))
        if sample_size is not None:
            mlflow.log_param("evaluation_sample_size", int(sample_size))
        for k, v in metrics.items():
            if isinstance(v, (int, float)) and np.isfinite(v):
                mlflow.log_metric(k, float(v))
        mlflow.log_artifact(str(out_pred_path))
        # Log plots if available
        if is_classification:
            for p in (locals().get("roc_path"), locals().get("pr_path"), locals().get("cm_path"), locals().get("prob_hist_path"), locals().get("calib_path")):
                if p is not None:
                    mlflow.log_artifact(str(p))
        else:
            for p in (locals().get("scatter_path"), locals().get("resid_path"), locals().get("pred_hist_path"), locals().get("reg_calib_path")):
                if p is not None:
                    mlflow.log_artifact(str(p))

    print(
        f"Evaluation complete. Model: {chosen_model_name}. Metrics: {metrics}\n"
        f"Saved predictions to: {out_pred_path}"
    )


if __name__ == "__main__":
    evaluate()
