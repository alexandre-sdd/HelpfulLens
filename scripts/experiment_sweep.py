from __future__ import annotations

import argparse
import copy
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features.build_features import build_feature_matrices, load_config
from src.models.train_and_evaluate import save_artifacts, select_best_model, train_models


def set_nested(cfg: Dict, dotted: str, value) -> None:
    """Set a nested config key using dotted notation."""
    parts = dotted.split(".")
    cur = cfg
    for key in parts[:-1]:
        if key not in cur or not isinstance(cur[key], dict):
            cur[key] = {}
        cur = cur[key]
    cur[parts[-1]] = value


def load_X_y(feature_dir: Path, target_col: str):
    X_train = pd.read_parquet(feature_dir / "X_train.parquet")
    y_train = pd.read_parquet(feature_dir / "y_train.parquet")[target_col]
    X_eval = pd.read_parquet(feature_dir / "X_eval.parquet")
    y_eval = pd.read_parquet(feature_dir / "y_eval.parquet")[target_col]
    return X_train, y_train, X_eval, y_eval


def _sanitize_xy(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    y_arr = y.to_numpy()
    mask = np.isfinite(y_arr) & (y_arr >= 0)
    if not mask.any():
        raise ValueError("No valid target values after filtering for finiteness and non-negativity.")
    return X.loc[mask].reset_index(drop=True), y.loc[mask].reset_index(drop=True)


def sweep(config_path: Path, scenarios: List[Dict[str, object]], summary_path: Path) -> None:
    base_cfg = load_config(config_path)
    summary_rows = []

    for scenario in scenarios:
        name = scenario.get("name", "scenario")
        overrides = scenario.get("overrides", {})
        cfg = copy.deepcopy(base_cfg)
        for key, val in overrides.items():
            set_nested(cfg, str(key), val)

        # isolate feature + artifact dirs per scenario
        feature_dir = Path(cfg.get("data", {}).get("features", {}).get("dir", "data/features")) / name
        set_nested(cfg, "data.features.dir", str(feature_dir))
        run_root = Path(cfg.get("modeling", {}).get("run_output_dir", "artifacts")) / "experiments"
        run_dir = run_root / name
        set_nested(cfg, "modeling.run_output_dir", str(run_root))

        run_dir.mkdir(parents=True, exist_ok=True)
        feature_dir.mkdir(parents=True, exist_ok=True)

        # Build features and load X/y
        build_feature_matrices(cfg)
        target_col = cfg.get("modeling", {}).get("target_column", "target_useful_votes")
        X_train, y_train, X_eval, y_eval = load_X_y(feature_dir, target_col)
        X_train, y_train = _sanitize_xy(X_train, y_train)
        X_eval, y_eval = _sanitize_xy(X_eval, y_eval)

        metrics, preds, models = train_models(X_train, y_train, X_eval, y_eval, cfg)
        best_model = select_best_model(metrics)
        save_artifacts(run_dir, metrics, preds, y_eval, X_eval, models, best_model)

        best_metrics = metrics.get(best_model, {})
        summary_rows.append(
            {
                "scenario": name,
                "best_model": best_model,
                "mae_log1p": best_metrics.get("mae_log1p"),
                "rmse_log1p": best_metrics.get("rmse_log1p"),
                "spearman": best_metrics.get("spearman"),
                "roc_auc": best_metrics.get("roc_auc"),
                "pr_auc": best_metrics.get("pr_auc"),
                "feature_dir": str(feature_dir),
                "run_dir": str(run_dir),
                "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
                "overrides": overrides,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_path, index=False)
    summary_path.with_suffix(".json").write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")
    print("Summary saved ->", summary_path.resolve())


def default_scenarios() -> List[Dict[str, object]]:
    return [
        {
            "name": "numeric_only",
            "overrides": {
                "features.tfidf.include_tfidf": False,
                "features.sentiment.include_sentiment_features": False,
                "modeling.min_total_votes": 0,
            },
        },
        {
            "name": "tfidf_3k",
            "overrides": {
                "features.tfidf.include_tfidf": True,
                "features.tfidf.max_tfidf_features": 3000,
                "features.tfidf.min_df": 5,
                "features.tfidf.ngram_range": [1, 2],
                "modeling.min_total_votes": 0,
            },
        },
        {
            "name": "sentiment_vader",
            "overrides": {
                "features.sentiment.include_sentiment_features": True,
                "modeling.min_total_votes": 0,
            },
        },
        {
            "name": "exposure_min3",
            "overrides": {
                "modeling.min_total_votes": 3,
                "features.tfidf.include_tfidf": False,
                "features.sentiment.include_sentiment_features": False,
            },
        },
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep multiple feature/model configs and collect metrics.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("src/config/config.yaml"),
        help="Base YAML config path.",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("artifacts/experiments/summary.csv"),
        help="Where to write the summary CSV (JSON saved alongside).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    scenarios = default_scenarios()
    sweep(args.config, scenarios, args.summary)
