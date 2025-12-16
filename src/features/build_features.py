"""Assemble modeling-ready feature matrices from prepared Yelp datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer

from src.data.make_dataset import EVAL_OUTPUT_NAME, TRAIN_OUTPUT_NAME
from src.utils.logging_utils import get_logger
from src.utils.schema_checks import (
    assert_aligned_columns,
    assert_no_duplicate_columns,
    assert_no_leakage,
    coerce_boolean_to_int,
)

LOGGER = get_logger("build_features")
CONFIG_PATH = Path("src/config/config.yaml")
DEFAULT_JOIN_KEY = "review_id"
DEFAULT_TARGET = "target_useful_votes"
FEATURE_FILENAMES = {
    "X_train": "X_train.parquet",
    "y_train": "y_train.parquet",
    "X_eval": "X_eval.parquet",
    "y_eval": "y_eval.parquet",
    "schema": "feature_schema.json",
}


def load_config(config_path: Path = CONFIG_PATH) -> Dict:
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _slugify(token: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in token.lower()).strip("_")
    return cleaned or "unk"


def _assert_paths_exist(paths: Sequence[Path]) -> None:
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required parquet(s): {missing}")


def _load_dataset(path: Path) -> pd.DataFrame:
    LOGGER.info("Loading %s", path)
    df = pd.read_parquet(path)
    assert_no_duplicate_columns(df, path.name)
    return df


def _merge_sentiment(
    df: pd.DataFrame,
    sentiment_path: Path,
    join_key: str,
) -> pd.DataFrame:
    if not sentiment_path.exists():
        LOGGER.warning("Sentiment path %s not found; skipping sentiment merge.", sentiment_path)
        return df

    sentiment = pd.read_parquet(sentiment_path)
    keep_cols = [c for c in sentiment.columns if c != join_key and not c.startswith("target_")]
    if join_key not in sentiment.columns:
        LOGGER.warning("Join key %s missing from sentiment parquet; skipping sentiment merge.", join_key)
        return df
    LOGGER.info("Merging sentiment features from %s (cols=%d)", sentiment_path, len(keep_cols))
    merged = df.merge(
        sentiment[[join_key] + keep_cols],
        on=join_key,
        how="left",
        validate="1:1",
        suffixes=("", "_sent"),
    )
    assert_no_duplicate_columns(merged, "post-sentiment-merge")
    return merged


def _extract_top_tokens(series: pd.Series, top_k: int) -> List[str]:
    tokens = (
        series.dropna()
        .astype(str)
        .str.strip()
        .replace("", np.nan)
        .dropna()
        .value_counts()
        .head(top_k)
        .index.tolist()
    )
    return tokens


def _extract_top_list_tokens(series: pd.Series, top_k: int) -> List[str]:
    exploded = series.dropna().explode()
    exploded = exploded.dropna().astype(str).str.strip()
    tokens = exploded.value_counts().head(top_k).index.tolist()
    return tokens


def _normalize_category_list(value) -> List[str]:
    """Normalize a category value into a deduplicated list.

    Args:
        value: Raw category field (string or list) from the business table.

    Returns:
        List of cleaned category tokens with generic parents removed.
    """
    if value is None:
        return []
    raw_tokens: List[str]
    if isinstance(value, list):
        raw_tokens = [str(v) for v in value]
    else:
        raw_tokens = [str(value)]

    parents = {
        "restaurants",
        "food",
        "bars",
        "nightlife",
        "event planning & services",
        "hotels",
        "hotels & travel",
        "shopping",
    }
    cleaned: List[str] = []
    seen = set()
    for token in raw_tokens:
        for part in token.split(","):
            t = part.strip()
            if not t:
                continue
            t_lower = t.lower()
            if t_lower in parents:
                continue
            if t_lower in seen:
                continue
            seen.add(t_lower)
            cleaned.append(t)
    return cleaned


def _encode_top_categories(
    train_series: pd.Series,
    eval_series: pd.Series,
    prefix: str,
    top_k: int,
    list_input: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, List[str]]]:
    if top_k <= 0:
        empty = pd.DataFrame(index=train_series.index)
        return empty, pd.DataFrame(index=eval_series.index), {}

    top_tokens = (
        _extract_top_list_tokens(train_series, top_k) if list_input else _extract_top_tokens(train_series, top_k)
    )
    if not top_tokens:
        empty = pd.DataFrame(index=train_series.index)
        return empty, pd.DataFrame(index=eval_series.index), {}

    col_names = [f"{prefix}_{_slugify(tok)}" for tok in top_tokens]

    def encode(series: pd.Series) -> pd.DataFrame:
        if list_input:
            series = series.apply(_normalize_category_list)
        df = pd.DataFrame(0, index=series.index, columns=col_names, dtype=np.int8)
        for tok, col in zip(top_tokens, col_names):
            df[col] = series.apply(
                lambda val: int(tok in val) if isinstance(val, list) else int(str(val).strip() == tok)
            )
        return df

    return encode(train_series), encode(eval_series), {prefix: top_tokens}


def _fit_tfidf(
    train_text: pd.Series,
    eval_text: pd.Series,
    max_features: int,
    min_df: int,
    ngram_range: Tuple[int, int],
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=min_df,
        ngram_range=ngram_range,
        stop_words="english",
    )
    LOGGER.info(
        "Fitting TF-IDF (max_features=%d, min_df=%d, ngram_range=%s) on %d rows",
        max_features,
        min_df,
        ngram_range,
        len(train_text),
    )
    tfidf_train = vectorizer.fit_transform(train_text.fillna("").astype(str))
    tfidf_eval = vectorizer.transform(eval_text.fillna("").astype(str))
    feature_names = [f"tfidf_{_slugify(t)}" for t in vectorizer.get_feature_names_out()]

    train_df = pd.DataFrame(tfidf_train.toarray(), columns=feature_names, index=train_text.index, dtype=np.float32)
    eval_df = pd.DataFrame(tfidf_eval.toarray(), columns=feature_names, index=eval_text.index, dtype=np.float32)
    meta = {
        "enabled": True,
        "vocab_size": len(feature_names),
        "max_features": max_features,
        "min_df": min_df,
        "ngram_range": list(ngram_range),
    }
    return train_df, eval_df, meta


def _prepare_numeric_features(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    target_col: str,
    drop_cols: Iterable[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    drop_set = set(drop_cols)
    numeric_cols: List[str] = []
    for col in train_df.columns:
        if col == target_col or col in drop_set:
            continue
        if pd.api.types.is_bool_dtype(train_df[col]):
            numeric_cols.append(col)
        elif pd.api.types.is_numeric_dtype(train_df[col]):
            numeric_cols.append(col)
    if not numeric_cols:
        return pd.DataFrame(index=train_df.index), pd.DataFrame(index=eval_df.index), []

    def coerce(df: pd.DataFrame) -> pd.DataFrame:
        subset = df[numeric_cols].copy()
        bool_cols = [c for c in subset.columns if pd.api.types.is_bool_dtype(subset[c])]
        subset = coerce_boolean_to_int(subset, bool_cols)
        subset = subset.fillna(0)
        return subset.astype(np.float32)

    return coerce(train_df), coerce(eval_df), numeric_cols


def build_feature_matrices(config: Dict) -> Dict[str, Path]:
    data_cfg = config.get("data", {})
    datasets_cfg = data_cfg.get("datasets", {})
    data_features_cfg = data_cfg.get("features", {})
    feature_root = Path(data_features_cfg.get("dir", "data/features"))
    feature_root.mkdir(parents=True, exist_ok=True)

    training_dir = Path(datasets_cfg.get("training_dir", "data/datasets/training"))
    eval_dir = Path(datasets_cfg.get("evaluation_dir", "data/datasets/evaluation"))
    train_path = training_dir / TRAIN_OUTPUT_NAME
    eval_path = eval_dir / EVAL_OUTPUT_NAME
    _assert_paths_exist([train_path, eval_path])

    modeling_cfg = config.get("modeling", {})
    target_col = modeling_cfg.get("target_column", DEFAULT_TARGET)
    join_key = DEFAULT_JOIN_KEY

    feature_cfg = config.get("features", {})
    include_sentiment = feature_cfg.get("sentiment", {}).get("include_sentiment_features", False)
    sentiment_path = Path(feature_cfg.get("sentiment", {}).get("sentiment_path", "data/features/review_features_with_sentiment.parquet"))
    tfidf_cfg = feature_cfg.get("tfidf", {})
    include_tfidf = tfidf_cfg.get("include_tfidf", False)
    max_tfidf = int(tfidf_cfg.get("max_tfidf_features", 3000))
    min_df = int(tfidf_cfg.get("min_df", 5))
    ngram_range = tuple(tfidf_cfg.get("ngram_range", [1, 2]))

    category_cfg = feature_cfg.get("categories", {})
    top_cities = int(category_cfg.get("top_cities", 50))
    top_states = int(category_cfg.get("top_states", 20))
    top_biz_cats = int(category_cfg.get("top_business_categories", 30))

    train_df = _load_dataset(train_path)
    eval_df = _load_dataset(eval_path)

    if include_sentiment:
        train_df = _merge_sentiment(train_df, sentiment_path, join_key)
        eval_df = _merge_sentiment(eval_df, sentiment_path, join_key)

    if target_col not in train_df.columns:
        raise KeyError(f"Target column '{target_col}' not found in training dataset.")
    if target_col not in eval_df.columns:
        raise KeyError(f"Target column '{target_col}' not found in evaluation dataset.")

    y_train = train_df[target_col].copy()
    y_eval = eval_df[target_col].copy()

    drop_cols = {
        target_col,
        "target_is_useful",
        "target_helpful_votes",
        "target_helpful_log1p",
        "useful",  # raw target proxy; drop to avoid leakage
        "total_votes",  # contains the target; drop to avoid leakage
        "useful_rate_smoothed",  # derived from target; drop to avoid leakage
        join_key,
    }
    text_col = "text" if "text" in train_df.columns else None
    if text_col:
        drop_cols.add(text_col)

    numeric_train, numeric_eval, numeric_cols = _prepare_numeric_features(
        train_df, eval_df, target_col, drop_cols
    )

    city_train, city_eval, city_map = _encode_top_categories(
        train_df["city"] if "city" in train_df.columns else pd.Series(index=train_df.index, dtype="object"),
        eval_df["city"] if "city" in eval_df.columns else pd.Series(index=eval_df.index, dtype="object"),
        prefix="city",
        top_k=top_cities,
        list_input=False,
    )
    state_train, state_eval, state_map = _encode_top_categories(
        train_df["state"] if "state" in train_df.columns else pd.Series(index=train_df.index, dtype="object"),
        eval_df["state"] if "state" in eval_df.columns else pd.Series(index=eval_df.index, dtype="object"),
        prefix="state",
        top_k=top_states,
        list_input=False,
    )
    cats_train, cats_eval, cat_map = _encode_top_categories(
        train_df["categories"] if "categories" in train_df.columns else pd.Series(index=train_df.index, dtype="object"),
        eval_df["categories"] if "categories" in eval_df.columns else pd.Series(index=eval_df.index, dtype="object"),
        prefix="cat",
        top_k=top_biz_cats,
        list_input=True,
    )

    tfidf_train = pd.DataFrame(index=train_df.index)
    tfidf_eval = pd.DataFrame(index=eval_df.index)
    tfidf_meta: Dict[str, object] = {"enabled": False}
    if include_tfidf and text_col:
        tfidf_train, tfidf_eval, tfidf_meta = _fit_tfidf(
            train_df[text_col],
            eval_df[text_col],
            max_features=max_tfidf,
            min_df=min_df,
            ngram_range=ngram_range,  # type: ignore[arg-type]
        )
    elif include_tfidf and not text_col:
        LOGGER.warning("TF-IDF requested but no text column found; skipping TF-IDF features.")

    X_train_parts = [numeric_train, city_train, state_train, cats_train, tfidf_train]
    X_eval_parts = [numeric_eval, city_eval, state_eval, cats_eval, tfidf_eval]
    X_train = pd.concat(X_train_parts, axis=1)
    X_eval = pd.concat(X_eval_parts, axis=1)

    X_train = X_train.fillna(0)
    X_eval = X_eval.fillna(0)

    assert_no_leakage(X_train.columns, [target_col])
    assert_aligned_columns(X_train, X_eval)

    feature_order = list(X_train.columns)
    dtypes_map = {col: str(dtype) for col, dtype in X_train.dtypes.items()}

    schema = {
        "feature_order": feature_order,
        "dtypes": dtypes_map,
        "target": target_col,
        "join_keys": [join_key],
        "one_hot_maps": {**city_map, **state_map, **cat_map},
        "tfidf": tfidf_meta,
        "imputations": {
            "numeric": 0.0,
            "categorical_missing_token": "",
        },
    }

    paths = {
        "X_train": feature_root / FEATURE_FILENAMES["X_train"],
        "y_train": feature_root / FEATURE_FILENAMES["y_train"],
        "X_eval": feature_root / FEATURE_FILENAMES["X_eval"],
        "y_eval": feature_root / FEATURE_FILENAMES["y_eval"],
        "schema": feature_root / FEATURE_FILENAMES["schema"],
    }

    LOGGER.info("Saving features to %s", feature_root)
    X_train.to_parquet(paths["X_train"], index=False)
    y_train.to_frame(name=target_col).to_parquet(paths["y_train"], index=False)
    X_eval.to_parquet(paths["X_eval"], index=False)
    y_eval.to_frame(name=target_col).to_parquet(paths["y_eval"], index=False)
    paths["schema"].write_text(json.dumps(schema, indent=2), encoding="utf-8")

    LOGGER.info("Feature build complete: %d train cols, %d rows", X_train.shape[1], X_train.shape[0])
    LOGGER.info("Schema saved -> %s", paths["schema"])
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build modeling feature matrices for Yelp helpfulness.")
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
    build_feature_matrices(config)


if __name__ == "__main__":
    main()
