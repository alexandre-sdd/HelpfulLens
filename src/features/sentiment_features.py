"""Reusable sentiment and feature engineering utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

try:
    from transformers import pipeline as hf_pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    hf_pipeline = None
    TRANSFORMERS_AVAILABLE = False

DEFAULT_SENTIMENT_BACKEND = "vader"
DEFAULT_TRANSFORMER_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
DEFAULT_TRANSFORMER_BATCH_SIZE = 32
DEFAULT_TRANSFORMER_MAX_LENGTH = 256


@dataclass
class SentimentConfig:
    backend: str = DEFAULT_SENTIMENT_BACKEND
    transformer_model: str = DEFAULT_TRANSFORMER_MODEL
    transformer_batch_size: int = DEFAULT_TRANSFORMER_BATCH_SIZE
    transformer_device: int = -1
    transformer_max_length: int = DEFAULT_TRANSFORMER_MAX_LENGTH


def sanitize_label(label: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in label.lower()).strip("_")
    return cleaned or "label"


def apply_with_progress(series: pd.Series, func, batch_size: int, desc: str) -> pd.Series | pd.DataFrame:
    """Apply a Python callback in chunks while keeping the script responsive."""
    total = len(series)
    if total == 0:
        return series.apply(func)

    outputs: list[pd.Series | pd.DataFrame] = []
    pbar = tqdm(total=total, desc=desc, unit="rows")
    try:
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            chunk = series.iloc[start:end].apply(func)
            outputs.append(chunk)
            pbar.update(end - start)
    finally:
        pbar.close()

    return pd.concat(outputs)


def compute_text_features(text: pd.Series) -> pd.DataFrame:
    """Derive lightweight textual statistics."""
    txt = text.fillna("").astype(str)

    def text_stats(s: str) -> pd.Series:
        words = s.split()
        word_count = len(words)
        avg_word_len = (sum(len(w) for w in words) / word_count) if word_count else 0.0
        uppercase = sum(ch.isupper() for ch in s)
        caps_ratio = uppercase / len(s) if s else 0.0
        return pd.Series(
            {
                "text_len_chars": len(s),
                "text_len_words": word_count,
                "avg_word_length": avg_word_len,
                "caps_ratio": caps_ratio,
                "exclaim_count": s.count("!"),
                "question_count": s.count("?"),
            }
        )

    stats = apply_with_progress(txt, text_stats, batch_size=50_000, desc="Text metrics")
    return stats.reset_index(drop=True)


def compute_vader_sentiment(text: pd.Series) -> pd.DataFrame:
    analyzer = SentimentIntensityAnalyzer()

    def score(s: str) -> pd.Series:
        scores = analyzer.polarity_scores(s or "")
        return pd.Series(
            {
                "sentiment_neg": scores["neg"],
                "sentiment_neu": scores["neu"],
                "sentiment_pos": scores["pos"],
                "sentiment_compound": scores["compound"],
            }
        )

    txt = text.fillna("").astype(str)
    sentiment = apply_with_progress(txt, score, batch_size=10_000, desc="Sentiment (VADER)")
    return sentiment.reset_index(drop=True)


def compute_transformer_sentiment(text: pd.Series, cfg: SentimentConfig) -> pd.DataFrame:
    if not TRANSFORMERS_AVAILABLE or hf_pipeline is None:
        raise ImportError(
            "transformers is not installed. Install via `pip install transformers torch` "
            "or use the VADER backend."
        )

    model_name = cfg.transformer_model or DEFAULT_TRANSFORMER_MODEL
    batch_size = max(cfg.transformer_batch_size, 1)
    max_length = max(cfg.transformer_max_length, 16)
    device = cfg.transformer_device if cfg.transformer_device is not None else -1

    classifier = hf_pipeline(
        task="sentiment-analysis",
        model=model_name,
        tokenizer=model_name,
        device=device,
    )

    txt = text.fillna("").astype(str)
    total = len(txt)
    outputs: list[pd.DataFrame] = []
    pbar = tqdm(total=total, desc="Sentiment (Transformer)", unit="rows")

    try:
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch_texts = txt.iloc[start:end].tolist()
            results = classifier(
                batch_texts,
                truncation=True,
                max_length=max_length,
                return_all_scores=True,
            )
            rows = []
            for scores in results:
                row = {}
                if not scores:
                    rows.append(row)
                    continue
                best = max(scores, key=lambda s: s.get("score", 0.0))
                row["sentiment_label"] = best.get("label")
                row["sentiment_score"] = best.get("score")
                for entry in scores:
                    label = entry.get("label", "")
                    score = entry.get("score", 0.0)
                    key = f"sentiment_{sanitize_label(label)}"
                    row[key] = score
                rows.append(row)
            outputs.append(pd.DataFrame(rows))
            pbar.update(end - start)
    finally:
        pbar.close()

    if not outputs:
        return pd.DataFrame()
    return pd.concat(outputs, ignore_index=True)


def compute_sentiment_features(text: pd.Series, cfg: SentimentConfig) -> pd.DataFrame:
    backend = (cfg.backend or DEFAULT_SENTIMENT_BACKEND).lower()
    if backend == "transformer":
        return compute_transformer_sentiment(text, cfg)
    return compute_vader_sentiment(text)


def engineer_features(
    reviews: pd.DataFrame,
    business: pd.DataFrame,
    users: pd.DataFrame | None,
    sentiment_cfg: SentimentConfig,
) -> pd.DataFrame:
    """Enrich cleaned reviews with sentiment, text stats, and joins."""
    df = reviews.copy()

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        date_series = df["date"]
    else:
        date_series = None

    if "review_year" in df.columns:
        df["review_year"] = df["review_year"].fillna(date_series.dt.year if date_series is not None else np.nan)
    elif date_series is not None:
        df["review_year"] = date_series.dt.year

    if "review_month" in df.columns:
        df["review_month"] = df["review_month"].fillna(date_series.dt.month if date_series is not None else np.nan)
    elif date_series is not None:
        df["review_month"] = date_series.dt.month

    if "review_day_of_week" in df.columns:
        df["review_day_of_week"] = df["review_day_of_week"].fillna(
            date_series.dt.dayofweek if date_series is not None else np.nan
        )
    elif date_series is not None:
        df["review_day_of_week"] = date_series.dt.dayofweek

    df["review_is_weekend"] = df["review_day_of_week"].isin([5, 6])

    df["target_helpful_votes"] = df["useful"].clip(lower=0)
    df["target_helpful_log1p"] = np.log1p(df["target_helpful_votes"].astype(float))
    if "date" in df.columns:
        max_date = df["date"].max()
        df["review_age_days"] = (max_date - df["date"]).dt.days
    else:
        df["review_age_days"] = np.nan

    text_features = compute_text_features(df["text"])
    sentiment_features = compute_sentiment_features(df["text"], sentiment_cfg)
    df = pd.concat([df.reset_index(drop=True), text_features, sentiment_features], axis=1)

    business_cols = [
        "business_id",
        "name",
        "city",
        "state",
        "stars",
        "review_count",
        "categories",
        "is_open",
    ]
    biz = business[business_cols].copy()
    biz.rename(
        columns={
            "name": "business_name",
            "stars": "business_stars",
            "review_count": "business_review_count",
            "categories": "business_categories",
            "is_open": "business_is_open",
        },
        inplace=True,
    )
    df = df.merge(biz, on="business_id", how="left")
    df["business_review_count"] = df["business_review_count"].fillna(0)
    df["business_review_density"] = df["business_review_count"] / (df["text_len_words"] + 1)
    df["business_category_count"] = df["business_categories"].apply(
        lambda cats: len(cats) if isinstance(cats, list) else 0
    )

    # Expand indicators for top categories in the sample
    if "business_categories" in df.columns:
        all_categories = (
            df["business_categories"]
            .explode()
            .dropna()
            .astype(str)
            .str.strip()
        )
        top_categories = all_categories.value_counts().head(15).index
        for cat in top_categories:
            slug = cat.lower().replace(" ", "_").replace("/", "_")
            df[f"cat_{slug}"] = df["business_categories"].apply(
                lambda cats: int(isinstance(cats, list) and cat in cats)
            )

    if users is not None and not users.empty:
        user_cols = [
            "user_id",
            "review_count",
            "fans",
            "elite_raw",
            "is_elite_ever",
            "average_stars",
            "useful",
            "funny",
            "cool",
        ]
        available_cols = [col for col in user_cols if col in users.columns]
        user_df = users[available_cols].copy()
        rename_map = {
            "review_count": "user_review_count",
            "fans": "user_fans",
            "elite_raw": "user_elite_raw",
            "is_elite_ever": "user_is_elite_ever",
            "average_stars": "user_average_stars",
            "useful": "user_useful_votes",
            "funny": "user_funny_votes",
            "cool": "user_cool_votes",
        }
        user_df.rename(columns={k: v for k, v in rename_map.items() if k in user_df.columns}, inplace=True)
        df = df.merge(user_df, on="user_id", how="left")

    return df


def load_clean_table(path: Path, columns: Iterable[str] | None = None, n_rows: int | None = None) -> pd.DataFrame:
    df = pd.read_parquet(path, columns=columns)
    if n_rows is not None and n_rows > 0:
        df = df.head(n_rows).copy()
    return df
