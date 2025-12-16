"""Build sentiment-enriched review features from cleaned Yelp tables."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features.sentiment_features import (
    DEFAULT_SENTIMENT_BACKEND,
    DEFAULT_TRANSFORMER_BATCH_SIZE,
    DEFAULT_TRANSFORMER_MAX_LENGTH,
    DEFAULT_TRANSFORMER_MODEL,
    SentimentConfig,
    engineer_features,
    load_clean_table,
)

CLEAN_DIR = Path("data/cleaned")
FEATURE_DIR = Path("data/features")
REVIEW_PATH = CLEAN_DIR / "reviews_clean.parquet"
BUSINESS_PATH = CLEAN_DIR / "business_clean.parquet"
USER_PATH = CLEAN_DIR / "users_clean.parquet"
OUTPUT_PATH = FEATURE_DIR / "review_features_with_sentiment.parquet"


def run_feature_engineering(n_reviews: int | None, sentiment_cfg: SentimentConfig) -> Path:
    """Engineer sentiment and text features, then persist them to parquet.

    Args:
        n_reviews: Optional cap on the number of reviews to process.
        sentiment_cfg: Backend configuration for sentiment scoring.

    Returns:
        Path to the saved feature parquet.
    """
    print("Feature engineering run started...")
    start = time.time()

    for path in (REVIEW_PATH, BUSINESS_PATH, USER_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path.resolve()}")

    reviews = load_clean_table(REVIEW_PATH, n_rows=n_reviews)
    business = load_clean_table(BUSINESS_PATH)
    users = load_clean_table(USER_PATH)

    print(
        f"Loaded reviews: {len(reviews):,} | business: {len(business):,} | users: {len(users):,}"
    )
    backend = sentiment_cfg.backend
    print(f"Sentiment backend: {backend}")
    if backend == "transformer":
        print(
            f"Transformer model: {sentiment_cfg.transformer_model} | "
            f"batch={sentiment_cfg.transformer_batch_size} | "
            f"device={sentiment_cfg.transformer_device}"
        )

    features = engineer_features(reviews, business, users, sentiment_cfg)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(OUTPUT_PATH, index=False)

    print(f"\nFeature set saved -> {OUTPUT_PATH.resolve()}")
    print("Total elapsed:", round(time.time() - start, 1), "s")
    return OUTPUT_PATH


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build review features with sentiment for usefulness modeling."
    )
    parser.add_argument(
        "--n-reviews",
        type=int,
        default=None,
        help="Optional cap on reviews loaded from the cleaned parquet.",
    )
    parser.add_argument(
        "--sentiment-backend",
        choices=["vader", "transformer"],
        default=DEFAULT_SENTIMENT_BACKEND,
        help="Sentiment engine to use (rule-based VADER or contextual Transformer).",
    )
    parser.add_argument(
        "--transformer-model",
        type=str,
        default=DEFAULT_TRANSFORMER_MODEL,
        help="Hugging Face model to use when --sentiment-backend transformer.",
    )
    parser.add_argument(
        "--transformer-batch-size",
        type=int,
        default=DEFAULT_TRANSFORMER_BATCH_SIZE,
        help="Batch size for transformer sentiment scoring.",
    )
    parser.add_argument(
        "--transformer-device",
        type=int,
        default=-1,
        help="Device for transformer inference (-1=CPU, >=0 GPU index).",
    )
    parser.add_argument(
        "--transformer-max-length",
        type=int,
        default=DEFAULT_TRANSFORMER_MAX_LENGTH,
        help="Max token length for transformer sentiment inputs.",
    )
    args = parser.parse_args()

    sentiment_cfg = SentimentConfig(
        backend=args.sentiment_backend,
        transformer_model=args.transformer_model,
        transformer_batch_size=args.transformer_batch_size,
        transformer_device=args.transformer_device,
        transformer_max_length=args.transformer_max_length,
    )

    try:
        run_feature_engineering(args.n_reviews, sentiment_cfg)
    except KeyboardInterrupt:
        raise SystemExit("Interrupted by user.")
