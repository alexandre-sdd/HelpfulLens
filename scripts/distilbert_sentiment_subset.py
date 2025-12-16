"""Score sentiment on a compact subset of reviews with DistilBERT."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.features.sentiment_subset import run_distilbert_subset

CLEAN_DIR = Path("data/cleaned")
FEATURE_DIR = Path("data/features")
DEFAULT_INPUT = CLEAN_DIR / "reviews_clean.parquet"
DEFAULT_OUTPUT = FEATURE_DIR / "distilbert_sentiment_200k.parquet"
DEFAULT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run DistilBERT SST-2 sentiment on up to 200k reviews that have "
            "non-null useful votes."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Clean review parquet path.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Destination parquet for the sentiment-enriched subset.",
    )
    parser.add_argument(
        "--max-reviews",
        type=int,
        default=200_000,
        help="Maximum number of reviews to score.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Transformer batch size.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
        help="Maximum token length for sentiment inputs.",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=-1,
        help="Device index for transformers pipeline (-1 for CPU).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Hugging Face identifier for the sentiment model.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random state used when sampling down to the requested limit.",
    )
    parser.add_argument(
        "--require-positive-useful",
        action="store_true",
        help="Keep only reviews with strictly positive useful votes.",
    )
    parser.add_argument(
        "--omit-text",
        action="store_true",
        help="Drop review text from the saved output to reduce file size.",
    )
    parser.add_argument(
        "--resume-path",
        type=Path,
        default=None,
        help="Optional JSON file to store a sentiment resume/summary.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_distilbert_subset(
        input_path=args.input,
        output_path=args.output,
        max_reviews=args.max_reviews,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=args.device,
        model_name=args.model,
        random_seed=args.random_seed,
        require_positive_useful=args.require_positive_useful,
        omit_text=args.omit_text,
        resume_path=args.resume_path,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit("Interrupted by user.")
