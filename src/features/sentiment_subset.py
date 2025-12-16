"""Helpers for scoring sentiment subsets with transformer pipelines."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

try:
    from transformers import pipeline as hf_pipeline
except ImportError as exc:  # pragma: no cover - runtime guard
    raise SystemExit(
        "transformers is required for this script. Install it via `pip install transformers torch`."
    ) from exc

try:
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover - optional optimization
    pq = None

from src.features.sentiment_features import sanitize_label

STREAM_BATCH_SIZE = 50_000


def read_review_subset(
    path: Path,
    limit: int,
    require_positive_useful: bool,
    random_state: int | None,
) -> pd.DataFrame:
    """Load a filtered subset of reviews for sentiment scoring.

    Args:
        path: Parquet path containing cleaned reviews.
        limit: Maximum number of rows to return.
        require_positive_useful: If True, keep only rows with useful > 0.
        random_state: Optional seed used when sampling down to ``limit``.

    Returns:
        DataFrame with review identifiers, text, and useful votes.

    Raises:
        FileNotFoundError: If the parquet file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path.resolve()}")

    columns = ["review_id", "user_id", "business_id", "text", "useful"]
    limit = max(limit, 1)

    def _filter(df: pd.DataFrame) -> pd.DataFrame:
        df = df[df["useful"].notna()]
        if require_positive_useful:
            df = df[df["useful"] > 0]
        return df

    if pq is None:
        df = pd.read_parquet(path, columns=columns)
        df = _filter(df)
    else:
        collected: list[pd.DataFrame] = []
        total = 0
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(columns=columns, batch_size=STREAM_BATCH_SIZE):
            pdf = batch.to_pandas()
            pdf = _filter(pdf)
            if pdf.empty:
                continue
            collected.append(pdf)
            total += len(pdf)
            if total >= limit:
                break
        if not collected:
            return pd.DataFrame(columns=columns)
        df = pd.concat(collected, ignore_index=True)

    if df.empty:
        return df

    if len(df) > limit:
        if random_state is None:
            df = df.head(limit)
        else:
            df = df.sample(n=limit, random_state=random_state)
    return df.reset_index(drop=True)


def prepare_text_column(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize the review text column to non-null strings.

    Args:
        df: Review DataFrame containing a ``text`` column.

    Returns:
        Copy with the ``text`` column filled and cast to string.
    """
    df = df.copy()
    df["text"] = df["text"].fillna("").astype(str)
    return df


def build_resume(df: pd.DataFrame) -> dict[str, object]:
    """Summarize sentiment outputs for logging or persistence.

    Args:
        df: DataFrame containing scored sentiment columns.

    Returns:
        Dictionary with row counts, label counts, and summary stats.
    """
    resume: dict[str, object] = {"total_reviews": int(len(df))}

    if "sentiment_label" in df.columns:
        counts = df["sentiment_label"].value_counts(dropna=False).to_dict()
        resume["label_counts"] = {str(k): int(v) for k, v in counts.items()}
    else:
        resume["label_counts"] = {}

    if "sentiment_score" in df.columns and df["sentiment_score"].notna().any():
        score_series = df["sentiment_score"].dropna()
        resume.update(
            {
                "sentiment_score_mean": float(score_series.mean()),
                "sentiment_score_median": float(score_series.median()),
                "sentiment_score_min": float(score_series.min()),
                "sentiment_score_max": float(score_series.max()),
            }
        )

    if "useful" in df.columns and df["useful"].notna().any():
        resume["useful_votes_mean"] = float(df["useful"].dropna().mean())

    return resume


def print_resume(resume: dict[str, object]) -> None:
    """Print a formatted sentiment summary.

    Args:
        resume: Summary dictionary produced by :func:`build_resume`.
    """
    print("\nSentiment resume:")
    print(f"Total reviews: {resume.get('total_reviews', 0):,}")
    label_counts = resume.get("label_counts", {})
    if label_counts:
        print("Label counts:")
        for label, count in label_counts.items():
            print(f"  - {label}: {count:,}")
    if "sentiment_score_mean" in resume:
        print(
            "Score stats:"
            f" mean={resume['sentiment_score_mean']:.4f}"
            f" | median={resume['sentiment_score_median']:.4f}"
            f" | min={resume['sentiment_score_min']:.4f}"
            f" | max={resume['sentiment_score_max']:.4f}"
        )
    if "useful_votes_mean" in resume:
        print(f"Avg useful votes: {resume['useful_votes_mean']:.2f}")


def score_sentiment(
    reviews: pd.DataFrame,
    model_name: str,
    batch_size: int,
    max_length: int,
    device: int,
) -> pd.DataFrame:
    """Score sentiment labels using a transformer pipeline.

    Args:
        reviews: Review subset with ``review_id`` and ``text`` columns.
        model_name: Hugging Face model identifier.
        batch_size: Number of rows scored per batch.
        max_length: Maximum token length during truncation.
        device: Device index passed to the transformers pipeline (-1 for CPU).

    Returns:
        DataFrame containing sentiment scores keyed by ``review_id``.
    """
    classifier = hf_pipeline(
        task="text-classification",
        model=model_name,
        tokenizer=model_name,
        device=device,
    )

    records: list[dict[str, object]] = []
    texts = reviews["text"].tolist()
    review_ids = reviews["review_id"].tolist()
    pbar = tqdm(total=len(texts), desc="Scoring sentiment", unit="reviews")

    try:
        for start in range(0, len(texts), batch_size):
            end = min(start + batch_size, len(texts))
            batch_texts = texts[start:end]
            batch_ids = review_ids[start:end]
            results = classifier(
                batch_texts,
                truncation=True,
                max_length=max_length,
                return_all_scores=True,
                batch_size=batch_size,
            )
            for rid, scores in zip(batch_ids, results):
                record = {"review_id": rid}
                if not scores:
                    records.append(record)
                    continue
                best = max(scores, key=lambda s: s.get("score", 0.0))
                record["sentiment_label"] = best.get("label")
                record["sentiment_score"] = float(best.get("score", 0.0))
                for entry in scores:
                    label = entry.get("label", "")
                    key = f"sentiment_{sanitize_label(label)}"
                    record[key] = float(entry.get("score", 0.0))
                records.append(record)
            pbar.update(end - start)
    finally:
        pbar.close()

    return pd.DataFrame(records)


def run_distilbert_subset(
    input_path: Path,
    output_path: Path,
    max_reviews: int,
    batch_size: int,
    max_length: int,
    device: int,
    model_name: str,
    random_seed: int | None,
    require_positive_useful: bool,
    omit_text: bool,
    resume_path: Path | None = None,
) -> dict[str, object]:
    """Run the subset sentiment workflow end to end.

    Args:
        input_path: Parquet source for cleaned reviews.
        output_path: Destination for the scored subset parquet.
        max_reviews: Maximum number of reviews to score.
        batch_size: Batch size for transformer inference.
        max_length: Maximum token length for transformer inputs.
        device: Device index passed to the transformers pipeline (-1 for CPU).
        model_name: Hugging Face model identifier.
        random_seed: Optional seed for sampling and reproducibility.
        require_positive_useful: If True, keep only reviews with useful > 0.
        omit_text: Whether to drop the ``text`` column before saving.
        resume_path: Optional JSON path for writing summary statistics.

    Returns:
        Summary dictionary produced by :func:`build_resume`.

    Raises:
        SystemExit: If no reviews satisfy the selection criteria.
    """
    t0 = time.time()
    reviews = read_review_subset(
        path=input_path,
        limit=max_reviews,
        require_positive_useful=require_positive_useful,
        random_state=random_seed,
    )

    if reviews.empty:
        raise SystemExit("No reviews matched the selection criteria.")

    print(f"Loaded {len(reviews):,} reviews for sentiment scoring.")
    print(f"Model: {model_name} | device={device} | batch={batch_size}")

    reviews = prepare_text_column(reviews)
    sentiments = score_sentiment(
        reviews=reviews,
        model_name=model_name,
        batch_size=max(batch_size, 1),
        max_length=max(max_length, 32),
        device=device,
    )

    merged = reviews.merge(sentiments, on="review_id", how="left", validate="one_to_one")
    if omit_text and "text" in merged.columns:
        merged = merged.drop(columns=["text"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(output_path, index=False)

    resume = build_resume(merged)
    print_resume(resume)
    if resume_path is not None:
        resume_path.parent.mkdir(parents=True, exist_ok=True)
        resume_path.write_text(json.dumps(resume, indent=2), encoding="utf-8")
        print(f"Resume saved -> {resume_path.resolve()}")

    elapsed = time.time() - t0
    print(f"Saved sentiment subset -> {output_path.resolve()}")
    print(f"Total elapsed: {elapsed:.1f}s for {len(merged):,} reviews.")
    return resume
