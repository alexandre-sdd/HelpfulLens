"""Clean Yelp parquet caches and align schemas for downstream modeling."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import pandas as pd
import yaml

from src.utils.logging_utils import get_logger

CONFIG_PATH = Path("src/config/config.yaml")
DEFAULT_DATASETS = ("reviews", "users", "business")

LOGGER = get_logger("clean_yelp")


# ---------- Core cleaning functions (pure, no I/O) ----------
def clean_reviews_df(df: pd.DataFrame, min_text_length: int = 10) -> pd.DataFrame:
    """Clean raw reviews DataFrame and return a cleaned version."""
    expected_cols = [
        "review_id",
        "user_id",
        "business_id",
        "stars",
        "date",
        "text",
        "useful",
        "funny",
        "cool",
    ]
    available_cols = [c for c in expected_cols if c in df.columns]
    missing_cols = set(expected_cols) - set(available_cols)
    if missing_cols:
        LOGGER.warning("Missing columns in reviews_raw: %s", sorted(missing_cols))

    df = df[available_cols].copy()

    id_cols = [c for c in ["review_id", "user_id", "business_id"] if c in df.columns]
    for col in id_cols:
        df = df[df[col].notna()]

    if "text" in df.columns:
        df["text"] = df["text"].astype("string")
        df = df[df["text"].notna()]
        df = df[df["text"].str.len() >= min_text_length]

    for col in ["useful", "funny", "cool"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("int64")

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df[df["date"].notna()]
        df["review_year"] = df["date"].dt.year
        df["review_month"] = df["date"].dt.month
        df["review_day_of_week"] = df["date"].dt.dayofweek

    return df


def parse_elite_column(elite_series: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """Parse Yelp 'elite' column into raw string + boolean."""
    elite_str = elite_series.astype("string").fillna("None")
    is_elite_ever = elite_str.str.lower().ne("none")
    return elite_str, is_elite_ever


def clean_users_df(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw users DataFrame and return a cleaned version."""
    expected_cols = [
        "user_id",
        "name",
        "review_count",
        "yelping_since",
        "useful",
        "funny",
        "cool",
        "fans",
        "elite",
        "average_stars",
    ]
    available_cols = [c for c in expected_cols if c in df.columns]
    missing_cols = set(expected_cols) - set(available_cols)
    if missing_cols:
        LOGGER.warning("Missing columns in users_raw: %s", sorted(missing_cols))

    df = df[available_cols].copy()
    df = df[df["user_id"].notna()]

    if "yelping_since" in df.columns:
        df["yelping_since"] = pd.to_datetime(df["yelping_since"], errors="coerce")

    if "elite" in df.columns:
        elite_raw, is_elite_ever = parse_elite_column(df["elite"])
        df["elite_raw"] = elite_raw
        df["is_elite_ever"] = is_elite_ever
    else:
        df["elite_raw"] = pd.NA
        df["is_elite_ever"] = False

    numeric_cols = [
        "review_count",
        "useful",
        "funny",
        "cool",
        "fans",
        "average_stars",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def clean_business_df(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw business DataFrame and return a cleaned version."""
    expected_cols = [
        "business_id",
        "name",
        "city",
        "state",
        "stars",
        "review_count",
        "categories",
        "is_open",
        "latitude",
        "longitude",
        "attributes",
        "hours",
    ]
    available_cols = [c for c in expected_cols if c in df.columns]
    missing_cols = set(expected_cols) - set(available_cols)
    if missing_cols:
        LOGGER.warning("Missing columns in business_raw: %s", sorted(missing_cols))

    df = df[available_cols].copy()
    df = df[df["business_id"].notna()]

    for col in ["stars", "review_count", "latitude", "longitude"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "categories" in df.columns:
        df["categories"] = (
            df["categories"]
            .astype("string")
            .fillna("")
            .apply(lambda x: [c.strip() for c in x.split(",")] if x else [])
        )

    if "is_open" in df.columns:
        df["is_open"] = (
            pd.to_numeric(df["is_open"], errors="coerce").fillna(0).astype("int64")
        )

    return df


# ---------- I/O wrappers ----------
def load_config(config_path: Path = CONFIG_PATH) -> Dict:
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _load_raw_table(dataset: str, raw_root: Path) -> pd.DataFrame | None:
    """Load concatenated parquet shards for a dataset."""
    dataset_dir = raw_root / dataset
    legacy_path = raw_root / f"{dataset}_raw.parquet"
    frames: List[pd.DataFrame] = []

    if dataset_dir.exists():
        frames.extend(
            pd.read_parquet(path)
            for path in sorted(dataset_dir.glob("*.parquet"))
            if path.is_file()
        )
    if not frames and legacy_path.exists():
        frames.append(pd.read_parquet(legacy_path))

    if not frames:
        LOGGER.warning("No parquet files found for dataset '%s' in %s", dataset, raw_root)
        return None
    return pd.concat(frames, ignore_index=True)


def _write_clean_table(dataset: str, df: pd.DataFrame, cleaned_root: Path) -> Path:
    cleaned_root.mkdir(parents=True, exist_ok=True)
    output_path = cleaned_root / f"{dataset}_clean.parquet"
    df.to_parquet(output_path, index=False)
    return output_path


def _normalize_dataset(name: str) -> str:
    lowered = name.strip().lower()
    if lowered in {"review", "reviews"}:
        return "reviews"
    if lowered in {"user", "users"}:
        return "users"
    if lowered in {"business", "businesses"}:
        return "business"
    return lowered


CLEANERS: Dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = {
    "reviews": clean_reviews_df,
    "users": clean_users_df,
    "business": clean_business_df,
}


def clean_dataset(
    dataset: str,
    raw_root: Path,
    cleaned_root: Path,
    limit: int | None = None,
) -> Path | None:
    """Load, clean, and persist a dataset."""
    cleaner = CLEANERS.get(dataset)
    if not cleaner:
        LOGGER.warning("No cleaner defined for dataset '%s'", dataset)
        return None

    df_raw = _load_raw_table(dataset, raw_root)
    if df_raw is None or df_raw.empty:
        return None

    if limit:
        df_raw = df_raw.head(limit)
    LOGGER.info("Cleaning %s (%d rows)", dataset, df_raw.shape[0])
    df_clean = cleaner(df_raw)
    output_path = _write_clean_table(dataset, df_clean, cleaned_root)
    LOGGER.info("Saved %s rows -> %s", df_clean.shape[0], output_path)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean Yelp parquet caches.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DEFAULT_DATASETS),
        help="Datasets to clean (reviews, users, business).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_PATH,
        help="Path to the shared YAML config.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row cap per dataset for smoke tests.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    data_cfg = config.get("data", {})
    raw_cfg = data_cfg.get("raw", {})
    cleaned_cfg = data_cfg.get("cleaned", {})
    raw_root = Path(raw_cfg.get("parquet_dir", "data/raw/parquet"))
    cleaned_root = Path(cleaned_cfg.get("dir", "data/cleaned"))

    datasets = [_normalize_dataset(name) for name in args.datasets]
    for dataset in datasets:
        clean_dataset(dataset, raw_root, cleaned_root, limit=args.limit)


if __name__ == "__main__":
    main()
