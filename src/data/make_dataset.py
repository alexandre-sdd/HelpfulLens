"""Assemble cleaned Yelp tables into train/eval modeling datasets."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from src.utils.logging_utils import get_logger

CONFIG_PATH = Path("src/config/config.yaml")
TRAIN_OUTPUT_NAME = "yelp_helpfulness_train.parquet"
EVAL_OUTPUT_NAME = "yelp_helpfulness_eval.parquet"

LOGGER = get_logger("make_dataset")


def load_config(config_path: Path = CONFIG_PATH) -> Dict:
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _load_cleaned_table(name: str, cleaned_dir: Path) -> pd.DataFrame | None:
    candidates = [
        cleaned_dir / f"{name}_clean.parquet",
        cleaned_dir / f"{name}.parquet",
    ]
    for path in candidates:
        if path.exists():
            return pd.read_parquet(path)
    LOGGER.warning("Cleaned table %s missing (looked for %s)", name, candidates)
    return None


def _engineer_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "review_text" in df.columns:
        df["review_char_len"] = df["review_text"].str.len()
        df["review_word_len"] = df["review_text"].str.split().str.len()
    if "review_date" in df.columns:
        df["review_is_weekend"] = df["review_date"].dt.dayofweek >= 5
        df["review_weekofyear"] = df["review_date"].dt.isocalendar().week.astype(int)
    target_src = None
    for candidate in ("review_useful", "useful"):
        if candidate in df.columns:
            target_src = candidate
            break
    if target_src:
        df["target_useful_votes"] = df[target_src].fillna(0).astype(int)
        df["target_is_useful"] = (df["target_useful_votes"] > 0).astype(int)
    else:
        df["target_useful_votes"] = 0
        df["target_is_useful"] = 0
    return df


def _merge_side_tables(
    reviews: pd.DataFrame, business: pd.DataFrame | None, users: pd.DataFrame | None
) -> pd.DataFrame:
    df = reviews.copy()
    if business is not None and "business_id" in business.columns:
        business_cols = [
            "business_id",
            "business_stars",
            "business_review_count",
            "city",
            "state",
            "categories",
            "is_open",
        ]
        available_cols = [col for col in business_cols if col in business.columns]
        df = df.merge(
            business[available_cols],
            on="business_id",
            how="left",
        )
    if users is not None and "user_id" in users.columns:
        user_cols = [
            "user_id",
            "user_review_count",
            "user_average_stars",
            "fans",
            "useful",
            "funny",
            "cool",
        ]
        available_cols = [col for col in user_cols if col in users.columns]
        df = df.merge(
            users[available_cols],
            on="user_id",
            how="left",
            suffixes=("", "_user"),
        )
    return df


def _split_train_eval(
    df: pd.DataFrame, validation_split: float, random_state: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty or len(df) < 2:
        return df, df.iloc[0:0]
    stratify = None
    if "target_is_useful" in df.columns:
        if df["target_is_useful"].nunique() > 1:
            stratify = df["target_is_useful"]
    train_df, eval_df = train_test_split(
        df,
        test_size=validation_split,
        random_state=random_state,
        shuffle=True,
        stratify=stratify,
    )
    return train_df, eval_df


def build_master_table(config: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    data_cfg = config.get("data", {})
    cleaned_cfg = data_cfg.get("cleaned", {})
    datasets_cfg = data_cfg.get("datasets", {})
    cleaned_dir = Path(cleaned_cfg.get("dir", "data/cleaned"))
    training_dir = Path(datasets_cfg.get("training_dir", "data/datasets/training"))
    eval_dir = Path(datasets_cfg.get("evaluation_dir", "data/datasets/evaluation"))
    training_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    reviews = _load_cleaned_table("reviews", cleaned_dir)
    if reviews is None:
        raise FileNotFoundError(
            f"No cleaned reviews parquet found in {cleaned_dir}. Run clean_yelp first."
        )
    business = _load_cleaned_table("business", cleaned_dir)
    users = _load_cleaned_table("users", cleaned_dir)

    LOGGER.info("Loaded %d review rows", len(reviews))

    assembled = _merge_side_tables(reviews, business, users)
    engineered = _engineer_basic_features(assembled)

    training_cfg = config.get("training", {})
    validation_split = training_cfg.get("validation_split", 0.2)
    random_state = training_cfg.get("random_state", 42)
    train_df, eval_df = _split_train_eval(engineered, validation_split, random_state)

    train_path = training_dir / TRAIN_OUTPUT_NAME
    eval_path = eval_dir / EVAL_OUTPUT_NAME
    train_df.to_parquet(train_path, index=False)
    eval_df.to_parquet(eval_path, index=False)
    LOGGER.info("Saved %d training rows -> %s", len(train_df), train_path)
    LOGGER.info("Saved %d evaluation rows -> %s", len(eval_df), eval_path)

    return train_df, eval_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
    build_master_table(config)


if __name__ == "__main__":
    main()
