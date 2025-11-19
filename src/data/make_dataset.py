"""Transform raw Yelp data into analysis-ready tables."""

from __future__ import annotations

from pathlib import Path

INTERIM_OUTPUT_DIR = Path("data/interim/cleaned")
TRAINING_OUTPUT_DIR = Path("data/processed/training")
EVAL_OUTPUT_DIR = Path("data/processed/evaluation")


def build_master_table() -> None:
    """Stub for building the modeling dataset."""
    # Expected stages:
    # 1. Merge review, business, user, and optional tip tables (see data/raw/*).
    # 2. Persist intermediary merged tables to data/interim/cleaned for auditing.
    # 3. Save the final train/eval splits into data/processed/* directories.
    _ = (
        INTERIM_OUTPUT_DIR,
        TRAINING_OUTPUT_DIR,
        EVAL_OUTPUT_DIR,
    )
    return None


if __name__ == "__main__":
    print("Build cleaned dataset (to be implemented)")
