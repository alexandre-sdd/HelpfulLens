"""Schema validation helpers for feature assembly and modeling."""

from __future__ import annotations

from typing import Iterable, Sequence

import pandas as pd


def assert_no_duplicate_columns(df: pd.DataFrame, frame_name: str) -> None:
    """Raise if a DataFrame contains duplicate column labels."""
    duplicated = df.columns[df.columns.duplicated()].tolist()
    if duplicated:
        raise ValueError(f"{frame_name} has duplicate columns: {duplicated}")


def assert_no_leakage(feature_cols: Iterable[str], target_cols: Iterable[str]) -> None:
    """Ensure feature columns do not include any target columns."""
    feature_set = set(feature_cols)
    overlap = feature_set.intersection(set(target_cols))
    if overlap:
        raise ValueError(f"Feature set contains target columns: {sorted(overlap)}")


def assert_aligned_columns(left: pd.DataFrame, right: pd.DataFrame) -> None:
    """Ensure two DataFrames share identical columns in identical order."""
    if list(left.columns) != list(right.columns):
        left_only = [c for c in left.columns if c not in right.columns]
        right_only = [c for c in right.columns if c not in left.columns]
        raise ValueError(
            "Feature matrices are not aligned.\n"
            f"Only in left: {left_only}\n"
            f"Only in right: {right_only}"
        )


def coerce_boolean_to_int(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Return a copy with boolean columns converted to integers."""
    updated = df.copy()
    for col in columns:
        if col in updated.columns and pd.api.types.is_bool_dtype(updated[col]):
            updated[col] = updated[col].astype("int8")
    return updated
