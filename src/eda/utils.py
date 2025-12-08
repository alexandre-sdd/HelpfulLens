"""Shared helpers for EDA scripts."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm.auto import tqdm

THREAD_ENV_VARS = [
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
]


def savefig(fig_dir: Path, name: str, plt) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(fig_dir / name, dpi=200)
    plt.close()


def assert_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path.resolve()}")


def configure_thread_env(max_threads: int | None) -> None:
    """Optionally cap numerical backend threads."""
    if max_threads is None or max_threads <= 0:
        return
    value = str(max_threads)
    for var in THREAD_ENV_VARS:
        os.environ[var] = value


def apply_with_progress(
    series: pd.Series,
    func,
    batch_size: int = 10_000,
    desc: str | None = None,
) -> pd.Series:
    """Apply a Python function to a Series with progress + responsive interrupts."""
    total = len(series)
    if total == 0:
        return series.apply(func)

    pbar = tqdm(total=total, desc=desc, unit="rows")
    chunks: list[pd.Series] = []
    try:
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            chunk = series.iloc[start:end].apply(func)
            chunks.append(chunk)
            pbar.update(end - start)
    finally:
        pbar.close()
    return pd.concat(chunks)


def to_categories(x) -> list[str]:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return []
    if isinstance(x, list):
        return [str(c).strip() for c in x if str(c).strip()]
    return [c.strip() for c in str(x).split(",") if c.strip()]


def filter_jsonl(path: Path, id_field: str, ids: set[str], keep_cols: list[str]) -> pd.DataFrame:
    """Stream JSON Lines file, keep only rows where obj[id_field] in ids."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc=f"Filtering {path.name}"):
            obj = json.loads(line)
            if obj.get(id_field) in ids:
                rows.append({k: obj.get(k) for k in keep_cols})
    return pd.DataFrame(rows)


def read_parquet_head(
    path: Path,
    n_rows: int | None = None,
    columns: Iterable[str] | None = None,
    batch_size: int = 10_000,
    desc: str | None = None,
) -> pd.DataFrame:
    """Read up to n_rows rows from a parquet file with optional progress updates."""
    pq_file = pq.ParquetFile(path)
    chunks: list[pd.DataFrame] = []
    rows_read = 0
    total_rows = pq_file.metadata.num_rows if pq_file.metadata is not None else None

    if n_rows is not None and n_rows > 0:
        target_rows = n_rows if total_rows is None else min(n_rows, total_rows)
    else:
        target_rows = total_rows

    pbar = tqdm(total=target_rows, unit="rows", desc=desc) if desc else None

    for batch in pq_file.iter_batches(batch_size=batch_size, columns=columns):
        chunk = batch.to_pandas()
        chunks.append(chunk)
        rows_read += len(chunk)
        if pbar:
            pbar.update(len(chunk))
        if n_rows is not None and n_rows > 0 and rows_read >= n_rows:
            break

    if pbar:
        pbar.close()

    if not chunks:
        return pd.DataFrame(columns=columns or [])

    df = pd.concat(chunks, ignore_index=True)
    if n_rows is not None and n_rows > 0 and len(df) > n_rows:
        df = df.iloc[:n_rows].copy()
    return df
