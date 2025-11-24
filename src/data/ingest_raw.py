"""Convert Yelp JSON dumps into parquet caches for faster iteration."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import pandas as pd
import yaml

from src.utils.logging_utils import get_logger

CONFIG_PATH = Path("src/config/config.yaml")
DEFAULT_DATASETS = ("business", "reviews", "users", "tips")
SUPPORTED_EXTENSIONS = (".json", ".json.gz")
FALLBACK_PATTERNS = {
    "business": "yelp_academic_dataset_business*.json*",
    "reviews": "yelp_academic_dataset_review*.json*",
    "users": "yelp_academic_dataset_user*.json*",
    "tips": "yelp_academic_dataset_tip*.json*",
}

LOGGER = get_logger("ingest_raw")


def load_config(config_path: Path = CONFIG_PATH) -> Dict[str, Dict[str, str]]:
    """Return the parsed YAML configuration."""
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _normalize_dataset_name(name: str) -> str:
    lowered = name.strip().lower()
    # accept either singular or plural names
    if lowered in {"review", "reviews"}:
        return "reviews"
    if lowered in {"business", "businesses"}:
        return "business"
    if lowered in {"user", "users"}:
        return "users"
    if lowered in {"tip", "tips"}:
        return "tips"
    return lowered


def _json_stem(json_path: Path) -> str:
    """Derive the parquet filename from a JSON shard."""
    name = json_path.name
    if name.endswith(".json.gz"):
        name = name[: -len(".json.gz")]
    elif name.endswith(".json"):
        name = name[: -len(".json")]
    return name


def _parquet_name(json_path: Path) -> str:
    return f"{_json_stem(json_path)}.parquet"


def _is_supported_file(path: Path) -> bool:
    if path.suffix in SUPPORTED_EXTENSIONS:
        return True
    return path.name.endswith(".json.gz")


def _collect_files_from_folder(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    if folder.is_file():
        return [folder] if _is_supported_file(folder) else []
    return sorted([path for path in folder.iterdir() if _is_supported_file(path)])


def _fallback_files(dataset: str, root_dir: Path) -> List[Path]:
    pattern = FALLBACK_PATTERNS.get(dataset)
    if not pattern:
        return []
    if not root_dir.exists():
        return []
    return sorted([path for path in root_dir.glob(pattern) if _is_supported_file(path)])


def _read_json(json_path: Path, chunk_size: int | None = None) -> pd.DataFrame:
    """Read a single Yelp JSON (or JSON.GZ) file."""
    if chunk_size and chunk_size > 0:
        chunks: List[pd.DataFrame] = []
        for chunk in pd.read_json(json_path, lines=True, chunksize=chunk_size):
            chunks.append(chunk)
        if not chunks:
            return pd.DataFrame()
        return pd.concat(chunks, ignore_index=True)
    return pd.read_json(json_path, lines=True)


def convert_json_to_parquet(
    dataset: str,
    files: Iterable[Path],
    output_root: Path,
    chunk_size: int | None = None,
    force: bool = False,
    limit: int | None = None,
    rows_per_chunk: int | None = None,
) -> List[Path]:
    """Convert JSON shards for a dataset into parquet files."""
    written: List[Path] = []
    target_dir = output_root / dataset
    target_dir.mkdir(parents=True, exist_ok=True)

    for idx, json_file in enumerate(sorted(files)):
        if limit is not None and idx >= limit:
            LOGGER.info(
                "Reached file limit (%s) for dataset '%s'; skipping remaining shards.",
                limit,
                dataset,
            )
            break

        if not _is_supported_file(json_file):
            LOGGER.debug("Skipping non-JSON file %s", json_file.name)
            continue

        base_name = _json_stem(json_file)
        if rows_per_chunk and rows_per_chunk > 0:
            existing_parts = list(target_dir.glob(f"{base_name}_part*.parquet"))
            if existing_parts and not force:
                LOGGER.info(
                    "Skipping %s (chunked parquet already exists: %d files)",
                    json_file.name,
                    len(existing_parts),
                )
                continue
            LOGGER.info(
                "Reading %s in chunks of %d rows", json_file, rows_per_chunk
            )
            chunk_iter = pd.read_json(
                json_file, lines=True, chunksize=rows_per_chunk
            )
            chunk_written = False
            for chunk_idx, chunk in enumerate(chunk_iter, start=1):
                chunk_path = target_dir / f"{base_name}_part{chunk_idx:04d}.parquet"
                LOGGER.info(
                    "Writing %s (%d rows)", chunk_path, chunk.shape[0]
                )
                chunk.to_parquet(chunk_path, index=False)
                written.append(chunk_path)
                chunk_written = True
            if not chunk_written:
                LOGGER.warning("No data written for %s", json_file)
            continue

        parquet_path = target_dir / _parquet_name(json_file)
        if parquet_path.exists() and not force:
            LOGGER.info("Skipping %s (parquet already exists)", json_file.name)
            continue

        LOGGER.info("Reading %s", json_file)
        df = _read_json(json_file, chunk_size=chunk_size)
        LOGGER.info("Writing %s (%d rows)", parquet_path, df.shape[0])
        df.to_parquet(parquet_path, index=False)
        written.append(parquet_path)
    return written


def discover_raw_files(
    config: Dict[str, Dict[str, str]],
    datasets: Sequence[str],
) -> Dict[str, List[Path]]:
    """Return JSON shards detected for each requested dataset."""
    data_cfg = config.get("data", {})
    raw_cfg = data_cfg.get("raw", {})
    root_dir = Path(raw_cfg.get("root_dir", "data/raw"))
    discovered: Dict[str, List[Path]] = {}
    for name in datasets:
        normalized = _normalize_dataset_name(name)
        dir_key = f"{normalized}_dir"
        raw_dir_entry = raw_cfg.get(dir_key)
        folder = Path(raw_dir_entry) if raw_dir_entry else root_dir / normalized
        files: List[Path] = _collect_files_from_folder(folder)
        if not files:
            fallback = _fallback_files(normalized, root_dir)
            if fallback:
                LOGGER.info(
                    "Using fallback pattern for dataset '%s' (found %d files in %s)",
                    normalized,
                    len(fallback),
                    root_dir,
                )
                files = fallback
            else:
                LOGGER.warning(
                    "Raw directory %s does not exist or contains no JSON files",
                    folder,
                )
        discovered[normalized] = files
    return discovered


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DEFAULT_DATASETS),
        help="Datasets to ingest (business, reviews, users, tips).",
    )
    parser.add_argument(
        "--chunk-size",
        dest="chunk_size",
        type=int,
        default=None,
        help="Optional pandas.read_json chunk size for huge files.",
    )
    parser.add_argument(
        "--rows-per-chunk",
        type=int,
        default=None,
        help="If set, write multiple parquet files per JSON (one per chunk of this many rows).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite parquet files even if they already exist.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N shards per dataset (useful for smoke tests).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_PATH,
        help="Path to the project config file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    data_cfg = config.get("data", {})
    raw_cfg = data_cfg.get("raw", {})
    raw_parquet_root = Path(raw_cfg.get("parquet_dir", "data/raw/parquet"))
    raw_parquet_root.mkdir(parents=True, exist_ok=True)

    datasets = [_normalize_dataset_name(name) for name in args.datasets]
    discovered = discover_raw_files(config, datasets)
    summary = []
    for dataset, files in discovered.items():
        if not files:
            LOGGER.warning("No raw files found for dataset '%s'; skipping.", dataset)
            continue
        written = convert_json_to_parquet(
            dataset,
            files,
            raw_parquet_root,
            chunk_size=args.chunk_size,
            force=args.force,
            limit=args.limit,
            rows_per_chunk=args.rows_per_chunk,
        )
        summary.append((dataset, len(written)))

    if summary:
        LOGGER.info("Ingestion complete:")
        for dataset, count in summary:
            LOGGER.info("  %s -> %d parquet file(s)", dataset, count)
    else:
        LOGGER.info("No datasets were ingested. Ensure raw JSON files exist.")


if __name__ == "__main__":
    main()
