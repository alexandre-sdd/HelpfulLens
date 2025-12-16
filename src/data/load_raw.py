"""Load raw Yelp JSON data into memory for downstream processing."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

RAW_ROOT = Path("data/raw")
FALLBACK_PATTERNS = {
    "business": "yelp_academic_dataset_business*.json*",
    "reviews": "yelp_academic_dataset_review*.json*",
    "users": "yelp_academic_dataset_user*.json*",
    "tips": "yelp_academic_dataset_tip*.json*",
}


def _collect_dataset_files(dataset: str, root: Path = RAW_ROOT) -> List[Path]:
    """Return JSON shards for a dataset, supporting both folders and flat files.

    Args:
        dataset: Dataset name (business, reviews, users, tips).
        root: Root directory containing raw files.

    Returns:
        Sorted list of candidate JSON paths (may be empty).
    """
    dataset_dir = root / dataset
    candidates: List[Path] = []
    if dataset_dir.exists() and dataset_dir.is_dir():
        candidates.extend(sorted(dataset_dir.glob("*.json*")))
    if not candidates:
        pattern = FALLBACK_PATTERNS.get(dataset)
        if pattern:
            candidates.extend(sorted(root.glob(pattern)))
    return candidates


def load_raw_yelp() -> Dict[str, List[Path]]:
    """Enumerate available Yelp JSON shards.

    Returns:
        Mapping of dataset name to discovered JSON file paths.
    """

    datasets = ("business", "reviews", "users", "tips")
    available_files: Dict[str, List[Path]] = {}
    for name in datasets:
        available_files[name] = _collect_dataset_files(name)
    return available_files


if __name__ == "__main__":
    files = load_raw_yelp()
    print("Discovered raw Yelp shards:")
    for dataset, shards in files.items():
        print(f"  {dataset}: {len(shards)} file(s)")
