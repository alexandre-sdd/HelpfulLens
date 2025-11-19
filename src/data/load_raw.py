"""Load raw Yelp JSON data into memory for downstream processing."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

RAW_SUBDIRS = {
    "business": Path("data/raw/business"),
    "reviews": Path("data/raw/reviews"),
    "users": Path("data/raw/users"),
    "tips": Path("data/raw/tips"),
}


def load_raw_yelp(raw_dirs: Dict[str, Path] | None = None) -> Dict[str, List[Path]]:
    """Stub loader for Yelp JSON dumps to pandas DataFrames.

    The new folder layout separates each JSON dump into its own subdirectory.
    This helper simply enumerates the available JSON files so downstream code
    knows what to load.
    """

    directories = raw_dirs or RAW_SUBDIRS
    available_files: Dict[str, List[Path]] = {}
    for name, folder in directories.items():
        available_files[name] = sorted(folder.glob("*.json"))
    return available_files


if __name__ == "__main__":
    files = load_raw_yelp()
    print("Discovered raw Yelp shards:")
    for dataset, shards in files.items():
        print(f"  {dataset}: {len(shards)} file(s)")
