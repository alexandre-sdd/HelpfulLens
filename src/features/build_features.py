"""Create modeling features from cleaned Yelp reviews."""

from __future__ import annotations

from pathlib import Path

FEATURE_CACHE_DIR = Path("data/features")


def make_features(df):
    """Stub for combining text and numeric feature engineering."""
    # Implementations should generate text features (e.g., TF-IDF vectors),
    # review metadata (length, sentiment proxies), and user/business attributes.
    # Intermediate matrices can be cached inside data/features.
    _ = FEATURE_CACHE_DIR
    return df


if __name__ == "__main__":
    print("Build features (to be implemented)")
