#!/usr/bin/env python
import argparse
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect a Parquet file quickly.")
    # Positional path allows: python scripts/inspect_parquet.py data/file.parquet
    parser.add_argument("path", nargs="?", help="Path to the Parquet file.")
    # Backward-compatible flag: --path
    parser.add_argument("--path", dest="path_opt", help="Alternative way to pass the Parquet path.")
    parser.add_argument("--cols", nargs="*", default=[], help="Optional subset of columns to read.")
    parser.add_argument("-n", "--nrows", type=int, default=5, help="Number of rows to display.")
    parser.add_argument("--dtypes", action="store_true", help="Also print dtypes.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path_str = args.path or args.path_opt
    if not path_str:
        print(
            "Usage: inspect_parquet.py <path> [--cols COL1 COL2 ...] [-n 10] [--dtypes]",
            file=sys.stderr,
        )
        sys.exit(2)

    parquet_path = Path(path_str)
    if not parquet_path.exists():
        print(f"File not found: {parquet_path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_parquet(parquet_path, columns=args.cols or None)
    print(f"path: {parquet_path.resolve()}")
    print("shape:", df.shape)
    print("columns:", list(df.columns))
    if args.dtypes:
        print("dtypes:", {col: str(dtype) for col, dtype in df.dtypes.items()})
    if len(df) == 0:
        print("(empty dataframe)")
    else:
        print(df.head(args.nrows))

    # Compute ratio for non-zero target_useful_votes and save outputs if possible
    needed = {"target_is_useful", "target_useful_votes"}
    missing = list(needed - set(df.columns))
    if missing:
        try:
            # Attempt to read required columns from the file
            req = pd.read_parquet(parquet_path, columns=list(needed))
        except Exception:
            req = None
    else:
        req = df[list(needed)].copy()

    if req is not None and needed.issubset(req.columns):
        votes = req["target_useful_votes"]
        is_useful = req["target_is_useful"]
        ratio = pd.Series(pd.NA, index=req.index, dtype="float64")
        nonzero = votes != 0
        ratio.loc[nonzero] = is_useful.loc[nonzero] / votes.loc[nonzero]
        ratio_name = "useful_over_votes"

        # Attach to the full df (align on index if shapes match)
        try:
            df[ratio_name] = ratio
        except Exception:
            # Fallback: create a stand-alone frame for saving/plotting
            pass

        # Derive output paths
        stem = parquet_path.with_suffix("").name
        out_parquet = parquet_path.with_name(f"{stem}_with_ratio.parquet")
        out_png = parquet_path.with_name(f"{stem}_useful_over_votes_hist.png")

        # Save updated parquet (best-effort)
        try:
            df.to_parquet(out_parquet, index=False)
            print(f"saved updated parquet with '{ratio_name}': {out_parquet}")
        except Exception as e:
            print(f"warning: failed to save updated parquet: {e}", file=sys.stderr)

        # Save histogram
        try:
            plt.figure(figsize=(8, 5))
            sns.histplot((df.get(ratio_name) or ratio).dropna(), bins=50, kde=False)
            plt.xlabel(ratio_name)
            plt.ylabel("count")
            plt.title("Histogram of useful_over_votes (non-zero votes)")
            plt.tight_layout()
            plt.savefig(out_png, dpi=150)
            plt.close()
            print(f"saved histogram: {out_png}")
        except Exception as e:
            print(f"warning: failed to save histogram: {e}", file=sys.stderr)
    else:
        print("note: required columns not available to compute ratio (need target_is_useful and target_useful_votes)")


if __name__ == "__main__":
    main()
    
   