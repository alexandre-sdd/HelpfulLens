from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm


def normalize_categories(series: pd.Series) -> pd.Series:
    parents = {
        # keep "restaurants" as its own category (do not drop)
        "food",
        "bars",
        "nightlife",
        "event planning & services",
        "hotels",
        "hotels & travel",
        "shopping",
    }

    def clean(val) -> list[str]:
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return []
        raw = val if isinstance(val, list) else [str(val)]
        out = []
        seen = set()
        for token in raw:
            for part in str(token).split(","):
                t = part.strip()
                if not t:
                    continue
                t_lower = t.lower()
                if t_lower in parents:
                    continue
                if t_lower in seen:
                    continue
                seen.add(t_lower)
                out.append(t)
        return out

    return series.apply(clean)


def savefig(fig_dir: Path, name: str, plt_mod) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt_mod.tight_layout()
    plt_mod.savefig(fig_dir / name, dpi=200)
    plt_mod.close()


def load_parquet(path: Path, cols: Iterable[str] | None = None, n_rows: int | None = None) -> pd.DataFrame:
    df = pd.read_parquet(path, columns=cols)
    if n_rows is not None and n_rows > 0:
        df = df.head(n_rows).copy()
    return df


def main(
    review_path: Path,
    business_path: Path,
    user_path: Path,
    out_dir: Path,
    limit: int | None = None,
    top_categories: int = 20,
    top_cities: int = 15,
) -> None:
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    review_cols = [
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
    reviews = load_parquet(review_path, cols=review_cols, n_rows=limit)
    reviews["date"] = pd.to_datetime(reviews["date"], errors="coerce")
    for col in ["useful", "funny", "cool"]:
        reviews[col] = pd.to_numeric(reviews[col], errors="coerce").fillna(0).astype(int)
    reviews["helpful"] = reviews["useful"].clip(lower=0)
    reviews["log_helpful"] = np.log1p(reviews["helpful"])
    reviews["text_len_words"] = reviews["text"].astype(str).str.split().str.len()

    business = load_parquet(business_path, cols=["business_id", "city", "state", "categories"])
    business["categories"] = normalize_categories(business["categories"])
    users = load_parquet(user_path, cols=["user_id", "fans", "review_count", "average_stars"])

    df = reviews.merge(business, on="business_id", how="left").merge(
        users, on="user_id", how="left", suffixes=("", "_user")
    )
    df["month"] = df["date"].dt.month
    df["weekday"] = df["date"].dt.day_name()

    # Helpful distribution
    plt.figure()
    plt.hist(df["log_helpful"], bins=60)
    plt.xlabel("log1p(helpful)")
    plt.ylabel("Count")
    plt.title("Distribution of log(1+helpful)")
    savefig(fig_dir, "helpful_distribution.png", plt)

    # Helpful vs cool/funny
    sample = df.sample(min(len(df), 50_000), random_state=42)
    plt.figure()
    plt.scatter(sample["cool"], sample["helpful"], alpha=0.2)
    plt.xlabel("cool")
    plt.ylabel("helpful")
    plt.title("Helpful vs Cool (sample)")
    savefig(fig_dir, "helpful_vs_cool.png", plt)

    plt.figure()
    plt.scatter(sample["funny"], sample["helpful"], alpha=0.2)
    plt.xlabel("funny")
    plt.ylabel("helpful")
    plt.title("Helpful vs Funny (sample)")
    savefig(fig_dir, "helpful_vs_funny.png", plt)

    # Length deciles
    df["len_decile"] = pd.qcut(df["text_len_words"], q=10, duplicates="drop")
    len_stats = df.groupby("len_decile")["helpful"].mean()
    plt.figure(figsize=(8, 4))
    plt.plot(range(len(len_stats)), len_stats.values, marker="o")
    plt.xlabel("Length decile (short -> long)")
    plt.ylabel("Mean helpful")
    plt.title("Helpful by length decile")
    savefig(fig_dir, "helpful_by_length_decile.png", plt)

    # Category and city
    cat_exp = df.explode("categories")
    top_cats = cat_exp["categories"].value_counts().head(top_categories).index
    cat_stats = (
        cat_exp[cat_exp["categories"].isin(top_cats)]
        .groupby("categories")["helpful"]
        .agg(mean="mean", median="median", n="size")
        .sort_values("mean", ascending=False)
    )
    plt.figure(figsize=(10, 5))
    plt.bar(cat_stats.index, cat_stats["mean"])
    plt.xticks(rotation=60, ha="right")
    plt.ylabel("Mean helpful")
    plt.title("Mean helpful by category (top)")
    savefig(fig_dir, "helpful_by_category.png", plt)

    # Meta-category indicators (keep restaurants as a top-level flag)
    meta_map = {
        "meta_restaurants": {"restaurants"},
        "meta_food_coffee": {"food", "coffee & tea"},
        "meta_bars_nightlife": {"bars", "nightlife"},
        "meta_hotels_travel": {"hotels", "hotels & travel"},
        "meta_event_services": {"event planning & services"},
        "meta_beauty_spas": {"beauty & spas", "hair salons", "nail salons"},
        "meta_shopping": {"shopping"},
    }

    for col, keywords in meta_map.items():
        df[col] = df["categories"].apply(
            lambda cats: int(any(cat.lower() in keywords for cat in cats)) if isinstance(cats, list) else 0
        )

    meta_means = {col: df[df[col] == 1]["helpful"].mean() for col in meta_map}
    meta_counts = {col: df[col].sum() for col in meta_map}
    meta_order = sorted(meta_means.keys(), key=lambda c: meta_means[c], reverse=True)

    plt.figure(figsize=(8, 4))
    plt.bar([m.replace("meta_", "") for m in meta_order], [meta_means[m] for m in meta_order])
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Mean helpful")
    plt.title("Mean helpful by meta-category")
    savefig(fig_dir, "helpful_by_meta_category.png", plt)

    top_city = df["city"].value_counts().head(top_cities).index
    city_stats = (
        df[df["city"].isin(top_city)]
        .groupby("city")["helpful"]
        .agg(mean="mean", median="median", n="size")
        .sort_values("mean", ascending=False)
    )
    plt.figure(figsize=(10, 4))
    plt.bar(city_stats.index, city_stats["mean"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Mean helpful")
    plt.title("Mean helpful by city (top)")
    savefig(fig_dir, "helpful_by_city.png", plt)

    # Temporal
    month_mean = df.groupby("month")["helpful"].mean().reindex(range(1, 13))
    plt.figure()
    plt.plot(month_mean.index, month_mean.values, marker="o")
    plt.xlabel("Month")
    plt.ylabel("Mean helpful")
    plt.title("Mean helpful by month")
    savefig(fig_dir, "helpful_by_month.png", plt)

    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    wd_mean = df.groupby("weekday")["helpful"].mean().reindex(weekday_order)
    plt.figure(figsize=(8, 4))
    plt.bar(wd_mean.index, wd_mean.values)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Mean helpful")
    plt.title("Mean helpful by weekday")
    savefig(fig_dir, "helpful_by_weekday.png", plt)

    # User reputation (fans)
    bins = [-1, 0, 5, 20, 100, 1e9]
    labels = ["0", "1-5", "6-20", "21-100", "100+"]
    df["fans_bin"] = pd.cut(df["fans"].fillna(0), bins=bins, labels=labels)
    fans_stats = df.groupby("fans_bin")["helpful"].mean()
    plt.figure()
    plt.bar(fans_stats.index.astype(str), fans_stats.values)
    plt.xlabel("Fans bin")
    plt.ylabel("Mean helpful")
    plt.title("Mean helpful by user fans bin")
    savefig(fig_dir, "helpful_by_user_fans.png", plt)

    # Correlation snapshot
    corr_cols = [
        c
        for c in [
            "helpful",
            "stars",
            "cool",
            "funny",
            "text_len_words",
            "fans",
            "review_count_user",
            "average_stars_user",
        ]
        if c in df.columns
    ]
    corr = df[corr_cols].corr(method="spearman")
    plt.figure(figsize=(8, 6))
    plt.imshow(corr, interpolation="nearest")
    plt.xticks(range(len(corr_cols)), corr_cols, rotation=90)
    plt.yticks(range(len(corr_cols)), corr_cols)
    plt.colorbar()
    plt.title("Spearman correlation (selected numeric)")
    savefig(fig_dir, "correlation_heatmap.png", plt)

    # Markdown summary
    summary = out_dir / "eda_refined_summary.md"
    summary.write_text(
        "\n".join(
            [
                "# Yelp EDA (refined)",
                f"- Reviews loaded: **{len(df):,}**",
                f"- Businesses merged: **{df['business_id'].nunique():,}**",
                f"- Users merged: **{df['user_id'].nunique():,}**",
                "",
                "## Helpful distribution (log1p)",
                df["helpful"].describe().to_frame("helpful").to_markdown(),
                "",
                "## Helpful by category (top)",
                cat_stats.head(15).to_markdown(),
                "",
                "## Helpful by city (top)",
                city_stats.head(15).to_markdown(),
                "",
                "## Helpful by meta-category",
                pd.DataFrame(
                    {
                        "mean_helpful": [meta_means[m] for m in meta_order],
                        "count": [meta_counts[m] for m in meta_order],
                    },
                    index=[m.replace(\"meta_\", \"\") for m in meta_order],
                ).to_markdown(),
            ]
        ),
        encoding="utf-8",
    )
    print("EDA figures ->", fig_dir.resolve())
    print("Summary ->", summary.resolve())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Refined EDA focusing on helpful votes vs. popularity/exposure.")
    parser.add_argument("--review-path", type=Path, default=Path("data/cleaned/reviews_clean.parquet"))
    parser.add_argument("--business-path", type=Path, default=Path("data/cleaned/business_clean.parquet"))
    parser.add_argument("--user-path", type=Path, default=Path("data/cleaned/users_clean.parquet"))
    parser.add_argument("--out-dir", type=Path, default=Path("reports/eda_refined"))
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on reviews loaded.")
    parser.add_argument("--top-categories", type=int, default=20, help="Top business categories to show.")
    parser.add_argument("--top-cities", type=int, default=15, help="Top cities to show.")
    args = parser.parse_args()

    try:
        main(
            review_path=args.review_path,
            business_path=args.business_path,
            user_path=args.user_path,
            out_dir=args.out_dir,
            limit=args.limit,
            top_categories=args.top_categories,
            top_cities=args.top_cities,
        )
    except KeyboardInterrupt:
        raise SystemExit("Interrupted by user.")
