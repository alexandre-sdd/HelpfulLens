from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyarrow.parquet as pq
from tqdm.auto import tqdm

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# =========================
# Config
# =========================
DEFAULT_N_REVIEWS = 50_000  # start here; increase to 100k/200k if it runs fine
DO_USER_MERGE = True        # set False if you want it faster
DO_TFIDF_WORDS = True       # set False if you want it faster
TFIDF_SAMPLE = 30_000       # TF-IDF runs on a subset to stay fast
WEEKDAY_NAMES = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
THREAD_ENV_VARS = [
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
]

CLEAN_DIR = Path("data/cleaned")
REVIEW_PATH = CLEAN_DIR / "reviews_clean.parquet"
BUSINESS_PATH = CLEAN_DIR / "business_clean.parquet"
USER_PATH = CLEAN_DIR / "users_clean.parquet"

OUT_DIR = Path("reports/eda")
FIG_DIR = OUT_DIR / "figures"


# =========================
# Helpers
# =========================
def savefig(name: str) -> None:
    plt.tight_layout()
    plt.savefig(FIG_DIR / name, dpi=200)
    plt.close()

def assert_exists(p: Path) -> None:
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {p.resolve()}")

def configure_thread_env(max_threads: int | None) -> None:
    """Optionally cap numerical backend threads."""
    if max_threads is None or max_threads <= 0:
        return
    value = str(max_threads)
    for var in THREAD_ENV_VARS:
        os.environ[var] = value

def read_parquet_head(
    path: Path,
    n_rows: int | None = None,
    columns: list[str] | None = None,
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


def compute_text_basic_features(
    series: pd.Series,
    chunk_size: int = 50_000,
) -> pd.DataFrame:
    """Compute text length + punctuation counts in manageable chunks."""
    total = len(series)
    if total == 0:
        return pd.DataFrame(
            {
                "text_len_chars": [],
                "text_len_words": [],
                "exclaim_count": [],
                "question_count": [],
            },
            index=series.index,
        )

    char_arr = np.zeros(total, dtype=np.int32)
    word_arr = np.zeros(total, dtype=np.int32)
    exclaim_arr = np.zeros(total, dtype=np.int32)
    question_arr = np.zeros(total, dtype=np.int32)

    pbar = tqdm(total=total, desc="Text length/count features", unit="rows")
    try:
        pos = 0
        for start in range(0, total, chunk_size):
            end = min(start + chunk_size, total)
            chunk = series.iloc[start:end]

            chunk_chars = chunk.str.len().fillna(0).astype(np.int32)
            chunk_words = chunk.str.split().str.len().fillna(0).astype(np.int32)
            chunk_exclaim = chunk.str.count("!").fillna(0).astype(np.int32)
            chunk_question = chunk.str.count(r"\?").fillna(0).astype(np.int32)

            char_arr[pos : pos + len(chunk)] = chunk_chars.to_numpy()
            word_arr[pos : pos + len(chunk)] = chunk_words.to_numpy()
            exclaim_arr[pos : pos + len(chunk)] = chunk_exclaim.to_numpy()
            question_arr[pos : pos + len(chunk)] = chunk_question.to_numpy()

            pos += len(chunk)
            pbar.update(len(chunk))
    finally:
        pbar.close()

    return pd.DataFrame(
        {
            "text_len_chars": char_arr,
            "text_len_words": word_arr,
            "exclaim_count": exclaim_arr,
            "question_count": question_arr,
        },
        index=series.index,
    )


def main(
    n_reviews: int | None = DEFAULT_N_REVIEWS,
    enable_tfidf: bool = DO_TFIDF_WORDS,
    max_threads: int | None = None,
    tfidf_jobs: int | None = None,
) -> None:
    print("EDA started...")
    print("Working directory:", Path.cwd())

    # output dirs
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    configure_thread_env(max_threads)

    # check files
    for p in [REVIEW_PATH, BUSINESS_PATH]:
        assert_exists(p)
    if DO_USER_MERGE:
        assert_exists(USER_PATH)

    t0 = time.time()

    # -------------------------
    # Load reviews (sample/full)
    # -------------------------
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
        "review_year",
        "review_month",
        "review_day_of_week",
    ]
    if n_reviews is None or n_reviews <= 0:
        review_rows = None
        review_desc = "all available"
    else:
        review_rows = n_reviews
        review_desc = f"{review_rows:,}"
    print(f"\nLoading {review_desc} reviews from {REVIEW_PATH.name} ...")
    reviews = read_parquet_head(
        REVIEW_PATH,
        n_rows=review_rows,
        columns=review_cols,
        desc=f"Reviews ({review_desc})",
    )

    # Basic cleaning (mostly already handled upstream in clean_yelp)
    if "date" in reviews.columns and not pd.api.types.is_datetime64_any_dtype(reviews["date"]):
        reviews["date"] = pd.to_datetime(reviews["date"], errors="coerce")
    for c in ["useful", "funny", "cool"]:
        if c in reviews.columns and not pd.api.types.is_integer_dtype(reviews[c]):
            reviews[c] = pd.to_numeric(reviews[c], errors="coerce").fillna(0).astype("int64")
    if "stars" in reviews.columns and not pd.api.types.is_numeric_dtype(reviews["stars"]):
        reviews["stars"] = pd.to_numeric(reviews["stars"], errors="coerce")

    # Target
    reviews["helpful"] = reviews["useful"].clip(lower=0)
    reviews["helpful_log1p"] = np.log1p(reviews["helpful"].astype(float))

    # Seasonality (prefer cleaned columns if available)
    if "review_month" in reviews.columns:
        reviews["month"] = reviews["review_month"]
    elif "date" in reviews.columns:
        reviews["month"] = reviews["date"].dt.month
    else:
        reviews["month"] = np.nan

    if "review_day_of_week" in reviews.columns:
        reviews["weekday"] = reviews["review_day_of_week"].map(
            lambda idx: WEEKDAY_NAMES[int(idx)] if pd.notna(idx) and 0 <= int(idx) < len(WEEKDAY_NAMES) else None
        )
    elif "date" in reviews.columns:
        reviews["weekday"] = reviews["date"].dt.day_name()
    else:
        reviews["weekday"] = None

    # Text features (NO emoji)
    txt = reviews["text"].astype(str)
    text_feature_df = compute_text_basic_features(txt, chunk_size=50_000)
    reviews = reviews.join(text_feature_df)

    def caps_ratio(s: str) -> float:
        if not s:
            return 0.0
        upper = sum(ch.isupper() for ch in s)
        return upper / max(len(s), 1)

    reviews["caps_ratio"] = apply_with_progress(
        txt,
        caps_ratio,
        batch_size=20_000,
        desc="Caps ratio",
    )

    # Sentiment (VADER baseline)
    print("Computing sentiment (VADER)...")
    analyzer = SentimentIntensityAnalyzer()
    reviews["sentiment_vader"] = apply_with_progress(
        txt,
        lambda s: analyzer.polarity_scores(s)["compound"],
        batch_size=5_000,
        desc="VADER sentiment",
    )

    print("Loaded + engineered reviews:", reviews.shape, "elapsed:", round(time.time() - t0, 1), "s")

    # IDs for merges
    biz_ids = set(reviews["business_id"].dropna().unique())
    user_ids = set(reviews["user_id"].dropna().unique())

    # -------------------------
    # Merge business: city + categories
    # -------------------------
    print("\nMerging business (city/categories) ...")
    business_cols = ["business_id", "city", "state", "categories"]
    business = pd.read_parquet(BUSINESS_PATH, columns=business_cols)
    business = business[business["business_id"].isin(biz_ids)]
    business["city"] = business["city"].fillna("Unknown")

    df = reviews.merge(business, on="business_id", how="left")
    df["city"] = df["city"].fillna("Unknown")

    # Category explode
    df["category_list"] = df["categories"].apply(to_categories)
    df_cat = df.explode("category_list")
    df_cat = df_cat[df_cat["category_list"].notna() & (df_cat["category_list"] != "")]

    # -------------------------
    # Optional: merge user
    # -------------------------
    if DO_USER_MERGE:
        print("\nMerging user features (fans, review_count, avg_stars) ...")
        user_cols = ["user_id", "review_count", "fans", "average_stars", "yelping_since"]
        users = pd.read_parquet(USER_PATH, columns=user_cols)
        users = users[users["user_id"].isin(user_ids)]
        users["yelping_since"] = pd.to_datetime(users["yelping_since"], errors="coerce")

        df = df.merge(users, on="user_id", how="left", suffixes=("", "_user"))
        # rename for clarity
        if "review_count_user" not in df.columns and "review_count" in users.columns:
            df.rename(columns={"review_count": "review_count_user"}, inplace=True)

    # -------------------------
    # Plots: Overview helpfulness vs features
    # -------------------------
    print("\nPlotting overview charts...")

    # helpful distribution (log)
    plt.figure()
    plt.hist(df["helpful_log1p"], bins=50)
    plt.title("Distribution of log(1 + helpful/useful votes)")
    plt.xlabel("log1p(helpful)")
    plt.ylabel("Count")
    savefig("helpful_distribution.png")

    # seasonality by month
    month_mean = df.groupby("month")["helpful"].mean().reindex(range(1, 13))
    plt.figure()
    plt.plot(month_mean.index, month_mean.values, marker="o")
    plt.title("Mean helpful votes by month")
    plt.xlabel("Month")
    plt.ylabel("Mean helpful")
    plt.xticks(range(1, 13))
    savefig("helpful_by_month.png")

    # weekday
    weekday_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    wd_mean = df.groupby("weekday")["helpful"].mean().reindex(weekday_order)
    plt.figure(figsize=(8, 4))
    plt.bar(wd_mean.index.astype(str), wd_mean.values)
    plt.title("Mean helpful votes by weekday")
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Mean helpful")
    savefig("helpful_by_weekday.png")

    # city (top 15 by count)
    top_cities = df["city"].value_counts().head(15).index
    city_stats = (df[df["city"].isin(top_cities)]
                  .groupby("city")["helpful"]
                  .agg(mean="mean", median="median", n="size")
                  .sort_values("mean", ascending=False))
    plt.figure(figsize=(10, 4))
    plt.bar(city_stats.index, city_stats["mean"])
    plt.title("Mean helpful votes by city (top 15 by count)")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Mean helpful")
    savefig("helpful_by_city.png")

    # category (top 20 by count)
    top_cats = df_cat["category_list"].value_counts().head(20).index
    cat_stats = (df_cat[df_cat["category_list"].isin(top_cats)]
                 .groupby("category_list")["helpful"]
                 .agg(mean="mean", median="median", n="size")
                 .sort_values("mean", ascending=False))
    plt.figure(figsize=(10, 5))
    plt.bar(cat_stats.index, cat_stats["mean"])
    plt.title("Mean helpful votes by category (top 20 by count)")
    plt.xticks(rotation=60, ha="right")
    plt.ylabel("Mean helpful")
    savefig("helpful_by_category.png")

    # useful vs cool / funny (scatter sample)
    sample = df.sample(min(len(df), 25_000), random_state=42)
    plt.figure()
    plt.scatter(sample["cool"], sample["helpful"], alpha=0.2)
    plt.title("Helpful (useful votes) vs Cool votes")
    plt.xlabel("cool")
    plt.ylabel("helpful")
    savefig("helpful_vs_cool.png")

    plt.figure()
    plt.scatter(sample["funny"], sample["helpful"], alpha=0.2)
    plt.title("Helpful (useful votes) vs Funny votes")
    plt.xlabel("funny")
    plt.ylabel("helpful")
    savefig("helpful_vs_funny.png")

    # -------------------------
    # Plots: Feature correlations to helpfulness
    # -------------------------
    print("Plotting correlation + length effects...")

    # length deciles
    df["len_decile"] = pd.qcut(df["text_len_words"], q=10, duplicates="drop")
    len_stats = df.groupby("len_decile")["helpful"].mean()
    plt.figure(figsize=(10, 4))
    plt.plot(range(len(len_stats)), len_stats.values, marker="o")
    plt.title("Mean helpful votes by review length decile")
    plt.xlabel("Length decile (short → long)")
    plt.ylabel("Mean helpful")
    savefig("helpful_by_length_decile.png")

    corr_cols = [
        "helpful", "stars", "cool", "funny",
        "text_len_words", "text_len_chars",
        "exclaim_count", "question_count", "caps_ratio",
        "sentiment_vader",
    ]
    if DO_USER_MERGE:
        # these may or may not exist depending on merge/renames
        for c in ["review_count_user", "fans", "average_stars"]:
            if c in df.columns:
                corr_cols.append(c)

    corr_cols = [c for c in corr_cols if c in df.columns]
    corr = df[corr_cols].corr(method="spearman")

    plt.figure(figsize=(10, 8))
    plt.imshow(corr, interpolation="nearest")
    plt.xticks(range(len(corr_cols)), corr_cols, rotation=90)
    plt.yticks(range(len(corr_cols)), corr_cols)
    plt.title("Spearman correlation (selected numeric features)")
    plt.colorbar()
    savefig("correlation_heatmap.png")

    # Optional user plots
    if DO_USER_MERGE and "fans" in df.columns:
        df["fans_bin"] = pd.cut(df["fans"].fillna(0), bins=[-1,0,5,20,100,1e9],
                                labels=["0","1-5","6-20","21-100","100+"])
        fans_stats = df.groupby("fans_bin")["helpful"].mean()
        plt.figure()
        plt.bar(fans_stats.index.astype(str), fans_stats.values)
        plt.title("Mean helpful votes by user fans bin")
        plt.xlabel("Fans bin")
        plt.ylabel("Mean helpful")
        savefig("helpful_by_user_fans.png")

    # -------------------------
    # “Kind of words” (TF-IDF)
    # -------------------------
    top_phrases_helpful = None
    if enable_tfidf:
        print("\nRunning TF-IDF to find phrases associated with helpful>0 ...")
        df_text = df.sample(min(len(df), TFIDF_SAMPLE), random_state=42).copy()
        y = (df_text["helpful"] > 0).astype(int).values

        vec = TfidfVectorizer(
            max_features=6000,
            ngram_range=(1, 2),
            min_df=5,
            stop_words="english",
        )
        X = vec.fit_transform(df_text["text"].astype(str))

        clf = (
            LogisticRegression(max_iter=400, n_jobs=tfidf_jobs)
            if tfidf_jobs is not None
            else LogisticRegression(max_iter=400)
        )
        clf.fit(X, y)

        terms = np.array(vec.get_feature_names_out())
        coef = clf.coef_[0]

        top_pos = terms[np.argsort(coef)[-25:]][::-1]
        top_neg = terms[np.argsort(coef)[:25]]
        top_phrases_helpful = (top_pos.tolist(), top_neg.tolist())
        print("Top phrases (helpful>0):", top_pos[:10])

    # -------------------------
    # Write markdown summary
    # -------------------------
    print("\nWriting summary markdown...")
    summary_path = OUT_DIR / "eda_summary.md"

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"# Yelp EDA (sample)\n\n")
        f.write(f"- Reviews loaded: **{len(reviews):,}**\n")
        f.write(f"- Merged businesses: **{len(business):,}**\n")
        if DO_USER_MERGE:
            f.write(f"- Merged users: **{df['user_id'].nunique():,}** (from review sample)\n")
        f.write("\n## Helpful (= useful votes) summary\n")
        f.write(df["helpful"].describe().to_frame("helpful").to_markdown() + "\n\n")

        f.write("## Helpful by City (top 10)\n")
        f.write(city_stats.head(10).to_markdown() + "\n\n")

        f.write("## Helpful by Category (top 10)\n")
        f.write(cat_stats.head(10).to_markdown() + "\n\n")

        f.write("## Spearman correlation (selected numeric features)\n")
        f.write(corr.to_markdown() + "\n\n")

        if top_phrases_helpful is not None:
            pos, neg = top_phrases_helpful
            f.write("## Kind of words (TF-IDF, logistic regression)\n")
            f.write("Phrases most associated with **helpful > 0**:\n\n")
            f.write("- " + "\n- ".join(pos[:20]) + "\n\n")
            f.write("Phrases most associated with **helpful == 0**:\n\n")
            f.write("- " + "\n- ".join(neg[:20]) + "\n\n")

    print("DONE ✅")
    print("Summary:", summary_path.resolve())
    print("Figures:", FIG_DIR.resolve())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Helpful Lens EDA")
    parser.add_argument(
        "--n-reviews",
        type=int,
        default=DEFAULT_N_REVIEWS,
        help="Number of reviews to load from the cleaned parquet. "
             "Set <=0 to load the full dataset.",
    )
    parser.add_argument(
        "--skip-tfidf",
        action="store_true",
        help="Skip TF-IDF/logistic regression (useful for constrained environments).",
    )
    parser.add_argument(
        "--max-threads",
        type=int,
        default=None,
        help="Optional cap on numerical backend threads (e.g., 1 to avoid OMP errors).",
    )
    parser.add_argument(
        "--tfidf-jobs",
        type=int,
        default=None,
        help="Workers for the TF-IDF logistic regression (set 1 to avoid loky warnings).",
    )
    args = parser.parse_args()

    n_reviews = None if args.n_reviews is None or args.n_reviews <= 0 else args.n_reviews
    tfidf_jobs = args.tfidf_jobs
    if tfidf_jobs is None and args.max_threads is not None and args.max_threads > 0:
        tfidf_jobs = args.max_threads
    try:
        main(
            n_reviews=n_reviews,
            enable_tfidf=not args.skip_tfidf and DO_TFIDF_WORDS,
            max_threads=args.max_threads,
            tfidf_jobs=tfidf_jobs,
        )
    except KeyboardInterrupt:
        print("\nInterrupted by user, exiting early.")
        sys.exit(130)
