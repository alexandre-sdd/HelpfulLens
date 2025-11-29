from __future__ import annotations

from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# =========================
# Config
# =========================
N_REVIEWS = 50_000          # start here; increase to 100k/200k if it runs fine
DO_USER_MERGE = True        # set False if you want it faster
DO_TFIDF_WORDS = True       # set False if you want it faster
TFIDF_SAMPLE = 30_000       # TF-IDF runs on a subset to stay fast

RAW_DIR = Path("data/raw")
REVIEW_PATH = RAW_DIR / "yelp_academic_dataset_review.json"
BUSINESS_PATH = RAW_DIR / "yelp_academic_dataset_business.json"
USER_PATH = RAW_DIR / "yelp_academic_dataset_user.json"

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

def filter_jsonl(path: Path, id_field: str, ids: set[str], keep_cols: list[str]) -> pd.DataFrame:
    """Stream JSON Lines file, keep only rows where obj[id_field] in ids."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc=f"Filtering {path.name}"):
            obj = json.loads(line)
            if obj.get(id_field) in ids:
                rows.append({k: obj.get(k) for k in keep_cols})
    return pd.DataFrame(rows)

def to_categories(x) -> list[str]:
    if pd.isna(x):
        return []
    return [c.strip() for c in str(x).split(",") if c.strip()]


def main() -> None:
    print("EDA started...")
    print("Working directory:", Path.cwd())

    # output dirs
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # check files
    for p in [REVIEW_PATH, BUSINESS_PATH]:
        assert_exists(p)
    if DO_USER_MERGE:
        assert_exists(USER_PATH)

    t0 = time.time()

    # -------------------------
    # Load reviews (sample)
    # -------------------------
    print(f"\nLoading {N_REVIEWS:,} reviews from {REVIEW_PATH.name} ...")
    reviews = pd.read_json(REVIEW_PATH, lines=True, nrows=N_REVIEWS)

    # Basic cleaning
    reviews["date"] = pd.to_datetime(reviews["date"], errors="coerce")
    for c in ["useful", "funny", "cool"]:
        reviews[c] = pd.to_numeric(reviews[c], errors="coerce").fillna(0).astype(int)
    reviews["stars"] = pd.to_numeric(reviews.get("stars"), errors="coerce")

    # Target
    reviews["helpful"] = reviews["useful"]
    reviews["helpful_log1p"] = np.log1p(reviews["helpful"])

    # Seasonality
    reviews["month"] = reviews["date"].dt.month
    reviews["weekday"] = reviews["date"].dt.day_name()

    # Text features (NO emoji)
    txt = reviews["text"].astype(str)
    reviews["text_len_chars"] = txt.str.len()
    reviews["text_len_words"] = txt.str.split().str.len()
    reviews["exclaim_count"] = txt.str.count("!")
    reviews["question_count"] = txt.str.count(r"\?")

    def caps_ratio(s: str) -> float:
        if not s:
            return 0.0
        upper = sum(ch.isupper() for ch in s)
        return upper / max(len(s), 1)

    reviews["caps_ratio"] = txt.apply(caps_ratio)

    # Sentiment (VADER baseline)
    print("Computing sentiment (VADER)...")
    analyzer = SentimentIntensityAnalyzer()
    reviews["sentiment_vader"] = txt.apply(lambda s: analyzer.polarity_scores(s)["compound"])

    print("Loaded + engineered reviews:", reviews.shape, "elapsed:", round(time.time() - t0, 1), "s")

    # IDs for merges
    biz_ids = set(reviews["business_id"].dropna().unique())
    user_ids = set(reviews["user_id"].dropna().unique())

    # -------------------------
    # Merge business: city + categories
    # -------------------------
    print("\nMerging business (city/categories) ...")
    business_cols = ["business_id", "city", "state", "categories"]
    business = filter_jsonl(BUSINESS_PATH, "business_id", biz_ids, business_cols)
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
        users = filter_jsonl(USER_PATH, "user_id", user_ids, user_cols)
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
    if DO_TFIDF_WORDS:
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

        clf = LogisticRegression(max_iter=400, n_jobs=-1)
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
    main()

