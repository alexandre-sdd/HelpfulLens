# check the /Users/jiangmengjie/Documents/HelpfulLens/data/datasets/evaluation/yelp_helpfulness_eval.parquet file to see the distribution of the target_useful_votes column

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_parquet("/Users/jiangmengjie/Documents/HelpfulLens/data/datasets/evaluation/yelp_helpfulness_eval.parquet")

# Select and clean the series
col = "target_useful_votes"
s = df[col].dropna()

# Print descriptive statistics with useful percentiles
print("shape:", df.shape)
print(f"non-null {col}:", len(s))
print(
    s.describe(
        percentiles=[0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    )
)

# Show unique count info and a preview of value counts (sorted by value)
vc = s.value_counts().sort_index()
print("\nunique values:", len(vc))
print("\nfirst 20 values (value -> count):")
print(vc.head(20))
print("\nlast 20 values (value -> count):")
print(vc.tail(20))

# Plot histograms: linear scale and log-scaled y to expose long tails
fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
bins = 50 if len(s) > 0 else 10

sns.histplot(s, bins=bins, ax=axes[0], color="#4C78A8")
axes[0].set_title("target_useful_votes (linear scale)")
axes[0].set_xlabel(col)
axes[0].set_ylabel("count")

sns.histplot(s, bins=bins, ax=axes[1], color="#4C78A8")
axes[1].set_yscale("log")
axes[1].set_title("target_useful_votes (log y-scale)")
axes[1].set_xlabel(col)
axes[1].set_ylabel("count (log)")

plt.show()