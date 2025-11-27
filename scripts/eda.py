from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = Path("data")
REPORTS_DIR = Path("reports")
FIG_DIR = REPORTS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

def find_data_file():
    # tweak extensions if needed
    for ext in ("*.csv", "*.parquet", "*.xlsx"):
        files = list(DATA_DIR.rglob(ext))
        if files:
            return files[0]
    return None

def load_df(path: Path) -> pd.DataFrame:
    if path.suffix == ".csv":
        return pd.read_csv(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix in (".xlsx", ".xls"):
        return pd.read_excel(path)
    raise ValueError(f"Unsupported file type: {path.suffix}")

path = find_data_file()
if path is None:
    raise FileNotFoundError("No data file found under /data (csv/parquet/xlsx).")

df = load_df(path)

# --- 1) Basic sanity checks ---
summary_lines = []
summary_lines.append(f"Data file: {path}")
summary_lines.append(f"Shape: {df.shape}")
summary_lines.append("\nDtypes:\n" + df.dtypes.to_string())

missing = df.isna().mean().sort_values(ascending=False)
summary_lines.append("\nMissing rate (top 20):\n" + missing.head(20).to_string())

dup_rows = df.duplicated().sum()
summary_lines.append(f"\nDuplicate rows: {dup_rows}")

# --- 2) Simple numeric/categorical split ---
num_cols = df.select_dtypes(include="number").columns.tolist()
cat_cols = [c for c in df.columns if c not in num_cols]

summary_lines.append(f"\n# numeric cols: {len(num_cols)}")
summary_lines.append(f"# non-numeric cols: {len(cat_cols)}")

# --- 3) Quick univariate plots ---
# Numeric histograms (first 12)
for col in num_cols[:12]:
    ax = df[col].dropna().plot(kind="hist", bins=30, title=f"Histogram: {col}")
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"hist_{col}.png", dpi=200)
    plt.close(fig)

# Categorical bar charts (first 12, top 15 categories)
for col in cat_cols[:12]:
    vc = df[col].astype("string").fillna("NaN").value_counts().head(15)
    ax = vc.plot(kind="bar", title=f"Top categories: {col}")
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"bar_{col}.png", dpi=200)
    plt.close(fig)

# --- 4) Correlation heatmap (if enough numeric cols) ---
if len(num_cols) >= 2:
    corr = df[num_cols].corr(numeric_only=True)
    plt.imshow(corr, interpolation="nearest")
    plt.xticks(range(len(num_cols)), num_cols, rotation=90)
    plt.yticks(range(len(num_cols)), num_cols)
    plt.title("Correlation (numeric)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "corr_numeric.png", dpi=200)
    plt.close()

# --- 5) Write a tiny report ---
report_path = REPORTS_DIR / "eda_summary.md"
report_path.write_text("\n".join(summary_lines), encoding="utf-8")

print(f"Saved report to: {report_path}")
print(f"Saved figures to: {FIG_DIR.resolve()}")
