# HelpfulLens

Lightweight tools for exploring Yelp review helpfulness without any MLflow or package-install ceremony. The repo covers ingest/clean steps for the public Yelp dataset, sentiment-enriched feature engineering, and fast EDA helpers.

## What you can do
- Convert Yelp JSON dumps into parquet caches and cleaned tables (`src/data`, `scripts/run_data_pipeline.sh`).
- Build sentiment-augmented feature sets with VADER or DistilBERT (`scripts/feature_engineering_sentiment.py`, `scripts/distilbert_sentiment_subset.py`).
- Generate reproducible EDA plots/notebooks for quick intuition (`scripts/eda.py`, `scripts/eda_full.py`, `notebooks/`).

## Repo layout
```
README.md
conda.yaml
requirements.txt
data/                  # placeholder READMEs describing expected drops
notebooks/             # exploratory notebooks
reports/eda/           # generated figures + summary markdown
scripts/               # pipeline + feature/EDA helpers
src/                   # reusable modules for ingest/clean/make_dataset/config/utils
```

## Setup
**Conda**
```bash
conda env create -f conda.yaml
conda activate helpfullens
```

**Virtualenv / pip**
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
No editable install is needed; run commands from the repo root. Transformers scripts rely on PyTorch, which is included in the requirements.

## Data prep
1) Drop the Yelp JSON exports under `data/raw/` (see `data/README.md` + subfolder READMEs for expected filenames).
2) Run ingest → clean → dataset assembly:
```bash
./scripts/run_data_pipeline.sh
```
Tunable env vars: `CONFIG` (alt config file), `INGEST_DATASETS`, `CLEAN_DATASETS`, `ROWS_PER_CHUNK`, `INGEST_LIMIT`, `CLEAN_LIMIT`, and `FORCE=1` to overwrite existing parquet outputs. Results land in `data/raw/parquet/`, `data/cleaned/`, and `data/datasets/{training,evaluation}/`.

## Feature engineering & sentiment
- Full feature pass (VADER by default):
```bash
python scripts/feature_engineering_sentiment.py --sentiment-backend vader
```
Use `--sentiment-backend transformer` plus `--transformer-device` if you want DistilBERT scoring during feature creation. Output: `data/features/review_features_with_sentiment.parquet`.

- DistilBERT sentiment on a compact subset (max 200k reviews):
```bash
python scripts/distilbert_sentiment_subset.py --max-reviews 200000 --omit-text
```
Add `--require-positive-useful` to keep only reviews with useful > 0. A summary JSON resume can be written via `--resume-path`.

## EDA
Run `python scripts/eda.py` for a fast sample EDA or `python scripts/eda_full.py` for a larger sweep. Figures + markdown summaries are written to `reports/eda/`.

## Configuration
`src/config/config.yaml` centralizes folder paths, feature toggles, and train/validation split settings used by the data pipeline.

## Notebook hygiene (optional)
Keep notebook diffs clean with nbstripout once dependencies are installed:
```bash
nbstripout --install --attributes .gitattributes
```
