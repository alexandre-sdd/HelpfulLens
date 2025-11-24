# HelpfulLens (yelp_helpfulness_mlflow)

End-to-end scaffold for predicting Yelp review helpfulness with MLflow-driven
experiment tracking, structured data pipelines, and reproducible project metadata.

## Highlights

- **MLflow project** with `train` and `evaluate` entry points plus local tracking
  in `mlruns/`.
- **Opinionated directory layout** covering raw dumps, cleaned tables, features,
  and a lightweight model registry (`models/registry/`).
- **Config-first design:** `src/config/config.yaml` centralizes paths,
  feature toggles, hyperparameters, and MLflow metadata.

## Project Structure

```
.
├─ README.md
├─ MLproject
├─ conda.yaml
├─ requirements.txt
├─ data/
│  ├─ raw/ (JSON dumps + Dataset_User_Agreement.pdf)
│  │  └─ parquet/          # JSON→parquet caches from ingest_raw
│  ├─ cleaned/             # outputs from clean_yelp.py
│  ├─ enriched/            # manual joins / aggregations
│  ├─ features/            # cached feature matrices
│  ├─ datasets/{training,evaluation}/
│  ├─ predictions/         # saved inference outputs
│  └─ external/            # reference lookups (holidays, etc.)
├─ models/
│  ├─ artifacts/              # serialized models + vectorizers
│  └─ registry/               # promoted versions (JSON descriptors)
├─ notebooks/                 # starter EDA/feature/model notebooks
├─ scripts/                   # MLflow convenience launchers
└─ src/
   ├─ config/
   ├─ data/
   ├─ features/
   ├─ models/
   └─ utils/
```

Each `data/` subfolder has a README describing expected contents so you can drop
Yelp JSON exports and processed tables without guesswork.

## Environment Setup

```bash
# Option A: Conda
conda env create -f conda.yaml
conda activate yelp_helpfulness_mlflow

# Option B: Virtualenv / pip
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Notebook Output Hygiene

We vendor `.gitattributes` rules so Jupyter notebooks are cleaned via
[`nbstripout`](https://github.com/kynan/nbstripout) before commits. Run the
install step once per clone (after the dependencies are installed):

```bash
nbstripout --install --attributes .gitattributes
nbstripout --status   # optional: confirm the filter is active
```

`nbstripout` strips execution counts, images, and other transient cell outputs so
you can `git push` notebooks without noisy diffs or binary blobs.

## Data Pipeline

The end-to-end Yelp data prep lives under `src/data/` and can be kicked off with
a single helper script:

```bash
# Optional env vars:
#   INGEST_DATASETS, CLEAN_DATASETS, CHUNK_SIZE, ROWS_PER_CHUNK,
#   INGEST_LIMIT, CLEAN_LIMIT, FORCE (0/1), CONFIG
./scripts/run_data_pipeline.sh
```

The script orchestrates three steps:

1. `python -m src.data.ingest_raw` converts JSON shards in `data/raw/` into
   fast-loading parquet caches inside `data/raw/parquet/`. Use
   `--rows-per-chunk` to write multiple parquet files per JSON so you can follow
   ingestion progress chunk-by-chunk.
2. `python -m src.data.clean_yelp` runs the `clean_*_df` functions shown in
   `src/data/clean_yelp.py`, persisting `*_clean.parquet` files to `data/cleaned/`.
3. `python -m src.data.make_dataset` joins the cleaned tables, engineers a
   handful of review-level features, and writes train/eval parquet files to
   `data/datasets/{training,evaluation}/`.

Smoke-test the pipeline on small subsets via `INGEST_LIMIT=1 CLEAN_LIMIT=1000 ./scripts/run_data_pipeline.sh`.

## Running the Project

```bash
# Train (logs run, params, metrics, dummy artifacts)
mlflow run . -e train -P experiment_name="yelp_helpfulness_baseline"

# Evaluate (logs placeholder eval metrics)
mlflow run . -e evaluate
```

Shortcut scripts are available under `scripts/` if you prefer `./scripts/run_training.sh`.

## Next Steps

1. Drop the official Yelp JSON dumps into `data/raw/*`.
2. Implement the loaders in `src/data/`, feature builders in `src/features/`,
   and real training/evaluation logic in `src/models/`.
3. Replace dummy artifacts/registry entries once genuine MLflow runs are produced.

## How the Pieces Fit Together

1. **Raw data ingestion** – JSON exports from Yelp go into `data/raw/`. The
   ingestion module mirrors them to `data/raw/parquet/` for faster reloads.
2. **Cleaning + merging** – `src/data/clean_yelp.py` standardizes each dataset
   and writes deterministic `*_clean.parquet` files to `data/cleaned/`.
3. **Feature engineering** – `src/features/build_features.py` (or notebooks)
   produce experiment-specific caches under `data/features/`.
4. **Processed datasets** – `src/data/make_dataset.py` joins the cleaned tables
   and writes final train/eval splits to `data/datasets/{training,evaluation}`.
5. **Modeling with MLflow** – the `train` entry point runs your training script,
   which logs parameters, metrics, artifacts, and a model registry pointer. The
   `evaluate` entry point consumes the saved model and logs evaluation metrics.

## Brand-New to Git or GitHub?

You can still use this project even if you have never touched Git before:

1. **Install Git** – download it from <https://git-scm.com/downloads>. Accept the
   defaults during installation.
2. **Clone the project** – open a terminal and run:
   ```bash
   git clone https://github.com/<your-account>/yelp_helpfulness_mlflow.git
   cd yelp_helpfulness_mlflow
   ```
   If you do not have a GitHub account, click “Download ZIP” on the repository
   page, unzip it, and continue in the extracted folder.
3. **Make changes safely** – run `git status` to see what changed, `git add <file>`
   to stage updates, and `git commit -m "short message"` to save them. When ready,
   `git push` uploads your commits to GitHub.
4. **No terminal?** – GitHub Desktop offers a friendly UI for the same commands
   (<https://desktop.github.com/>). You can drag-and-drop files to commit.

### Tips for Large/Data-Heavy Work

- Keep bulky raw data outside Git; only the folder structure and README files are
  tracked. You can add your own `.env` file or use `data/raw/.gitkeep` if needed.
- When running notebooks, start with small samples (e.g., a few hundred reviews)
  placed in the `data/raw/` folders to verify code without waiting on the full
  dataset.
- Use the provided README files in each data subfolder as a checklist so you know
  exactly where every file should go.
- If you are unsure about using the command line, run the scripts via the
  notebooks or copy/paste commands from this README—they are ready as-is.

## Step-by-Step Quickstart (No Experience Required)

1. **Install the tools**  
   - Python 3.10: <https://www.python.org/downloads/> (check “Add to PATH” on Windows).  
   - Git: <https://git-scm.com/downloads>.  
   - Optional: VS Code or JupyterLab for editing notebooks.
2. **Download the project**  
   - Easy mode: click “Code → Download ZIP” on GitHub, unzip anywhere (e.g., Desktop).  
   - Git mode: open *Terminal* (macOS/Linux) or *PowerShell* (Windows), then run:  
     ```bash
     git clone https://github.com/<your-account>/HelpfulLens.git
     cd HelpfulLens
     ```
3. **Create a Python environment**  
   - Windows: `python -m venv .venv` then `.\.venv\Scripts\activate`.  
   - macOS/Linux: `python3 -m venv .venv` then `source .venv/bin/activate`.  
   - Install packages: `pip install -r requirements.txt`.
4. **Check everything works**  
   - Run `python -m pip show mlflow`. If it prints details, MLflow is installed.  
   - Execute `mlflow run . -e train` to generate the dummy model run.
5. **Add data later**  
   - Copy Yelp JSON files into the matching `data/raw/*` folders.  
   - Open `notebooks/01_eda.ipynb` to explore, or modify `src/data/` scripts to load the files.

## What Does `.gitignore` Mean?

- `.gitignore` is a list of files/folders Git should *not* track. This project
  ignores heavy folders like `data/raw/`, `data/cleaned/`, `data/datasets/`,
  `mlruns/`, and `models/artifacts/` so gigabytes of data stay on your computer only.
- You can still create those folders locally (they already exist). Git simply
  skips them when committing. This keeps the repository fast and avoids uploading
  private Yelp datasets.
- If you ever need to share a sample dataset, put it somewhere else (e.g.,
  `data/sample/`) or remove that path from `.gitignore` before committing.
