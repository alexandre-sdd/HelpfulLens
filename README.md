# yelp_helpfulness_mlflow

End-to-end scaffold for predicting Yelp review helpfulness with MLflow-driven
experiment tracking, structured data pipelines, and reproducible project metadata.

## Highlights

- **MLflow project** with `train` and `evaluate` entry points plus local tracking
  in `mlruns/`.
- **Opinionated directory layout** for raw/interim/processed data, feature caches,
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
│  ├─ raw/{business,reviews,users,tips}/
│  ├─ interim/{cleaned,features}/
│  └─ processed/{training,evaluation}/
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

1. **Raw data ingestion** – JSON exports from Yelp go into the `data/raw/*`
   folders. The loader scripts only need the files to exist; no database setup is
   required.
2. **Cleaning + merging** – `src/data/make_dataset.py` will merge reviews,
   businesses, users, and optional tips into intermediate CSV/Parquet tables,
   saved in `data/interim/cleaned`.
3. **Feature engineering** – `src/features/build_features.py` creates numeric and
   text features and stores cache files in `data/interim/features`.
4. **Processed datasets** – final train/eval splits land in
   `data/processed/{training,evaluation}`.
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
     git clone https://github.com/<your-account>/yelp_helpfulness_mlflow.git
     cd yelp_helpfulness_mlflow
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
  ignores heavy folders like `data/raw/`, `data/processed/`, `mlruns/`, and
  `models/artifacts/` so gigabytes of data stay on your computer only.
- You can still create those folders locally (they already exist). Git simply
  skips them when committing. This keeps the repository fast and avoids uploading
  private Yelp datasets.
- If you ever need to share a sample dataset, put it somewhere else (e.g.,
  `data/sample/`) or remove that path from `.gitignore` before committing.
