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
