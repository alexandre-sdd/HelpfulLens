# HelpfulLens Data Layout

The repository tracks only lightweight README files so you can rebuild the full
pipeline locally without bloating Git history. Each subfolder maps directly to a
stage in `scripts/run_data_pipeline.sh`.

```
data/
  raw/
    business/ reviews/ users/ tips/   # untouched JSON exports from Yelp
  interim/
    raw_parquet/                      # columnar copies of the raw shards
    cleaned/                          # outputs from src/data/clean_yelp.py
    features/                         # optional caches from src/features/*
  processed/
    training/                         # train splits written by make_dataset.py
    evaluation/                       # hold-out splits written by make_dataset.py
```

## Folder contracts

| Folder | Contents | Produced by |
| --- | --- | --- |
| `data/raw/*` | The official Yelp Academic Dataset JSON (or `.json.gz`) files. No edits, no commits. | Humans download + drop the files here. |
| `data/interim/raw_parquet/` | Automatically generated parquet shards (per dataset) for fast reloads. | `python -m src.data.ingest_raw` |
| `data/interim/cleaned/` | Canonical cleaned tables such as `reviews_clean.parquet`. | `python -m src.data.clean_yelp` |
| `data/interim/features/` | Optional feature matrices or embeddings if you cache them. | `src/features/build_features.py` (manual) |
| `data/processed/training/` | Final modeling datasets (features + label) for training. | `python -m src.data.make_dataset` |
| `data/processed/evaluation/` | Hold-out datasets mirrored from the training schema. | `python -m src.data.make_dataset` |

## Workflow summary

1. Drop the raw Yelp dumps in the matching `data/raw/<dataset>/` folders.
2. Run `./scripts/run_data_pipeline.sh` (or call each module manually).
3. Inspect the parquet outputs in `data/interim/` and `data/processed/`.
4. Point notebooks or MLflow entry points at the processed training/eval files.

Everything inside `data/` is intentionally ignored by Git except for these
documentation breadcrumbs, so feel free to regenerate artifacts whenever needed.
