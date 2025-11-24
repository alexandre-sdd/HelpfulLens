# HelpfulLens Data Layout

Only lightweight README files are versioned under `data/`; every other artifact
is regenerated locally by the pipeline.

```
data/
  raw/                 # untouched Yelp JSON exports + Dataset_User_Agreement.pdf
    parquet/           # JSON → parquet caches produced by ingest_raw
  cleaned/             # outputs of src/data/clean_yelp.py
  enriched/            # joins/aggregations/external merges
  features/            # experiment-specific feature matrices (X only)
  datasets/            # final train/val/test bundles (X + y)
    training/
    evaluation/
  predictions/         # scored outputs for evaluation/monitoring
  external/            # reference tables (holidays, FX, mappings, etc.)
```

## Folder contracts

| Folder | Contents | Produced by |
| --- | --- | --- |
| `data/raw/` | Official Yelp JSON dumps and other immutable source files. | Manual download from Yelp. |
| `data/raw/parquet/` | Columnar caches (optionally chunked) mirroring the JSON shards. | `python -m src.data.ingest_raw` |
| `data/cleaned/` | Schema-stable review/user/business tables (e.g., `reviews_clean.parquet`). | `python -m src.data.clean_yelp` |
| `data/enriched/` | Intermediate joins/aggregations once multiple sources are combined. | Custom notebooks/scripts. |
| `data/features/` | Feature matrices or embeddings prior to modeling. | `src/features/build_features.py` (or similar). |
| `data/datasets/training/` | Train-ready bundles that include both features and labels. | `python -m src.data.make_dataset` |
| `data/datasets/evaluation/` | Hold-out bundles with the same schema as training. | `python -m src.data.make_dataset` |
| `data/predictions/` | Saved inference runs for audits or monitoring. | Any scoring job. |
| `data/external/` | Reference lookups (holiday calendars, FX rates, etc.). | Manual curation. |

## Workflow summary

1. Drop the downloaded Yelp JSON files into `data/raw/`.
2. Run `./scripts/run_data_pipeline.sh` (or call each module) to populate
   `raw/parquet`, `cleaned`, and `datasets`.
3. Optionally stage advanced joins in `data/enriched/` and cached features in
   `data/features/`.
4. Log inference outputs to `data/predictions/` whenever you score a dataset.

Everything except these README breadcrumbs stays out of Git, so you can freely
delete or regenerate artifacts as needed.
