# HelpfulLens Data Lifecycle

The `data/` folder now mirrors the full journey from raw JSON dumps to model
predictions. Each stage is its own directory so teammates can easily find the
right files without worrying about machine-learning jargon.

```
data/
  raw/            # untouched source files (e.g., Yelp JSON exports)
  cleaned/        # schema-aligned tables after quality checks
  enriched/       # joins + aggregations that add context
  features/       # model-ready matrices (X only) per cut/experiment
  datasets/       # final train/val/test bundles (X + y)
  predictions/    # saved model outputs for evaluation or monitoring
```

## What belongs where

| Stage | What goes here | Who usually touches it |
| --- | --- | --- |
| `raw/` | Vendor dumps, CSV/JSON exports, data agreements. Nothing is edited in place. | Anyone collecting data. |
| `cleaned/` | Sanitized tables with consistent column names, types, and obvious errors removed. | Data wranglers. |
| `enriched/` | Outputs after combining sources (joins), aggregations, and derived business metrics. | Data + domain experts. |
| `features/` | Numerical matrices or embeddings used to train models. There is no target column here. | ML engineers. |
| `datasets/` | Final train/validation/test splits containing both features (`X`) and labels (`y`). | ML engineers. |
| `predictions/` | Inference outputs (e.g., CSV/Parquet with `id`, `prediction`, optional metadata). | Anyone running/evaluating models. |

## How we use the folders

1. **Drop new source files in `raw/`.** Nothing else should change inside those
   files so we always have an audit trail back to the original dataset.
2. **Create cleaned versions in `cleaned/`.** Use consistent naming such as
   `YYYYMMDD_cleaned_reviews.parquet` to make it obvious when a file was
   produced and which source it came from.
3. **Add context in `enriched/`.** Typical examples are user-level aggregates,
   business metadata joins, or any feature engineering that still preserves the
   target column.
4. **Generate experiment-specific matrices in `features/`.** Store one file per
   experimental configuration (e.g., `tfidf_v1/`, `llm_embeddings_v2/`), keeping
   only the feature columns.
5. **Package modeling datasets in `datasets/`.** Save artifacts such as
   `helpfulness_v1_train.parquet`, `helpfulness_v1_val.parquet`, and
   `helpfulness_v1_test.parquet` that include both features and the label.
6. **Track model outputs in `predictions/`.** When you score a dataset, export
   predictions here so others can review performance or run monitoring checks.

Keeping this flow consistent lets non-ML teammates know exactly where to read
from or write to, and it makes automated pipelines much easier to script.
