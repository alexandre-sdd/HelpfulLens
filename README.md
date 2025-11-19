# yelp_helpfulness_mlflow

Predict the helpfulness of Yelp reviews with a structured machine learning workflow.
This project scaffold is pre-configured to use MLflow for experiment tracking, model
versioning, and reproducible runs.

## Example MLflow run

```bash
mlflow run . -e train -P experiment_name="yelp_helpfulness_baseline"
```
