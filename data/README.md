# Data Directory Structure

This folder mirrors the full Yelp helpfulness data lifecycle:

- `raw/`
  - `business/`: Drop exported business.json files here.
  - `reviews/`: Store review.json splits.
  - `users/`: Contains user.json exports.
  - `tips/`: Optional tip.json files.
- `interim/`
  - `cleaned/`: Staging tables after basic cleaning/merging.
  - `features/`: Cached feature matrices (e.g., TF-IDF, aggregated stats).
- `processed/`
  - `training/`: Final training-ready tables.
  - `evaluation/`: Held-out and scoring tables.

Each subdirectory also contains a short README to explain what belongs there.
