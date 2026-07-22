# Decisions Log

## 7/16/26 - Use a virtual environment to run python scripts.

Using a virtual environment (Python 3.13) ensures that package versions do not conflict with other projects. Dependencies are tracked in requirements.txt (generated via pip freeze) and committed to the repo, while the venv folder itself is excluded from version control, making it easy for others to replicate the exact environment the code was run on. Disk space is not a concern for this project.

## 7/17/26 - Consider which users to map to R

Only map users in the train data to R. This means that any "new" users in the dev data will not have profiles. However, the R matrix for the model will not be updated throughout the dev data regardless. At this point, the plan is to recommend based purely on news item popularity for completely new users and match new users with some data with existing users through top-k similarity scoring.

## 7/18/26 - Store processed data as parquet files

Processed data is stored as Parquet files because they are columnar, compress well, and embed schema/type metadata directly in the file, avoiding CSV's need to re-infer types on every load. This also enables efficient column-selective and batched reads, which matters given the streaming/partitioned scale of this project (155 train parts, ~77M rows). ORC offers similar columnar benefits but has weaker Python/PyArrow support compared to Parquet's ecosystem maturity.

## 7/18/26 - Combine train and dev news metadata in one dataset

Because news items are so heavily shared between train and dev datasets, it's more efficient to combine them together. This also allows for a check to make sure the metadata matches between both the train and dev news datasets.

## 7/20/26 - Filter train/dev interaction tables to positives only before building sparse matrices

`train_with_news` / `dev_with_news` are built by merging *all* parsed interaction rows (history + impressions, both click=0 and click=1) with news metadata. Before constructing the CSR interaction matrices, rows must be filtered to only `source == "history"` or (`source == "impression"` and `click == 1`). Otherwise click=0 (exposed-but-not-clicked) impressions would be counted as positive interactions. This filter is applied at matrix-construction time rather than at merge time, so `train_with_news`/`dev_with_news` retain full impression logs (including click=0) for potential future use, e.g. negative sampling.

## 7/21/26 - Define popularity as raw train interaction count

Popularity is defined as the raw count of train interactions per item, not click-through rate or recency-weighted score. This keeps the baseline simple and static, serving as a floor for comparison against later models rather than a competitive recommender itself.

## 7/21/26 - Exclude already-seen items using train data only

Recommendations exclude items a user has already interacted with in train, not dev. This keeps the baseline consistent with a real deployment scenario, where only past (train-time) interactions would be known at recommendation time.

## 7/21/26 - Cold-start users return an empty recommendation list

Users not present in `user_idx_map` (i.e., not in train) return an empty list from `recommend_popularity` rather than falling back to unpersonalized top-K. This keeps the popularity baseline strictly warm-start for now; cold-start handling is deferred to a later phase (KNN-based or content-based fallback).