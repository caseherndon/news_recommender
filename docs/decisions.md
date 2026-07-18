# Decisions Log

## 7/16/26 - Use a virtual environment to run python scripts.

Using a virtual environment (Python 3.13) ensures that package versions do not conflict with other projects. Dependencies are tracked in requirements.txt (generated via pip freeze) and committed to the repo, while the venv folder itself is excluded from version control, making it easy for others to replicate the exact environment the code was run on. Disk space is not a concern for this project.

## 7/17/26 - Consider which users to map to R

Only map users in the train data to R. This means that any "new" users in the dev data will not have profiles. However, the R matrix for the model will not be updated throughout the dev data regardless. At this point, the plan is to recommend based purely on news item popularity for completely new users and match new users with some data with existing users through top-k similarity scoring.