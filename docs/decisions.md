# Decisions Log

## 7/16/26 - Use a virtual environment to run python scripts.

Using a virtual environment (Python 3.13) ensures that package versions do not conflict with other projects. Dependencies are tracked in requirements.txt (generated via pip freeze) and committed to the repo, while the venv folder itself is excluded from version control, making it easy for others to replicate the exact environment the code was run on. Disk space is not a concern for this project.