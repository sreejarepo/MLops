# Migration notes — what to change in your repo

## 1. Fix the README merge conflict
Replace README.md with the clean version in this package, or run:
    git checkout main -- README.md   # then edit
Better: overwrite with the provided README.md and commit.

## 2. Purge committed junk (run once, from repo root)
    git rm -r --cached mlruns mlflow.db .env .pytest_cache .ruff_cache 2>/dev/null
    git rm -r --cached $(git ls-files | grep __pycache__) 2>/dev/null
    git commit -m "Remove tracked runtime artifacts (mlruns, db, caches, env)"
.env, mlruns/ and mlflow.db are already in .gitignore — after the purge they
will stay untracked.

## 3. Add new files
    cp -r mlops-updates/notebooks/01_eda.ipynb   notebooks/
    cp    mlops-updates/src/tune.py              src/
    cp    mlops-updates/src/inference.py         src/
    cp    mlops-updates/README.md                README.md
    cp    mlops-updates/.github/workflows/ci.yml   .github/workflows/ci.yml
    cp    mlops-updates/.github/workflows/tune.yml .github/workflows/tune.yml
    cp    mlops-updates/screenshots/architecture_diagram.png screenshots/
Fill report/REPORT.md with your numbers and screenshots, export to PDF.

## 4. Add a tuning test (optional, keeps CI honest)
Create tests/test_tune.py:
    from src.preprocessing import make_models
    def test_param_grids_match_pipelines():
        from src.tune import PARAM_GRIDS
        models = make_models(__import__("pandas").DataFrame({"age":[1],"sex":[1]}))
        assert set(PARAM_GRIDS) == set(models)

## 5. New commands to verify everything
    python -m src.train
    python -m src.tune
    python -m src.inference --sample
    jupyter nbconvert --to notebook --execute notebooks/01_eda.ipynb
