# Diabetes Risk Predictor: an end-to-end MLOps + DevOps project (v1)

A public web app that estimates a person's risk of diabetes or prediabetes. It's a
learning project that demonstrates, in one small system:

- **MLOps**: data validation, model selection (bias vs. variance), experiment tracking, a model
  registry, a quality gate, an orchestrated pipeline, serving, drift monitoring and automatic retraining.
- **DevOps**: everything as code, reproducible environments, pre-commit checks, pull requests with
  branch protection, CI with test coverage, container smoke tests, security scanning, automated
  dependency updates, infrastructure as code, automated deployment, production checks, releases and rollback.

Every file is commented for novices, and two classroom notebooks (`demo/`) present the whole thing.

Data: the CDC Diabetes Health Indicators dataset (BRFSS 2015 survey, ~250,000 people).

> This is an educational project, not a medical device.

## How it fits together

```
 MLOps (the model)                                DevOps (the software)
 ─────────────────                                ─────────────────────
 User ─▶ Streamlit UI ─▶ FastAPI /predict          💻 change on a branch
                              │                         │ pre-commit hooks (format, lint, lockfile)
                              ▼                         ▼
                   prediction log (SQLite)         🔀 pull request
                              │                         │ CI: checks · tests+coverage · model gate
                              ▼                         │     · security scan · containers+smoke test
          Prefect monitoring flow (PSI drift)           ▼
                              │ drift + new labels  ✅ merge into protected main
                              ▼                         │
          Prefect training flow ─▶ MLflow registry      ▼
                              │ quality gate +      🚀 Render deploys after checks pass (render.yaml)
                              │ beats champion?     🌐 Streamlit Cloud deploys main
                              ▼                         │
          models/ updated ─▶ commit ─▶ (DevOps path)    ▼
                                                    🔍 daily production smoke test · 🤖 Dependabot updates
```

## Project structure

```
diabetes-mlops/
├── src/                       # reusable logic (tested in tests/)
│   ├── config.py              # ALL tunable project settings (start here to change anything)
│   ├── schema.py              # data contract: features, allowed ranges, the shared example input
│   ├── data.py                # download, validate, split, assemble training data
│   ├── models.py              # candidate model definitions
│   ├── evaluation.py          # metrics, cross-validation, bias/variance, learning curve
│   ├── quality_gate.py        # minimum standard for promoting AND deploying a model
│   ├── registry.py            # MLflow registration + promotion to production
│   ├── drift.py               # PSI drift detection
│   ├── timing.py              # step timings printed and logged by the pipeline
│   └── prediction_log.py      # where the API logs predictions (read by monitoring)
├── pipeline.py                # Prefect training flow (orchestration only)
├── monitor.py                 # Prefect monitoring flow (drift -> retrain)
├── simulate_traffic.py        # replays "future users" through the API
├── api/main.py                # FastAPI service (/health, /predict)
├── ui/app.py                  # Streamlit web app for end users
├── scripts/
│   ├── check_model.py         # runs the quality gate on models/ (CI "model-gate" job)
│   └── smoke_test.py          # checks a running deployment (CI + daily production check)
├── demo/
│   ├── mlops_demo.ipynb       # classroom demo: the MLOps loop (~9 min)
│   ├── devops_demo.ipynb      # classroom demo: the DevOps loop (~8 min)
│   ├── common.py              # shared demo machinery (modes, fallback, display)
│   ├── mlops_helpers.py       # MLOps demo steps + snapshot/reset
│   ├── devops_helpers.py      # DevOps demo steps
│   ├── baseline/              # the demos' starting model (committed)
│   └── fallback/              # saved rehearsal results (committed)
├── tests/                     # pytest unit tests (synthetic data, temporary folders)
├── models/                    # the production model (committed: Git history = model releases)
├── .github/
│   ├── workflows/ci.yml       # CI: 5 jobs on every pull request and merge
│   ├── workflows/production-check.yml   # daily smoke test of the deployed app
│   ├── dependabot.yml         # weekly dependency update pull requests
│   └── pull_request_template.md
├── .pre-commit-config.yaml    # checks that run before every commit
├── render.yaml                # infrastructure as code for the API on Render
├── Dockerfile, ui/Dockerfile, docker-compose.yml, .dockerignore
├── .vscode/                   # shared VS Code settings + recommended extensions
├── quick.env                  # switches any command to the quick run profile
├── pyproject.toml             # dependencies + tool settings (ruff, pytest, coverage, uv)
├── uv.lock                    # exact versions of every package (generated, committed)
└── .python-version            # Python version for this project (3.13)
```

## Which file does which stage

**MLOps**

| Stage | File(s) | Tool |
|---|---|---|
| Environment + dependencies | `pyproject.toml`, `uv.lock`, `.python-version` | uv |
| Data ingestion and validation | `src/data.py`, `src/schema.py` | pandas |
| Model selection | `src/models.py`, `src/evaluation.py` | scikit-learn |
| Experiment tracking + registry | `pipeline.py`, `src/registry.py` | MLflow |
| Quality gate | `src/quality_gate.py` | (shared by pipeline and CI) |
| Orchestration | `pipeline.py`, `monitor.py` | Prefect |
| Serving + user interface | `api/main.py`, `ui/app.py` | FastAPI, Streamlit |
| Monitoring + retraining | `src/drift.py`, `monitor.py` | PSI, Evidently, Prefect |

**DevOps**

| Practice | File(s) | Tool |
|---|---|---|
| Everything as code, one home per setting | `src/config.py`, `pyproject.toml`, workflow files | |
| Checks before commit | `.pre-commit-config.yaml` | pre-commit, ruff, uv |
| Pull requests + branch protection | `.github/pull_request_template.md`, GitHub ruleset | GitHub |
| Continuous integration | `.github/workflows/ci.yml` | GitHub Actions |
| Test coverage | `pyproject.toml` (`[tool.coverage]`) | pytest-cov |
| Containers + smoke test | `Dockerfile`, `ui/Dockerfile`, `scripts/smoke_test.py` | Docker |
| Security scanning | `ci.yml` (security + containers jobs) | Trivy |
| Dependency updates | `.github/dependabot.yml` | Dependabot |
| Infrastructure as code + deployment | `render.yaml` | Render |
| Production monitoring | `.github/workflows/production-check.yml` | GitHub Actions |
| Releases + rollback | Git tags, `models/` history | Git, Render |
| Presenting it all | `demo/*.ipynb` | Jupyter |

## Design principles

- **Single source of truth.** Every value and rule is defined once (see the next section).
- **Separation of concerns.** `src/` holds the logic; `pipeline.py`, `monitor.py`, `scripts/` and the demos
  only orchestrate or present it. The quality gate is one module used by both the pipeline and CI.
- **Small, testable functions.** Most functions take data in and return results, so `tests/` can check
  them with tiny synthetic datasets in seconds.
- **Open for extension.** To try a new model, add one line to `MODEL_CLASSES` in `src/models.py` and one
  entry to `CANDIDATE_MODELS` in `src/config.py`.
- **Reproducible environments.** `uv.lock` pins every package; the same Python 3.13 + uv + lockfile are
  used on your laptop, in CI and in Docker (environment parity).
- **Automate the safety net.** Checks run before commit, on every pull request, before every deploy and
  every day in production. Nobody has to remember to run them.
- **Least privilege.** CI workflows can only read the repository; containers run as a non-root user;
  the demo reset can only touch the files it's meant to.
- **Lean, secure deployables.** The API image installs only runtime dependencies.

## One home per kind of setting

No numbers are scattered through the code. Each kind of setting has exactly one home:

| Kind of setting | Its home | Examples |
|---|---|---|
| Project settings | `src/config.py` | data split shares (50/10/40%), drift threshold, quality-gate minimum, smoke-test timeouts, demo speed, quick-run settings |
| Run profile (full or quick) | `quick.env` | `RUN_PROFILE=quick` |
| Fixed facts / data contract | `src/schema.py` | feature names, allowed ranges, the example input |
| Tool settings | `pyproject.toml` | ruff line length, pytest paths, coverage minimum |
| App version | `pyproject.toml` (`[project] version`) | read by `src/config.py`, shown by `/health` |
| CI settings | `env:` block at the top of each workflow file | uv version, image names, scan severity |
| Cloud service settings | `render.yaml` | region, plan, health check path |
| Values only the deployment knows | GitHub repository variables | `DEPLOYED_API_URL`, `DEPLOYED_UI_URL` |
| Secrets | never in Git (GitHub secrets / environment variables) | (none needed in v1) |

Each training run logs all project settings to MLflow as `config.json`.

The data is split only once. After changing `TRAIN_SHARE`, `TEST_SHARE` or `STREAM_SHARE`, delete
`data/train.csv`, `test.csv`, `stream.csv`, `stream_offset.txt` and `new_labeled.csv` so the pipeline re-splits.

## The comment standard

Everything is written to be explained to beginners:

- **Every file starts with a header**: what it is, why it exists, how to use it, which concepts it shows.
- **Every setting** says what it is, why it has this value, what changing it would do, and (where useful)
  an everyday analogy.
- **Every CI step** says what happens, why, and what a failure means and how to fix it.
- **`CONCEPT -` comments** mark where an ML, MLOps or DevOps idea is applied.
- **Every function** documents its parameters and return value in plain language.

---

# Part 1: Build and run it (MLOps)

## Step 0: Set up your machine

You need **uv**, **Git** and **Docker Desktop**. You do NOT need to install Python yourself:
uv reads `.python-version` and downloads Python 3.13 automatically.

**1. Install uv** (skip if `uv --version` already works; this project needs 0.13 or newer):

```powershell
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```
```bash
# Mac/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**2. Create the lockfile and the environment** (inside the project folder):

```bash
uv lock     # resolve exact versions of every package -> uv.lock (first time only)
uv sync     # download Python 3.13 if needed, create .venv, install everything
```

**3. Turn on the pre-commit hooks** (once per clone), then run them on everything once.
The first run also formats every file and generates `ui/requirements.txt` from `uv.lock`:

```bash
uv run pre-commit install
uv run pre-commit run --all-files    # run it again if it changed files; the second run should pass
```

**4. Point VS Code at the environment.** `.vscode/settings.json` already points to `.venv`. Accept the
prompt to install the recommended extensions (Python, Jupyter, Ruff, Even Better TOML, YAML).

**How to run commands:** prefix them with `uv run`, e.g. `uv run pytest`. You never need to "activate" anything.

**Your first run:** try a quick run (`uv run --env-file quick.env pipeline.py`, see the next section)
to check everything works in seconds, then follow Steps 1–8 for the full runs.

**Adding a package later:** `uv add <package>` (or `uv add --group pipeline <package>`), then commit
`pyproject.toml`, `uv.lock` and `ui/requirements.txt`. The pre-commit hooks keep the last two in sync.

## Quick runs: check everything works in seconds

A **quick run** executes the whole pipeline end to end on a small slice of the real data, with
lighter settings, so it finishes in seconds instead of minutes. It's a rehearsal: it answers
**"does everything work?"**, while a full run answers **"how good is the model?"**

Use a quick run:
- as your **very first run** on a new machine (it also downloads the dataset, once, for both profiles),
- after changing code, **before** starting a full run,
- while debugging, when you may rerun many times in a row.

**How to switch:** add `--env-file quick.env` to any `uv run` command. The same command works on
Windows, Mac and Linux. Without it, everything runs in the full profile.

```bash
uv run --env-file quick.env pipeline.py                  # quick training run
uv run --env-file quick.env uvicorn api.main:app         # serve the quick model
uv run --env-file quick.env simulate_traffic.py          # 500 ordinary simulated users
uv run --env-file quick.env simulate_traffic.py --drift  # 500 heavier, older users
uv run --env-file quick.env monitor.py                   # drift check (+ retraining)
```

**What data it uses:** the same real dataset, sampled with stratification (the diabetes rate stays
about 14%) and a fixed random seed (every quick run uses the same people):

| | Full run | Quick run |
|---|---|---|
| Rows used from the dataset | all ~253,700 | 20,000 (about 8%) |
| Training / test / future users (50/10/40%) | ~126,800 / ~25,400 / ~101,500 | 10,000 / 2,000 / 8,000 |
| Rows used to compare the six models | 40,000, 5 folds | 5,000, 2 folds |
| Simulated users per batch | 2,000 | 500 |
| Predictions checked for drift / new labels needed to retrain | 2,000 / 2,000 | 500 / 1,000 |

All of these live in `src/config.py`, where each setting shows its full and quick values side by side
(`_by_profile(full=..., quick=...)`).

**It can never touch production.** Quick runs use their own data folder (`data/quick/`), model folder
(`models_quick/`, ignored by Git and Docker), prediction log (`logs/predictions_quick.db`), reports
folder and MLflow experiment and registry name (`diabetes-risk-quick`). Only the raw download
(`data/raw.csv`) is shared. A quick model can't be committed, can't pass through CI (which checks
`models/`) and can't be deployed. `tests/test_quick_mode.py` checks this guarantee.

**A quick model still faces the same quality gate.** With 10,000 training rows, a decent model passes
comfortably; if one failed, that would be a useful warning, not something to hide.

**Step timings:** every run, quick or full, ends with a table showing how long each step took, and logs
each step's duration (`step_seconds_...`) and each model's average training time (`cv_fit_seconds`) to
MLflow. If a run is slow, this shows which step to look at.

## Step 1: Get and validate the data

```bash
uv run python -m src.data
```

This downloads the dataset into `data/raw.csv`, prints the row count and diabetes
rate, and runs the validation checks.

If the download fails, download `diabetes_binary_health_indicators_BRFSS2015.csv`
from Kaggle (search "Diabetes Health Indicators Dataset") and save it as `data/raw.csv`.

**Learn:** open `src/schema.py`. It is the data contract: allowed columns and value
ranges. The validator, the API and monitoring all use it, so they can never disagree.

## Step 2: Run the training pipeline

```bash
uv run pipeline.py
```

The pipeline:
1. splits the data once into `train` (50%), `test` (10%) and `stream` (40%, our pretend future users),
2. validates it,
3. compares six candidate models with 5-fold cross-validation and diagnoses each one,
4. draws a learning curve for the winner,
5. retrains the winner on all training data and scores it on the test set,
6. logs everything to MLflow and registers a new model version,
7. runs the **quality gate** (`src/quality_gate.py`): minimum ROC-AUC, beats guessing the average,
   valid probabilities,
8. compares it with the current production model (none yet, so it wins),
9. promotes it only if it passed the gate AND won, exporting it to `models/` with metadata and a
   reference sample for monitoring.

It takes about a minute. Model comparison uses a 40,000-row sample (`CV_SAMPLE_SIZE` in
`src/config.py`) because k-NN is slow on large data; the winner is still retrained on all rows.

**Learn:** run it a second time. The new version scores the same as the champion,
so it is *not* promoted. That is champion vs. challenger in action.

Optional: watch runs in the Prefect UI.

```bash
uv run prefect server start      # in a second terminal, then open http://127.0.0.1:4200
```

## Step 3: Compare models and read bias vs. variance

The six candidates are chosen to show different points on the bias-variance spectrum:

| Candidate | What it is | Typical behavior |
|---|---|---|
| `logistic_regression` | a weighted sum of features | simple, stable; may underfit |
| `decision_tree_deep` | one tree, no depth limit | memorizes training data: overfits |
| `decision_tree_pruned` | one tree, limited depth | much less overfitting |
| `knn` | votes of the 50 most similar people | `n_neighbors` controls bias vs. variance |
| `neural_network` | small MLP (32 → 16 neurons) | flexible; regularized with `alpha` and early stopping |
| `gradient_boosting` | many small trees correcting each other | usually best on tabular data |

`gradient_boosting` is scikit-learn's `HistGradientBoostingClassifier`. It is the same family
of algorithm as XGBoost and LightGBM, without an extra dependency.

The pipeline prints a table like this (your numbers will differ):

```
               model  train_score  val_score    gap  diagnosis
   gradient_boosting       0.84       0.83     0.01  good balance
      neural_network       0.83       0.83     0.00  good balance
 logistic_regression       0.81       0.81     0.00  high bias (underfitting)
decision_tree_pruned       0.83       0.81     0.02  high bias (underfitting)
                 knn       0.85       0.79     0.06  high variance (overfitting)
  decision_tree_deep       1.00       0.60     0.40  high variance (overfitting)
```

How to read it:
- **train_score**: how well the model fits data it learned from.
- **val_score**: how well it does on held-out folds. This is what matters, and what selection uses.
- **gap** = train − validation. A large gap means the model memorized the training data
  (**high variance / overfitting**). The deep decision tree scores close to a perfect 1.0 on training
  data and much worse on new data: a textbook example.
- A small gap but a score well below the best model means the model is too simple
  (**high bias / underfitting**).
- The thresholds for these labels are `VARIANCE_GAP_THRESHOLD` and `BIAS_SHORTFALL_THRESHOLD`
  in `src/config.py`. They are rules of thumb, not laws.

Open MLflow to see the details:

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

At http://127.0.0.1:5000, expand the latest `train-manual` run: each candidate is a nested
`cv-...` run. Select them and click **Compare**. In the parent run's artifacts you'll find:
- `learning_curve.png`: if the two curves are close and flat, more data won't help much;
  if there's a gap that shrinks as rows increase, more data would help (variance).
- `candidate_comparison.json`: the table above.
- `config.json`: every setting used.

Open **Models → diabetes-risk** to see versions and which one has the `production` alias.

**Try:** in `src/config.py`, change the pruned tree's `max_depth` to 3 (more bias) or 20
(more variance), or k-NN's `n_neighbors` to 5 (more variance) or 500 (more bias), rerun,
and watch the diagnosis change.

If k-NN ever wins, note that its saved model contains all training rows (~20 MB), which
makes the Git repo and Docker image noticeably larger. The pre-commit large-file guard
(2 MB limit) would even stop you committing it. That's a real deployment trade-off.

## Step 4: Serve the model with an API

```bash
uv run uvicorn api.main:app --reload
```

Open http://127.0.0.1:8000/docs, click **POST /predict → Try it out**, and send a request.
Then try a BMI of 500 and see the API reject it with a 422 error.

Also open http://127.0.0.1:8000/health: it reports the app version (from `pyproject.toml`) and
the model version, which is how you confirm which release is running.

**Learn:** every prediction is logged to `logs/predictions.db`. That log is what monitoring reads.

## Step 5: Add the user interface

With the API still running, in a new terminal:

```bash
uv run streamlit run ui/app.py
```

Open http://localhost:8501, fill in the form and check a risk score.

## Step 6: Run everything with Docker

Stop the API and UI (Ctrl+C), then:

```bash
docker compose up --build
```

UI: http://localhost:8501 · API docs: http://localhost:8000/docs

**Learn:** this is exactly what will run in the cloud. "Works on my machine" becomes "works in this container."
Run `docker compose ps`: the API shows as **healthy** thanks to the healthcheck in `docker-compose.yml`,
and the web app only starts once it is.

## Step 7: Simulate users, detect drift, retrain

Start the API locally (`docker compose up` or `uv run uvicorn api.main:app`), then:

**9a. Normal traffic, no drift expected:**
```bash
uv run simulate_traffic.py --n 2000
uv run monitor.py --no-retrain
```
You should see low PSI values and "No significant drift."

**9b. A changed population (heavier and older):**
```bash
uv run simulate_traffic.py --n 2000 --drift
uv run monitor.py
```
You should see **DRIFT DETECTED** for BMI and Age. Because 4,000 new labels have now
"arrived", monitoring triggers the training pipeline, which trains a challenger on
old + new data, runs the quality gate, and compares it with the champion on the most recent labeled data.

Look at:
- `reports/drift_*.json` for PSI scores and `reports/drift_*.html` for the Evidently report,
- the new `train-drift` run in MLflow, with its `promoted` tag,
- whether the challenger was promoted.

**Learn:** the challenger may well *not* be promoted. Shifting everyone's BMI and Age
moves risk scores for everyone in a similar way, so the ranking quality (ROC-AUC) barely changes.
Drift is a warning, not proof the model is worse. That's why the pipeline checks
performance before replacing anything. Compare `eval_brier` with `champion_brier` in MLflow too:
it measures calibration and usually shows the champion getting worse under drift.

To start the simulation over: delete `data/stream_offset.txt`, `data/new_labeled.csv` and `logs/`.

## Step 8: Automate the schedules

With `uv run prefect server start` running in one terminal:

```bash
uv run prefect config set PREFECT_API_URL=http://127.0.0.1:4200/api
uv run pipeline.py --serve     # retrains on the 1st of every month
uv run monitor.py --serve      # checks drift every hour (separate terminal)
```

You can trigger and watch both flows from the Prefect UI.


---

# Part 2: Ship it safely (DevOps)

## Step 9: Put it on GitHub and configure the safety settings

**1. Create the repository and push** (a public repository gets free GitHub Actions minutes):

```bash
git add .
git commit -m "Diabetes risk MLOps + DevOps v1"
git branch -M main
git remote add origin https://github.com/<you>/diabetes-mlops.git
git push -u origin main
```

Check that `uv.lock`, `ui/requirements.txt`, `models/`, `demo/baseline/` and `demo/fallback/` were committed.
Data, logs, `.venv` and MLflow files are excluded by `.gitignore`.

**2. Protect `main`** (Settings → Rules → Rulesets → New branch ruleset):
- **Name:** `protect-main`; **Enforcement status:** Active; **Target branches:** add the default branch.
- Tick **Restrict deletions** and **Block force pushes**.
- Tick **Require a pull request before merging**, with **Required approvals: 0** (you're working solo;
  raise it when others join).
- Tick **Require status checks to pass**, then add these four checks (they appear after CI has run once):
  `Checks (format, lint, lockfile)`, `Tests and coverage`, `Model quality gate`,
  `Containers (build, smoke test, image scan)`.
  The security scan is deliberately not required: it reports, it doesn't block.

**3. Turn on GitHub's free security features** (Settings → Advanced Security, or "Code security"):
- **Dependabot alerts** and **Dependabot security updates**: warnings and fix pull requests for
  vulnerable packages.
- **Secret scanning** and **Push protection**: stops you accidentally pushing a password or key.

**4. Add repository variables** for the production check (after Step 13 gives you the URLs):
Settings → Secrets and variables → Actions → **Variables** → `DEPLOYED_API_URL`, `DEPLOYED_UI_URL`.

## Step 10: Make changes with the pull-request workflow

From now on, changes reach `main` only through pull requests:

```bash
git switch -c improve-readme          # 1. a new branch for one change
# ... edit files ...
git add .
git commit -m "Clarify setup steps"   # 2. pre-commit hooks run here; if they fix files, add and commit again
git push -u origin improve-readme     # 3. send the branch to GitHub
```

4. On GitHub, click **Compare & pull request**. The template's checklist appears; fill it in.
5. Wait for the CI checks. Click any ❌ to read why (each step's comment in `ci.yml` explains failures).
6. When the required checks are ✅, click **Merge**, then delete the branch.
7. Locally: `git switch main` and `git pull`.

## Step 11: Read the CI jobs

`.github/workflows/ci.yml` runs five jobs in parallel. Each is a separate check on the pull request:

| Job | What it checks | If it fails |
|---|---|---|
| Checks (format, lint, lockfile) | the same pre-commit hooks as your machine | run `uv run pre-commit run --all-files`, commit the changes |
| Tests and coverage | all tests pass; coverage ≥ the minimum | fix the failing test, or add tests for the "Missing" lines |
| Model quality gate | the committed model is good enough to serve | retrain (`uv run pipeline.py`) or revert the `models/` change |
| Security scan (warning only) | known vulnerabilities in `uv.lock` | never fails; read the Summary page; merge Dependabot's fixes |
| Containers | images build, start, pass the smoke test; image scan (warning) | read the container logs printed at the end of the job |

Each run's **Summary** page shows the coverage table and the scan results. Full scan reports are
downloadable at the bottom of the run page.

## Step 12: Test coverage

**What it is:** while the tests run, coverage records which lines of our code were executed. "85% coverage"
means 85% of the code ran during the tests. Code no test runs could be broken without anyone noticing.

**Run it locally:**

```bash
uv run pytest --cov                        # table in the terminal (the "Missing" column lists untested lines)
uv run pytest --cov --cov-report=html      # also writes htmlcov/index.html: open it in a browser
```

The HTML report shows each file with untested lines highlighted. (A plain `uv run pytest`, or running
tests from VS Code's Testing panel, skips coverage, so running one test alone doesn't "fail" the minimum.)

**The settings** live in `pyproject.toml` under `[tool.coverage]`, each one commented:
- `source`: what's measured (`src`, `api`, `scripts`, `demo`).
- `omit`: `src/registry.py`, because it needs MLflow, which CI deliberately doesn't install; it's
  exercised every time the pipeline runs.
- `branch = true`: both outcomes of every `if` must be tested, not just each line.
- `fail_under = 80`: the minimum. Below it, CI's tests job fails. Not 100%, because the last few percent
  cost the most and tend to produce meaningless tests. Raise it over time; don't lower it without a reason.
- `# pragma: no cover` in the code marks functions that need live services or real data (the running
  API, the deployed app, the downloaded dataset). Each one says why; they're exercised by CI's container
  smoke test and the demo rehearsals instead.

**The key limitation:** coverage shows which code **ran**, not whether the tests **checked** the right
things. It's a guide for finding untested code, not a goal in itself.

## Step 13: Deploy so real users can access it

**API on Render, from `render.yaml` (infrastructure as code):**
1. Sign up at render.com with GitHub → **New → Blueprint** → pick your repository.
2. Render reads `render.yaml` and shows the service it will create (Docker, free plan, Singapore region,
   health check on `/health`, deploys only after CI checks pass). Click **Apply**.
3. When it's live, open `https://<your-service>.onrender.com/health`.

Changes to `render.yaml`, merged into `main`, update the service. Note: the region can't be changed after
creation.

**Web app on Streamlit Community Cloud** (configured in its dashboard; it has no config file):
1. Sign in at share.streamlit.io with GitHub → **Create app** → pick your repository, branch `main`.
2. Main file path: `ui/app.py`. It installs `ui/requirements.txt` (exported from `uv.lock`).
3. **Advanced settings:** Python **3.13**; under **Secrets** add
   `API_URL = "https://<your-service>.onrender.com"`.
4. Deploy. Because `main` is protected, Streamlit only ever deploys code that passed CI.

Free Render services sleep when idle, so the first request can take up to a minute. Render's free disk is
temporary, so production prediction logs are lost on restart (v2 moves them to a hosted database).

## Step 14: Verify and watch production

- **Right after a release:** open `/health` and check the `app_version` and `model_version`.
- **Every day automatically:** `.github/workflows/production-check.yml` smoke-tests the deployed app at
  07:00 Brisbane time, using the repository variables from Step 9. A failure turns the run red and GitHub
  emails you. Run it any time from **Actions → Production check → Run workflow**.
- **By hand:**
  ```bash
  uv run python -m scripts.smoke_test --api-url https://<your-service>.onrender.com --ui-url https://<your-app>.streamlit.app --deployed
  ```

## Step 15: Releases and rollback

**Versioning:** the version lives in `pyproject.toml` (`version = "1.0.0"`), following
MAJOR.MINOR.PATCH: breaking change . new feature . fix.

**Making a release:**
1. In a pull request, bump the version in `pyproject.toml` and describe the change.
2. Merge it. Render deploys after CI passes.
3. Tag it and publish release notes:
   ```bash
   git switch main && git pull
   git tag v1.1.0
   git push origin v1.1.0
   ```
   On GitHub: **Releases → Draft a new release** → choose the tag → **Generate release notes** → Publish.
4. Verify: `/health` shows the new version; run the Production check workflow.

**Shipping a new model:** when the pipeline promotes a model, `models/` changes. Ship it like code:
a branch, a commit (`Promote model version N`), a pull request. CI's model gate checks it before it can merge.

**Rolling back:**
- **Code, fastest:** Render dashboard → the service → **Events** → pick the last good deploy → **Rollback**.
- **Code, permanently:** `git revert <bad-commit>` in a pull request, then merge.
- **A model:** every model release is a commit touching `models/`. Find it with
  `git log --oneline -- models/`, then `git revert <commit>` in a pull request.

## Step 16: Dependency updates and security scans

- **Dependabot** (`.github/dependabot.yml`) opens pull requests every week: grouped small updates for
  Python packages, plus GitHub Actions and Docker base images. CI tests each one; merge the green ones.
- **scikit-learn is excluded on purpose:** upgrading it means retraining the model. To upgrade: change the
  pin in `pyproject.toml`, `uv lock`, retrain, run the model gate, re-snapshot the demo baseline.
- **Security scans** (Trivy, in CI) check `uv.lock` and the API image. Policy: **warning only**. Findings
  appear on the run's Summary page with a ⚠️ annotation; fixes arrive via Dependabot security updates.

---

# Running the demos (classroom presentations)

Two notebooks present the project to beginners, each with **talking points** (plain-language
explanations that double as a handout) and 🎤 presenter cues for timing and window switches:

- `demo/mlops_demo.ipynb` (~9 min): data, model comparison, serving, drift and automatic retraining,
  running the real Prefect flows.
- `demo/devops_demo.ipynb` (~8 min): settings as code, pre-commit, CI and coverage, the quality gate,
  deployment and smoke tests, keeping production healthy.

Both run the project's real code. Three modes, set at the top: `"live"` (run for real, fall back to a saved
result if a step fails), `"record"` (rehearsal: run for real and save the results) and `"saved"` (run nothing live).

**One-time setup** (after Part 1 works, and Part 2 for the DevOps demo):

1. Save the current production model as the MLOps demo's starting point:
   ```bash
   uv run python -m demo.mlops_helpers snapshot
   ```
2. Open each notebook in VS Code; when asked for a kernel, choose the `.venv` environment.
3. Fill in the settings cell (your deployed URLs).
4. Rehearse each notebook once with `MODE = "record"` (follow the checklist in its section 0).
5. Set `MODE = "live"` again, then reset: `uv run python -m demo.mlops_helpers reset`.
6. Commit `demo/baseline/` and `demo/fallback/` so a fresh clone has the safety net.

**Before presenting:** follow the checklist in section 0 (start the services, open browser tabs, run the
reset and pre-flight cells; all ✅). For the DevOps demo, prepare a small pull request so its CI checks are visible.

**After presenting the MLOps demo:** the drift section may promote a new model. Run
`uv run python -m demo.mlops_helpers reset` so `models/` matches the committed baseline again.

**Making the handouts** (committing notebooks with their outputs):
1. Reset, then **Restart and Run All** in `"live"` mode.
2. The last code cell must say every step ran live.
3. Reset again from the terminal (not the notebook, so its outputs stay as they are).
4. Commit the notebook in a pull request.

Outputs show only paths relative to the project, never your personal folders. CI lints the notebooks but
doesn't run them, because they need the running services.

**If you deliberately retrain and commit a better model later**, update the baseline with
`uv run python -m demo.mlops_helpers snapshot --force`, then re-record the fallbacks.

---

## Ideas for v2

- Build each image once in CI, push it to a registry, and deploy that exact image (no rebuild on Render).
- A staging environment, smoke-tested before production.
- Hosted MLflow (DagsHub) and Prefect, so training and monitoring don't depend on a laptop; continuous
  delivery of models.
- Service monitoring and alerts (response times, error rates), and prediction logs in a hosted database.
- Real later BRFSS years instead of simulated drift; Optuna tuning and XGBoost; weighted and
  sliding-window retraining; SHAP explanations; fairness checks; probability calibration.
- Type checking (mypy), CodeQL code scanning, and blocking security scans.
- Settings in a `config.yaml`, with `uv run pipeline.py --config experiments/x.yaml`.
