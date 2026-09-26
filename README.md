# Zepto AI/ML Capstone

This repository brings together three small, reproducible pieces of an AI/ML workflow: a Books to Scrape data pipeline, a Titanic analysis and modeling workflow, and a policy assistant for Zepto. Each module has its own dependencies and run instructions below.

## Setup

Use Python 3.11 or newer. Each module has a separate `requirements.txt`; install only the requirements for the module you plan to run.

```powershell
python -m pip install -r data_pipeline/requirements.txt
python -m pip install -r analytics/requirements.txt
python -m pip install -r support_assistant/requirements.txt
```

## Data pipeline

From the repository root:

```powershell
# Zepto Data & AI Platform

One repository containing the three capstone modules: scraped catalog data, a Titanic analytics/modeling workflow, and a grounded Zepto policy assistant. Each module has its own `requirements.txt`; install only the module dependencies you plan to run. Python 3.11 or newer is recommended. No paid service or API key is needed for the required baseline.

## Setup

Run from the repository root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r data_pipeline/requirements.txt
python -m pip install -r analytics/requirements.txt
python -m pip install -r support_assistant/requirements.txt
```

The analytics command uses `sns.load_dataset("titanic")` once by default; the committed `analytics/titanic.csv` supports offline evaluation. The support assistant downloads `all-MiniLM-L6-v2` on first use if it is not cached; embedding and ChromaDB retrieval are local afterward.

## Run Modules

### Data Pipeline

```powershell
python data_pipeline/pipeline.py
```

Scrapes and paginates four Books to Scrape categories, cleans fields, converts GBP to INR at the fixed project rate **1 GBP = 105.50 INR**, loads normalized SQLite tables, executes the saved SQL, and verifies the SQL join against an in-memory pandas merge. `python data_pipeline/pipeline.py --offline` runs a clearly labeled synthetic demonstration fixture, not the required live scrape. Details and outputs: [data_pipeline/README.md](data_pipeline/README.md).

### Analytics

```powershell
python analytics/analysis.py
```

`python analytics/analysis.py --offline` uses the committed Titanic CSV. The workflow preserves the raw CSV, reports missingness and EDA findings, writes plots and model comparisons, trains/tunes leakage-safe pipelines, evaluates linear fare regression, and saves/reloads a complete classifier pipeline. Details: [analytics/README.md](analytics/README.md).

### Support Assistant

```powershell
$env:MOCK_LLM = "1"
python -m uvicorn main:app --app-dir support_assistant --host 127.0.0.1 --port 7860
```

Post `{"query":"How long does delivery take?"}` to `http://127.0.0.1:7860/ask`. The default mock path makes no LLM API call; policy questions still use MiniLM embeddings and ChromaDB retrieval. `MOCK_LLM=0` enables the optional OpenAI-compatible endpoint when configured with environment variables. Build locally with `docker build -t zepto-support support_assistant` and run with `docker run --rm -p 7860:7860 zepto-support`. Details, transcripts, and architecture: [support_assistant/README.md](support_assistant/README.md).

## Design Summary

- **Data engineering:** homepage-discovered category URLs avoid stale slugs; malformed numeric fields use median imputation, unparseable availability rows are dropped, and a normalized SQLite foreign key keeps category names out of book rows.
- **Analytics:** a single raw dataset is saved before cleaning; EDA cleaning decisions are threshold-based, while model imputation/encoding/scaling live inside train-fitted scikit-learn pipelines. Classification and regression metrics are reported separately.
- **GenAI:** each exact policy document is embedded as one chunk in a persistent cosine ChromaDB collection. LangGraph routes policy queries to retrieval and general queries to a fixed mock answer; Pydantic validates the response shape.

The GitHub submission is one public repository link for this root and all three module folders. The required feature-branch/merge history is a repository-level workflow item and should be visible in `git log --graph --all`.
