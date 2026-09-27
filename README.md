# Zepto AI/ML Capstone

An end-to-end capstone with three independently runnable modules: a book catalog data pipeline, Titanic analytics and modeling, and a grounded Zepto policy assistant.

Each module keeps its own dependencies and documentation. The required baseline runs locally without a paid API key.

## Quick start

Use Python 3.11 or newer from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r data_pipeline/requirements.txt
python -m pip install -r analytics/requirements.txt
python -m pip install -r support_assistant/requirements.txt
```

The support assistant downloads the `all-MiniLM-L6-v2` embedding model the first time it starts. The model and ChromaDB index are then used locally.

## Modules

### 1. Data pipeline

```powershell
python data_pipeline/pipeline.py
```

The pipeline discovers category links from Books to Scrape, paginates listings, cleans numeric and availability fields, converts GBP to INR at `1 GBP = 105.50 INR`, loads normalized SQLite tables, runs saved SQL queries, and checks the SQL join against a pandas merge.

Use `python data_pipeline/pipeline.py --offline` for a deterministic demonstration run without network access. See [data_pipeline/README.md](data_pipeline/README.md) for the data model and generated files.

### 2. Analytics

```powershell
python analytics/analysis.py
```

Use `python analytics/analysis.py --offline` to run against the committed Titanic CSV. The workflow profiles missingness, creates EDA plots, compares three classifiers, evaluates fare regression, tunes a random forest, and saves a complete preprocessing-plus-model pipeline under `analytics/outputs/`.

See [analytics/README.md](analytics/README.md) for the measured findings and model interpretation.

### 3. Support assistant

```powershell
$env:MOCK_LLM = "1"
python -m uvicorn main:app --app-dir support_assistant --host 127.0.0.1 --port 7860
```

Ask the API with:

```powershell
Invoke-RestMethod -Method Post `
	-Uri http://127.0.0.1:7860/ask `
	-ContentType "application/json" `
	-Body '{"query":"How long does delivery take?"}'
```

Policy questions use MiniLM embeddings and ChromaDB retrieval. General questions receive a clear policy-only response. Run `python support_assistant/smoke_test.py` to exercise both graph branches without starting the server. See [support_assistant/README.md](support_assistant/README.md) for the optional OpenAI-compatible LLM configuration and Docker commands.

## Engineering notes

- The data pipeline uses a normalized `categories` table and a foreign key from `books`.
- Modeling transformations are fitted inside scikit-learn pipelines after the train/test split to avoid leakage.
- The assistant keeps retrieval local, validates API responses with Pydantic, and uses LangGraph for explicit intent routing.
- Offline modes are labeled as demonstrations; the normal pipeline paths remain available for the required live or downloaded datasets.

## Verification

From the repository root:

```powershell
python data_pipeline/pipeline.py --offline
python analytics/analysis.py --offline
$env:MOCK_LLM = "1"
python support_assistant/smoke_test.py
```

The expected checks are 60 offline books across four categories, regenerated analytics outputs, and two valid assistant responses with source IDs for the policy query.
