# Module 3: Zepto Support Assistant

This module is a small policy assistant for Zepto. It keeps the useful part of a RAG system local: policy documents are embedded and searched in ChromaDB, while the default mock response makes the project easy to run without an API key or an LLM service.

## Setup

From this directory, install the module dependencies:

```powershell
python -m pip install -r requirements.txt
```

The first run may download `all-MiniLM-L6-v2` if it is not already cached. Once available, the eight policy documents are embedded into the persistent `chroma_db/` directory. The code requires embeddings and ChromaDB retrieval in both mock and real-LLM modes; if the embedding model cannot load, startup fails rather than using keyword search.

## Run the API

```powershell
$env:MOCK_LLM = "1"
python -m uvicorn main:app --host 127.0.0.1 --port 7860
```

Send requests to `POST http://127.0.0.1:7860/ask`:

```powershell
Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:7860/ask `
  -ContentType "application/json" `
  -Body '{"query":"How long does delivery take?"}'

Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:7860/ask `
  -ContentType "application/json" `
  -Body '{"query":"What is the weather today?"}'
```

The two graph branches can be checked without starting a server:

```powershell
python smoke_test.py
```

## Mock-mode example responses

With `MOCK_LLM` unset or set to `1`, `python smoke_test.py` exercises both graph paths using the real MiniLM embeddings and ChromaDB index. The current verified raw JSON responses are:

Policy retrieval (`How long does delivery take?`):

```json
{"answer":"Based on the retrieved context: Delivery Policy: \"Zepto delivers grocery and household essentials to serviceable pin codes within 10 to 30 minutes of order confirmation, depending on the customer's delivery zone and current order vo","sources":["doc_01","doc_02","doc_04"],"confidence":1.0}
```

General question (`What is the weather today?`):

```json
{"answer":"I can only answer questions about Zepto policies right now.","sources":[],"confidence":1.0}
```

The policy response retrieved `doc_01` for the delivery query. Source ordering may vary if the embedding model or corpus changes.

## Architecture

The pipeline in `main.py` follows ingestion -> embedding -> retrieval -> generation:

1. **Ingestion:** `PolicyIndex._read_documents()` loads `docs/doc_01.txt` through `docs/doc_08.txt`. Each document is used as one chunk because the policy files are short. The file stem is the chunk/document ID.
2. **Embedding:** `PolicyIndex._build()` embeds all eight chunks with `all-MiniLM-L6-v2` and stores the normalized vectors in the persistent ChromaDB collection `zepto_policies` under `chroma_db/`.
3. **Retrieval:** the LangGraph node `retrieve_and_answer` embeds each policy query and asks ChromaDB for the top three nearest chunks using the collection's cosine distance configuration.
4. **Generation:** in mock mode, `retrieve_and_answer` creates `Based on the retrieved context: ...` from the first result. For general questions, `direct_answer` returns the fixed policy-only response. Both paths validate the final result with the `AskResponse` Pydantic model.

The flow is:

```text
POST /ask -> graph -> classify_intent
                         | policy_question -> retrieve_and_answer -> AskResponse
                         | general_question -> direct_answer         -> AskResponse
```

`classify_intent`, `retrieve_and_answer`, and `direct_answer` are the three LangGraph nodes. The conditional edge is implemented by `route()`. In the default mock mode, intent classification uses the required keyword heuristic and the answer nodes make no LLM call. Retrieval and embeddings still run locally for policy questions.

When `MOCK_LLM=0`, the optional real-LLM path uses the OpenAI-compatible Groq endpoint configured by `GROQ_API_KEY`, `LLM_BASE_URL`, and `LLM_MODEL`. Classification and final generation then use the LLM, while ChromaDB retrieval remains local. `PROMPT_TEMPLATE` contains the required role, context, task, format, length, negative constraint, and few-shot example. `real_response()` validates JSON and retries malformed LLM output up to two additional times before returning a marked error response.

## Docker

Build and run locally:

```powershell
docker build -t zepto-support .
docker run --rm -p 7860:7860 -e MOCK_LLM=1 zepto-support
```

The container serves the same `POST /ask` endpoint on `http://127.0.0.1:7860`. The Dockerfile copies the policy documents into the image; the ChromaDB index is created from those documents when the app starts.
