# Module 3 – Policy Support Assistant

This module provides a local policy-question service built with sentence-transformer embeddings, ChromaDB, LangGraph and FastAPI. The default configuration is deterministic and does not require a provider key.

The service accepts a question through `POST /ask`, classifies it, retrieves relevant Zepto policy documents when required and returns a validated JSON response.

## Project structure

```text
support_assistant/
├── docs/
│   ├── doc_01.txt
│   ├── doc_02.txt
│   ├── doc_03.txt
│   ├── doc_04.txt
│   ├── doc_05.txt
│   ├── doc_06.txt
│   ├── doc_07.txt
│   └── doc_08.txt
├── main.py
├── graph.py
├── rag.py
├── prompt.py
├── schemas.py
├── test_app.py
├── requirements.txt
├── Dockerfile
└── README.md
```

## Default mode

The default value of `MOCK_LLM` is `1`. In this mode:

- Intent classification uses the required keyword rules.
- Policy retrieval still uses real local embeddings and ChromaDB similarity search.
- Policy answers use the most relevant retrieved document excerpt.
- General questions receive a fixed response.
- No provider key is required.
- No provider request is made.

Setting `MOCK_LLM=0` enables the optional provider-backed path. This path requires `GROQ_API_KEY`. It is not required for the default project workflow.

## Installation

From the repository root, create a virtual environment:

```bash
python -m venv .venv
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install the dependencies:

```bash
pip install -r support_assistant/requirements.txt
```

The `all-MiniLM-L6-v2` embedding model is downloaded the first time it is used and then loaded from the local cache.

## Running the service

Change to the module directory and start FastAPI:

```bash
cd support_assistant
MOCK_LLM=1 uvicorn main:app --host 0.0.0.0 --port 7860
```

On Windows PowerShell:

```powershell
cd support_assistant
$env:MOCK_LLM="1"
uvicorn main:app --host 0.0.0.0 --port 7860
```

API documentation is available at:

```text
http://localhost:7860/docs
```

## Request and response schema

The endpoint accepts:

```json
{
  "query": "What is the delivery fee?"
}
```

Every response is validated with the `AskResponse` Pydantic model:

```json
{
  "answer": "string",
  "sources": ["document ids"],
  "confidence": 1.0
}
```

`confidence` must be between 0 and 1. General-question responses return an empty `sources` list.

## LangGraph workflow

The graph uses a `TypedDict` state and the following three nodes:

1. `classify_intent` classifies the request as `policy_question` or `general_question`.
2. `retrieve_and_answer` retrieves the three closest policy documents and creates a grounded response.
3. `direct_answer` handles questions unrelated to the policy corpus.

A conditional edge after `classify_intent` sends policy questions to `retrieve_and_answer` and other questions to `direct_answer`. Both answer nodes then finish at the graph's `END` state.

In default mode, a query is classified as a policy question when it contains one of these required keywords:

```text
delivery, return, refund, membership, tracking, cancel,
gift card, support hours
```

## Retrieval pipeline architecture

### 1. Ingestion

`rag.py` reads the eight files from the `docs` directory. Each short policy document is treated as one chunk and uses its filename as the chunk ID, such as `doc_01`.

### 2. Embedding

`PolicyStore` loads the local `all-MiniLM-L6-v2` sentence-transformer model. It converts every policy document into a normalized vector and stores the vectors, document text and source IDs in the ChromaDB collection named `zepto_policies`. ChromaDB is configured to use cosine distance.

### 3. Retrieval

For a policy question, `retrieve_and_answer` converts the query into an embedding with the same model. ChromaDB returns the three most similar policy chunks and their source IDs.

### 4. Generation

With `MOCK_LLM=1`, `retrieve_and_answer` builds a deterministic answer from the first 200 characters of the highest-ranked chunk. `direct_answer` returns a fixed message for unrelated questions. With `MOCK_LLM=0`, these nodes use the structured prompt from `prompt.py`.

Intent classification and answer generation branch on `MOCK_LLM`. Document ingestion, embedding and ChromaDB retrieval operate the same way in both modes.

## Structured prompt

The optional provider-backed prompt in `prompt.py` contains all required sections:

- Role: defines the service as a Zepto policy support assistant.
- Context: supplies only the retrieved policy excerpts.
- Task: asks for an accurate answer based on those excerpts.
- Format: requires an `answer`, `sources` and `confidence` JSON object.
- Length: limits the answer to fewer than 100 words.

The prompt includes the negative constraint:

```text
Do not answer using information that is not present in the provided context.
```

It also contains a complete example question and JSON response about the standard-delivery threshold.

If an optional provider response does not match the Pydantic schema, the application sends a correction instruction and retries up to two additional times. After three unsuccessful attempts, it returns a clearly marked error response with zero confidence.

## Tested example 1 – Policy question

Request:

```bash
curl -X POST http://localhost:7860/ask \
  -H "Content-Type: application/json" \
  -d '{"query":"What is the delivery fee?"}'
```

Raw response:

```json
{"answer":"Based on the retrieved context: Zepto delivers grocery and household essentials to serviceable pin codes within 10 to 30 minutes of order confirmation, depending on the customer's delivery zone and current order volume. Standard del","sources":["doc_01","doc_05","doc_02"],"confidence":1.0}
```

This request follows the `policy_question` route. The highest-ranked result is `doc_01`, which contains the delivery-fee policy.

## Tested example 2 – General question

Request:

```bash
curl -X POST http://localhost:7860/ask \
  -H "Content-Type: application/json" \
  -d '{"query":"What is the capital of Canada?"}'
```

Raw response:

```json
{"answer":"I can only answer questions about Zepto policies right now.","sources":[],"confidence":1.0}
```

This request follows the `general_question` route. It does not run document retrieval and returns an empty source list.

## Running the tests

From the `support_assistant` directory:

```bash
MOCK_LLM=1 pytest -q
```

The tests verify that:

- All eight policy documents are stored in ChromaDB.
- A delivery question retrieves `doc_01` as the first result.
- The policy answer follows the required retrieved-context template.
- A general question uses the direct-answer route.
- Both responses match the required Pydantic schema.

## Docker

Build the image from the module directory:

```bash
docker build -t zepto-policy-support .
```

Run the container:

```bash
docker run --rm -p 7860:7860 zepto-policy-support
```

The Docker image downloads and caches the embedding model during the build. The running container uses `MOCK_LLM=1` by default and serves `POST /ask` on port 7860.
