# Chainlit RAG App (GSKI)

This folder contains the Chainlit app plus scripts to:
- export source PDFs with Docling,
- ingest documents into Qdrant,
- run the chat UI.

## Quickstart (recommended)

From `apps/chainlit`:

```bash
cp .env.example .env
# set LITELLM_BASE_URL + LITELLM_API_KEY (+ CHAT_MODEL/EMBED_MODEL if needed)
make up
make logs
```

Open:
- `http://localhost:8000`

Useful day-to-day commands:

```bash
make down         # stop stack
make reingest     # force recreate + reingest collection
make ps           # show service status
```

If you do not want `make`, run directly:

```bash
docker compose up -d --build
docker compose logs -f chainlit
```

## 1) Prerequisites

- Python 3.12+
- Docker (recommended for Qdrant)
- A running LiteLLM endpoint (for chat + embeddings)

From this folder:

```bash
cd apps/chainlit
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Copy env file and fill values:

```bash
cp .env.example .env
```

Required in `.env`:
- `LITELLM_BASE_URL`
- `LITELLM_API_KEY`
- `CHAT_MODEL`
- `EMBED_MODEL`
- `QDRANT_URL` (default `http://localhost:6333`)
- `QDRANT_COLLECTION`
- `CHAT_DB_PATH` (optional, default `./.chainlit/chat_history.sqlite3`)
- `CHAT_EXPORT_DIR` (optional, default `./.files/chat_exports`)

## 2) Start Vector DB (Qdrant)

Recommended (Docker):

```bash
docker run -d \
  --name qdrant \
  -p 6333:6333 \
  -v qdrant_data:/qdrant/storage \
  qdrant/qdrant
```

Check health:

```bash
curl http://localhost:6333/collections
```

## 2b) Start everything with Docker Compose (recommended)

From `apps/chainlit`:

```bash
cp .env.example .env
# set LITELLM_BASE_URL and LITELLM_API_KEY in .env
docker compose up --build
```

This starts:
- `chainlit` on `http://localhost:8000`
- `postgres` on `localhost:5432` (native Chainlit thread persistence)
- `qdrant` on `http://localhost:6333`
- `langflow` on `http://localhost:7860`
- `ingest` one-shot container (runs `ingest_docling.py` before Chainlit starts)

Note:
- Inside Docker, services talk to each other via service DNS names (`qdrant`, `postgres`), not `localhost`.
- Compose forces `QDRANT_URL=http://qdrant:6333` for `chainlit` and `ingest`.

For native left sidebar history, keep these env vars set:
- `DATABASE_URL`
- `CHAINLIT_AUTH_SECRET`
- `CHAINLIT_AUTH_USERNAME`
- `CHAINLIT_AUTH_PASSWORD`

Optional Langflow routing:
- `LANGFLOW_ENABLED=true`
- `LANGFLOW_BASE_URL=http://langflow:7860` (Docker) or `http://localhost:7860` (local)
- `LANGFLOW_FLOW_ID=<your_flow_id>`
- `LANGFLOW_API_KEY=<optional_api_key>`

When enabled:
- Chainlit bypasses the local `rag_retrieve` tool loop for user chat turns.
- Langflow becomes responsible for retrieval + answer generation.
- Chainlit still extracts `Anschlussfragen`, renders citations, persists chat state, and restores citation history on resume.
- If Langflow is unavailable or returns unusable citation data, Chainlit falls back to the current LiteLLM + local `rag_retrieve` path for that turn.

Langflow flow contract:
- Return assistant text that already contains in-text citations like `Quelle 1: ... (S.x-y)`.
- Return the final `Anschlussfragen:` block in the assistant text.
- Return structured citation metadata that Chainlit can map into the current citation sidebar.

Expected structured citation fields:
- `citation_number`
- `file`
- `page_start`
- `page_end`
- `section_title`
- `evidence`

The flow can use returned source documents internally, but prose-only references are not enough to preserve the existing citation UI.

Auto-ingestion controls (in `.env`):
- `INGEST_DOCLING_JSON_DIR` (default `/data/data_docling_json_ocr`)
- `INGEST_RECREATE=true|false` (recreate Qdrant collection)
- `INGEST_BATCH_SIZE`
- `INGEST_MAX_BATCH_CHARS`

Notes:
- Chainlit waits for the ingest service (`depends_on: service_completed_successfully`).
- Ingest auto-skips when the target collection already exists.
- Set `INGEST_RECREATE=true` to force re-ingestion.

## 3) Export PDFs with Docling (optional, but recommended)

Export JSON (keeps page/provenance metadata):

```bash
source .venv/bin/activate
python export_docling_md.py \
  --pdf-dir ../../data/data_raw \
  --out-dir ../../data/data_docling_json_ocr \
  --format json \
  --device cpu \
  --ocr \
  --ocr-engine tesseract \
  --ocr-lang eng deu \
  --pretty-json \
  --skip-existing
```

Notes:
- Use `--ocr-engine mac` on macOS if you want Vision OCR.
- Keep `--format json` for citation/page metadata.

## 4) Ingest into Qdrant

### Option A: Ingest Docling JSON (current primary flow)

```bash
source .venv/bin/activate
python ingest_docling.py \
  --docling-json-dir ../../data/data_docling_json_ocr \
  --collection ${QDRANT_COLLECTION:-grundschutz} \
  --recreate \
  --batch-size 256 \
  --max-batch-chars 20000
```

Use this for section/page-aware metadata and better citations.

### Option B: Ingest preprocessed JSON (`data_preprocessed`)

```bash
source .venv/bin/activate
python ingest.py --source all --recreate --batch-size 256
```

## 5) Start Chainlit

From `apps/chainlit`:

```bash
source .venv/bin/activate
chainlit run app.py -w
```

Open:
- `http://localhost:8000`

Use a different port (example `8001`) if `8000` is occupied:

```bash
source .venv/bin/activate
chainlit run app.py -w --port 8001
```

Then open:
- `http://localhost:8001`

## 6) Typical workflow

1. Start Qdrant
2. Export Docling JSON (if source PDFs changed)
3. Ingest (`ingest_docling.py --recreate`)
4. Start Chainlit

## 7) Chat history + export (new)

Two history layers are available:

- Native Chainlit thread history (left sidebar): backed by Postgres (`DATABASE_URL`) + login.
- Local SQLite export helper: used by slash commands (`/history`, `/export`) from earlier setup.

- DB file (default): `apps/chainlit/.chainlit/chat_history.sqlite3`
- Export folder (default): `apps/chainlit/.files/chat_exports`

In the chat UI:

- `/history` -> list recent saved chat sessions
- `/history <session_id>` -> show last messages from one session
- `/export` -> export current chat as OpenAI-format JSON (`messages` with `role` + `content`, roles: `user`, `assistant`, `tool`)
- `/export <session_id>` -> export one session as OpenAI-format JSON (`system` is excluded)
- `/export all` -> export all chats as OpenAI-format JSONL

Native sidebar export button:
- A custom left-sidebar button **Export all chats** is injected via `public/custom.js`.
- It triggers `/export all` and returns an OpenAI-format JSONL export.

CLI export:

```bash
source .venv/bin/activate
python export_chats.py --format all
```

Examples:

```bash
# Export one session as JSON
python export_chats.py --session-id <session_id>

# Export all sessions as JSONL only
python export_chats.py --format jsonl

# Export all sessions as CSV only
python export_chats.py --format csv
```

## 8) Useful env knobs

- `TOP_K` retrieval size
- `MAX_SOURCE_LINKS` limits how many PDF source links are shown (default `8`)
- `DATA_RAW_DIR` is the source directory for PDFs served via `/sources/pdf/{file_name}`
- `STARTER_QUESTIONS` for Chainlit starter prompts (`||` separated)

## 9) Troubleshooting

- **`Connection refused` to Qdrant**  
  Qdrant not running or wrong `QDRANT_URL`.

- **Embedding 400 / context window exceeded**  
  Lower `--max-batch-chars` and/or `--batch-size` during ingest.

- **Large payload 400 from Qdrant**  
  Lower ingest batch size (e.g. 256 -> 128).

- **Port 8000 already in use**  
  Stop existing Chainlit process or run with another port, e.g.:
  `chainlit run app.py -w --port 8001`

- **PDF viewer/citation mismatches**  
  Re-ingest with `--recreate` and ensure `DATA_RAW_DIR` points to the correct PDFs.

- **`401 Unauthorized` on `/sources/pdf/...`**  
  Log in again and verify the request is sent with an authenticated Chainlit session.

- **`404 Source PDF not found` on `/sources/pdf/...`**  
  Ensure the requested filename exists as a `.pdf` directly under `DATA_RAW_DIR`.
