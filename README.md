<div style="background-color: #ffffff; color: #000000; padding: 10px;">
<img src="assets/logo_aisc_bmftr.jpg">
<h1>Workshop: Agentic RAG</h1>
</div>

Build an agentic Retrieval-Augmented Generation (RAG) system using LLM tool-calling, Qdrant vector search, and Chainlit.

## Architecture

```
workshop-agentic-rag/
  app/            # Chainlit chat app with RAG agent
  ingestion/      # Chainlit UI for document upload & vector ingestion
  k8s/            # Kubernetes deployment manifests
  Dockerfile      # Multi-target build (app / ingestion)
  system.md       # System prompt for the RAG agent
```

### Components

| Service | Description | Port |
|---------|-------------|------|
| **app** | Chainlit chat interface with LLM agent + Qdrant RAG tool | 8000 |
| **ingestion** | Chainlit upload UI — parse docs via Docling, embed, store in Qdrant | 8001 |
| **Qdrant** | Vector database for document embeddings | 6333 |
| **PostgreSQL** | Chat thread persistence & authentication | 5432 |
| **LiteLLM** | LLM proxy for chat and embedding models | 4000 |

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Access to a LiteLLM proxy (or OpenAI-compatible API)

### 1. Configure

```bash
# Chat app
cp app/.env.example app/.env
# Edit app/.env with your LiteLLM endpoint and API key

# Ingestion app
cp ingestion/.env.example ingestion/.env
# Edit ingestion/.env with matching Qdrant + LiteLLM settings
```

### 2. Start services

```bash
cd app
docker compose up -d
```

This starts the chat app, PostgreSQL, and Qdrant.

### 3. Ingest documents

Option A — **Upload via UI**: Start the ingestion app and upload PDFs through the browser.

```bash
cd ingestion
docker build -t workshop-ingestion .
docker run --rm --env-file .env -p 8001:8001 workshop-ingestion
```

Option B — **Bulk ingest**: Use the CLI script in `app/` for batch ingestion.

```bash
cd app
python ingest_docling.py --docling-json-dir /path/to/json/exports
```

### 4. Chat

Open http://localhost:8000 and start asking questions.

## Kubernetes Deployment

See [k8s/](k8s/) for deployment manifests.

## License

See [LICENSE](LICENSE).

---

## Acknowledgements
<img src="assets/logo_bmftr_de.png" alt="drawing" style="width:170px;"/>

The [AI Service Centre Berlin Brandenburg](http://hpi.de/kisz) is funded by the [Federal Ministry of Research, Technology and Space](https://www.bmbf.de/) under the funding code 16IS22092.
