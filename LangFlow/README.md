# Agentic RAG with Langflow — Workshop

A hands-on workshop that walks through five progressive agentic RAG patterns using [Langflow](https://www.langflow.org/), a visual low-code builder for LLM workflows.

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Docker Desktop | [Download](https://www.docker.com/products/docker-desktop/) — the only required install |
| HPI AISC API key | Provided by the workshop organiser |
| (Optional) Tavily API key | Free at [tavily.com](https://tavily.com) — only needed for Flow 5 |

You do **not** need Python, LangChain, or any other library installed locally.

---

## Quickstart (5 minutes)

**1. Copy the environment file and fill in your API key:**

```bash
cd ~/Workshops/LangFlow/setup
cp .env.example .env
# Open .env and replace "your_hpi_aisc_api_key_here" with your real key
```

**2. Start Langflow and Qdrant:**

```bash
docker compose up
```

Wait for the line: `Langflow is ready at http://0.0.0.0:7860`

**3. Open Langflow in your browser:**

```
http://localhost:7860
```

**4. Set your API keys as Global Variables** (do this once, all flows will use them):

- Go to **Settings** (gear icon, bottom-left) → **Global Variables**
- Add `AISC_API_KEY` → paste your HPI AISC key
- Add `AISC_BASE_URL` → `https://api.aisc.hpi.de`
- Add `QDRANT_URL` → `http://qdrant:6333`

---

## How to Import a Flow

1. In the Langflow sidebar, click **New Flow** → **Import**
2. Select one of the `.json` files from the `flows/` folder
3. The flow opens in the canvas — you can explore, run, and modify it

---

## How to Upload Custom Components

The `custom_components/` folder contains production components from the reference app. Upload them once and they become available in all flows:

1. In the sidebar, click **Components** (puzzle icon)
2. Click **Upload Component** (top-right of the panel)
3. Select all `.py` files from `custom_components/` (except `__init__.py`)
4. They appear under **Custom Components** in the component list

---

## Workshop Flows

Work through these flows in order. Each builds on concepts from the previous one.

### Flow 1 — Basic RAG Pipeline (`flows/01_basic_rag.json`)

**Duration:** ~20 min

The simplest possible RAG system. No agent — just a chain.

```
PDF Loader → Text Splitter → Embeddings → Qdrant (ingest)
Chat Input → Qdrant Retriever → Prompt → LLM → Chat Output
```

**What you'll learn:**
- How documents are chunked and stored as vectors
- How a query is turned into a vector and matched against stored chunks
- How retrieved context is injected into the LLM prompt

---

### Flow 2 — Agentic RAG (`flows/02_agentic_rag.json`)

**Duration:** ~20 min

The same retrieval logic, but now the LLM is an **agent** that decides *when* to use retrieval as a tool.

```
Chat Input → Agent (with RAG Tool) → Chat Output
```

**What you'll learn:**
- The difference between a chain (always retrieves) and an agent (retrieves only when needed)
- How to expose a component as a Tool in Langflow (toggle "Tool Mode")
- How agents reason step-by-step before responding

---

### Flow 3 — Multi-Query RAG (`flows/03_multi_query_rag.json`)

**Duration:** ~25 min

Instead of one retrieval query, the LLM generates multiple query variants and fuses the results — the same strategy used in the reference application.

```
Chat Input → Query Expander LLM → [3× Qdrant Retriever] → Parse/Merge → LLM → Chat Output
```

**What you'll learn:**
- Why a single query often misses relevant chunks (vocabulary mismatch)
- How Reciprocal Rank Fusion (RRF) combines results from multiple queries
- How to fan out and merge data flows in Langflow

---

### Flow 4 — Adaptive RAG (`flows/04_adaptive_rag.json`)

**Duration:** ~25 min

Uses the `adaptive_qdrant_retriever` custom component, which retries retrieval with progressively larger result sets (3 → 5 → 8 → 10) until the LLM says the context is sufficient.

```
Chat Input → Adaptive Qdrant Retriever (custom) → LLM → Chat Output
```

**What you'll learn:**
- How custom components work in Langflow (upload + use as any built-in)
- Adaptive / self-correcting retrieval patterns
- LLM-as-judge for retrieval sufficiency

---

### Flow 5 — Full Agentic RAG System (`flows/05_full_agentic_rag.json`)

**Duration:** ~30 min

The complete architecture: an agent with multiple tools (adaptive RAG + web search), conversation memory, and structured citation output. This mirrors the production system in `workshop-agentic-rag`.

```
Chat Input → Agent
              ├── Tool: Adaptive RAG (custom component)
              ├── Tool: Web Search (Tavily)
              └── Memory (session-based)
           → SourceDocumentsBuilder → AnswerEnvelopeBuilder → Output
```

**What you'll learn:**
- Multi-tool agents: how the agent decides which tool to call
- Citation / source tracking in agentic workflows
- Session memory for multi-turn conversations
- How this flow maps to the production Chainlit app

---

## Folder Structure

```
LangFlow/
├── README.md                    ← this file
├── setup/
│   ├── docker-compose.yml       ← starts Langflow + Qdrant
│   └── .env.example             ← template for API keys (copy → .env)
├── custom_components/           ← upload these once in the Langflow UI
│   ├── adaptive_qdrant_retriever.py
│   ├── agent_tool_envelope_extractor.py
│   ├── answer_envelope_builder.py
│   ├── envelope_output.py
│   ├── grundschutz_search_tool.py
│   ├── source_documents_builder.py
│   └── text_output.py
└── flows/
    ├── 01_basic_rag.json
    ├── 02_agentic_rag.json
    ├── 03_multi_query_rag.json
    ├── 04_adaptive_rag.json
    └── 05_full_agentic_rag.json
```

---

## LLM Model Selection

All flows use the **OpenAI** component pointed at the HPI AISC model hub:

| Langflow field | Value |
|---|---|
| Base URL | `https://api.aisc.hpi.de` |
| API Key | your `AISC_API_KEY` global variable |
| Model | `gpt-oss-120b` (or any model listed on the hub) |

To switch models: click the **OpenAI** component in any flow → change the **Model Name** field.

---

## Troubleshooting

**Port 7860 already in use:**
```bash
docker compose down && docker compose up
```
Or change `7860:7860` → `7861:7860` in `docker-compose.yml`.

**Qdrant connection refused in a flow:**
Make sure the Qdrant URL in the component matches your Global Variable: `http://qdrant:6333` (inside Docker) or `http://localhost:6333` (if running Langflow without Docker).

**"API key not found" error:**
Re-check **Settings → Global Variables** — the variable name must match exactly (case-sensitive).

**Custom component not appearing:**
After uploading, refresh the page. Components appear under **Custom Components** at the bottom of the sidebar panel.

**Embedding dimension mismatch after changing models:**
Delete the Qdrant collection in the vector store component and re-ingest your documents. Different embedding models produce vectors of different dimensions.

---

## Reference Application

The production app this workshop is based on lives in `../workshop-agentic-rag/`. It uses the same Qdrant + LLM stack but wraps everything in a Chainlit chat UI with multi-user support, PostgreSQL persistence, and a full REST API. The custom components in `custom_components/` are taken directly from that app.
