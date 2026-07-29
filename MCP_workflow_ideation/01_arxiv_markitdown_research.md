# Example 3 — Personalized ArXiv Research Assistant (arXiv MCP + mem0)

## 1. The idea, in one sentence
A research assistant that **remembers who the user is and what they work on**, then **finds and summarises recent arXiv papers** tailored to those interests.

## 2. Goal & example prompts
- **Goal:** combine a *memory* MCP (mem0) with a *research* MCP (arXiv) so answers are personalised, not generic — and orchestrate them with an agent.
- **Example user prompts (same chat, two turns):**
  1. *"I'm an AI engineer at the AI Service Center, interested in reinforcement learning for control."* — agent **stores** this profile to mem0.
  2. *"Show me influential arXiv papers on reinforcement learning in control from the past month and summarise them."* — agent **recalls** the interests, resolves "past month" via the Current Date tool, searches arXiv, and returns a ranked, summarised list.

## 3. Why is this *agentic*?
The agent runs a multi-step plan and picks tools as needed: *recall profile (mem0) → resolve the date window → search arXiv → read/abstract the top hits → rank & summarise → optionally store a new preference.* Nothing is hardcoded — it decides which of the ~25 available tools to call, and when it has enough to answer.

## 4. MCP servers — **two, one remote + one local**

| Field | **mem0** (memory) | **arXiv** (research) |
|---|---|---|
| Provider | mem0.ai — first-party hosted | community (`blazickjp/arxiv-mcp-server`) |
| What it does | store / search the user's long-term profile & preferences | search, read, and analyse arXiv papers |
| Transport | **Remote** (Streamable HTTP) | **Local stdio** (launched via `uvx`) |
| URL / command | `https://mcp.mem0.ai/mcp` | `command: uvx`, `args: ["arxiv-mcp-server"]` |
| Auth & cost | API key (Bearer) or OAuth; free Hobby tier | **none** — free |
| Tools (loaded) | `add_memory`, `search_memories`, `get_memories`, `update_memory`, `delete_memory`, … (11) | `search_papers`, `semantic_search`, `get_abstract`, `read_paper`, `citation_graph`, `get_paper_latex`, … (14) |
| Trust check | ☑ first-party hosted — but **your data leaves your machine** | ☐ community — **runs code locally with your user privileges**; pin a version / review source / or containerise |

## 5. Langflow components (as built)
☑ Chat Input ☑ Agent ☑ LiteLLM Proxy (AISC hub) ☑ System prompt (see below)
☑ **two** MCP Tools nodes (`mem0` + `arxiv`) ☑ **Current Date** node ☑ Chat Output

The **Current Date** node is wired into the Agent so relative timeframes like *"past month"* resolve to real dates.

The Agent's system prompt is the research-assistant prompt in
`MCP_workflow_ideation/arxiv_research_agent_prompt.md` (it pins `user_id = workshop_demo_user`).

## 6. Flow sketch (matches the built flow)
```
Chat Input ─────────────► Agent ─────────► Chat Output
LiteLLM Proxy (model) ──► Agent
                           ▲ tools
                           ├── MCP: mem0          (remote — recall / store user profile)
                           ├── MCP: arxiv         (local stdio via uvx — search / read papers)
                           └── Current Date       (resolve "past month", "this week", …)
```

## 7. One-time setup
1. **mem0 (remote):** get an API key at `https://app.mem0.ai`. In Langflow → Settings → MCP Servers, ensure the `mem0` server has your key (Streamable HTTP, URL `https://mcp.mem0.ai/mcp`, header `Authorization: Bearer <KEY>`). Open the mem0 node → **Refresh Tools**.
2. **arXiv (local stdio):** install `uv` where Langflow runs (`brew install uv`). Configure the arXiv MCP node as stdio — `command: uvx`, `args: ["arxiv-mcp-server"]` — then **Refresh Tools** (should list 14 tools).
   - **Safety:** this executes a third-party package on your machine with your privileges. Pin a version (`arxiv-mcp-server==<v>`), glance at `https://github.com/blazickjp/arxiv-mcp-server`, and/or run it as an isolated HTTP sidecar container if you want the code sandboxed.
3. **System prompt:** paste the block from `arxiv_research_agent_prompt.md` into the Agent's System Prompt.
4. **Tool Mode on** for both MCP nodes; wire each node's Toolset output + the Current Date node into the Agent's Tools input.

## 8. Try it — success test
Turn 1: *"I'm an AI engineer at the AI Service Center, interested in reinforcement learning for control."*
→ Agent calls `add_memory` (user_id `workshop_demo_user`).
Turn 2: *"Influential arXiv papers on RL in control from the past month — summarise them."*
→ Agent calls `search_memories` (recalls the interest), uses Current Date for the window, calls `search_papers`/`semantic_search`, and returns a ranked summary — noting honestly that "influence" is limited for very recent papers.

## 9. Difficulty & likely gotchas
- **Difficulty:** ⭐⭐⭐⭐ — but it's the best **capstone**: two MCP servers + memory + a date tool working together.
- **Gotchas:**
  - **mem0 `user_id` scoping** — store and search must use the *same* `user_id`. The prompt pins `workshop_demo_user`; keep it identical to your mem0 (Example 1) flow or the recall comes back empty.
  - **arXiv needs `uvx` where Langflow runs** — native Langflow, or `uv` inside the container. Otherwise use the HTTP-sidecar approach.
  - **stdio args must include the package** — the config is `command: uvx`, `args: ["arxiv-mcp-server"]`; `uvx` alone won't launch it. Verify if Refresh Tools comes back empty.
  - **"Influential" in a recent window is hard** — citations lag; the agent ranks by relevance and can use `citation_graph` where data exists, and says so.
  - **Local-code trust** — pin the version / review the source / containerise for isolation.
  - **Two toolkits = ~25 tools** — keep the system prompt clear so the agent routes well; disable tools you don't need to reduce confusion.
