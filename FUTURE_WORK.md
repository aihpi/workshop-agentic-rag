# Future Work

Running list of known gaps and follow-up work. Order ≈ priority, not strict.

## Admin UI: model selection (primary + fallback)

Today the primary chat model is pinned via `CHAT_MODEL` in the sealed secret and the fallback via `FALLBACK_CHAT_MODEL` as a plain deployment env. To change either, an admin has to re-seal the secret / edit the manifest, push, and wait for ArgoCD to roll — slow and requires git access.

Shape: dropdowns in `/admin/app` for primary + fallback (options populated by `GET /v1/models` on the LiteLLM gateway — LiteLLM exposes this endpoint natively). Persist the admin's choice into a new `app_config` SQLite table (single row). Both `chat()` and `stream_chat()` in [app/core/llm.py](app/core/llm.py) read the persisted choice first, fall back to env, then fall back to the hard-coded default.

Estimate: ~2–3 hours. Env vars stay as the bootstrap default + emergency override path.

## **P0 — GPU Docling service** (blocks multi-user upload)

Current state (after 2026-04-22 demo prep): CPU Docling works in the `workshop-app` pod (16Gi mem limit), but a single 2 MB text PDF takes **~30s** to parse. With 10 workshop participants uploading concurrently we'd see three failure modes: serial queueing (~5 min wait for the last user), pod OOM (peak memory 6–8 Gi per parse × concurrent uploads trips the 16 Gi limit), and chat freeze (CPU starvation of the serving path). [k8s/docling-service.yaml](k8s/docling-service.yaml) ships the Deployment + Service skeleton with `replicas: 0` and a GPU resource request; the integration is what's missing.

Three pieces to finish:

1. **Service code.** `uvicorn docling_service:app` expects `app/docling_service.py` — a small FastAPI wrapper around `docling.DocumentConverter`. Minimal shape:
   - `POST /process` (multipart PDF) → returns Docling JSON structure matching what `kb.ingest_docling._build_docs_from_docling_json` already consumes.
   - `GET /healthz` → `200` once the DocumentConverter is warm; used by the readiness probe.
   - Load the DocumentConverter once at startup so the first request doesn't pay the model-load cost.
2. **App-side integration.** [app/kb/ingestion_pipeline.py](app/kb/ingestion_pipeline.py) `parse_pdf()` currently runs Docling in-process. Swap in an optional HTTP path: if `DOCLING_SERVICE_URL` is set, `POST` the PDF there; otherwise fall back to local. Per-deploy control without a hard dependency.
3. **GPU allocation.** A30 is MIG-capable (1g.10gb slices fit Docling comfortably). Confirm with `kubectl get nodes -o json | jq '.items[].status.capacity'` — if MIG is exposed, request `nvidia.com/mig-1g.10gb: 1` and up to 4 docling-service pods share one A30. Else fall back to `nvidia.com/gpu: 1` (whole device).
4. **Schedule-aware scaling** (smaller win, can wait):
   - **Cron-based** (simpler, no new infra): two `CronJob` manifests scaling the Deployment up 15 min before workshops and down 30 min after. Dates hardcoded but honest.
   - **App-driven**: app sees upload, checks `docling-service` ready replicas, scales 0→1 via k8s API, polls readiness, then forwards. Needs ServiceAccount with `apps/deployments/scale` verb on just this one Deployment.

Ties into "Strip CUDA from the image" below — once the main app image is CPU-only, this service becomes the single place CUDA wheels live, and the image split pays for itself in pull time.

## LaTeX / math rendering — off by default, delimiter mismatch

[app/.chainlit/config.toml:34](app/.chainlit/config.toml#L34) has `latex = false`, so math in model responses renders as raw text (user saw `(W^{Q}_{i}, W^{K}_{i}, W^{V}_{i})` and `d_{\text{model}}` unrendered). Chainlit's comment warns that turning it on clashes with literal `$` characters — a real tradeoff for a chat app where users may paste code or prices.

Second, even with `latex = true`, the model's current output uses bare `(...)` as delimiters, which KaTeX doesn't recognize. Supported: `$...$` / `\(...\)` inline, `$$...$$` / `\[...\]` block. Either the system prompt needs to instruct the model on delimiters, or a post-processor needs to rewrite `\(...\)` → `$...$` before render.

Proposed shape — per-user toggle in the ingestion/settings panel (reuse `user_settings` SQLite table):
- Add `math_rendering` boolean default `false`.
- Inject a `<style>` toggle or a JS shim that flips `latex` at render time; OR simpler, render with `latex = true` globally and escape `$` in user-authored messages client-side when the toggle is off.
- Update system prompt / RAG prompt to always emit `$...$` when the flag is on.

Estimate: ~½ day if done as a pure toggle; more if the `$`-escape preprocessor has to survive Chainlit's markdown pipeline cleanly.

## Strip CUDA from the image (~4 GB win)

Built image is ~6 GB; roughly 4 of that is `nvidia-*` / `cuda-*` / `cudnn` / `nccl` that we never execute — k8s target is CPU-only. Root cause: `docling` pulls `torch`, and pip's default Linux torch wheel is the CUDA variant.

Fix — one line in both [Dockerfile](Dockerfile) and [app/Dockerfile](app/Dockerfile), before `pip install -e .`:
```dockerfile
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
```
Installing CPU torch first satisfies Docling's requirement so pip won't resolve the CUDA build. Expected image size: ~1.5–2 GB. Wins: faster GHA build/push, faster k8s image pulls on cold nodes, CI cache stays under GHA's 10 GB quota longer.

If GPU Docling (OCR/layout speedup) is ever wanted, that's a separate GPU image variant — not the default.

## Consolidate Dockerfiles

[Dockerfile](Dockerfile) (used by CI/CD) and [app/Dockerfile](app/Dockerfile) (used by compose) have drifted. Root is minimal; `app/` adds libgl/libxcb/libsm/libxrender for OpenCV (Docling dep). Pick one, delete the other, point compose at it. Risk of divergence otherwise: CI passes, local breaks, or vice versa.

## Orphaned upload cleanup

`DATA_KB_DOCS_DIR` is keyed by `doc_id`. If the SQLite row goes away without hitting the delete path (manual DB surgery, crash mid-upload), the file lingers forever. Low priority — waste disk, don't break anything — but a periodic "orphan sweep" job (files without a matching `kb_documents.id`) would be cheap insurance.

## Multi-arch image builds

CD workflow builds `linux/amd64` only. Add `linux/arm64` to [.github/workflows/publish-app-image.yaml](.github/workflows/publish-app-image.yaml) if we ever target arm nodes.

## GHCR retention policy

Every `main` push accumulates a `sha-*` tag forever. Add a cleanup workflow (keep last N, or anything referenced by a running manifest) before the registry gets crowded.

## PR preview builds

CI workflow only builds, doesn't push. A `type=ref,event=pr` tag + per-PR image would let reviewers pull and run before merge. Useful once we have reviewers.
