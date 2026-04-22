# Future Work

Running list of known gaps and follow-up work. Order ≈ priority, not strict.

## User delete endpoint

Today there's no way for a user (or admin) to delete their account + data. Data is spread across Postgres, SQLite, Qdrant, and disk, so accounts linger and accumulate.

Implementation sketch — compose three existing helpers into one `DELETE /api/users/me` (and admin `DELETE /api/users/{id}`):

1. **Postgres** ([app/chat/native_chat.py:56-65](app/chat/native_chat.py#L56-L65)) — threads first (FK is `ON DELETE SET NULL`, so otherwise orphaned), then user. Steps/Elements/Feedback cascade via `threadId`.
   ```sql
   DELETE FROM "Thread" WHERE "userId" = (SELECT id FROM "User" WHERE identifier = $1);
   DELETE FROM "User" WHERE identifier = $1;
   ```
2. **SQLite** — generalize [delete_user_profile](app/chat/chat_history.py#L505) to one transaction over `knowledge_bases`, `kb_documents`, `user_settings`, `user_starters`, `user_profiles`.
3. **Qdrant + uploaded PDFs** — reuse the per-KB teardown from [remove_knowledge_base](app/api/api_routes.py#L235): iterate the user's KBs, `drop_collection` each, unlink persisted PDFs via [_persisted_doc_path](app/api/api_routes.py#L61).

UI: small "delete my account" button in the ingestion panel; admin-only list + delete in a separate admin view. Estimate ~½ day.

## LaTeX / math rendering — off by default, delimiter mismatch

[app/.chainlit/config.toml:34](app/.chainlit/config.toml#L34) has `latex = false`, so math in model responses renders as raw text (user saw `(W^{Q}_{i}, W^{K}_{i}, W^{V}_{i})` and `d_{\text{model}}` unrendered). Chainlit's comment warns that turning it on clashes with literal `$` characters — a real tradeoff for a chat app where users may paste code or prices.

Second, even with `latex = true`, the model's current output uses bare `(...)` as delimiters, which KaTeX doesn't recognize. Supported: `$...$` / `\(...\)` inline, `$$...$$` / `\[...\]` block. Either the system prompt needs to instruct the model on delimiters, or a post-processor needs to rewrite `\(...\)` → `$...$` before render.

Proposed shape — per-user toggle in the ingestion/settings panel (reuse `user_settings` SQLite table):
- Add `math_rendering` boolean default `false`.
- Inject a `<style>` toggle or a JS shim that flips `latex` at render time; OR simpler, render with `latex = true` globally and escape `$` in user-authored messages client-side when the toggle is off.
- Update system prompt / RAG prompt to always emit `$...$` when the flag is on.

Estimate: ~½ day if done as a pure toggle; more if the `$`-escape preprocessor has to survive Chainlit's markdown pipeline cleanly.

## Nutzungsbedingungen modal not shown after login

Terms modal never opens for logged-in users. Root cause in [app/public/custom.js:313-324](app/public/custom.js#L313-L324): `loadAndMaybeShowTerms()` runs at IIFE time AND on `window.load`, which both fire on the `/login` page — before any auth cookie exists. `/api/terms` requires auth → 401 → `data === null` → silent return. But `termsChecked = true` is set *before* the fetch, so the latch is held. Chainlit's post-login hand-off is client-side routed (no full reload), so the check never re-runs.

Fix (~5 lines):
- Early-return from `loadAndMaybeShowTerms` if on `/login` (reuse the `onLogin` regex from `markLoginPage`).
- Move `termsChecked = true` *inside* the `.then` block where we got valid JSON — not before the fetch.
- Re-run on `popstate` / after the login route transition, the same pattern `markLoginPage` already uses.

Verification: wipe a user with `DELETE FROM "User" WHERE identifier = 'X'`, log in fresh, modal should appear. With the current code it silently doesn't.

## GPU Docling service — finish the scaffold

`k8s/docling-service.yaml` ships a Deployment + Service skeleton with `replicas: 0`, GPU resource request, and `imagePullPolicy: IfNotPresent`. Manual scale today (`kubectl scale deploy/docling-service --replicas=1`), but three pieces are missing:

1. **Service code.** `uvicorn docling_service:app` expects `app/docling_service.py` — a small FastAPI wrapper around `docling.DocumentConverter`. Minimal shape:
   - `POST /process` (multipart PDF) → returns Docling JSON structure matching what `kb.ingest_docling._build_docs_from_docling_json` already consumes.
   - `GET /healthz` → `200` once the DocumentConverter is warm; used by the readiness probe.
   - Load the DocumentConverter once at startup so the first request doesn't pay the model-load cost.

2. **App-side integration.** [app/kb/ingestion_pipeline.py](app/kb/ingestion_pipeline.py) currently runs Docling in-process. Swap in an optional HTTP path: if `DOCLING_SERVICE_URL` is set, `POST` the PDF there; otherwise fall back to local. Gives us per-deploy control without a hard dependency.

3. **Schedule-aware scaling.** Two options when we get there:
   - **Cron-based** (simpler, no new infra): two `CronJob` manifests scaling the Deployment up 15 min before workshops and down 30 min after. Dates hardcoded, so annoying but honest.
   - **App-driven** (more elegant): app sees an upload, checks `docling-service` ready replicas, scales 0→1 via k8s API if needed, polls readiness, then forwards. Needs a ServiceAccount with `apps/deployments/scale` verb on just this one Deployment. Small blast radius.

Ties into the "Strip CUDA from the image" entry below — once the main app image is CPU-only, this service becomes the single place CUDA wheels live, and the image split pays for itself in pull time.

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
