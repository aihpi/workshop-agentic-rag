"""FastAPI endpoints for per-user knowledge bases, system prompt, and starters.

Registered by `register_user_api_routes()` from inside the Chainlit
`@cl.on_app_startup` handler. Each endpoint requires an authenticated user
(`Depends(get_current_user)`) and operates only on that user's rows.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

from chainlit.auth import get_current_user
from fastapi import Depends, File, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from kb.ingestion_pipeline import (
    SUPPORTED_EXTENSIONS,
    delete_document_points,
    drop_collection,
    ingest_sections,
    parse_file,
    validate_pdf_upload,
)
from chat.chat_history import hard_delete_user_data
from chat.native_chat import (
    accept_terms,
    create_user_with_password,
    delete_user_cascade,
    get_terms_status,
    get_user_role_status,
    list_all_users,
    update_user_admin_fields,
)

import bcrypt
from api.prompt_generator import generate_starters, generate_system_prompt
from core.settings import (
    CHAINLIT_AUTH_USERNAME,
    CHAT_DB_PATH,
    DATA_KB_DOCS_DIR,
    DATABASE_URL,
    MAX_FILE_SIZE_MB,
    QDRANT_COLLECTION,
    TERMS_VERSION,
)
from kb.user_kb import (
    SHARED_KB_ID,
    create_kb,
    delete_document,
    delete_kb,
    get_document,
    get_kb,
    get_user_system_prompt,
    list_documents,
    list_kbs,
    list_user_starters,
    record_document,
    replace_user_starters,
    set_user_system_prompt,
    update_kb,
)


def _persisted_doc_path(doc_id: str, file_name: str) -> Path:
    """Where bytes for a document live on disk. Keyed by doc_id, suffix from filename."""
    ext = Path(file_name).suffix.lower()
    return DATA_KB_DOCS_DIR / f"{doc_id}{ext}"


def _user_id(current_user: Any) -> str:
    if current_user is None:
        raise HTTPException(status_code=401, detail="Unauthorized")
    identifier = getattr(current_user, "identifier", None)
    if not identifier:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return str(identifier)


def _assert_owns_kb(current_user: Any, kb_id: str) -> dict[str, Any]:
    user_id = _user_id(current_user)
    if kb_id == SHARED_KB_ID:
        raise HTTPException(status_code=403, detail="Die gemeinsame Wissensdatenbank ist schreibgeschützt.")
    kb = get_kb(CHAT_DB_PATH, kb_id)
    if kb is None:
        raise HTTPException(status_code=404, detail="Wissensdatenbank nicht gefunden")
    if kb["user_id"] != user_id:
        raise HTTPException(status_code=403, detail="Zugriff verweigert")
    return kb


def _resolve_readable_kb(current_user: Any, kb_id: str) -> dict[str, Any]:
    """Like _assert_owns_kb but also admits the shared read-only KB."""
    user_id = _user_id(current_user)
    if kb_id == SHARED_KB_ID:
        return {"id": SHARED_KB_ID, "qdrant_collection": QDRANT_COLLECTION}
    kb = get_kb(CHAT_DB_PATH, kb_id)
    if kb is None:
        raise HTTPException(status_code=404, detail="Wissensdatenbank nicht gefunden")
    if kb["user_id"] != user_id:
        raise HTTPException(status_code=403, detail="Zugriff verweigert")
    return kb


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------

class KbCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)
    description: str = Field(default="", max_length=2000)


class KbUpdateRequest(BaseModel):
    name: str | None = Field(default=None, max_length=120)
    description: str | None = Field(default=None, max_length=2000)


class SystemPromptRequest(BaseModel):
    system_prompt: str | None = None


class StarterItem(BaseModel):
    label: str = Field(default="", max_length=120)
    message: str = Field(..., min_length=1, max_length=2000)


class StartersRequest(BaseModel):
    starters: list[StarterItem] = Field(default_factory=list)


class UserAdminUpdateRequest(BaseModel):
    status: str | None = Field(default=None, pattern="^(approved|blocked)$")
    role: str | None = Field(default=None, pattern="^(user|admin)$")


class UserCreateRequest(BaseModel):
    identifier: str = Field(..., min_length=1, max_length=120)
    password: str = Field(..., min_length=8, max_length=200)
    email: str | None = Field(default=None, max_length=200)
    role: str | None = Field(default="user", pattern="^(user|admin)$")


class TermsAcceptRequest(BaseModel):
    version: str = Field(..., min_length=1, max_length=120)


_SUMMARY_DE = """- **Nur für Forschung und Test, keine produktive Nutzung.** Die Dienste des AISC-BB dürfen ausschließlich im Testbetrieb genutzt werden (§ 5.4).
- **Keine sensiblen oder personenbezogenen Daten hochladen.** Verwenden Sie ausschließlich synthetische oder vollständig anonymisierte Daten. Die Verantwortung hierfür liegt bei Ihnen (§ 6.2, § 6.3).
- **Zugangsdaten schützen.** Ihre Zugangsdaten dürfen nicht an Dritte weitergegeben werden (§ 4.2).
- **Keine Gewährleistung.** Die Dienste werden ohne Verfügbarkeitsgarantie und Haftung bereitgestellt (§ 2.1, § 12).
- **Missbrauch**, insbesondere Verstöße gegen geltendes Recht oder diese Nutzungsbedingungen, kann zur fristlosen Kündigung führen (§ 14.2).

Der vollständige Nutzungsvertrag steht unten als PDF und in der Web-Ansicht zur Verfügung. Mit Ihrer Zustimmung bestätigen Sie, die vollständigen Nutzungsbedingungen gelesen, verstanden und akzeptiert zu haben."""

_SUMMARY_EN = """- **Research and testing only, no production use.** AISC-BB services may only be used in test mode (§ 5.4).
- **Do not upload sensitive or personal data.** Use synthetic or fully anonymised data only. The responsibility for this lies with you (§ 6.2, § 6.3).
- **Protect your access credentials.** Your access credentials must not be shared with third parties (§ 4.2).
- **No warranty.** The services are provided without availability guarantee and without liability (§ 2.1, § 12).
- **Misuse**, in particular violations of applicable law or these Terms of Use, may result in immediate termination without notice (§ 14.2).

The full usage agreement is available below as a PDF and as a web view. By giving your consent, you confirm that you have read, understood, and accepted the full Terms of Use."""


async def _require_admin(current_user: Any) -> str:
    """Return identifier if the user's DB role is admin; 403 otherwise."""
    identifier = _user_id(current_user)
    if not DATABASE_URL:
        raise HTTPException(status_code=503, detail="User directory not configured")
    pair = await get_user_role_status(DATABASE_URL, identifier)
    if pair is None or pair[0] != "admin":
        raise HTTPException(status_code=403, detail="Admin-Berechtigung erforderlich")
    if pair[1] == "blocked":
        raise HTTPException(status_code=403, detail="Konto gesperrt")
    return identifier


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register_user_api_routes(fastapi_app: Any, default_system_prompt: str | None) -> list[str]:
    """Attach the /api/* routes to the given FastAPI app.

    Returns the list of registered paths so the caller can nudge them above
    the Chainlit catch-all route.
    """

    @fastapi_app.get("/api/kbs")
    async def list_knowledge_bases(current_user=Depends(get_current_user)):
        user_id = _user_id(current_user)
        rows = list_kbs(CHAT_DB_PATH, user_id)
        shared = {
            "id": SHARED_KB_ID,
            "user_id": None,
            "name": "IT-Grundschutz (gemeinsam)",
            "description": (
                "Gemeinsame, schreibgeschützte Wissensbasis mit dem "
                "IT-Grundschutz-Kompendium und den BSI-Standards."
            ),
            "qdrant_collection": QDRANT_COLLECTION,
            "created_at": None,
            "document_count": None,
            "readonly": True,
        }
        return {"knowledge_bases": [shared, *rows]}

    @fastapi_app.post("/api/kbs")
    async def create_knowledge_base(
        request: KbCreateRequest,
        current_user=Depends(get_current_user),
    ):
        user_id = _user_id(current_user)
        name = request.name.strip()
        if not name:
            raise HTTPException(status_code=400, detail="Name darf nicht leer sein")
        try:
            kb = create_kb(
                CHAT_DB_PATH,
                user_id=user_id,
                name=name,
                description=request.description.strip(),
            )
        except Exception as exc:  # UNIQUE(user_id, name)
            if "UNIQUE" in str(exc):
                raise HTTPException(status_code=409, detail="Name bereits vergeben") from exc
            raise
        return kb

    @fastapi_app.patch("/api/kbs/{kb_id}")
    async def patch_knowledge_base(
        kb_id: str,
        request: KbUpdateRequest,
        current_user=Depends(get_current_user),
    ):
        _assert_owns_kb(current_user, kb_id)
        update_kb(
            CHAT_DB_PATH,
            kb_id,
            name=request.name.strip() if isinstance(request.name, str) else None,
            description=request.description.strip() if isinstance(request.description, str) else None,
        )
        return get_kb(CHAT_DB_PATH, kb_id)

    @fastapi_app.delete("/api/kbs/{kb_id}")
    async def remove_knowledge_base(kb_id: str, current_user=Depends(get_current_user)):
        kb = _assert_owns_kb(current_user, kb_id)
        # Unlink persisted bytes for every document in this KB before the cascade
        # delete drops the DB rows and we lose the (id, file_name) mapping.
        for doc in list_documents(CHAT_DB_PATH, kb_id):
            _persisted_doc_path(doc["id"], doc["file_name"]).unlink(missing_ok=True)
        drop_collection(kb["qdrant_collection"])
        delete_kb(CHAT_DB_PATH, kb_id)
        return {"deleted": kb_id}

    @fastapi_app.get("/api/kbs/{kb_id}/documents")
    async def list_kb_documents(kb_id: str, current_user=Depends(get_current_user)):
        _assert_owns_kb(current_user, kb_id)
        return {"documents": list_documents(CHAT_DB_PATH, kb_id)}

    @fastapi_app.post("/api/kbs/{kb_id}/documents")
    async def upload_kb_document(
        kb_id: str,
        file: UploadFile = File(...),
        current_user=Depends(get_current_user),
    ):
        kb = _assert_owns_kb(current_user, kb_id)
        original_name = file.filename or "upload"
        ext = Path(original_name).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Nicht unterstützter Dateityp {ext or '(leer)'}. Erlaubt: {sorted(SUPPORTED_EXTENSIONS)}",
            )

        max_bytes = MAX_FILE_SIZE_MB * 1024 * 1024
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            total = 0
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    tmp.close()
                    Path(tmp.name).unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=413,
                        detail=f"Datei überschreitet {MAX_FILE_SIZE_MB} MB",
                    )
                tmp.write(chunk)
            tmp_path = Path(tmp.name)

        # Content-Type header is advisory (easily faked), but if the browser
        # explicitly labelled a different type we should trust that signal
        # and reject early — saves a magic-byte read and returns a clearer error.
        if file.content_type and file.content_type != "application/pdf":
            tmp_path.unlink(missing_ok=True)
            raise HTTPException(
                status_code=415,
                detail=f"Content-Type {file.content_type} nicht unterstützt. Nur application/pdf.",
            )

        # Magic-byte check — the actual server-side guard. Catches renamed
        # files (foo.exe → foo.pdf) that pass the extension check.
        try:
            validate_pdf_upload(tmp_path)
        except ValueError as exc:
            tmp_path.unlink(missing_ok=True)
            raise HTTPException(status_code=415, detail=str(exc))

        try:
            # Stage 1 — Docling parse. CPU-heavy; on OOM/timeout this is where it dies.
            # Wrapped in asyncio.to_thread so it doesn't block the event loop —
            # otherwise a single upload freezes chat streaming for all users.
            t0 = time.monotonic()
            try:
                sections = await asyncio.to_thread(parse_file, tmp_path)
            except Exception:
                logger.exception(
                    "upload.parse_failed kb_id=%s filename=%s size=%d elapsed=%.1fs",
                    kb_id, original_name, total, time.monotonic() - t0,
                )
                raise
            logger.info(
                "upload.parse_ok kb_id=%s filename=%s sections=%d elapsed=%.1fs",
                kb_id, original_name, len(sections) if sections else 0, time.monotonic() - t0,
            )
            if not sections:
                raise HTTPException(
                    status_code=422,
                    detail="Kein Textinhalt aus der Datei extrahiert",
                )

            # Seed the document record first so we have a stable id to tag points with.
            doc_row = record_document(
                CHAT_DB_PATH,
                kb_id=kb_id,
                file_name=original_name,
                size_bytes=total,
                chunk_count=0,
            )

            # Stage 2 — persist to DATA_KB_DOCS_DIR (PVC-backed in k8s). Defensive mkdir
            # so a freshly-attached PVC or a manually-deleted dir self-heals.
            persisted_path = _persisted_doc_path(doc_row["id"], original_name)
            try:
                persisted_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(tmp_path, persisted_path)
            except Exception:
                logger.exception(
                    "upload.persist_failed kb_id=%s doc_id=%s dest=%s",
                    kb_id, doc_row["id"], persisted_path,
                )
                delete_document(CHAT_DB_PATH, doc_row["id"])
                raise

            try:
                # Point the sections at the upload's original name for citation display.
                for section in sections:
                    section["file"] = original_name

                # Stage 3 — embed + upsert to Qdrant. Hits LiteLLM + Qdrant over network.
                t1 = time.monotonic()
                try:
                    chunk_count = await ingest_sections(
                        sections,
                        collection=kb["qdrant_collection"],
                        kb_id=kb_id,
                        document_id=doc_row["id"],
                    )
                except Exception:
                    logger.exception(
                        "upload.embed_failed kb_id=%s doc_id=%s collection=%s elapsed=%.1fs",
                        kb_id, doc_row["id"], kb["qdrant_collection"], time.monotonic() - t1,
                    )
                    raise
                logger.info(
                    "upload.embed_ok kb_id=%s doc_id=%s chunks=%d elapsed=%.1fs",
                    kb_id, doc_row["id"], chunk_count, time.monotonic() - t1,
                )

                # Update chunk_count after ingest.
                import sqlite3

                with sqlite3.connect(str(CHAT_DB_PATH)) as conn:
                    conn.execute(
                        "UPDATE kb_documents SET chunk_count = ? WHERE id = ?",
                        (chunk_count, doc_row["id"]),
                    )
                    conn.commit()
                doc_row["chunk_count"] = chunk_count
                return doc_row
            except Exception:
                # Roll back: remove persisted bytes and DB row before re-raising.
                persisted_path.unlink(missing_ok=True)
                delete_document(CHAT_DB_PATH, doc_row["id"])
                raise
        finally:
            tmp_path.unlink(missing_ok=True)

    @fastapi_app.delete("/api/kbs/{kb_id}/documents/{doc_id}")
    async def remove_kb_document(
        kb_id: str,
        doc_id: str,
        current_user=Depends(get_current_user),
    ):
        kb = _assert_owns_kb(current_user, kb_id)
        doc = get_document(CHAT_DB_PATH, doc_id)
        if doc is None or doc["kb_id"] != kb_id:
            raise HTTPException(status_code=404, detail="Dokument nicht gefunden")
        delete_document_points(kb["qdrant_collection"], doc_id)
        delete_document(CHAT_DB_PATH, doc_id)
        _persisted_doc_path(doc_id, doc["file_name"]).unlink(missing_ok=True)
        return {"deleted": doc_id}

    @fastapi_app.get("/api/settings/system-prompt")
    async def read_system_prompt(current_user=Depends(get_current_user)):
        user_id = _user_id(current_user)
        custom = get_user_system_prompt(CHAT_DB_PATH, user_id)
        return {
            "system_prompt": custom,
            "default_system_prompt": default_system_prompt or "",
            "using_default": custom is None,
        }

    @fastapi_app.put("/api/settings/system-prompt")
    async def update_system_prompt(
        request: SystemPromptRequest,
        current_user=Depends(get_current_user),
    ):
        user_id = _user_id(current_user)
        set_user_system_prompt(CHAT_DB_PATH, user_id, request.system_prompt)
        return {
            "system_prompt": get_user_system_prompt(CHAT_DB_PATH, user_id),
            "default_system_prompt": default_system_prompt or "",
            "using_default": get_user_system_prompt(CHAT_DB_PATH, user_id) is None,
        }

    @fastapi_app.get("/api/settings/starters")
    async def read_starters(current_user=Depends(get_current_user)):
        user_id = _user_id(current_user)
        return {"starters": list_user_starters(CHAT_DB_PATH, user_id)}

    @fastapi_app.put("/api/settings/starters")
    async def update_starters(
        request: StartersRequest,
        current_user=Depends(get_current_user),
    ):
        user_id = _user_id(current_user)
        payload = [s.model_dump() for s in request.starters]
        return {"starters": replace_user_starters(CHAT_DB_PATH, user_id, payload)}

    @fastapi_app.post("/api/kbs/{kb_id}/generate/system-prompt")
    async def generate_kb_system_prompt(
        kb_id: str,
        current_user=Depends(get_current_user),
    ):
        kb = _resolve_readable_kb(current_user, kb_id)
        reference = default_system_prompt or ""
        try:
            prompt = await generate_system_prompt(kb["qdrant_collection"], reference)
        except RuntimeError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"system_prompt": prompt}

    @fastapi_app.post("/api/kbs/{kb_id}/generate/starters")
    async def generate_kb_starters(
        kb_id: str,
        current_user=Depends(get_current_user),
    ):
        kb = _resolve_readable_kb(current_user, kb_id)
        reference = default_system_prompt or ""
        try:
            starters = await generate_starters(kb["qdrant_collection"], reference)
        except RuntimeError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"starters": starters}

    @fastapi_app.get("/api/me")
    async def read_current_user(current_user=Depends(get_current_user)):
        identifier = _user_id(current_user)
        role, status = "user", "approved"
        if DATABASE_URL:
            pair = await get_user_role_status(DATABASE_URL, identifier)
            if pair is not None:
                role, status = pair
        return {"identifier": identifier, "role": role, "status": status}

    @fastapi_app.get("/api/terms")
    async def read_terms_status(current_user=Depends(get_current_user)):
        identifier = _user_id(current_user)
        accepted_at: str | None = None
        accepted_version: str | None = None
        if DATABASE_URL:
            info = await get_terms_status(DATABASE_URL, identifier)
            if info is not None:
                accepted_at = info.get("accepted_at")
                accepted_version = info.get("accepted_version")
        return {
            "version": TERMS_VERSION,
            "accepted_at": accepted_at,
            "accepted_version": accepted_version,
            "up_to_date": accepted_version == TERMS_VERSION and accepted_at is not None,
            "summary": {"de": _SUMMARY_DE, "en": _SUMMARY_EN},
            "pdf": {
                "de": "https://docs.sc.hpi.de/attachments/aisc/nutzungsvertrag_kisz_de.pdf",
                "en": "https://docs.sc.hpi.de/attachments/aisc/nutzungsvertrag_kisz_en.pdf",
            },
        }

    @fastapi_app.post("/api/terms/accept")
    async def accept_terms_route(
        request: TermsAcceptRequest,
        current_user=Depends(get_current_user),
    ):
        identifier = _user_id(current_user)
        if request.version != TERMS_VERSION:
            raise HTTPException(status_code=409, detail="Veraltete Nutzungsbedingungen")
        if not DATABASE_URL:
            raise HTTPException(status_code=503, detail="User directory not configured")
        row = await accept_terms(DATABASE_URL, identifier, TERMS_VERSION)
        if row is None:
            raise HTTPException(status_code=404, detail="Benutzer nicht gefunden")
        return {
            "version": TERMS_VERSION,
            "accepted_at": row["accepted_at"],
            "accepted_version": row["accepted_version"],
        }

    @fastapi_app.get("/api/admin/users")
    async def admin_list_users(current_user=Depends(get_current_user)):
        await _require_admin(current_user)
        return {"users": await list_all_users(DATABASE_URL)}

    @fastapi_app.patch("/api/admin/users/{user_id}")
    async def admin_update_user(
        user_id: str,
        request: UserAdminUpdateRequest,
        current_user=Depends(get_current_user),
    ):
        admin_identifier = await _require_admin(current_user)
        if request.status is None and request.role is None:
            raise HTTPException(status_code=400, detail="Keine Änderung angegeben")
        # Look up target identifier up front so we can refuse self-lockout
        # before touching the row.
        existing = [u for u in await list_all_users(DATABASE_URL) if u["id"] == user_id]
        if not existing:
            raise HTTPException(status_code=404, detail="Benutzer nicht gefunden")
        if existing[0]["identifier"] == admin_identifier and (
            request.role == "user" or request.status == "blocked"
        ):
            raise HTTPException(
                status_code=400,
                detail="Du kannst deine eigene Admin-Rolle nicht entfernen oder dich sperren.",
            )
        try:
            row = await update_user_admin_fields(
                DATABASE_URL,
                user_id,
                status=request.status,
                role=request.role,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if row is None:
            raise HTTPException(status_code=404, detail="Benutzer nicht gefunden")
        return row

    async def _hard_delete_user(identifier: str) -> dict[str, Any]:
        """Orchestrate a full user teardown across Qdrant, disk, SQLite, Postgres.

        Used by both DELETE /api/me (self) and DELETE /api/admin/users/{id} (admin).
        The ordering (Qdrant/disk → SQLite → Postgres) is deliberate: we need the
        KB rows to enumerate collections+files, so drop external resources first
        while the rows still exist; then SQLite; then the Postgres User row.
        Partial failures leave orphaned external state but never orphaned auth
        data — better than the reverse.
        """
        qdrant_collections_dropped = 0
        pdfs_unlinked = 0

        # Step 1 — Qdrant collections + disk PDFs for every KB this user owns.
        for kb in list_kbs(CHAT_DB_PATH, identifier):
            for doc in list_documents(CHAT_DB_PATH, kb["id"]):
                path = _persisted_doc_path(doc["id"], doc["file_name"])
                if path.exists():
                    path.unlink(missing_ok=True)
                    pdfs_unlinked += 1
            drop_collection(kb["qdrant_collection"])
            qdrant_collections_dropped += 1

        # Step 2 — SQLite: all 7 user-scoped tables in one transaction.
        sqlite_counts = hard_delete_user_data(CHAT_DB_PATH, identifier)

        # Step 3 — Postgres: Threads (cascades to Steps/Elements/Feedback), then User.
        pg_counts = {"threads": 0, "users": 0}
        if DATABASE_URL:
            pg_counts = await delete_user_cascade(DATABASE_URL, identifier)

        logger.info(
            "user.hard_delete identifier=%s qdrant=%d pdfs=%d sqlite=%s pg=%s",
            identifier, qdrant_collections_dropped, pdfs_unlinked,
            sqlite_counts, pg_counts,
        )
        return {
            "identifier": identifier,
            "qdrant_collections_dropped": qdrant_collections_dropped,
            "pdfs_unlinked": pdfs_unlinked,
            "sqlite": sqlite_counts,
            "postgres": pg_counts,
        }

    def _is_env_admin(identifier: str) -> bool:
        """True if this identifier is the env-admin whose credentials come
        from CHAINLIT_AUTH_USERNAME/PASSWORD. Deleting that user is pointless
        because the auth callback's fallback re-creates it on the next login,
        and the re-created row has no accepted-terms / KBs, which just
        confuses the operator."""
        return bool(CHAINLIT_AUTH_USERNAME) and identifier == CHAINLIT_AUTH_USERNAME

    @fastapi_app.delete("/api/me")
    async def delete_own_account(current_user=Depends(get_current_user)):
        identifier = _user_id(current_user)
        if _is_env_admin(identifier):
            raise HTTPException(
                status_code=400,
                detail=(
                    "Das Umgebungs-Admin-Konto kann nicht gelöscht werden — "
                    "es würde beim nächsten Login automatisch neu angelegt. "
                    "Zum Testen der Löschfunktion bitte einen regulären "
                    "Benutzer im Admin-Bereich anlegen und damit testen."
                ),
            )
        result = await _hard_delete_user(identifier)
        return {"deleted": True, **result}

    @fastapi_app.delete("/api/admin/users/{user_id}")
    async def admin_delete_user(user_id: str, current_user=Depends(get_current_user)):
        admin_identifier = await _require_admin(current_user)
        # Resolve UUID → identifier (admin.html knows the UUID from the list).
        target = next(
            (u for u in await list_all_users(DATABASE_URL) if u["id"] == user_id),
            None,
        )
        if target is None:
            raise HTTPException(status_code=404, detail="Benutzer nicht gefunden")
        if target["identifier"] == admin_identifier:
            raise HTTPException(
                status_code=400,
                detail="Zum Löschen des eigenen Kontos bitte /api/me verwenden.",
            )
        if _is_env_admin(target["identifier"]):
            raise HTTPException(
                status_code=400,
                detail=(
                    "Das Umgebungs-Admin-Konto kann nicht gelöscht werden — "
                    "es würde beim nächsten Login automatisch neu angelegt."
                ),
            )
        result = await _hard_delete_user(target["identifier"])
        return {"deleted": True, **result}

    @fastapi_app.post("/api/admin/users")
    async def admin_create_user(
        request: UserCreateRequest,
        current_user=Depends(get_current_user),
    ):
        await _require_admin(current_user)
        identifier = request.identifier.strip()
        if not identifier:
            raise HTTPException(status_code=400, detail="Benutzername erforderlich")
        if len(request.password) < 8:
            raise HTTPException(
                status_code=400,
                detail="Passwort muss mindestens 8 Zeichen lang sein",
            )
        role = request.role or "user"
        if role not in ("user", "admin"):
            raise HTTPException(status_code=400, detail=f"Ungültige Rolle: {role}")

        password_hash = bcrypt.hashpw(
            request.password.encode("utf-8"), bcrypt.gensalt()
        ).decode("utf-8")

        row = await create_user_with_password(
            DATABASE_URL,
            identifier,
            password_hash,
            email=(request.email or None),
            role=role,
        )
        if row is None:
            raise HTTPException(
                status_code=409,
                detail=f"Benutzer mit dieser Kennung existiert bereits: {identifier}",
            )
        return row

    return [
        "/api/kbs",
        "/api/kbs/{kb_id}",
        "/api/kbs/{kb_id}/documents",
        "/api/kbs/{kb_id}/documents/{doc_id}",
        "/api/kbs/{kb_id}/generate/system-prompt",
        "/api/kbs/{kb_id}/generate/starters",
        "/api/settings/system-prompt",
        "/api/settings/starters",
        "/api/me",
        "/api/admin/users",
        "/api/admin/users/{user_id}",
        "/api/terms",
        "/api/terms/accept",
    ]
