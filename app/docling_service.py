"""GPU-backed Docling preprocessing service.

Thin FastAPI wrapper that calls the same `parse_file()` function the main
app uses — but runs on a pod with a GPU resource request so Docling's
DocumentConverter auto-detects CUDA and processes PDFs ~10x faster than
the CPU path.

Deployed via [k8s/docling-service.yaml](../k8s/docling-service.yaml) with
replicas: 0 by default; scale manually around workshop hours. The main
app checks this service on each upload (see `_parse_pdf_remote` in
`kb.ingestion_pipeline`) and falls back to local CPU Docling on any
connect/read failure — so scaling replicas to 0 reverts to CPU without
any env change.
"""

from __future__ import annotations

import asyncio
import logging
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from core.settings import MAX_FILE_SIZE_MB
from kb.ingestion_pipeline import SUPPORTED_EXTENSIONS, parse_file, validate_pdf_upload

logger = logging.getLogger(__name__)

# Pre-warm flag. Flips True at startup after the first DocumentConverter
# construction finishes (which loads layout + table models into GPU memory).
# /healthz fails until this flips, so the readiness probe blocks traffic
# during model load.
_ready = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm up Docling at startup.

    Constructing a DocumentConverter is cheap — the expensive bit (pipeline
    init + model weights → GPU memory) only happens on the first actual
    .convert() call. So we prime the shared converter AND run one real
    parse against a known-good baked-in PDF. By the time the readiness
    probe flips green, the first user upload skips the model-load cost.
    """
    global _ready
    logger.info("docling_service.startup begin")
    try:
        from kb.ingestion_pipeline import _get_pdf_converter

        # Prime the module-level cache.
        converter = _get_pdf_converter()

        # Force pipeline init + weight load via a real parse. The BSI standard
        # PDFs are baked into the image under /data/data_raw/; any one works.
        warmup_pdf = Path("/data/data_raw/standard_200_1.pdf")
        if warmup_pdf.is_file():
            logger.info("docling_service.warmup parsing=%s", warmup_pdf)
            converter.convert(str(warmup_pdf))
            logger.info("docling_service.warmup done")
        else:
            logger.warning(
                "docling_service.warmup skipped — %s missing; first user upload will pay model-load cost",
                warmup_pdf,
            )

        _ready = True
        logger.info("docling_service.startup ready")
    except Exception:
        logger.exception("docling_service.startup failed")
        # Leave _ready False so readiness probe fails and traffic is blocked.
    yield
    logger.info("docling_service.shutdown")


app = FastAPI(title="docling-service", lifespan=lifespan)


@app.get("/healthz")
async def healthz():
    if not _ready:
        raise HTTPException(status_code=503, detail="Docling not warm yet")
    return {"status": "ok"}


@app.post("/process")
async def process(file: UploadFile = File(...)):
    """Parse an uploaded PDF and return the section list the main app's
    ingest pipeline already expects. Same output shape as calling
    `parse_file()` locally.

    Defence-in-depth: the app upload endpoint already does extension +
    Content-Type + magic-byte checks before forwarding here, but we
    re-validate at the service so cluster-internal callers (debug scripts,
    future workers) can't bypass.
    """
    original_name = file.filename or "upload"
    ext = Path(original_name).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file extension {ext or '(empty)'}; allowed: {sorted(SUPPORTED_EXTENSIONS)}",
        )
    if file.content_type and file.content_type != "application/pdf":
        raise HTTPException(
            status_code=415,
            detail=f"Content-Type {file.content_type} not supported; only application/pdf.",
        )

    # Stream body to a tempfile, enforcing the same size cap as the app endpoint.
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

    # Magic-byte guard — rejects renamed files before Docling touches them.
    try:
        validate_pdf_upload(tmp_path)
    except ValueError as exc:
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=415, detail=str(exc))

    try:
        # parse_file is sync + CPU/GPU-bound → to_thread to keep the event loop free.
        sections = await asyncio.to_thread(parse_file, tmp_path)
    except Exception:
        logger.exception("docling_service.parse_failed filename=%s size=%d", original_name, total)
        raise HTTPException(status_code=500, detail="Docling parse failed")
    finally:
        tmp_path.unlink(missing_ok=True)

    return JSONResponse({"sections": sections})
