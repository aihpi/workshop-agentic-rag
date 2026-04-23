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

from kb.ingestion_pipeline import SUPPORTED_EXTENSIONS, parse_file

logger = logging.getLogger(__name__)

# Pre-warm flag. Flips True at startup after the first DocumentConverter
# construction finishes (which loads layout + table models into GPU memory).
# /healthz fails until this flips, so the readiness probe blocks traffic
# during model load.
_ready = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm up Docling at startup so the first /process call doesn't
    pay the 10–15s model-load cost."""
    global _ready
    logger.info("docling_service.startup begin")
    try:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption

        # Construct a throwaway DocumentConverter to trigger model download
        # and GPU memory allocation. Subsequent parse_file() calls re-use
        # the cached weights (torch/Docling's own model cache).
        pdf_opts = PdfPipelineOptions(do_ocr=False)
        DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_opts)},
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
    """Parse an uploaded PDF (or txt/md) and return the section list the
    main app's ingest pipeline already expects. Same output shape as
    calling `parse_file()` locally."""
    original_name = file.filename or "upload"
    ext = Path(original_name).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file extension {ext or '(empty)'}; allowed: {sorted(SUPPORTED_EXTENSIONS)}",
        )

    # Stream body to a tempfile so we don't buffer the full PDF in memory.
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        total = 0
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            tmp.write(chunk)
        tmp_path = Path(tmp.name)

    try:
        # parse_file is sync + CPU/GPU-bound → to_thread to keep the event loop free.
        sections = await asyncio.to_thread(parse_file, tmp_path)
    except Exception:
        logger.exception("docling_service.parse_failed filename=%s size=%d", original_name, total)
        raise HTTPException(status_code=500, detail="Docling parse failed")
    finally:
        tmp_path.unlink(missing_ok=True)

    return JSONResponse({"sections": sections})
