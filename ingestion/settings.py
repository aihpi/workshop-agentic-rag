from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env", override=False)


def _getenv(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    return value if value is not None else default


LITELLM_BASE_URL = _getenv("LITELLM_BASE_URL")
LITELLM_API_KEY = _getenv("LITELLM_API_KEY")
EMBED_MODEL = _getenv("EMBED_MODEL", "text-embedding-3-large")

QDRANT_URL = _getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = _getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = _getenv("QDRANT_COLLECTION", "grundschutz")

MAX_FILE_SIZE_MB = int(_getenv("MAX_FILE_SIZE_MB", "50"))
CHUNK_MAX_CHARS = int(_getenv("CHUNK_MAX_CHARS", "3000"))
CHUNK_OVERLAP = int(_getenv("CHUNK_OVERLAP", "300"))
EMBED_BATCH_SIZE = int(_getenv("EMBED_BATCH_SIZE", "64"))
EMBED_MAX_BATCH_CHARS = int(_getenv("EMBED_MAX_BATCH_CHARS", "20000"))
