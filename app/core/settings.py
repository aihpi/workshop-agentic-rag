from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env", override=False)


def _getenv(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    return value if value is not None else default


def _getenv_list(name: str, default: list[str] | None = None, sep: str = "||") -> list[str]:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default or []
    return [item.strip() for item in value.split(sep) if item.strip()]


LITELLM_BASE_URL = _getenv("LITELLM_BASE_URL")
LITELLM_API_KEY = _getenv("LITELLM_API_KEY")
CHAT_MODEL = _getenv("CHAT_MODEL", "gpt-4o-mini")
FALLBACK_CHAT_MODEL = _getenv("FALLBACK_CHAT_MODEL")
EMBED_MODEL = _getenv("EMBED_MODEL", "text-embedding-3-large")

QDRANT_URL = _getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = _getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = _getenv("QDRANT_COLLECTION", "grundschutz")

# Optional GPU Docling service. When set, parse_pdf POSTs PDFs to this URL
# instead of running Docling in-process. On connect error / timeout the
# client falls back to local Docling — so scaling the service replicas to
# 0 reverts behavior to local CPU without any env change.
DOCLING_SERVICE_URL = (_getenv("DOCLING_SERVICE_URL", "") or "").rstrip("/")

MAX_FILE_SIZE_MB = int(_getenv("MAX_FILE_SIZE_MB", "50"))
# Per-chat session uploads (PDF/MD/TXT injected into the system prompt).
# Smaller cap than KB uploads because the content goes straight into the
# LLM context window — 10 MB roughly matches 8 000 tokens of textual PDF.
MAX_SESSION_FILE_SIZE_MB = int(_getenv("MAX_SESSION_FILE_SIZE_MB", "10"))
SESSION_DOC_TOKEN_LIMIT = int(_getenv("SESSION_DOC_TOKEN_LIMIT", "8000"))
CHUNK_MAX_CHARS = int(_getenv("CHUNK_MAX_CHARS", "3000"))
CHUNK_OVERLAP = int(_getenv("CHUNK_OVERLAP", "300"))
EMBED_BATCH_SIZE = int(_getenv("EMBED_BATCH_SIZE", "64"))
EMBED_MAX_BATCH_CHARS = int(_getenv("EMBED_MAX_BATCH_CHARS", "20000"))
TOP_K = int(_getenv("TOP_K", "5"))
MAX_TOP_K = int(_getenv("MAX_TOP_K", str(TOP_K)))
MAX_SOURCE_LINKS = int(_getenv("MAX_SOURCE_LINKS", "8"))
SCORE_THRESHOLD = float(_getenv("SCORE_THRESHOLD", "0.0"))
STREAMING_ENABLED = (_getenv("STREAMING_ENABLED", "false") or "false").lower() == "true"
STREAMING_DOUBLE_PASS = (_getenv("STREAMING_DOUBLE_PASS", "false") or "false").lower() == "true"
LANGFLOW_ENABLED = (_getenv("LANGFLOW_ENABLED", "false") or "false").lower() == "true"
LANGFLOW_BASE_URL = (_getenv("LANGFLOW_BASE_URL", "http://localhost:7860") or "http://localhost:7860").rstrip("/")
LANGFLOW_FLOW_ID = _getenv("LANGFLOW_FLOW_ID")
LANGFLOW_API_KEY = _getenv("LANGFLOW_API_KEY")
LANGFLOW_OUTPUT_COMPONENT = _getenv("LANGFLOW_OUTPUT_COMPONENT")

SYSTEM_PROMPT_PATH = Path(
    _getenv(
        "SYSTEM_PROMPT_PATH",
        str((BASE_DIR / ".." / "system.md").resolve()),
    )
)

DATA_RAW_DIR = Path(
    _getenv(
        "DATA_RAW_DIR",
        str((BASE_DIR / ".." / "data" / "data_raw").resolve()),
    )
)

DATA_KB_DOCS_DIR = Path(
    _getenv(
        "DATA_KB_DOCS_DIR",
        str((BASE_DIR / ".." / "data" / "kb_docs").resolve()),
    )
)
DATA_KB_DOCS_DIR.mkdir(parents=True, exist_ok=True)

GRUNDSCHUTZ_SOURCE_PDF = (
    _getenv("GRUNDSCHUTZ_SOURCE_PDF", "IT_Grundschutz_Kompendium_Edition2023.pdf")
    or "IT_Grundschutz_Kompendium_Edition2023.pdf"
)

CITATION_MAP_PATH = Path(
    _getenv(
        "CITATION_MAP_PATH",
        str((BASE_DIR / "citation_map.json").resolve()),
    )
)

STARTER_QUESTIONS = _getenv_list(
    "STARTER_QUESTIONS",
    default=[
        "Was ist der Unterschied zwischen Prozess- und Systembausteinen?",
        "Welche Schritte umfasst die Basis-Absicherung nach BSI-Standard 200-2?",
        "Wie müssen Passwörter bei der Authentisierung am Webserver gesichert werden?",
    ],
)

CHAT_DB_PATH = Path(
    _getenv(
        "CHAT_DB_PATH",
        str((BASE_DIR / ".chainlit" / "chat_history.sqlite3").resolve()),
    )
)

CHAT_EXPORT_DIR = Path(
    _getenv(
        "CHAT_EXPORT_DIR",
        str((BASE_DIR / ".files" / "chat_exports").resolve()),
    )
)

DATABASE_URL = _getenv("DATABASE_URL")
CHAINLIT_AUTH_USERNAME = _getenv("CHAINLIT_AUTH_USERNAME", "admin")
CHAINLIT_AUTH_PASSWORD = _getenv("CHAINLIT_AUTH_PASSWORD", "admin")
CHAINLIT_INIT_DB = (_getenv("CHAINLIT_INIT_DB", "true") or "true").lower() == "true"

# Comma-separated list of GitHub logins / usernames auto-promoted to role=admin
# on each login. Empty ⇒ admin role can only be granted manually via the admin UI.
ADMIN_IDENTIFIERS = _getenv_list("ADMIN_IDENTIFIERS", default=[], sep=",")

# Version string for the usage agreement. Bump to force every user to re-accept
# (e.g. after a wording change). Users whose stored version differs from this
# are re-prompted on next login.
TERMS_VERSION = _getenv("TERMS_VERSION", "hpi-aisc-bb-2026-01") or "hpi-aisc-bb-2026-01"

# ---------------------------------------------------------------------------
# Personalization Settings
# ---------------------------------------------------------------------------
PERSONALIZATION_ENABLED = (_getenv("PERSONALIZATION_ENABLED", "true") or "true").lower() == "true"
PROFILE_MIN_MESSAGES = int(_getenv("PROFILE_MIN_MESSAGES", "5"))
PROFILE_TOPIC_LIMIT = int(_getenv("PROFILE_TOPIC_LIMIT", "8"))
PROFILE_RELEVANCE_THRESHOLD = float(_getenv("PROFILE_RELEVANCE_THRESHOLD", "0.3"))
PERSONALIZED_FOLLOWUPS_COUNT = int(_getenv("PERSONALIZED_FOLLOWUPS_COUNT", "2"))

# Validate personalization settings at import time
if PROFILE_MIN_MESSAGES < 1:
    raise ValueError(f"PROFILE_MIN_MESSAGES must be >= 1, got {PROFILE_MIN_MESSAGES}")
if PROFILE_TOPIC_LIMIT < 0:
    raise ValueError(f"PROFILE_TOPIC_LIMIT must be >= 0, got {PROFILE_TOPIC_LIMIT}")
if not (0.0 <= PROFILE_RELEVANCE_THRESHOLD <= 1.0):
    raise ValueError(f"PROFILE_RELEVANCE_THRESHOLD must be between 0.0 and 1.0, got {PROFILE_RELEVANCE_THRESHOLD}")
if PERSONALIZED_FOLLOWUPS_COUNT < 0:
    raise ValueError(f"PERSONALIZED_FOLLOWUPS_COUNT must be >= 0, got {PERSONALIZED_FOLLOWUPS_COUNT}")
