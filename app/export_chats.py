from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from chat_history import (
    export_all_messages_csv,
    export_all_sessions_jsonl,
    export_session_json,
    init_chat_db,
)
from settings import CHAT_DB_PATH, CHAT_EXPORT_DIR


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def main() -> int:
    parser = argparse.ArgumentParser(description="Export saved Chainlit chat history.")
    parser.add_argument("--db-path", type=Path, default=CHAT_DB_PATH, help="Path to SQLite chat DB.")
    parser.add_argument("--out-dir", type=Path, default=CHAT_EXPORT_DIR, help="Directory for export files.")
    parser.add_argument("--session-id", type=str, default=None, help="Export only one session ID (JSON).")
    parser.add_argument(
        "--format",
        choices=["json", "jsonl", "csv", "all"],
        default="all",
        help="Export format for all sessions. Ignored when --session-id is set.",
    )
    args = parser.parse_args()

    init_chat_db(args.db_path)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = _stamp()

    if args.session_id:
        out = args.out_dir / f"chat-export-{args.session_id}-{stamp}.json"
        export_session_json(args.db_path, args.session_id, out)
        print(out)
        return 0

    exported: list[Path] = []
    if args.format in {"jsonl", "all"}:
        out_jsonl = args.out_dir / f"chat-export-all-{stamp}.jsonl"
        export_all_sessions_jsonl(args.db_path, out_jsonl)
        exported.append(out_jsonl)
    if args.format in {"csv", "all"}:
        out_csv = args.out_dir / f"chat-export-all-{stamp}.csv"
        export_all_messages_csv(args.db_path, out_csv)
        exported.append(out_csv)
    if args.format == "json":
        # Per-session JSON is only supported with --session-id to keep output deterministic.
        raise SystemExit("--format json requires --session-id")

    for path in exported:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
