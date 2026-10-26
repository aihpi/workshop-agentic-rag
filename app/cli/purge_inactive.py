"""One-shot purge of users inactive for >= PURGE_INACTIVE_DAYS (default 90).

Runs daily inside the app pod via `purge_loop()` (started in app.py
on_app_startup). It used to be a k8s CronJob, but that opened the SQLite
file at CHAT_DB_PATH from a second pod on the NFS PVC and took the app down
(issue #1). Never run it from another pod. Manual run, same pod:

    kubectl -n workshop-agentic-rag exec deploy/workshop-app -- \
        python -m cli.purge_inactive

Reuses the same hard_delete_user orchestrator as the HTTP delete endpoints so
cascade semantics stay in one place.

Env-admin (CHAINLIT_AUTH_USERNAME) is skipped here AND refused by
hard_delete_user — belt + braces, because the auth callback would just
recreate that row on next login anyway.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys

from api.api_routes import hard_delete_user
from chat.native_chat import list_inactive_users
from core.settings import CHAINLIT_AUTH_USERNAME, DATABASE_URL


log = logging.getLogger("purge_inactive")


def _days() -> int:
    raw = os.getenv("PURGE_INACTIVE_DAYS", "90")
    try:
        value = int(raw)
    except ValueError as exc:
        raise SystemExit(f"PURGE_INACTIVE_DAYS must be an int, got {raw!r}") from exc
    if value < 1:
        raise SystemExit(f"PURGE_INACTIVE_DAYS must be >= 1, got {value}")
    return value


async def main() -> int:
    if not DATABASE_URL:
        log.error("DATABASE_URL not configured; nothing to do")
        return 1

    days = _days()
    candidates = await list_inactive_users(DATABASE_URL, days)
    log.info("purge_start days=%d candidates=%d", days, len(candidates))

    purged = failed = skipped = 0
    for row in candidates:
        identifier = row["identifier"]
        if identifier == CHAINLIT_AUTH_USERNAME:
            skipped += 1
            log.info("purge_skip_env_admin identifier=%s", identifier)
            continue
        try:
            await hard_delete_user(identifier)
            purged += 1
            log.info(
                "purged identifier=%s last_login=%s",
                identifier, row.get("lastLoginAt"),
            )
        except Exception:
            failed += 1
            log.exception("purge_failed identifier=%s", identifier)

    log.info(
        "purge_summary purged=%d failed=%d skipped=%d",
        purged, failed, skipped,
    )
    return 0 if failed == 0 else 1


async def purge_loop() -> None:
    """Long-running background task: purge 10 min after start, then daily.

    A pod restart resets the 24h interval; harmless because the purge is
    idempotent and the 90-day window has slack.
    """
    await asyncio.sleep(600)
    while True:
        try:
            await main()
        except Exception:
            log.exception("purge_loop_failed")
        await asyncio.sleep(24 * 3600)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )
    raise SystemExit(asyncio.run(main()))
