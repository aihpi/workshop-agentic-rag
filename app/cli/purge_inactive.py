"""One-shot purge of users inactive for >= PURGE_INACTIVE_DAYS (default 90).

Invoked by the k8s CronJob `purge-inactive-users` in [k8s/purge-cronjob.yaml].
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


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )
    raise SystemExit(asyncio.run(main()))
