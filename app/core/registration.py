"""Invite-code gate for self-registration.

The code is DERIVED, not stored: HMAC(secret, UTC date) → 6 digits, prefixed.
So it rotates every 24h on its own — no table, no cron job, no cleanup, and
nothing to revoke. Learning one day's code reveals nothing about any other.

Stdlib only on purpose: app.py cannot be imported without pulling in chainlit,
litellm and qdrant, so gate logic living there would have no runnable check.
Run this file directly to exercise the asserts at the bottom.
"""
from __future__ import annotations

import hmac
import time
from collections import deque
from datetime import date, datetime, timedelta, timezone
from hashlib import sha256

CODE_PREFIX = "workshop-rag-"
CODE_DIGITS = 6

# ponytail: in-process failure counter — per-replica, no persistence, and a pod
# restart clears the lock. A true global cap only while replicas=1
# (k8s/deployment.yaml:8); scaling out silently multiplies the effective limit.
# Move to a Postgres row if this ever runs more than one replica.
FAIL_LIMIT = 20
FAIL_WINDOW_SECONDS = 600

# maxlen does the eviction, so there is no pruning loop and both operations are
# O(1). _failures[0] is the oldest of the last FAIL_LIMIT failures.
_failures: deque[float] = deque(maxlen=FAIL_LIMIT)


def today_utc() -> date:
    """Rotation is keyed on the UTC date so it doesn't depend on host TZ."""
    return datetime.now(timezone.utc).date()


def code_for_day(secret: str, day: date) -> str:
    """The invite code valid on `day`. Empty string when no secret is set."""
    if not secret:
        return ""
    digest = hmac.new(
        secret.encode("utf-8"), day.isoformat().encode("ascii"), sha256
    ).hexdigest()
    number = int(digest[:8], 16) % (10 ** CODE_DIGITS)
    return f"{CODE_PREFIX}{number:0{CODE_DIGITS}d}"


def current_code(secret: str, day: date | None = None) -> str:
    return code_for_day(secret, day or today_utc())


def valid_codes(secret: str, day: date | None = None) -> list[str]:
    """Today's code plus yesterday's.

    The grace period means a code handed out shortly before the UTC rollover
    keeps working, so nobody is locked out mid-session. Consequence: any given
    code is accepted for up to 48h.
    """
    day = day or today_utc()
    return [code_for_day(secret, day), code_for_day(secret, day - timedelta(days=1))]


def next_rotation_utc(day: date | None = None) -> datetime:
    """When the current code stops being the current one (for display)."""
    day = day or today_utc()
    return datetime.combine(day + timedelta(days=1), datetime.min.time(), timezone.utc)


def code_matches(submitted: str, secret: str, day: date | None = None) -> bool:
    """Constant-time check against every currently valid code.

    False when no secret is configured — that's the kill switch.
    """
    if not secret:
        return False
    # compare_digest raises TypeError on str with non-ASCII characters, so
    # compare bytes. Codes are ASCII by construction; submissions may not be.
    candidate = submitted.strip().encode("utf-8")
    # any() short-circuits, which leaks only *which* of two codes matched —
    # not the code itself. Each individual comparison stays constant-time.
    return any(
        hmac.compare_digest(candidate, code.encode("ascii"))
        for code in valid_codes(secret, day)
    )


def is_locked(now: float | None = None) -> bool:
    """True once FAIL_LIMIT failures have landed inside FAIL_WINDOW_SECONDS."""
    now = time.monotonic() if now is None else now
    return len(_failures) == FAIL_LIMIT and now - _failures[0] < FAIL_WINDOW_SECONDS


def record_failure(now: float | None = None) -> None:
    """Record one wrong-code attempt.

    Callers must return 429 *before* calling this while already locked —
    otherwise failures during a lock slide the window and the 10-minute lock
    becomes indefinite, keeping legitimate users out.
    """
    _failures.append(time.monotonic() if now is None else now)


if __name__ == "__main__":
    d = date(2026, 9, 2)
    code = current_code("seed", d)

    # shape: workshop-rag-<6 digits>
    assert code.startswith(CODE_PREFIX), code
    digits = code[len(CODE_PREFIX):]
    assert len(digits) == CODE_DIGITS and digits.isdigit(), code

    # deterministic per (secret, day), and rotates
    assert current_code("seed", d) == code
    assert current_code("seed", d + timedelta(days=1)) != code
    assert current_code("other-seed", d) != code, "secret must matter"

    # no secret ⇒ closed, and no code is guessable into existence
    assert current_code("", d) == ""
    assert not code_matches("", "", d)
    assert not code_matches(CODE_PREFIX + "000000", "", d)

    # matching, incl. the grace period and whitespace
    assert code_matches(code, "seed", d)
    assert code_matches(f"  {code}  ", "seed", d)
    assert not code_matches(code.upper(), "seed", d), "codes are case-sensitive"
    assert not code_matches(CODE_PREFIX + "999999x", "seed", d)
    yesterday = current_code("seed", d - timedelta(days=1))
    assert code_matches(yesterday, "seed", d), "yesterday's code still valid"
    two_days = current_code("seed", d - timedelta(days=2))
    assert not code_matches(two_days, "seed", d), "grace is one day only"
    assert code_matches("cödeü", "cödeü") is False, "non-ASCII must not raise"

    # rotation timestamp is the following UTC midnight
    assert next_rotation_utc(d) == datetime(2026, 9, 3, tzinfo=timezone.utc)

    # failure window
    assert not is_locked(0.0)
    for i in range(FAIL_LIMIT - 1):
        record_failure(float(i))
    assert not is_locked(float(FAIL_LIMIT)), "one under the limit stays open"
    record_failure(float(FAIL_LIMIT - 1))
    assert is_locked(float(FAIL_LIMIT)), "FAIL_LIMIT failures in-window ⇒ locked"
    assert not is_locked(FAIL_WINDOW_SECONDS + FAIL_LIMIT), "window ages out"

    print("ok —", code, "expires", next_rotation_utc(d).isoformat())
