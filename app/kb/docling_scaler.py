"""In-process autoscaler for the docling-service Deployment.

Pre-flight scaling: each upload triggers `ensure_capacity()` before queueing
its chunks; a background task scales back to 0 after
`DOCLING_IDLE_SCALEDOWN_SECONDS` of zero in-flight requests.

Single-pod design: this module keeps `last_request_ts` and `in_flight` in
process memory and is **not** safe across multiple app replicas. The main
app deployment is `replicas: 1, strategy: Recreate` today
([k8s/deployment.yaml](../../k8s/deployment.yaml)). If we ever go
multi-replica, two pods will both observe their own in-flight counters and
race on scale-up/down — k8s API patches are idempotent so the failure mode
is over-provisioning, not data corruption. Move state to Redis +
leader-election when that day comes.

Failure modes:
- not running in-cluster (local dev) → `_SCALER_DISABLED=True` after first
  call, all subsequent calls are silent no-ops.
- RBAC denied (manifest not applied) → same path; one WARNING log, then
  silence. Manual `kubectl scale` remains the backstop.

ArgoCD ignores `/spec/replicas` on docling-service
([k8s/argocd/application.yaml](../../k8s/argocd/application.yaml)) so the
scale patches we issue here aren't reverted by selfHeal.
"""

from __future__ import annotations

import asyncio
import logging
import time

from core.settings import (
    DOCLING_DEPLOYMENT_NAME,
    DOCLING_DEPLOYMENT_NAMESPACE,
    DOCLING_IDLE_SCALEDOWN_SECONDS,
    DOCLING_MAX_REPLICAS,
)

logger = logging.getLogger(__name__)


# Module-level state. Bumped/read by ingestion_pipeline.parse_pdf_async and
# the idle-scaledown loop below. Plain ints are fine because the GIL makes
# `+=` atomic for CPython integers and we only ever do compare-then-act
# inside a single coroutine — no cross-coroutine read-modify-write.
_in_flight: int = 0
_last_request_ts: float = 0.0

_SCALER_DISABLED: bool = False
_disabled_reason: str | None = None
_apps_v1 = None  # kubernetes.client.AppsV1Api or None


def _try_load() -> None:
    """Load in-cluster k8s config once. Sets _SCALER_DISABLED on failure."""
    global _SCALER_DISABLED, _disabled_reason, _apps_v1
    if _SCALER_DISABLED or _apps_v1 is not None:
        return
    try:
        from kubernetes import client, config  # type: ignore
    except ImportError as exc:
        _SCALER_DISABLED = True
        _disabled_reason = f"kubernetes lib missing: {exc}"
        logger.warning("docling_scaler.disabled reason=%s", _disabled_reason)
        return
    try:
        config.load_incluster_config()
    except Exception as exc:  # noqa: BLE001
        _SCALER_DISABLED = True
        _disabled_reason = f"not in-cluster: {type(exc).__name__}"
        logger.warning("docling_scaler.disabled reason=%s", _disabled_reason)
        return
    _apps_v1 = client.AppsV1Api()
    logger.info(
        "docling_scaler.loaded namespace=%s deployment=%s max_replicas=%d idle_scaledown_s=%d",
        DOCLING_DEPLOYMENT_NAMESPACE,
        DOCLING_DEPLOYMENT_NAME,
        DOCLING_MAX_REPLICAS,
        DOCLING_IDLE_SCALEDOWN_SECONDS,
    )


def mark_request_start() -> None:
    """Bump in_flight + last_request_ts before any await.

    Called by parse_pdf_async at the very top so the idle-scaledown task's
    snapshot can never observe `in_flight==0 AND idle` for a request that
    has actually arrived.
    """
    global _in_flight, _last_request_ts
    _in_flight += 1
    _last_request_ts = time.monotonic()


def mark_request_end() -> None:
    global _in_flight
    _in_flight = max(0, _in_flight - 1)


def _read_replicas_blocking() -> tuple[int, int]:
    """Returns (spec_replicas, ready_replicas). Sync; call via to_thread.

    Plain GET on the Deployment — needs `get deployments`, not the
    deployments/status subresource verb. Both spec.replicas and
    status.readyReplicas come back in one call.
    """
    assert _apps_v1 is not None
    dep = _apps_v1.read_namespaced_deployment(
        name=DOCLING_DEPLOYMENT_NAME, namespace=DOCLING_DEPLOYMENT_NAMESPACE,
    )
    spec_replicas = int(getattr(dep.spec, "replicas", 0) or 0)
    ready = int(getattr(dep.status, "ready_replicas", 0) or 0)
    return spec_replicas, ready


def _patch_scale_blocking(replicas: int) -> None:
    """Patch the scale subresource. Sync; call via to_thread."""
    assert _apps_v1 is not None
    _apps_v1.patch_namespaced_deployment_scale(
        name=DOCLING_DEPLOYMENT_NAME,
        namespace=DOCLING_DEPLOYMENT_NAMESPACE,
        body={"spec": {"replicas": replicas}},
    )


async def ensure_capacity(desired: int) -> None:
    """Scale up to min(desired, DOCLING_MAX_REPLICAS) if currently lower.

    Idempotent. No-ops in local dev / RBAC-denied. Errors are logged and
    swallowed — the upload should proceed and try the existing replicas;
    if there are none, the readiness wait will surface the failure.
    """
    _try_load()
    if _SCALER_DISABLED:
        return
    target = max(1, min(desired, DOCLING_MAX_REPLICAS))
    try:
        spec_replicas, _ = await asyncio.to_thread(_read_replicas_blocking)
    except Exception as exc:  # noqa: BLE001
        logger.warning("docling_scaler.read_failed err=%s", exc)
        return
    if spec_replicas >= target:
        return
    try:
        await asyncio.to_thread(_patch_scale_blocking, target)
        logger.info(
            "docling_scaler.scale_up from=%d to=%d (desired=%d)",
            spec_replicas, target, desired,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("docling_scaler.patch_failed target=%d err=%s", target, exc)


async def wait_for_ready_replicas(min_ready: int, timeout_s: float) -> bool:
    """Poll Deployment.status.readyReplicas until >= min_ready or timeout.

    Returns True on success, False on timeout. In disabled / error mode
    returns True immediately so callers don't block uploads in local dev —
    they'll either hit a real Docling at the configured URL, or fail at the
    httpx call with ConnectError, which existing fallbacks already handle.
    """
    _try_load()
    if _SCALER_DISABLED:
        return True
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            _, ready = await asyncio.to_thread(_read_replicas_blocking)
        except Exception as exc:  # noqa: BLE001
            logger.warning("docling_scaler.wait_read_failed err=%s", exc)
            return True  # don't block the upload on a transient API hiccup
        if ready >= min_ready:
            return True
        if time.monotonic() >= deadline:
            logger.warning(
                "docling_scaler.wait_timeout min_ready=%d timeout_s=%.0f",
                min_ready, timeout_s,
            )
            return False
        await asyncio.sleep(2.0)


async def idle_scaledown_loop(stop_event: asyncio.Event | None = None) -> None:
    """Long-running background task. Scales to 0 when truly idle.

    Intentionally simple: every 60s, if `in_flight == 0` AND
    `monotonic() - last_request_ts > IDLE_SECONDS`, patch replicas=0.
    Both conditions together close the race with `mark_request_start`,
    which bumps in_flight before any await.
    """
    _try_load()
    if _SCALER_DISABLED:
        return
    logger.info(
        "docling_scaler.idle_loop_start interval_s=60 idle_threshold_s=%d",
        DOCLING_IDLE_SCALEDOWN_SECONDS,
    )
    while True:
        if stop_event is not None and stop_event.is_set():
            return
        await asyncio.sleep(60.0)
        if _in_flight > 0:
            continue
        idle_for = time.monotonic() - _last_request_ts
        if idle_for < DOCLING_IDLE_SCALEDOWN_SECONDS:
            continue
        try:
            spec_replicas, _ = await asyncio.to_thread(_read_replicas_blocking)
        except Exception as exc:  # noqa: BLE001
            logger.warning("docling_scaler.idle_read_failed err=%s", exc)
            continue
        if spec_replicas == 0:
            continue
        try:
            await asyncio.to_thread(_patch_scale_blocking, 0)
            logger.info(
                "docling_scaler.scale_down from=%d to=0 idle_for_s=%.0f",
                spec_replicas, idle_for,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("docling_scaler.idle_patch_failed err=%s", exc)


def in_flight() -> int:
    """Read-only accessor for tests/diagnostics."""
    return _in_flight
