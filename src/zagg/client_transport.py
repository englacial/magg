"""v2 Event-transport status channel (issue #327): per-shard status objects.

The fleet-scale alternative to holding one synchronous connection per in-flight
shard: workers record each shard's outcome as a tiny JSON object under a
per-run prefix SIBLING to the output store::

    <store>.status/run-<run_id>/shard-<decimal-key>.json

and the client resolves futures by polling that prefix instead of reading an
invoke response. Both halves of the channel live here so the schema has one
home:

- **Worker half** (:func:`write_shard_status`): called from the Lambda
  handler's top-level seam on every per-unit response — every status branch,
  including the caught-error 500 envelope. Always-on (espg ratification on
  issue #327: every run, regardless of dispatcher) and emphatically
  **fail-open**: a status PUT failure logs and never affects the shard result.
  Written through the same ``open_object_store`` factory + output credentials
  as every other worker write, so the dispatcher-never-writes rule (D8) is
  preserved by construction.
- **Client half** (later phases of issue #327): the polling resolver that
  turns one LIST of the run prefix per tick into future resolutions.

Naming: the shard component is the shard key's plain decimal rendering
(``str(int(shard_key))``) — derivable on every handler branch including the
caught-error path, where the grid (and so the issue #199 morton label) may not
be constructible. Windowed units (issue #246) suffix the window label,
mirroring the leaf naming, so two windows of one shard cannot clobber each
other's object. The prefix is a sibling of the store (never inside it): the
store stays vanilla zarr v3, the benchmark object model never sees telemetry,
and an S3 lifecycle/delete rule on the store's key prefix string catches the
status objects with it (espg ruling on issue #327 — accumulation hygiene is
the existing prefix delete policy, no in-band pruning).

Lambda Event-invoke semantics this channel is built for (issue #327
amendments): the fleet's ``EventInvokeConfig`` pins ``MaximumRetryAttempts: 0``
(no service retries — attempt-id dedupe is belt-and-braces, and it also
dedupes the client's own re-dispatches) and ``MaximumEventAgeInSeconds: 60``
(an invoke that cannot start within 60 s is silently dropped — the client
resolves a status-less shard as ``failed-unknown`` after the drop deadline).
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any

from zagg.dispatch import BENIGN_ERRORS

logger = logging.getLogger(__name__)

#: Version stamped into every status object; bump on any key change.
STATUS_SCHEMA_VERSION = 1


def run_status_prefix(store_path: str, run_id: str) -> str:
    """The run's status prefix: ``<store>.status/run-<run_id>`` (a store SIBLING)."""
    return f"{store_path.rstrip('/')}.status/run-{run_id}"


def shard_status_key(shard_key, window: str | None = None) -> str:
    """One shard's status object name under the run prefix.

    ``shard-<decimal-key>.json``, with the window label suffixed for windowed
    units (validated against the frozen window grammar — a label is a path
    component here, exactly as in the leaf/sidecar names).
    """
    base = f"shard-{int(shard_key)}"
    if window is None:
        return f"{base}.json"
    from zagg.windows import validate_label

    validate_label(str(window))
    return f"{base}_{window}.json"


def build_shard_status(event: dict, response: dict) -> tuple[str, dict] | None:
    """(object key, status object) for one per-unit response, or ``None``.

    ``None`` when the event carries no run identity (``run_id`` /
    ``store_path`` / ``shard_key``) — an old dispatcher, or a non-spatial
    unit — so the write is skipped and behavior is byte-identical to
    pre-#327 for those events.

    Status classification mirrors the dispatch accumulators: a 200 envelope
    with no error is ``ok``; a benign no-work error (:data:`BENIGN_ERRORS`)
    is ``no_data``; everything else — including the handler's caught-error
    500 — is ``failed``. ``timings`` is the same ``phase_timings`` dict the
    response body carries; ``body`` is the full envelope body so the poller
    can synthesize the exact per-shard result dict the sync transport returns
    (stats record, counters, container telemetry). ``attempt_id`` is a fresh
    uuid per EXECUTION: a duplicate execution (or a client re-dispatch)
    overwrites the object with a new id, which is what keeps the client's
    retry accounting straight.
    """
    run_id = event.get("run_id")
    store_path = event.get("store_path")
    shard_key = event.get("shard_key")
    if not run_id or not store_path or shard_key is None:
        return None
    try:
        body = json.loads(response.get("body") or "{}")
    except (json.JSONDecodeError, TypeError):
        body = {}
    if not isinstance(body, dict):
        body = {}
    error = body.get("error")
    code = response.get("statusCode")
    if error in BENIGN_ERRORS:
        status = "no_data"
    elif code == 200 and not error:
        status = "ok"
    else:
        status = "failed"
    window = (event.get("window") or {}).get("label")
    obj = {
        "schema_version": STATUS_SCHEMA_VERSION,
        "status": status,
        "attempt_id": uuid.uuid4().hex,
        "run_id": run_id,
        "shard": str(int(shard_key)),
        "window": window,
        "timings": body.get("phase_timings"),
        "error": str(error) if error else None,
        "status_code": code,
        "body": body,
    }
    return shard_status_key(shard_key, window=window), obj


def write_shard_status(event: dict, response: dict, store_kwargs: dict[str, Any]) -> None:
    """PUT the per-shard status object for one response — fail-open, always.

    The worker-half entry point, called from the Lambda handler's top-level
    seam on every per-unit response. ANY exception — building the object,
    opening the store, the PUT itself — is logged and swallowed (espg
    ratification on issue #327: the status PUT must never fail a shard whose
    result is otherwise in hand). ``store_kwargs`` is the handler's own
    output-store resolution, so the write uses exactly the credentials the
    shard's data writes used.
    """
    try:
        built = build_shard_status(event, response)
        if built is None:
            return
        key, obj = built
        import obstore

        from zagg.store import open_object_store

        prefix = run_status_prefix(event["store_path"], event["run_id"])
        obstore.put(open_object_store(prefix, **store_kwargs), key, json.dumps(obj).encode())
        logger.info(f"Wrote shard status ({obj['status']}) to {prefix}/{key}")
    except Exception as e:
        logger.warning(f"shard status write failed (fail-open, issue #327): {e}")
