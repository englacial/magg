"""The sweep-admission lease: per-store serialization of pyramid sweeps (#384).

**Why this exists (the espg ruling).** A column is a multi-object artifact
under D4's stamp-last discipline, which proves completeness only under a
single writer — two concurrent sweeps interleaving PUTs can produce a
*chimera* column (objects from run A, stamp from run B) that validates as
complete: silent corruption. Cross-sweep admission is therefore a
correctness requirement, not operator discipline.

**Control plane, explicitly.** No data object is ever locked — the lease is
an *intent object* at the store root, and it is what makes "every data
object has exactly one writer, ever" true ACROSS runs, extending (never
amending) the no-locking law. It is **store-granular by correctness**, not
scope-granular: scope-disjointness does not imply write-disjointness,
because disjoint-leaf sweeps still converge on shared coarse ancestors (base
cells, the root ``coverage.moc``, the manifest).

**The concurrency matrix (ruled):** fleet ∥ fleet is governed by the
existing leaf single-writer law (the lease adds nothing); fleet ∥ sweep is
allowed (disjoint object sets; the stage worker's stamp validation makes
reads airtight, and a mid-sweep append is recorded-and-healed
under-coverage); sweep ∥ sweep is **serialized per store** — by this module.

**Mechanism.** Admission is one atomic conditional PUT (``If-None-Match:
*``, obstore ``mode="create"``) of :data:`LEASE_NAME` carrying the run id,
scope, heartbeat and TTL. A live intent refuses admission NAMING the running
sweep; an expired heartbeat is claimable (crash recovery): delete, re-create
conditionally, then read back and verify ownership — the delete/create pair
is not atomic, so the verify closes most of the window and the stage
workers' foreign-fresh-stamp abort (:class:`zagg.sweep_stage.ForeignSweepError`)
backstops the rest. The holder heartbeats between stages; the FINISHER
deletes the intent as its final act (lease release rides the singleton
owner). A run that dies mid-sweep leaves its intent to expire — exactly the
claimable-partial-run state the ratchet already heals.
"""

from __future__ import annotations

import json
import logging

logger = logging.getLogger(__name__)

#: The intent object's store-root basename (control plane, never data).
LEASE_NAME = "sweep.lease.json"
#: Envelope version of the intent payload.
LEASE_SPEC = "zagg-sweep-lease/1"
#: Default heartbeat TTL: a holder silent this long is claimable.
DEFAULT_TTL_S = 900


class SweepRefusedError(RuntimeError):
    """Admission refused: another sweep holds a live lease on this store.

    The message names the running sweep (run id, acquisition time, heartbeat
    age) so the operator can find its record or wait it out.
    """


def _payload(run_id: str, scope, ttl_s: int, *, acquired_at: str | None = None) -> dict:
    from zagg.hive import _utcnow

    now = _utcnow()
    return {
        "spec": LEASE_SPEC,
        "run_id": str(run_id),
        "scope": None if scope is None else [str(s) for s in scope],
        "acquired_at": acquired_at or now,
        "heartbeat_at": now,
        "ttl_s": int(ttl_s),
    }


def read_lease(store_root: str, *, store_kwargs: dict | None = None) -> dict | None:
    """The current intent object, or ``None`` (absent/unparsable reads free)."""
    import obstore
    from obstore.exceptions import NotFoundError

    from zagg.store import open_object_store

    store = open_object_store(store_root, **dict(store_kwargs or {}))
    try:
        data = obstore.get(store, LEASE_NAME).bytes()
    except (FileNotFoundError, NotFoundError):
        return None
    try:
        lease = json.loads(bytes(data))
    except ValueError:
        logger.warning(f"sweep lease at {store_root} is not JSON; treating as claimable debris")
        return None
    return lease if isinstance(lease, dict) and lease.get("spec") == LEASE_SPEC else None


def _expired(lease: dict) -> bool:
    """Whether the intent's heartbeat is older than its TTL (UTC ISO math)."""
    from datetime import datetime, timedelta, timezone

    from zagg.windows import parse_utc

    try:
        beat = parse_utc(lease.get("heartbeat_at"))
        ttl = timedelta(seconds=int(lease.get("ttl_s") or DEFAULT_TTL_S))
    except (TypeError, ValueError):
        return True  # malformed heartbeat: claimable debris, never a live lock
    return datetime.now(timezone.utc) - beat > ttl


def acquire_lease(
    store_root: str,
    *,
    run_id: str,
    scope=None,
    ttl_s: int = DEFAULT_TTL_S,
    store_kwargs: dict | None = None,
) -> dict:
    """Admit one sweep: conditional-PUT the intent object, or refuse loudly.

    Returns the payload written (with ``claimed_from`` naming the expired
    prior holder on a crash-recovery claim). Raises :class:`SweepRefusedError`
    when a live foreign intent holds the store — the refusal names it.
    """
    import obstore
    from obstore.exceptions import AlreadyExistsError

    from zagg.store import open_object_store

    store_kwargs = dict(store_kwargs or {})
    store = open_object_store(store_root, **store_kwargs)
    lease = _payload(run_id, scope, ttl_s)
    try:
        obstore.put(store, LEASE_NAME, json.dumps(lease, indent=1).encode(), mode="create")
        return lease
    except AlreadyExistsError:
        pass
    existing = read_lease(store_root, store_kwargs=store_kwargs)
    if existing is not None and existing.get("run_id") == str(run_id):
        return existing  # already ours (an idempotent re-admission)
    if existing is not None and not _expired(existing):
        raise SweepRefusedError(
            f"a sweep already holds this store: run {existing.get('run_id')!r} "
            f"(acquired {existing.get('acquired_at')}, last heartbeat "
            f"{existing.get('heartbeat_at')}, ttl {existing.get('ttl_s')}s) — sweeps "
            f"serialize per store; wait for it, or claim after the heartbeat expires"
        )
    # Expired (or debris): claim. Re-read immediately before the delete and
    # refuse if the intent MOVED since the expiry read (review finding: an
    # unconditional delete could remove a sibling claimant's already-verified
    # fresh intent, admitting two holders) — then delete + conditional create
    # + verify. The remaining GET->DELETE window is the documented residual
    # the stage workers' foreign-fresh-stamp abort backstops.
    recheck = read_lease(store_root, store_kwargs=store_kwargs)
    if (recheck or None) != (existing or None):
        raise SweepRefusedError(
            f"lost the claim race for {store_root}: the intent moved while claiming "
            f"(now held by run {None if recheck is None else recheck.get('run_id')!r})"
        )
    prior = (existing or {}).get("run_id")
    try:
        obstore.delete(store, LEASE_NAME)
    except Exception as e:
        logger.debug(f"sweep lease claim: delete of expired intent raced ({e})")
    claim = _payload(run_id, scope, ttl_s)
    if prior is not None:
        claim["claimed_from"] = prior
    try:
        obstore.put(store, LEASE_NAME, json.dumps(claim, indent=1).encode(), mode="create")
    except AlreadyExistsError:
        pass
    verify = read_lease(store_root, store_kwargs=store_kwargs)
    if verify is None or verify.get("run_id") != str(run_id):
        raise SweepRefusedError(
            f"lost the claim race for {store_root}: run "
            f"{None if verify is None else verify.get('run_id')!r} holds the lease now"
        )
    return claim


def heartbeat_lease(store_root: str, lease: dict, *, store_kwargs: dict | None = None) -> dict:
    """Refresh the holder's heartbeat; refuse if the intent is no longer ours."""
    import obstore

    from zagg.store import open_object_store

    store_kwargs = dict(store_kwargs or {})
    current = read_lease(store_root, store_kwargs=store_kwargs)
    if current is None or current.get("run_id") != lease.get("run_id"):
        raise SweepRefusedError(
            f"sweep lease on {store_root} is no longer held by run "
            f"{lease.get('run_id')!r} (found {None if current is None else current.get('run_id')!r})"
            f" — another run claimed an expired heartbeat; aborting"
        )
    fresh = dict(current)
    from zagg.hive import _utcnow

    fresh["heartbeat_at"] = _utcnow()
    obstore.put(
        open_object_store(store_root, **store_kwargs),
        LEASE_NAME,
        json.dumps(fresh, indent=1).encode(),
    )
    return fresh


def release_lease(store_root: str, *, run_id: str, store_kwargs: dict | None = None) -> bool:
    """Delete the intent object iff this run holds it; the finisher's final act."""
    import obstore

    from zagg.store import open_object_store

    store_kwargs = dict(store_kwargs or {})
    current = read_lease(store_root, store_kwargs=store_kwargs)
    if current is None:
        return False
    if current.get("run_id") != str(run_id):
        logger.warning(
            f"sweep lease on {store_root} belongs to run {current.get('run_id')!r}, "
            f"not {run_id!r}; leaving it in place"
        )
        return False
    obstore.delete(open_object_store(store_root, **store_kwargs), LEASE_NAME)
    return True
