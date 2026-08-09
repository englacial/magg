"""The staged sweep RUN: pass driver, finisher, and orchestration (issue #384).

:mod:`zagg.sweep_stage` owns one stage worker's machinery (the planner, the
fold kernels, the writers); this module owns a whole run of them:

- :func:`sweep_stage_pass` — every tuple, finest first, over one dirty set
  (the in-process driver; the lease heartbeat rides its ``on_stage`` seam);
- :func:`run_finisher` — the designated finisher-worker (espg ruling): the
  root singletons, exactly once, lease release last;
- :func:`run_stage_sweep` — admission (:mod:`zagg.sweep_lease`), discovery
  (listing-based — the run records via :func:`zagg.sweep.discover_leaves`;
  the root ``coverage.moc`` is an accelerator, never truth), ``partitions=``
  composition, the run record, and the post-fleet chaining seam
  (``output.sweep: "stages"``, opt-in per the recorded lean).

The CLI backstop is ``python -m zagg.sweep <root> --stages``
(:mod:`zagg.sweep` routes here). Raster hive stores are column-less by
construction and refuse at the ``/2``-declaration gate — issue #399's
reducer-keyed fold family joins this orchestration later, one schema.
"""

from __future__ import annotations

import json
import logging
import time

import numpy as np

from zagg.sweep_stage import (
    DEFAULT_TUPLE_WIDTH,
    _node_at,
    compose_scope,
    ladder_entries,
    normalize_scope,
    partition_words,
    scope_admits,
    stage_node,
    stage_tuples,
)

logger = logging.getLogger(__name__)


def sweep_stage_pass(
    store_root: str,
    manifest: dict,
    by_shard: dict,
    *,
    scope: np.ndarray | None = None,
    tuple_width: int = DEFAULT_TUPLE_WIDTH,
    run_id: str,
    run_started: str | None = None,
    store_kwargs: dict | None = None,
    on_stage=None,
    level_actuals: dict | None = None,
) -> dict:
    """One staged pass over the dirty set: every tuple, finest first.

    The in-process driver the orchestration (phase 4) wraps with admission,
    discovery, the finisher and the run record. ``by_shard`` is the
    normalized dirty set (``{shard_decimal: {window, ...}}``); candidate
    children per node come from it unioned with the root ``coverage.moc``
    (an ACCELERATOR — a stale root MOC degrades coverage of untouched
    siblings, recorded as ``root_moc_stale``, never discovery, which is
    listing-based upstream). ``scope`` (a normalized MOC) filters DISPATCH
    nodes; a dispatched worker folds all children on disk. ``on_stage`` is
    called after each tuple (the lease heartbeat's seam). Returns the pass
    summary with per-stage rows.
    """
    from zagg.hive import _utcnow
    from zagg.store import open_object_store
    from zagg.sweep_overview import _candidate_decimals, _window_work

    store_kwargs = dict(store_kwargs or {})
    run_started = run_started or _utcnow()
    pyramid = manifest.get("pyramid") or {}
    shard_order = int(manifest["shard_order"])
    cell_order = int(manifest["cell_order"])
    levels = ladder_entries(pyramid, shard_order)
    decl = pyramid.get("overview") if isinstance(pyramid.get("overview"), dict) else {}
    fields = {
        n: dict(m)
        for n, m in (decl.get("fields") or {}).items()
        if isinstance(m, dict) and m.get("class") in ("exact", "approximate")
    }
    summary: dict = {"run_id": run_id, "tuple_width": int(tuple_width), "stages": []}
    if not fields:
        logger.info("stage sweep: no composable fields declared; nothing to generate")
        return summary
    windowed = manifest.get("temporal") is not None
    candidates, moc_stale = _candidate_decimals(store_root, shard_order, by_shard, store_kwargs)
    if moc_stale:
        summary["root_moc_stale"] = True
    if not candidates:
        return summary
    store = open_object_store(store_root, **store_kwargs)
    if level_actuals is None:
        level_actuals = {}  # callers may pass one to accumulate across passes
    from zagg.windows import SCHEDULE_NONE_TOKEN

    for stage in stage_tuples(shard_order, tuple_width=tuple_width):
        t0 = time.perf_counter()
        counts = {
            "written": 0,
            "current": 0,
            "empty": 0,
            "failed": 0,
            "under_covered": 0,
            "columns_written": 0,
            "columns_current": 0,
            "revalidated": 0,
        }
        nodes = sorted({_node_at(d, stage["dispatch"]) for d in candidates})
        nodes = [n for n in nodes if scope_admits(n, scope)]
        for node in nodes:
            dirty_windows: set = set()
            for d in by_shard:
                if d.startswith(node):
                    dirty_windows |= by_shard[d]
            from zagg.sweep_overview import _read_envelope

            envelope = _read_envelope(store, node)
            entries = dict((envelope or {}).get("windows") or {})
            for key, fold_windows in _window_work(decl, windowed, dirty_windows, entries):
                stage_node(
                    store,
                    store_root,
                    node,
                    stage,
                    levels,
                    fields,
                    key=key,
                    fold_windows=fold_windows,
                    all_time=bool(windowed and key == SCHEDULE_NONE_TOKEN),
                    windowed=windowed,
                    shard_order=shard_order,
                    cell_order=cell_order,
                    candidates=candidates,
                    run_id=run_id,
                    run_started=run_started,
                    counts=counts,
                    store_kwargs=store_kwargs,
                    level_actuals=level_actuals,
                )
        row = {
            "dispatch_order": stage["dispatch"],
            "orders": list(stage["orders"]),
            "nodes": len(nodes),
            **counts,
            "duration_s": time.perf_counter() - t0,
        }
        summary["stages"].append(row)
        if on_stage is not None:
            on_stage(row)
    summary["levels"] = {str(k): dict(v) for k, v in sorted(level_actuals.items(), reverse=True)}
    return summary


# ---------------------------------------------------------------------------
# The designated finisher-worker — the root singletons, exactly once.
# ---------------------------------------------------------------------------


def run_finisher(
    store_root: str,
    manifest: dict,
    by_shard: dict,
    level_actuals: dict,
    *,
    run_id: str,
    store_kwargs: dict | None = None,
    release=None,
) -> dict:
    """The designated finisher-worker (espg ruling): root singletons, once.

    Fired by the orchestrator after the root tuple completes — never
    distributed across the 12 base cells (plural writers of singleton
    objects would breach the single-writer law). Owns, in order:

    1. the root ``coverage.moc`` refresh (GET-union-PUT, ``source:
       "sweep"``) — sequenced HERE, before any later scoped fan-out reads
       it: the #380 read-side obligation the partition machinery deferred;
    2. the manifest RMW nesting per-entry actuals inside the level entries
       of ``pyramid.overviews`` (#381 point (7); readers MUST tolerate the
       added key). The leaf entry records the ``leaf-column`` law
       (merges-from-raw 1); ladder entries record what this run observed —
       ``stage-gather`` at 1, ``stage-merge`` at 2, never 3 (gen 3 is
       append-later cascade territory only). This re-PUT also refreshes the
       manifest's ``LastModified``, satisfying the PR #397 lifecycle
       root-touch for ``morton_hive.json`` (and step 1 for the root MOC)
       WITHOUT duplicating it — only ``aggregation.yaml`` still needs the
       explicit touch, step 3. The family dict's order-keyed
       ``materialized`` is deliberately NOT written on ``/2`` stores: the
       per-entry actuals are the one source of truth (the flatten-ruling
       principle; any /1-era inventory is preserved verbatim);
    3. the ``aggregation.yaml`` lifecycle touch (issue #388's machinery);
    4. lease release via ``release()`` — the FINAL act, so a finisher that
       failed midway leaves the lease held and the run claimable/idempotent
       (re-invoking re-runs steps 1-3 harmlessly).

    Failures in steps 1-2 RAISE (the orchestrator records the incomplete
    finish and leaves the lease for recovery); step 3 is fail-open telemetry.
    """
    import obstore

    from zagg.grids.morton import morton_word
    from zagg.hive import (
        AGGREGATION_CORE_NAME,
        MANIFEST_NAME,
        _utcnow,
        build_root_coverage,
        read_manifest,
        write_root_coverage,
    )
    from zagg.lifecycle import touch_unit_footprint
    from zagg.store import open_object_store

    store_kwargs = dict(store_kwargs or {})
    out = {
        "root_moc": False,
        "manifest_updated": False,
        "objects_touched": 0,
        "touch_failures": 0,
        "lease_released": False,
    }
    shard_order = int(manifest["shard_order"])
    if by_shard:
        envelope = build_root_coverage(
            [morton_word(d) for d in by_shard], shard_order, source="sweep"
        )
        write_root_coverage(store_root, envelope, **store_kwargs)
        out["root_moc"] = True
    fresh = read_manifest(store_root, **store_kwargs)
    if fresh is None:
        raise ValueError(f"no {MANIFEST_NAME} at {store_root} — cannot record actuals")
    now = _utcnow()
    changed = False
    for entry in (fresh.get("pyramid") or {}).get("overviews") or []:
        node = int(entry.get("node", -1))
        if node == shard_order and level_actuals:
            actuals = {"regime": "leaf-column", "merges_from_raw": 1, "generated_at": now}
        elif node in level_actuals:
            a = level_actuals[node]
            actuals = {
                "regime": a["regime"],
                "merges_from_raw": int(a["merges_from_raw"]),
                "source_children": dict(a["source_children"]),
                "run_id": run_id,
                "generated_at": now,
            }
        else:
            continue
        if entry.get("actuals") != actuals:
            entry["actuals"] = actuals
            changed = True
    if changed or level_actuals:
        obstore.put(
            open_object_store(store_root, **store_kwargs),
            MANIFEST_NAME,
            json.dumps(fresh, indent=1).encode(),
        )
        out["manifest_updated"] = True
    touch = touch_unit_footprint(
        [], [f"{str(store_root).rstrip('/')}/{AGGREGATION_CORE_NAME}"], store_kwargs=store_kwargs
    )
    out["objects_touched"] = touch["touched"]
    out["touch_failures"] = touch["failed"]
    if release is not None:
        out["lease_released"] = bool(release())
    return out


# ---------------------------------------------------------------------------
# Phase 4: dispatch orchestration — admission, discovery, record, chaining.
# ---------------------------------------------------------------------------


def run_stage_sweep(
    store_root: str,
    leaves=None,
    *,
    scope=None,
    tuple_width: int = DEFAULT_TUPLE_WIDTH,
    partitions: int | None = None,
    store_kwargs: dict | None = None,
    record: bool = True,
    run_id: str | None = None,
    lease_ttl_s: int | None = None,
) -> dict:
    """One admitted staged sweep, end to end: lease -> stages -> finisher.

    ``leaves`` is the dirty ``(shard_key, window)`` work set; ``None`` means
    UNSCOPED DISCOVERY, which is listing-based by the espg ruling — the run
    records (:func:`zagg.sweep.discover_leaves`), never the root
    ``coverage.moc`` (a fleet append with no subsequent sweep leaves the
    root MOC stale, and the ratchet only heals nodes a sweep visits; the MOC
    stays an in-pass accelerator for sibling candidates only).

    ``scope`` is the optional node-prefix MOC (#381 point (11) — decimals,
    words, or a shardmap whose keys are the prefixes); ``partitions=``
    composes by intersection, each of the ``2^n`` partitions swept in turn
    UNDER THE SAME LEASE (the lease is store-granular by correctness: scope
    disjointness does not imply write disjointness). Admission is the
    :mod:`zagg.sweep_lease` conditional PUT — a live foreign intent raises
    :class:`zagg.sweep_lease.SweepRefused` naming the runner; an expired
    heartbeat is claimed and the claimant simply completes the partial prior
    run (the ratchet's ordinary posture). The lease heartbeats after every
    stage; the finisher releases it as its final act. On ANY failure the
    lease is deliberately left held — the run record says what happened, and
    the intent expires into claimability rather than admitting a sibling
    into a half-written store.

    The summary is PUT at the store root as the run record
    (``sweep_stats_{ts}_stages.json``) with the lease intent/completion, the
    per-stage rows, the per-level actuals, and the finisher counts
    (``objects_touched``/``touch_failures`` — the PR #397 shape).
    """
    import uuid
    from datetime import datetime, timezone

    from zagg.hive import MANIFEST_NAME, _utcnow, read_manifest
    from zagg.sweep import _normalize_leaves, discover_leaves
    from zagg.sweep_lease import (
        DEFAULT_TTL_S,
        acquire_lease,
        heartbeat_lease,
        release_lease,
    )

    t0 = time.perf_counter()
    store_kwargs = dict(store_kwargs or {})
    manifest = read_manifest(store_root, **store_kwargs)
    if manifest is None:
        raise ValueError(f"no {MANIFEST_NAME} at {store_root} — not a hive store root")
    shard_order = int(manifest["shard_order"])
    ladder_entries(manifest.get("pyramid") or {}, shard_order)  # loud /2 gate
    run_id = run_id or (
        f"stage-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:6]}"
    )
    if leaves is None:
        leaves = discover_leaves(store_root, store_kwargs=store_kwargs)
    by_shard, skipped = _normalize_leaves(leaves, shard_order)
    scope_words = normalize_scope(scope)
    lease = acquire_lease(
        store_root,
        run_id=run_id,
        scope=None if scope_words is None else [int(w) for w in scope_words],
        ttl_s=int(lease_ttl_s or DEFAULT_TTL_S),
        store_kwargs=store_kwargs,
    )
    run_started = _utcnow()
    summary: dict = {
        "run_id": run_id,
        "store_root": store_root,
        "shard_order": shard_order,
        "tuple_width": int(tuple_width),
        "n_leaves": sum(len(w) for w in by_shard.values()),
        "skipped_leaves": skipped,
        "scope": None if scope_words is None else [str(int(w)) for w in scope_words],
        "partitions": partitions,
        "lease": {
            "acquired_at": lease.get("acquired_at"),
            "claimed_from": lease.get("claimed_from"),
            "ttl_s": lease.get("ttl_s"),
            "released": False,
        },
        "stages": [],
        "levels": {},
    }
    level_actuals: dict = {}
    try:
        indexes = [None] if not partitions or partitions == 1 else list(range(int(partitions)))
        for index in indexes:
            part_scope = (
                scope_words
                if index is None
                else compose_scope(scope_words, partition_words(int(partitions), index))
            )
            if part_scope is not None and part_scope.size == 0:
                continue  # scope ∩ partition selects nothing
            part = sweep_stage_pass(
                store_root,
                manifest,
                by_shard,
                scope=part_scope,
                tuple_width=tuple_width,
                run_id=run_id,
                run_started=run_started,
                store_kwargs=store_kwargs,
                on_stage=lambda row: heartbeat_lease(store_root, lease, store_kwargs=store_kwargs),
                level_actuals=level_actuals,
            )
            rows = part["stages"]
            if index is not None:
                for row in rows:
                    row["partition"] = {"index": index, "of": int(partitions)}
            summary["stages"].extend(rows)
            if part.get("root_moc_stale"):
                summary["root_moc_stale"] = True
        summary["levels"] = {
            str(k): dict(v) for k, v in sorted(level_actuals.items(), reverse=True)
        }
        summary["finisher"] = run_finisher(
            store_root,
            manifest,
            by_shard,
            level_actuals,
            run_id=run_id,
            store_kwargs=store_kwargs,
            release=lambda: release_lease(store_root, run_id=run_id, store_kwargs=store_kwargs),
        )
        summary["lease"]["released"] = bool(summary["finisher"].get("lease_released"))
    except BaseException as e:
        # The lease stays HELD: it expires into claimability, and a sibling
        # admitted now would write into a half-swept store. Record and raise.
        summary["error"] = f"{type(e).__name__}: {e}"
        summary["duration_s"] = time.perf_counter() - t0
        if record:
            summary["record"] = _write_stage_record(store_root, summary, store_kwargs)
        raise
    summary["duration_s"] = time.perf_counter() - t0
    if record:
        summary["record"] = _write_stage_record(store_root, summary, store_kwargs)
    return summary


def _write_stage_record(store_root: str, summary: dict, store_kwargs: dict) -> str | None:
    """PUT the staged run record at the store root; its key, or ``None``.

    ``sweep_stats_{ts}_stages.json`` — the :func:`zagg.sweep._write_sweep_record`
    naming family (timestamp-first, outside the ``stats_*.parquet`` glob),
    with a ``_stages`` tag so a staged run never reads as a families pass.
    The rows extend the settled #397 record shape: the lease block is the
    intent/completion row, per-stage rows carry written/current/failed/
    under-coverage, and the finisher block carries
    ``objects_touched``/``touch_failures``. Fail-open (telemetry, D9).
    """
    from datetime import datetime, timezone

    import obstore

    from zagg.store import open_object_store
    from zagg.sweep import SWEEP_SPEC

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    key = f"sweep_stats_{ts}_stages.json"
    try:
        obstore.put(
            open_object_store(store_root, **store_kwargs),
            key,
            json.dumps({"spec": SWEEP_SPEC, "mode": "stages", **summary}, indent=1).encode(),
        )
    except Exception as e:
        logger.warning(f"stage sweep: run record write failed (fail-open, D9): {e}")
        return None
    return key


def stage_sweep_after_run(store_root: str, leaves, *, store_kwargs: dict | None = None):
    """Post-fleet chaining: the ``output.sweep: "stages"`` opt-in, fail-open.

    The dispatcher fires the staged sweep immediately after the fleet lands,
    auto-scoped to the fleet's own footprint (#381 point (11): the run's
    shard set IS the touched shardmap, and scope spelled as shard prefixes
    invokes exactly their ancestor chain). Fail-open like
    :func:`zagg.sweep.sweep_after_run` — every stage artifact is regenerable
    (D9), so a refused lease or a failed stage costs one later
    ``python -m zagg.sweep --stages`` pass, never a wrong answer. Local
    backend only for now (D8: the Lambda dispatcher never writes; the
    worker-transport forward is the recorded open question (b)).
    """
    from zagg.grids.morton import morton_decimal

    try:
        scope = sorted({morton_decimal(int(k)) for k, _w in (tuple(r) for r in leaves)})
        if not scope:
            return None
        summary = run_stage_sweep(store_root, leaves, scope=scope, store_kwargs=store_kwargs)
        logger.info(
            f"Post-run staged sweep: {[(s['dispatch_order'], s['written']) for s in summary['stages']]}"
        )
        return summary
    except Exception as e:
        logger.warning(f"post-run staged sweep failed (fail-open, D9 — regenerable): {e}")
        return None
