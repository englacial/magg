"""Leaf-column backfill for pre-column stores (issue #520): the ``/1 -> /2`` bridge.

Leaf columns (:mod:`zagg.column`, issues #383/#391 — the gen-1 partials the
staged sweep folds) are written only by pyramid-ON builds whose D24
classifiers admit the fields. Every store built before the 0.50 classifiers
is column-less, and a column-less store cannot take the ``/2`` staged sweep
at any transport. This module is the missing bridge: one pass that visits
each committed leaf, recomputes its column from the leaf's own STORED bytes
(:func:`zagg.column.column_from_leaf`) and writes it — so an existing
published store is upgraded **without re-aggregation**, at the cost of one
leaf-reading pass, paid once.

It is a **sweep family** (``columns``, registered in :mod:`zagg.sweep`), not
a mode of its own: the family registry is the already-Lambda-wired
``mode: "sweep"`` transport, so partitioning (issue #377), the work-set
normalization and the discovery seam come for free, with no handler change
and no dependency on #519. It is deliberately NOT in
:data:`zagg.sweep.DEFAULT_FAMILIES` — a backfill is an explicit upgrade
operation, never something a routine rollup sweep does behind an operator's
back. Spell it: ``python -m zagg.sweep <root> --families columns``.

**Declaration-driven, never guessed.** The gate is the manifest's own
``zagg-pyramid/2`` declaration (:func:`zagg.column.manifest_column_plan`) —
a ``/1`` schedule, a declared-off block, or a block whose every field is D24
``class: "none"`` refuses BY NAME and says re-declare first. Those are
declaration bugs, and backfilling one would publish a morton-only column (or
no artifact) as an upgrade.

**Who may write a column.** Spec §4.6's single-writer law is written for the
fleet: a column's writer is its leaf's worker. This pass is the one
sanctioned exception, and it inherits the law rather than repealing it —
:mod:`zagg.sweep_lease` serializes it against every other sweep, and it must
run against a store **no aggregation run is writing**. That second half is an
operator precondition, not something the lease can enforce: the lease's ruled
concurrency matrix allows fleet ∥ sweep precisely because their object sets
are disjoint, and this is the one pass for which they are not.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

#: Leaves between lease heartbeats. The pass is one fold + one multi-object
#: write per leaf, so a store large enough to outlive the TTL is ordinary;
#: refreshing per leaf would be a PUT per leaf against the intent object.
HEARTBEAT_EVERY = 64


def backfill_columns(
    store_root: str,
    manifest: dict,
    by_shard: dict,
    store_kwargs: dict | None = None,
    min_order: int = 0,
    *,
    force: bool = False,
    run_id: str | None = None,
    lease: bool = True,
) -> dict:
    """Write the missing leaf column for every committed leaf in the work set.

    The ``columns`` family's whole-tree hook (:meth:`SweepFamily.sweep_store`'s
    signature, so :func:`zagg.sweep.run_sweep` dispatches straight to it).
    ``by_shard`` is the normalized ``{shard decimal: {window, ...}}`` work set
    the engine already filtered to this partition; every column lives at its
    own leaf's node prefix, at the shard order, which is at or below any
    partition split order — so partition disjointness holds by construction
    and ``min_order`` is accepted (the issue #377 contract is that a hook
    reads it, not swallows it) and recorded, never used to move a write.

    Per ``(leaf, window)``: an uncommitted leaf contributes nothing
    (``empty``); a column already current under
    :func:`zagg.column.column_is_current` is left alone (``current``, unless
    ``force``); anything else is recomputed from the leaf's stored bytes and
    written wholesale (``written``). A leaf that cannot be read or folded is
    logged and counted (``failed``), never fatal — the families sweep's
    posture, and a half-upgraded store is exactly as readable as an
    un-upgraded one, since readers never require a column (§4.6).

    ``lease=False`` skips admission — for callers that already hold the store's
    sweep lease. Otherwise the pass takes it for its whole duration
    (:mod:`zagg.sweep_lease`), heartbeating every
    :data:`HEARTBEAT_EVERY` leaves and releasing in a ``finally``.

    Returns the family summary the sweep engine records:
    ``written``/``current``/``empty``/``failed`` counts plus the declaration
    it worked to.
    """
    import uuid

    from zagg.column import manifest_column_plan
    from zagg.sweep_lease import acquire_lease, heartbeat_lease, release_lease

    store_kwargs = dict(store_kwargs or {})
    # BEFORE admission: a store that must be re-declared should say so without
    # first taking (and having to release) the store's sweep lease.
    plan = manifest_column_plan(manifest)
    counts = {"written": 0, "current": 0, "empty": 0, "failed": 0}
    run_id = run_id or f"backfill-{uuid.uuid4().hex[:12]}"
    held = None
    if lease:
        held = acquire_lease(store_root, run_id=run_id, store_kwargs=store_kwargs)
    try:
        seen = 0
        for decimal in sorted(by_shard):
            for window in sorted(by_shard[decimal], key=lambda w: (w is not None, w or "")):
                _backfill_leaf(store_root, decimal, window, plan, counts, store_kwargs, force=force)
                seen += 1
                if held is not None and seen % HEARTBEAT_EVERY == 0:
                    held = heartbeat_lease(store_root, held, store_kwargs=store_kwargs)
    finally:
        if held is not None:
            release_lease(store_root, run_id=run_id, store_kwargs=store_kwargs)
    return {
        **counts,
        "run_id": run_id,
        "min_order": int(min_order),
        "resolutions": list(plan.resolutions),
        "fields": sorted(plan.fields),
    }


def _backfill_leaf(store_root, decimal, window, plan, counts, store_kwargs, *, force) -> None:
    """One ``(leaf, window)``: gate, fold from stored bytes, write. Never raises.

    Split out of the driver so a fold that blows up on one leaf costs that
    leaf and nothing else — the ``failed`` count the families sweep reports.
    The read order is cheapest-first: the leaf's commit stamp (a leaf that is
    not committed has no column to write, and unstamped debris is invisible
    exactly as the walk treats it), then the column's own stamp and attrs for
    the skip test, and only then the leaf's arrays.
    """
    from zagg.column import COLUMN_ATTR, column_from_leaf, column_is_current, write_column
    from zagg.grids.morton import morton_word

    try:
        shard_key = morton_word(decimal)
        leaf_stamp = _leaf_stamp(store_root, shard_key, window, store_kwargs)
        if leaf_stamp is None:
            counts["empty"] += 1
            return
        stamp, attrs = _column_state(store_root, shard_key, window, store_kwargs)
        if not force:
            current, reason = column_is_current(
                leaf_stamp,
                stamp,
                attrs.get(COLUMN_ATTR),
                node_order=plan.node_order,
                cell_order=plan.cell_order,
                resolutions=plan.resolutions,
                fields=plan.fields,
            )
            if current:
                counts["current"] += 1
                return
            logger.debug(f"backfill: leaf {decimal} window {window!r} not current ({reason})")
        folded = column_from_leaf(
            store_root,
            shard_key,
            plan.fields,
            node_order=plan.node_order,
            cell_order=plan.cell_order,
            resolutions=plan.resolutions,
            window=window,
            store_kwargs=store_kwargs,
        )
        write_column(
            store_root,
            shard_key,
            folded,
            plan.fields,
            node_order=plan.node_order,
            cell_order=plan.cell_order,
            window=window,
            time_range=leaf_stamp.get("time_range"),
            granule_count=int(leaf_stamp.get("granule_count") or 0),
            store_kwargs=store_kwargs,
        )
        counts["written"] += 1
    except Exception as e:
        logger.warning(f"backfill: leaf {decimal} window {window!r} failed ({e})")
        counts["failed"] += 1


def _leaf_stamp(store_root, shard_key, window, store_kwargs):
    """The leaf's D4 commit stamp, or ``None`` for absent/unstamped debris."""
    from zagg.hive import read_commit, shard_leaf_path
    from zagg.store import open_store

    leaf = shard_leaf_path(store_root, shard_key, window=window)
    return read_commit(open_store(leaf, read_only=True, **store_kwargs))


def _column_state(store_root, shard_key, window, store_kwargs) -> tuple:
    """This ``(leaf, window)``'s stored column as ``(commit stamp, root attrs)``.

    ``(None, {})`` when nothing is there — the ordinary pre-column case this
    whole module exists for, and the same answer an unreadable prefix gives:
    the verdict either way is "not current", which rewrites.
    """
    import zarr

    from zagg.column import column_name
    from zagg.hive import COMMIT_ATTR, shard_leaf_path
    from zagg.store import open_store

    leaf = shard_leaf_path(store_root, shard_key, window=window)
    path = f"{leaf.rstrip('/').rsplit('/', 1)[0]}/{column_name(window)}"
    try:
        group = zarr.open_group(
            open_store(path, read_only=True, **store_kwargs), path="", mode="r", zarr_format=3
        )
    except Exception:
        return None, {}
    attrs = dict(group.attrs)
    stamp = attrs.get(COMMIT_ATTR)
    return (dict(stamp) if isinstance(stamp, dict) else None), attrs
