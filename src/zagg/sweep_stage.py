"""Staged dense sweep for ``zagg-pyramid/2`` stores (issue #384; umbrella #381).

The ``/2`` pyramid's above-shard ladder is materialized by **stage workers**,
never by a leaf-reading walk: the fleet's leaf columns (issue #383) already
carry every within-footprint member plus the node-order **universal partial**,
so the sweep reads columns only — a raw leaf is never opened above the shard.

**Tuple grouping is orchestration-only** (#381 point (6)). Ladder orders are
grouped into dispatch tuples of ``tuple_width`` consecutive orders (default
3), dispatched at nodes whose order is ``0 mod tuple_width`` — ``[8,7,6] ->
[5,4,3] -> [2,1,0]`` on an o9 store, ragged finest tuple when ``shard_order``
is not a multiple of the width. A worker owning an order-``D`` subtree is a
:mod:`zagg.sweep_partition` partition with the split at ``D``: same prefix
ownership, same disjointness.

**The merge-source law (espg ruling, 2026-08-09, issue #384).** A stage
worker reads exactly its ``4^width`` immediate child columns, ``width``
orders down — nothing ever reads deeper. Each worker's own column carries,
as a pure gather, the **leaf node-order partial set for its whole subtree**
(the relay) alongside whatever gatherable members the parent tuple needs.
Every merge, at every level, in every tuple, consumes **only the relayed
gen-1 partials** — never the worker's own outputs, never a previously merged
tier — so the merge tree is a fixed function of the store, independent of
grouping: builds at ``tuple_width=1`` and ``tuple_width=3`` are
byte-identical, and every upfront merge level records ``merges_from_raw: 2``
uniformly (gather levels carry gen-1 content untouched, ``merges_from_raw:
1``). Gen 3 belongs only to the append-later cascade regime (#381 point (7)),
which is unchanged and remains the path for pre-column ``/1`` stores.

**Source classification is derived, not hardcoded**: a level ``(node k,
cells r)`` is a **gather** when ``r >= shard_order`` (its cells nest within
single child footprints — concatenation of child members) and a **merge**
otherwise (its cells span leaves — a k-way fold of relayed partials).

**Scope** (#381 point (11)) is an optional node-prefix set — a MOC, the same
ownership predicate as partitions; a shardmap is sugar (its keys are already
shard prefixes). Scope selects which dispatch nodes are invoked; a dispatched
worker folds **all** children on disk, so an update adjacent to prior data
folds the old neighbors in automatically. Scope composes with ``partitions=``
by MOC intersection. Unscoped discovery is listing-based (the run records,
:func:`zagg.sweep.discover_leaves`); the root ``coverage.moc`` is an
**accelerator only, never truth** — a fleet append with no subsequent sweep
leaves it stale, and discovery must still find the new leaves (espg ruling).

**Concurrency.** Sweeps serialize per store via the admission lease
(:mod:`zagg.sweep_lease` — control plane: no data object is ever locked).
Fleets run concurrently with a live sweep: the worker validates each column's
stamp before and after reading its groups and re-reads on movement, so a
mid-read leaf rewrite never feeds a torn column into a merge; a mid-sweep
append is ordinary under-coverage, recorded and healed by the next sweep.
Stage-written stamps carry the run id; a skip-if-current read that sees a
FOREIGN fresh stamp aborts loudly (:class:`ForeignSweepError`) — the backstop
for residual races the lease cannot see (TTL clock skew, zombie workers).

**Soft barriers.** Inter-stage barriers are scheduling preferences, not
correctness (#381 point (6)): a stage run before its finer tuple landed
under-covers loudly (``source_children``) and self-heals on the next pass —
the skip gate keys on summed child generations, so a healed child moves the
parent's generation and forces the rewrite.

**The finisher** (espg ruling: a single designated finisher-worker, never
the 12 base cells) owns the root singletons after the root tuple completes:
the root ``coverage.moc`` refresh, the manifest RMW writing per-entry
actuals into ``pyramid.overviews`` (which also satisfies the PR #397
lifecycle root-touch for the manifest), and lease release as its final act.

Raster hive stores are column-less by construction (§4.6 is written for the
aggregation pipeline): the sweep refuses them loudly; issue #399's
reducer-keyed folds join this orchestration later under the same schema.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

#: Per-artifact attrs revision for stage-written ladder overviews — the
#: ``/2`` shape §4.4 deferred to the writers: one resolution group per
#: artifact at the entry's ``cells`` member (``k + d``, not ``/1``'s
#: constant-depth formula), regime + merges-from-raw + source_children.
OVERVIEW_SPEC_V2 = "zagg-overview/2"
#: #381 point (7) regimes a stage-written level records.
STAGE_GATHER = "stage-gather"
STAGE_MERGE = "stage-merge"
#: Dispatch cadence default (#381 point (6)).
DEFAULT_TUPLE_WIDTH = 3


class ForeignSweepError(RuntimeError):
    """A foreign run's FRESH stamp was seen mid-run: two sweeps are live.

    The lease (:mod:`zagg.sweep_lease`) makes this unreachable in normal
    operation; reaching it means a residual race (TTL-expiry clock skew, a
    zombie worker from a crashed run) — abort loudly, never fold through it.
    """


# ---------------------------------------------------------------------------
# Phase 1: the stage planner — pure functions over the expanded manifest list.
# ---------------------------------------------------------------------------


def ladder_entries(pyramid: dict, shard_order: int) -> list[dict]:
    """The above-shard ladder from a manifest ``pyramid.overviews`` list.

    Returns the entries the STAGED sweep owns — every ``node < shard_order``
    — sorted finest first, each normalized to ``{"node": int, "cells":
    [int]}``. The leaf entry (``node == shard_order``) is the fleet's own
    column and is excluded. Refuses by name a non-``/2`` block, an empty
    list, or a ladder entry carrying more than one member (the fixed ladder
    guarantees exactly one, §4.4) — the sweep must never widen a malformed
    declaration into a plausible schedule.
    """
    from zagg.pyramid import PYRAMID_SPEC_V2

    if not isinstance(pyramid, dict) or pyramid.get("spec") != PYRAMID_SPEC_V2:
        raise ValueError(
            f"staged sweep requires a {PYRAMID_SPEC_V2!r} manifest pyramid declaration "
            f"(got spec {pyramid.get('spec') if isinstance(pyramid, dict) else None!r}); "
            f"/1 stores keep the zagg.sweep_overview path (the retrofit regime)"
        )
    overviews = pyramid.get("overviews")
    if not isinstance(overviews, list) or not overviews:
        raise ValueError("manifest pyramid.overviews is absent or empty — nothing to sweep")
    entries = []
    for e in overviews:
        node, cells = int(e["node"]), [int(c) for c in e["cells"]]
        if node >= int(shard_order):
            continue  # the leaf entry: the fleet's column, never the sweep's
        if len(cells) != 1:
            raise ValueError(
                f"ladder entry {e!r} carries {len(cells)} members; the fixed every-order "
                f"ladder guarantees exactly one (§4.4) — refusing a malformed declaration"
            )
        entries.append({"node": node, "cells": cells})
    return sorted(entries, key=lambda e: -e["node"])


def stage_tuples(shard_order: int, *, tuple_width: int = DEFAULT_TUPLE_WIDTH) -> list[dict]:
    """Group ladder orders into dispatch tuples, finest tuple first.

    Dispatch nodes sit at orders ``0 mod tuple_width``, so every tuple below
    the finest spans exactly ``tuple_width`` orders and the finest tuple is
    ragged when ``shard_order`` is not a multiple of the width. Each item is
    ``{"dispatch": D, "orders": [finest..D], "child_order": C}`` where ``C``
    is the order of the child columns the tuple's workers read — the previous
    tuple's dispatch order, or ``shard_order`` (the leaf columns) for the
    finest tuple. The grouping changes no bytes (#381 point (6) + the
    merge-source law): it is a dispatch knob, never grammar.
    """
    shard_order, tuple_width = int(shard_order), int(tuple_width)
    if tuple_width < 1:
        raise ValueError(f"tuple_width must be >= 1 (got {tuple_width})")
    if shard_order < 1:
        raise ValueError(f"shard_order {shard_order} has no above-shard ladder to sweep")
    tuples = []
    for dispatch in range(0, shard_order, tuple_width):
        child_order = min(dispatch + tuple_width, shard_order)
        tuples.append(
            {
                "dispatch": dispatch,
                "orders": list(range(child_order - 1, dispatch - 1, -1)),
                "child_order": child_order,
            }
        )
    return list(reversed(tuples))


def classify_level(cells: int, *, shard_order: int) -> str:
    """``stage-gather`` or ``stage-merge`` for a ladder level's cell resolution.

    Derived, never hardcoded: cells at or finer than the shard order nest
    within single child footprints (leaf columns carry those members, stage
    columns relay them) — concatenation. Coarser cells span leaves — a k-way
    merge of the relayed gen-1 node-order partials.
    """
    return STAGE_GATHER if int(cells) >= int(shard_order) else STAGE_MERGE


def column_members(levels: list, node_order: int, *, shard_order: int) -> list[int]:
    """Resolutions a stage column at ``node_order`` carries, finest first.

    The relay member (``shard_order`` — the subtree's leaf node-order
    partials, the ruled merge-source tier) unconditionally, plus every
    gatherable member (``cells >= shard_order``) some coarser level
    (``node < node_order``) will gather. All members are pure gathers of the
    child columns' members at the same resolution — gen-1 content, untouched,
    so ``merges_from_raw`` stays 1 for every group and a parent merge that
    consumes the relay is exactly 2 merges from raw.
    """
    node_order, shard_order = int(node_order), int(shard_order)
    gatherable = {
        int(c)
        for e in levels
        for c in e["cells"]
        if int(e["node"]) < node_order and int(c) >= shard_order
    }
    return sorted(gatherable | {shard_order}, reverse=True)


def normalize_scope(scope) -> np.ndarray | None:
    """An optional sweep scope, canonicalized to a morton-word MOC.

    ``None`` means whole store. Accepted spellings: an iterable of morton
    words (ints) or D1 decimal strings — a node-prefix set at any mix of
    orders — or a mapping (a shardmap: its KEYS are the prefixes, already
    ancestor-coarsened by construction; values are ignored). Returns sorted
    unique ``uint64`` words. Empty scopes refuse by name: an empty MOC means
    "sweep nothing", which is never what an operator typed on purpose.
    """
    from zagg.grids.morton import morton_word

    if scope is None:
        return None
    if isinstance(scope, dict):
        scope = list(scope.keys())
    words = [morton_word(s) if isinstance(s, str) else int(s) for s in scope]
    if not words:
        raise ValueError("an empty sweep scope selects nothing — pass None for whole-store")
    return np.unique(np.asarray(words, dtype=np.uint64))


def partition_words(partitions: int, index: int) -> np.ndarray:
    """One ``2^n`` partition as a MOC: its 12 order-``k`` subtree roots.

    The :mod:`zagg.sweep_partition` ownership predicate spelled as words, so
    ``partitions=`` composes with a scope by plain MOC intersection
    (:func:`compose_scope`). ``partitions=1`` returns the 12 base cells —
    the identity scope.
    """
    from zagg.grids.morton import morton_word
    from zagg.hive import _rank_tail
    from zagg.sweep_partition import normalize_partition, partition_split_order

    normalize_partition({"index": index, "of": partitions})
    k = partition_split_order(partitions)
    tail = _rank_tail(int(index), k)
    bases = [str(b) for b in range(1, 7)] + [f"-{b}" for b in range(1, 7)]
    return np.unique(np.asarray([morton_word(b + tail) for b in bases], dtype=np.uint64))


def compose_scope(scope: np.ndarray | None, partition: np.ndarray | None) -> np.ndarray | None:
    """Scope ∩ partition (either may be ``None`` — the identity)."""
    from mortie import moc_and

    if scope is None:
        return partition
    if partition is None:
        return scope
    # TODO(espg/mortie#173, espg/mortie PR 174): swap the per-pair scalar
    # ``moc_and`` for the batch ``mocs_and`` (1xN broadcast with the hoisted
    # shared operand) when mortie 0.9.6 releases; do not depend on it before.
    return moc_and(scope, partition)


def scope_admits(decimal: str, scope: np.ndarray | None) -> bool:
    """Whether a node's subtree intersects the scope MOC (``None`` admits all).

    Containment resolves in either direction (mortie's ``moc_and`` semantics
    over mixed-order covers), so a scope spelled as shard prefixes admits
    every ancestor node above them — "only ancestor prefixes of the touched
    shardmap are invoked" (#381 point (11)).
    """
    from mortie import moc_and

    from zagg.grids.morton import morton_word

    if scope is None:
        return True
    # TODO(espg/mortie#173, espg/mortie PR 174): batch ``mocs_and`` replaces
    # this per-node scalar call once mortie 0.9.6 releases.
    return moc_and(np.asarray([morton_word(decimal)], dtype=np.uint64), scope).size > 0
