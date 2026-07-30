"""Sweep rollup engine: the shared node fold under every family (issue #330).

Split out of the single-file ``zagg.sweep`` (issue #330 phase 3); the public
surface is unchanged and re-exported from :mod:`zagg.sweep`. Holds the
shard-node and interior-node folds, the generation/ancestor arithmetic, and
the rollup object read/write pair — the family-agnostic half of the sweep,
driven by :func:`zagg.sweep._sweep_family`.
"""

from __future__ import annotations

import json
import logging

from zagg.sweep_families import SWEEP_SPEC

logger = logging.getLogger(__name__)


def _rollup_shard_node(
    store_root, store, fam, decimal, windows, shard_order, spec, store_kwargs, counts
) -> dict | None:
    """Fold one shard node's window-leaf artifacts into its rollup.

    The window set is the union of the run's dirty windows and the windows the
    existing rollup already merged (recorded in its ``windows`` key), so an
    append run that touches one window never drops its siblings.
    """
    existing = _read_rollup(store, fam, decimal)
    known = set(windows)
    if existing is not None:
        known |= {None if w is None else str(w) for w in existing.get("windows") or []}
    parts = []
    for window in sorted(known, key=lambda w: (w is not None, w or "")):
        try:
            got = fam.read_leaf(store_root, decimal, window, spec, store_kwargs)
        except Exception as e:
            logger.warning(
                f"sweep[{fam.name}]: leaf read failed at {decimal} window {window!r}; "
                f"skipping that leaf ({e})"
            )
            counts["failed"] += 1
            got = None
        if got is not None:
            parts.append((window, *got))
    if not parts:
        counts["empty"] += 1
        return None
    generation = _generation(len(parts), [ts for _w, _p, ts in parts])
    payload = _merged(fam, [p for _w, p, _ts in parts], decimal, shard_order, counts)
    if payload is None:
        return None
    if (
        existing is not None
        and existing.get("generation") == generation
        and existing.get("payload") == payload
    ):
        counts["current"] += 1
        return existing
    envelope = {
        "spec": SWEEP_SPEC,
        "family": fam.name,
        "node": decimal,
        "order": shard_order,
        "generation": generation,
        "windows": [w for w, _p, _ts in parts],
        "payload": payload,
    }
    _put_rollup(store, fam, decimal, envelope)
    counts["written"] += 1
    return envelope


def _rollup_interior(store, fam, node, computed, counts) -> dict | None:
    """Fold a digit node's four candidate children rollups into its own.

    A child freshly computed this pass is used in memory; any other candidate
    is probed on the store (<= 4 GETs, no LIST) so prior runs' siblings keep
    contributing. Generation is the children's sum/max — fold-of-folds equals
    the direct leaf fold because every family's merge is associative (§8.3).

    The store is append-only at the leaf level (leaf deletion/GC is the
    registered debris family, deliberately stubbed), so a child that emptied
    this pass returns ``None`` from its shard rollup and this fallback picks up
    its prior on-store rollup — an emptied/vanished leaf keeps contributing to
    its parent until a deletion-aware family exists. Under append-only + D9
    (rollups are regenerable caches) that is intended, not stale-serving.
    """
    from zagg.hive import _decimal_order

    children = []
    for digit in "1234":
        art = computed.get(node + digit)
        if art is None:
            art = _read_rollup(store, fam, node + digit)
        if art is not None:
            children.append(art)
    if not children:
        counts["empty"] += 1
        return None
    generation = _generation(
        sum(int(a["generation"]["n_leaves"]) for a in children),
        [a["generation"].get("max_leaf_timestamp") for a in children],
    )
    existing = _read_rollup(store, fam, node)
    payload = _merged(fam, [a["payload"] for a in children], node, _decimal_order(node), counts)
    if payload is None:
        return None
    if (
        existing is not None
        and existing.get("generation") == generation
        and existing.get("payload") == payload
    ):
        counts["current"] += 1
        return existing
    envelope = {
        "spec": SWEEP_SPEC,
        "family": fam.name,
        "node": node,
        "order": _decimal_order(node),
        "generation": generation,
        "payload": payload,
    }
    _put_rollup(store, fam, node, envelope)
    counts["written"] += 1
    return envelope


def _merged(fam, payloads, node, order, counts) -> dict | None:
    """The family fold, fail-open per node: unmergeable -> logged + skipped."""
    try:
        return fam.merge(payloads, node=node, order=order)
    except Exception as e:
        logger.warning(f"sweep[{fam.name}]: merge failed at node {node}; skipping ({e})")
        counts["failed"] += 1
        return None


def _generation(n_leaves: int, timestamps) -> dict:
    """The D22 generation stamp: merged-leaf count + max leaf timestamp."""
    stamps = [t for t in timestamps if t is not None]
    return {"n_leaves": int(n_leaves), "max_leaf_timestamp": max(stamps) if stamps else None}


def _ancestor(decimal: str) -> str | None:
    """The parent prefix of a node decimal (``None`` at a base component)."""
    from zagg.hive import _decimal_base

    return None if decimal == _decimal_base(decimal) else decimal[:-1]


def _node_rel(decimal: str) -> str:
    """A node decimal's relative digit path (``-311`` -> ``-3/1/1``)."""
    from zagg.hive import _decimal_base

    base = _decimal_base(decimal)
    return "/".join([base, *decimal[len(base) :]])


def _rollup_key(fam, decimal: str) -> str:
    return f"{_node_rel(decimal)}/{fam.rollup_name}"


def _read_rollup(store, fam, decimal: str) -> dict | None:
    """A node's stored rollup envelope, or ``None`` — strict, cache posture.

    Missing, unparsable, wrong-spec/family, or stamp-less objects all read as
    absent (debug-logged): a corrupt rollup is a regenerable cache (D9) and is
    simply rebuilt, never half-trusted.
    """
    import obstore
    from obstore.exceptions import NotFoundError

    try:
        data = obstore.get(store, _rollup_key(fam, decimal)).bytes()
    except (FileNotFoundError, NotFoundError):
        return None
    try:
        envelope = json.loads(bytes(data))
    except ValueError:
        envelope = None
    generation = envelope.get("generation") if isinstance(envelope, dict) else None
    usable = (
        isinstance(envelope, dict)
        and envelope.get("spec") == SWEEP_SPEC
        and envelope.get("family") == fam.name
        and isinstance(generation, dict)
        and isinstance(generation.get("n_leaves"), int)
        and "payload" in envelope
    )
    if not usable:
        logger.debug(
            f"sweep[{fam.name}]: unusable rollup at node {decimal}; ignoring "
            f"(regenerable cache, D9)"
        )
        return None
    return envelope


def _put_rollup(store, fam, decimal: str, envelope: dict) -> None:
    import obstore

    obstore.put(store, _rollup_key(fam, decimal), json.dumps(envelope, indent=1).encode())
