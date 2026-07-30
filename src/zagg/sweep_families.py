"""Sweep families: the #300 stats / MOC / sub-shardmap fold implementations.

Split out of the single-file ``zagg.sweep`` (issue #330 phase 3, which passed
the CLAUDE.md §4 ~1000-line limit at 1,034); the public surface is unchanged
and re-exported from :mod:`zagg.sweep`. Holds the :class:`SweepFamily`
contract, the four D22 families and the leaf sub-map helpers they share, and
the ``FAMILIES`` registry that is D22's plug-in point. Sibling module rather
than a package because ``zagg.sweep`` is a CLI entry point
(``python -m zagg.sweep``) and the tree already splits this way —
:mod:`zagg.sweep_overview` is the existing precedent.
"""

from __future__ import annotations

import json
import logging

logger = logging.getLogger(__name__)

SWEEP_SPEC = "zagg-sweep/1"

#: Families swept when the caller does not choose (the four D22 families).
#: ``moc`` deliberately precedes ``overview``: the overview fold discovers
#: untouched sibling shards through the root ``coverage.moc``, which the MOC
#: family refreshes earlier in the same pass.
DEFAULT_FAMILIES = ("stats", "moc", "submap", "overview")

#: Grid types already warned about as unsupported leaf sub-maps (once-ish, so a
#: raster sweep does not warn-spam per shard). Never security-load-bearing.
_warned_unsupported_submap: set[str | None] = set()


def _warn_unsupported_submap(grid_type: str | None) -> None:
    """Warn once per grid type that a non-HEALPix leaf sub-map was skipped."""
    if grid_type in _warned_unsupported_submap:
        return
    _warned_unsupported_submap.add(grid_type)
    logger.warning(
        f"sweep[submap]: skipping non-HEALPix leaf sub-map (grid type {grid_type!r}); "
        f"the sub-shardmap fold is HEALPix-only (issue #300)"
    )


class SweepFamily:
    """One derived-artifact family (D22): how leaves read and payloads fold.

    Subclasses implement :meth:`read_leaf` (one leaf's contribution + its
    staleness timestamp) and :meth:`merge` (the associative payload fold);
    :meth:`finish` is an optional post-walk hook fed the top-level (base-node)
    artifacts — the seam the MOC family uses to refresh the store-root
    ``coverage.moc``. A family whose artifact is not a per-node JSON rollup
    (the overview family's zarrs, issue #201) instead defines ``sweep_store``
    — the whole-tree hook :func:`run_sweep` dispatches to with the manifest
    and the normalized dirty work set. Families registered with
    ``available = False`` are visible slots that :func:`get_family` refuses
    with their ``reason``.
    """

    name = ""
    available = True
    reason = ""

    @property
    def rollup_name(self) -> str:
        """The family's per-node rollup object name."""
        return f"{self.name}.rollup.json"

    def read_leaf(self, store_root, decimal, window, spec, store_kwargs):
        """One leaf's ``(payload, timestamp)`` contribution, or ``None``.

        ``None`` means the leaf carries no artifact for this family (e.g. a
        fail-open sidecar PUT that never landed) — it is skipped, not fatal.
        ``spec`` is the manifest's store spec string, threaded so spec-keyed
        sidecar naming (the PR #307 D23 seam) resolves per store.
        """
        raise NotImplementedError

    def merge(self, payloads: list, *, node, order) -> dict:
        """Fold payloads into one (associative — rollup == direct, §8.3).

        ``node``/``order`` name the target rollup node (decimal prefix and its
        morton order) for families whose fold is order-aware (the sub-shardmap
        reproject); order-free families ignore them.
        """
        raise NotImplementedError

    def finish(self, store_root, tops, shard_order, store_kwargs) -> dict:
        """Post-walk hook over the base-node artifacts; extra summary keys."""
        return {}


class StatsFamily(SweepFamily):
    """Stats/cost rollups (D20 fold): leaf ``stats.json`` sidecars up-tree.

    The payload is a stats record as :func:`zagg.telemetry.merge` produces —
    mergeable by construction (counts/sums/min-max, never stored means), so
    interior payloads re-merge exactly and the rollup at any node equals the
    direct fold of every leaf record beneath it.
    """

    name = "stats"

    def read_leaf(self, store_root, decimal, window, spec, store_kwargs):
        from zagg.grids.morton import morton_word
        from zagg.hive import shard_leaf_path
        from zagg.telemetry import read_sidecar

        leaf = shard_leaf_path(store_root, morton_word(decimal), window=window)
        record = read_sidecar(leaf, spec, **store_kwargs)
        if record is None:
            return None
        return record, record.get("timestamp")

    def merge(self, payloads: list, *, node, order) -> dict:
        from zagg.telemetry import merge

        return merge(payloads)


class MocFamily(SweepFamily):
    """MOC regen (D8/D9): committed-leaf coverage folded up-tree + root refresh.

    A leaf contributes iff its D4 commit stamp is present (unstamped debris is
    invisible, exactly as the walk treats it); the staleness timestamp is the
    stamp's ``written_at``. Payloads are shard-order ranges bodies (the root
    ``coverage.moc``'s O1 encoding, minus its carrier fields) plus the D15
    time union, so the fold is a word union and the base-node artifacts
    compose directly into the store root. :meth:`finish` refreshes the root
    ``coverage.moc`` from those base folds via the GET-union-PUT writer —
    the sweep is the REGENERATOR; the runner's end-of-run fail-open write
    stays the fast path (O7/D9) — skipping the PUT when the existing root
    already covers the folded words and time range (sweep idempotence). The
    O8 in-leaf bitmap contract is untouched: this family reads only the
    stamp envelope, never the cell-order bitmap sidecar.
    """

    name = "moc"

    def read_leaf(self, store_root, decimal, window, spec, store_kwargs):
        # ``spec`` is unused here: leaf PATHS are the frozen /1-/2 grammar
        # (shard_leaf_path); the D23 /3 leaf naming has no writer yet, and
        # adopting it is the issue #299 flip, which lands in shard_leaf_path.
        from zagg.grids.morton import morton_word
        from zagg.hive import read_commit, shard_leaf_path
        from zagg.store import open_store

        leaf = shard_leaf_path(store_root, morton_word(decimal), window=window)
        stamp = read_commit(open_store(leaf, **store_kwargs))
        if stamp is None:
            return None  # absent leaf or unstamped debris (D4)
        payload = _moc_payload([morton_word(decimal)], stamp.get("time_range"))
        return payload, stamp.get("written_at")

    def merge(self, payloads: list, *, node, order) -> dict:
        import numpy as np

        from zagg.hive import root_coverage_words
        from zagg.windows import union_time_range

        words = np.unique(np.concatenate([root_coverage_words(p) for p in payloads]))
        return _moc_payload(words, union_time_range(*(p.get("time_range") for p in payloads)))

    def finish(self, store_root, tops, shard_order, store_kwargs) -> dict:
        """Refresh the store-root ``coverage.moc`` from the base-node folds.

        Unions with the existing root object (the sweep may cover only the
        dirty subtrees — untouched bases must keep their listing), via the
        same :func:`zagg.hive.write_root_coverage` transport the runner uses.
        No PUT when the existing root already lists every folded word and
        covers the folded time range, so an unchanged tree re-sweep is a
        no-op here too.
        """
        import numpy as np

        from zagg.hive import (
            build_root_coverage,
            read_root_coverage,
            root_coverage_words,
            write_root_coverage,
        )
        from zagg.windows import union_time_range

        if not tops:
            return {"root_moc_written": False}
        words = np.unique(np.concatenate([root_coverage_words(t["payload"]) for t in tops]))
        time_range = union_time_range(*(t["payload"].get("time_range") for t in tops))
        try:
            existing = read_root_coverage(store_root, **store_kwargs)
        except ValueError:
            existing = None  # unparsable root -> regenerate (D9)
        if isinstance(existing, dict):
            try:
                covered = bool(np.isin(words, root_coverage_words(existing)).all())
                covered = covered and (
                    union_time_range(existing.get("time_range"), time_range)
                    == existing.get("time_range")
                )
                if covered:
                    return {"root_moc_written": False}
            except (KeyError, TypeError, ValueError):
                pass  # malformed cache cannot vouch for coverage -> rewrite
        write_root_coverage(
            store_root,
            build_root_coverage(words, shard_order, source="sweep", time_range=time_range),
            **store_kwargs,
        )
        return {"root_moc_written": True}


def _moc_payload(words, time_range) -> dict:
    """A rollup's shard-order ranges body (deterministic — no carrier fields).

    Reuses :func:`zagg.hive.build_root_coverage` for the O1 range encoding and
    drops ``source``/``generated_at`` (they would defeat skip-if-current
    byte stability) and ``spec`` (the rollup rides the ``zagg-sweep/1``
    envelope; the root object written by :meth:`MocFamily.finish` carries the
    full ``morton-moc/1`` carrier as usual). ``time_range`` is normalized
    through the D15 union so leaf and merged payloads compare equal.
    """
    from zagg.grids.morton import morton_decimal
    from zagg.hive import _decimal_order, build_root_coverage
    from zagg.windows import union_time_range

    order = _decimal_order(morton_decimal(int(words[0])))
    envelope = build_root_coverage(words, order, time_range=union_time_range(time_range))
    payload = {"encoding": "ranges", "order": envelope["order"], "ranges": envelope["ranges"]}
    if "time_range" in envelope:
        payload["time_range"] = envelope["time_range"]
    return payload


#: Leaf sub-map object name (bare leaves); windowed and ``/3`` leaves derive
#: their names through :func:`submap_key`, single-sourced on the stats-sidecar
#: naming seam.
SUBMAP_NAME = "shardmap.json"


def submap_key(leaf_name: str, spec: str | None = None) -> str:
    """Sub-map object name for a leaf zarr basename, keyed by store spec.

    Single-sourced on the PR #307 sidecar-naming seam
    (:func:`zagg.telemetry.sidecar_key`) with the ``shardmap`` stem swapped in:
    legacy stores get ``shardmap.json`` / ``shardmap_{window}.json``,
    ``morton-hive/3`` stores get ``{window}.shardmap.json`` — so the issue
    #299 writer flip renames both sidecars through one seam. Raises on an
    unknown spec, exactly as the seam does.
    """
    from zagg.telemetry import sidecar_key

    key = sidecar_key(leaf_name, spec)
    if key.endswith(".stats.json"):  # D23 /3 grammar: {stem}.stats.json
        return key.removesuffix(".stats.json") + ".shardmap.json"
    return "shardmap" + key.removeprefix("stats")  # legacy: stats[_{window}].json


def write_leaf_submap(
    store_root: str,
    shard_key,
    granules,
    *,
    grid_signature: dict,
    metadata: dict | None = None,
    window: str | None = None,
    spec: str | None = None,
    store_kwargs: dict | None = None,
) -> None:
    """PUT one leaf's sub-map — full ShardMap JSON (D22, ratified) — at its prefix.

    The payload is a one-shard :class:`~zagg.catalog.shardmap.ShardMap` in the
    standard JSON schema (``metadata``/``grid_signature``/``shard_keys``/
    ``granules``) plus a top-level ``written_at`` staleness stamp — extra keys
    are ignored by ``ShardMap.from_json``, so the object stays loadable as-is.
    ``granules`` are the unit's ShardMap entries, copied verbatim (windowed
    units carry their window's subset, so the shard-node fold's window union
    reassembles the shard). ``metadata`` is the run catalog's, with the
    whole-catalog counts and build fields rewritten to describe this sub-map
    (the same fields ``reproject`` strips from derived maps). Written by the
    worker on success, sibling to the stats sidecar — call sites fail open.
    """
    import obstore

    from zagg.hive import _utcnow, shard_leaf_path
    from zagg.store import open_object_store

    entries = [dict(g) for g in granules]
    meta = dict(metadata or {})
    for stale in ("aoi_mask", "build_wall_s", "reproject"):
        meta.pop(stale, None)
    meta.update(total_shards=1, total_granules=len(entries), total_pairs=len(entries))
    payload = {
        "metadata": meta,
        "grid_signature": dict(grid_signature),
        "shard_keys": [int(shard_key)],
        "granules": [entries],
        "written_at": _utcnow(),
    }
    leaf = shard_leaf_path(store_root, int(shard_key), window=window)
    prefix, _, name = leaf.rstrip("/").rpartition("/")
    obstore.put(
        open_object_store(prefix, **(store_kwargs or {})),
        submap_key(name, spec),
        json.dumps(payload, indent=1).encode(),
    )


def submap_emittable(grid_signature: dict | None, granules) -> bool:
    """Whether a unit's leaf sub-map can be folded by :class:`SubmapFamily`.

    The sub-shardmap fold is HEALPix-morton only — ``reproject`` coarsens by
    ``parent_order``/``child_order`` and the merge keys entries by granule
    ``id``. A rectilinear raster signature (grid ``type != "healpix"``, no
    ``parent_order``/``child_order``) or id-less granule entries would fold to
    a ``failed`` node the sweep can never consume, so the emission sites (issue
    #300) check this and skip the write instead of persisting an unmergeable
    payload. ``store_layout: hive`` is HEALPix-only by validation, so the
    realistic skip is id-less raster entries under a rectilinear signature.
    """
    sig = grid_signature or {}
    if sig.get("type") != "healpix" or "parent_order" not in sig or "child_order" not in sig:
        return False
    return all("id" in g for g in granules)


class SubmapFamily(SweepFamily):
    """Sub-shardmap rollups (D22): leaf ShardMap JSON folded via ``reproject``.

    Leaf artifact: the full-ShardMap-JSON sub-map the worker writes next to
    the leaf (:func:`write_leaf_submap`). Shard-node fold: the windows' entry
    lists union, deduplicated by granule ``id`` (a granule spanning two
    windows counts once). Interior fold: children sub-maps concatenate and
    coarsen to the node's order via :meth:`ShardMap.reproject` — the #294
    exact pure regroup (granule union deduped by id), now over stored
    artifacts — so the rollup at order N equals a direct reproject of the
    leaf-level map to N (§8.3). Every rollup's ``payload`` is a plain
    ShardMap JSON dict (the sweep envelope wraps it, as for every family).
    """

    name = "submap"

    def read_leaf(self, store_root, decimal, window, spec, store_kwargs):
        import obstore
        from obstore.exceptions import NotFoundError

        from zagg.grids.morton import morton_word
        from zagg.hive import shard_leaf_path
        from zagg.store import open_object_store

        leaf = shard_leaf_path(store_root, morton_word(decimal), window=window)
        prefix, _, name = leaf.rstrip("/").rpartition("/")
        try:
            data = obstore.get(
                open_object_store(prefix, **store_kwargs), submap_key(name, spec)
            ).bytes()
        except (FileNotFoundError, NotFoundError):
            return None
        sub = json.loads(bytes(data))
        ok = isinstance(sub, dict) and all(
            k in sub for k in ("grid_signature", "shard_keys", "granules")
        )
        if not ok:
            # Corrupt (missing required keys): a present-but-malformed sub-map
            # means a broken writer; the engine catches per leaf and counts it
            # failed — the loud, not-absent signal.
            raise ValueError(f"malformed leaf sub-map next to {leaf}")
        sig = sub["grid_signature"] or {}
        if sig.get("type") != "healpix" or "parent_order" not in sig or "child_order" not in sig:
            # Unsupported (not corrupt): the sub-shardmap fold is HEALPix-only
            # (reproject coarsens by parent/child order). A non-HEALPix leaf —
            # e.g. a rectilinear raster sub-map slipped past the emission guard
            # (submap_emittable) — is a SKIP, not a broken writer: return None so
            # the engine counts it "empty", never "failed" (data-corruption).
            _warn_unsupported_submap(sig.get("type"))
            return None
        timestamp = sub.pop("written_at", None)
        return sub, timestamp

    def merge(self, payloads: list, *, node, order) -> dict:
        """Fold child sub-maps into this node's rollup (§8.3).

        Identity metadata (``collection``/``short_name``/``version``/
        ``footprint``) is inherited from the first child payload
        (``payloads[0]``); only ``total_granules`` is recomputed. The
        one-store-per-product invariant (D7/D19: one product tree per semantic
        core, one catalog family) makes those fields uniform across a node's
        children by construction, so the first child is representative. A store
        that accreted leaves from more than one catalog under one root would
        need identity reconciliation this fold does not attempt (the rollup
        would advertise one arbitrary child's identity). ``grid_signature`` is
        safe regardless — all children of a node share ``child_order``.
        """
        from zagg.catalog.shardmap import ShardMap

        # Same-key union first (several windows of one shard), deduplicated by
        # granule id — the same rule reproject's coarsen applies across shards.
        buckets: dict[int, dict] = {}
        for p in payloads:
            for key, entries in zip(p["shard_keys"], p["granules"]):
                bucket = buckets.setdefault(int(key), {})
                for entry in entries:
                    # read_leaf already filters non-HEALPix/id-less leaves to the
                    # "empty" skip path, so an id-less entry reaching the fold is a
                    # genuinely broken artifact — name the field cleanly ("failed"),
                    # never a bare KeyError.
                    if "id" not in entry:
                        raise ValueError("leaf sub-map granule entry missing required 'id' field")
                    bucket[entry["id"]] = dict(entry)
        keys = sorted(buckets)
        granules = [list(buckets[k].values()) for k in keys]
        signature = dict(payloads[0]["grid_signature"])
        meta = dict(payloads[0].get("metadata") or {})
        meta.pop("reproject", None)  # never inherit a child fold's stamp
        meta["total_granules"] = len({e["id"] for g in granules for e in g})
        folded = ShardMap(signature, keys, granules, meta)
        if int(signature["parent_order"]) == int(order):
            # Shard-node fold: already at the node's order — a window union,
            # not a reprojection; keep counts honest without a noop stamp.
            meta.update(total_shards=len(keys), total_pairs=sum(len(g) for g in granules))
        else:
            # Interior fold: coarsen the children to this node's order. The
            # resulting metadata["reproject"]["source_parent_order"] records the
            # immediately-lower fold level (payloads[0]'s parent_order, one order
            # below this node) — the LAST hop, not the leaf/shard origin order.
            # Coarsen is transitive, so the folded data equals a direct reproject
            # of the leaf-level map straight to this order (§8.3,
            # test_rollup_equals_direct_reproject); the stamp naming the last hop
            # is by design, not a data discrepancy.
            folded = folded.reproject(_ReprojectTarget(signature, order))
        return {
            "metadata": folded.metadata,
            "grid_signature": folded.grid_signature,
            "shard_keys": [int(k) for k in folded.shard_keys],
            "granules": folded.granules,
        }


class _ReprojectTarget:
    """Minimal coarsen-target shim for :meth:`ShardMap.reproject`.

    ``reproject`` consumes exactly ``parent_order``/``child_order`` and
    ``spatial_signature()``. The sweep has no run config to build a full
    :class:`~zagg.grids.healpix.HealpixGrid` from, and the target signature IS
    the source's with the shard order swapped — reproject changes only the
    dispatch order, never the leaf DGGS resolution.
    """

    def __init__(self, signature: dict, parent_order: int):
        self._signature = {**signature, "parent_order": int(parent_order)}
        self.parent_order = int(parent_order)
        self.child_order = int(signature["child_order"])

    def spatial_signature(self) -> dict:
        return dict(self._signature)


class OverviewFamily(SweepFamily):
    """Overview zarrs at ancestor nodes (D11/D22/D24 — issue #201).

    Unlike the JSON-rollup families, the artifact is a ZARR at manifest-
    declared orders only, folded from leaf DATA (per-field D24 merge laws),
    so this family overrides :attr:`sweep_store` — the whole-tree hook the
    engine dispatches to instead of the generic bottom-up JSON fold. The
    implementation lives in :mod:`zagg.sweep_overview`.
    """

    name = "overview"

    def sweep_store(self, store_root, manifest, by_shard, store_kwargs) -> dict:
        from zagg.sweep_overview import sweep_overviews

        return sweep_overviews(store_root, manifest, by_shard, store_kwargs=store_kwargs)


class DebrisFamily(SweepFamily):
    """Debris collection (D22 optional audit-class fifth family) — stub."""

    name = "debris"
    available = False
    reason = (
        "debris collection (deleting unstamped .zarr/ prefixes past a "
        "declared horizon, D22's optional fifth family) is not implemented"
    )


#: Family registry (D22's plug-in point): name -> class. Unavailable entries
#: are visible slots that :func:`get_family` refuses with their ``reason``.
FAMILIES: dict[str, type[SweepFamily]] = {
    cls.name: cls for cls in (StatsFamily, MocFamily, SubmapFamily, OverviewFamily, DebrisFamily)
}


def get_family(name: str) -> SweepFamily:
    """Instantiate a registered family; loud on unknown or stubbed names."""
    cls = FAMILIES.get(name)
    if cls is None:
        raise ValueError(f"unknown sweep family {name!r}; registered: {sorted(FAMILIES)}")
    if not cls.available:
        raise NotImplementedError(f"sweep family {name!r} is registered but stubbed: {cls.reason}")
    return cls()
