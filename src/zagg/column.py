"""Leaf-worker pyramid column folds (issue #383; umbrella #381 points (1)-(3)).

A **column artifact** is the leaf worker's own pyramid contribution, computed
at aggregation time while the shard's cell data is resident: one zarr per
``(leaf, window)`` under the leaf's node prefix, holding a resolution group
for every leaf-node level the ``zagg-pyramid/2`` declaration carries, every
member a coarser declaration implies within the leaf's footprint, and the
node-order member (``cells == node`` — the leaf's whole-footprint aggregate,
its **universal partial** for every coarser cell; there is no ``partial/``
grammar, #381 point (2)).

Every group folds directly from the leaf's raw resident cell slabs —
merges-from-raw 1 for all leaf-written content, the #381 point (1) regime
law; a group is never folded from another group. Exact classes reduce via
:func:`zagg.sweep_overview.fold_dense`, approximate (t-digest) classes via
the order-independent k-way merge (:func:`zagg.sweep_overview.fold_digests`,
the issue #370 fold law) — the same kernels the sweep's from-leaves fold
runs over the same per-cell inputs in the same ascending order, so column
bytes are parity-equal with that fold by construction.

This module owns the fold core (pure functions over in-memory slabs) and the
column writer (one artifact per ``(leaf, window)``, D4 write discipline: a
wholesale template, every resolution group, the role/provenance attrs, ONE
commit stamp last covering the whole column, and the D20 stats sidecar after
the stamp). The worker integration is the last phase of issue #383.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

#: Envelope version of the column artifact's provenance attrs payload.
COLUMN_SPEC = "zagg-column/1"
#: Column basename suffix: the D23 window stem carries it in place of
#: ``.zarr`` (:func:`column_name`). The ONE definition of the name seam —
#: store walkers that must tell a column from a source leaf key off it
#: (:func:`zagg.coverage.refresh_root_coverage`), never off a second literal.
COLUMN_SUFFIX = ".pyramid.zarr"
#: Root-group attrs key carrying the column provenance payload (D11).
COLUMN_ATTR = "zagg_column"
#: ``role`` attrs value classifying a column zarr (D11: classification by
#: attrs, never tree position; source leaves carry no role — absence means
#: source — and the sweep's overview family keeps its own ``overview`` role).
COLUMN_ROLE = "column"
#: The #381 point (7) regime every leaf-written group records: folded from
#: the leaf's own resident cells. No ``source_children`` rides this regime
#: (PR #379 precedent: coverage counts ride ``cascade``) — a leaf column's
#: source is complete by construction, and its merges-from-raw is 1.
LEAF_REGIME = "leaf-column"


def column_resolutions(levels: list, node_order: int) -> list[int]:
    """The resolutions a leaf-node column carries, finest first (issue #383).

    ``levels`` is the NORMALIZED ``zagg-pyramid/2`` grouped form — the
    manifest block's ``overviews`` list (:func:`zagg.pyramid.normalize_overviews`,
    the ``output.pyramid.overviews`` knob). The column holds every declared
    resolution that is complete within one leaf footprint (``cells >=
    node_order`` — the leaf-node groups plus the members coarser declarations
    imply, which makes their levels pure gathers, #381 point (3)) plus the
    node-order member unconditionally. Resolutions coarser than the node need
    no member of their own: the leaf's contribution to ANY coarser cell is
    its whole-footprint aggregate — the node-order member itself.

    Empty when ``levels`` declares no ``node == node_order`` entry: a
    schedule that starts coarser than the shard node declares no leaf-written
    column (the #383 gate), and the sweep owns whatever it materializes.
    """
    node_order = int(node_order)
    if not any(int(e["node"]) == node_order for e in levels or []):
        return []
    within = {int(c) for e in levels for c in e["cells"] if int(c) >= node_order}
    return sorted(within | {node_order}, reverse=True)


def composable_fields(fields: dict) -> dict:
    """The declared fields a column fold may carry: the two composable classes.

    The D24 ``class: "none"`` entries — expressions, vector fields,
    chunk-resolution companions, located ragged, and the derived statistics
    (:func:`zagg.semantics.field_composability`, recorded by
    :func:`zagg.pyramid.declared_fields`) — exist at native resolution ONLY,
    and no coarser fold of them is defined. The fold core filters them here,
    the same posture the sweep takes before ``_fold_node``
    (:func:`zagg.sweep_overview.sweep_overviews`), so handing a whole
    declaration's ``fields`` map straight through can neither refuse a leaf
    over a non-cell-extent vector slab nor materialize an all-empty ragged
    group for a companion the column has no business carrying.
    """
    return {
        n: m
        for n, m in (fields or {}).items()
        if isinstance(m, dict) and m.get("class") in ("exact", "approximate")
    }


def leaf_slabs(staged: dict, fields: dict, *, group_path: str, n_cells: int) -> dict:
    """``{field: cell slab}`` fold inputs from the leaf writer's staged sink.

    ``staged`` is the issue #342 staged-array record
    (``{f"{group_path}/{name}": slab}`` — the exact in-memory values the leaf
    write PUT), so the fold consumes what the leaf stores, byte-for-byte,
    with no read-back. A declared field absent from the sink contributes
    fill — the leaf writers skip an all-empty ragged array entirely
    (``write_ragged_leaf_to_zarr``), and its stored cells are the ``b""``
    fill regardless — so the synthesized slab is exactly what a read-back
    would return — synthesized by :func:`zagg.sweep_overview._empty_slab`, the
    same seed the sweep's own fold starts from, so the fill semantics the two
    machineries must agree on stay one definition.

    ``fields`` is filtered to the composable classes first
    (:func:`composable_fields`), which is what makes the ``(n_cells,)`` extent
    check sound: those two classes admit nothing but cell-resolution scalars
    and unlocated ragged payloads, so a staged slab of any other extent really
    is a sink that disagrees with the grid — and folding it would write a
    wrong column, so it raises.
    """
    from zagg.sweep_overview import _empty_slab

    slabs: dict = {}
    for name, meta in composable_fields(fields).items():
        slab = staged.get(f"{group_path}/{name}")
        if slab is None:
            slab = _empty_slab(meta, n_cells)
        else:
            slab = np.asarray(slab)
            if slab.shape != (int(n_cells),):
                raise ValueError(
                    f"staged slab for field {name!r} has shape {slab.shape}, not the "
                    f"leaf's ({int(n_cells)},) cell extent — refusing to fold a column "
                    f"from a sink that disagrees with the grid"
                )
        slabs[name] = slab
    return slabs


def fold_column(slabs: dict, fields: dict, *, cell_order: int, resolutions: list) -> dict:
    """Fold the leaf's resident cell slabs into ``{resolution: {field: slab}}``.

    Each resolution group folds INDEPENDENTLY from the raw cell slabs (never
    group from group — merges-from-raw stays 1 for every group, #381 point
    (1)): ``4^(cell_order - resolution)`` consecutive leaf cells share one
    target cell (the ascending packed-word leaf invariant). Exact fields
    reduce under their declared merge law; approximate fields decode each
    child cell's payload and k-way merge the non-empty digests per target
    cell, in ascending cell order — input-identical to the sweep's
    from-leaves fold (:func:`zagg.sweep_overview._fold_node` +
    :func:`zagg.sweep_overview.fold_digests`) of the committed leaf, which is
    the issue #383 byte-parity contract. The node-order resolution is the
    degenerate 1-cell group: the leaf's whole-footprint aggregate.

    ``fields`` is filtered to the composable classes (:func:`composable_fields`)
    — a D24 ``none`` field has no coarser fold and never becomes a group. A
    resolution FINER than ``cell_order`` is refused by name: it would ask for
    a fractional fold factor, which no guard downstream can read as a divisor
    (both classes would surface it as an opaque numpy failure instead).
    """
    from zagg.sweep_overview import decode_digest, fold_dense, fold_digests

    cell_order = int(cell_order)
    fields = composable_fields(fields)
    out: dict = {}
    for res in resolutions:
        res = int(res)
        if res > cell_order:
            raise ValueError(
                f"cannot fold a column group at resolution {res}: it is FINER than the "
                f"leaf's cell order {cell_order}, so the fold factor "
                f"4^({cell_order} - {res}) is fractional — a column group is a fold of "
                f"the leaf's own cells, never an upsample of them"
            )
        factor = 4 ** (cell_order - res)
        groups: dict = {}
        for name, meta in fields.items():
            slab = slabs[name]
            if meta["class"] == "exact":
                groups[name] = fold_dense(
                    slab, factor, meta.get("method"), meta.get("fill_value", "NaN")
                )
            else:
                dtype = meta.get("dtype") or "float32"
                inner = tuple(meta.get("inner_shape") or (2,))
                delta = int(meta.get("delta") or 512)
                if slab.shape[0] % factor:
                    raise ValueError(
                        f"cannot fold {slab.shape[0]} cells {factor}-to-one for {name!r}"
                    )
                folded = np.full(slab.shape[0] // factor, b"", dtype=object)
                for j in range(folded.shape[0]):
                    cell = [
                        decode_digest(payload, dtype, inner)
                        for payload in slab[j * factor : (j + 1) * factor]
                        if payload is not None and len(payload)
                    ]
                    if cell:
                        folded[j] = fold_digests(cell, delta=delta, dtype=dtype)
                groups[name] = folded
        out[res] = groups
    return out


def column_name(window: str | None) -> str:
    """The column basename: the D23 window stem + ``.pyramid.zarr``.

    ``{window}.pyramid.zarr``, with ``all.pyramid.zarr`` for the unwindowed /
    schedule-none leaf — the same window-only stem the overview writer uses
    unconditionally (:func:`zagg.windows.leaf_name_v3`), plus a ``.pyramid``
    stem marker so the name is disjoint from every leaf and overview basename
    (both end at ``{window}.zarr``). Proposed on the issue #383 PR, flagged
    there as a naming question.
    """
    from zagg.windows import leaf_name_v3

    return leaf_name_v3(window).removesuffix(".zarr") + COLUMN_SUFFIX


def write_column(
    store_root: str,
    shard_key,
    folded: dict,
    fields: dict,
    *,
    node_order: int,
    cell_order: int,
    window: str | None = None,
    time_range=None,
    granule_count: int = 0,
    store_kwargs: dict | None = None,
) -> str:
    """Write one leaf's column artifact under its node prefix; returns its basename.

    ``folded`` is :func:`fold_column`'s output; ``fields`` the declaration's
    composable map (the template and the provenance attrs are derived from
    it, exactly as the overview writer derives them). The write order is the
    leaf's own D4 discipline: template (wholesale — the prefix is DELETED
    first, the issue #341 semantics, so an idempotent re-run replaces the
    column entirely and a prior torn write never survives) -> every
    resolution group's ``morton`` + ``{order}/{field}`` arrays -> the
    role/provenance attrs -> ONE commit stamp LAST covering the whole
    column. The clear also removes any PRIOR sidecar (fail-open, sibling to
    the prefix it cannot reach): a rewrite whose fresh sidecar PUT then fails
    leaves the record ABSENT — unverifiable, §5.3 — never STALE, which would
    verify as a mismatch and read as a false tamper signal (D20).
    There is no partial-column failure state: an interrupted writer
    leaves an unstamped prefix — ignorable debris, and repair is re-invoking
    the idempotent leaf (never a sweep-side fallback to raw cells). The D20
    stats sidecar is a SIBLING object PUT after the stamp (fail-open,
    telemetry class), so the stamp stays the column's own final write.

    The commit stamp's ``cells_with_data`` counts the populated mask of the
    column's FINEST group — the one denominator a leaf or an overview has
    implicitly, but a column (N grids) does not, so the group it counts is
    recorded as ``cells_with_data_order`` in the ``zagg_column`` attrs: the
    number is declaration-dependent (a coarser base declaration yields a
    different count for the same leaf) and a reader cannot infer it.

    A ``folded`` without the node-order member is refused by name: that
    member is the leaf's universal partial for every coarser cell (#381
    point (2)), the one group #384's gather may assume, so stamping a
    column without it would publish a complete-looking artifact that folds
    short. The same guard covers an empty ``folded`` (the
    ``column_resolutions() == []`` gate belongs to the caller).

    Single-writer law (#381 point (2)): the column lives only under its
    leaf's node prefix and has exactly one writer, ever — no locking.
    """
    import zarr
    from mortie import generate_morton_children
    from pydantic_zarr.experimental.v3 import GroupSpec
    from zarr import config as zarr_config
    from zarr import open_array
    from zarr.core.sync import sync

    from zagg.grids.base import vlen_dtype_warning_suppressed
    from zagg.grids.healpix import HealpixGrid
    from zagg.grids.morton import morton_decimal
    from zagg.hive import _utcnow, shard_leaf_path, stamp_commit
    from zagg.store import open_store
    from zagg.sweep_overview import (
        ROLE_ATTR,
        _field_provenance,
        _overview_config,
        _populated_mask,
    )
    from zagg.windows import SCHEDULE_NONE_TOKEN

    store_kwargs = dict(store_kwargs or {})
    node_order, cell_order = int(node_order), int(cell_order)
    fields = composable_fields(fields)
    resolutions = sorted((int(r) for r in folded), reverse=True)
    if node_order not in resolutions:
        raise ValueError(
            f"a column must carry the node-order member ({node_order}) — it is the leaf's "
            f"universal partial for every coarser cell (#381 point (2)), and no coarse level "
            f"ever rewrites a leaf; got resolutions {resolutions}"
        )
    leaf_path = shard_leaf_path(store_root, shard_key, window=window)
    node_prefix = leaf_path.rstrip("/").rsplit("/", 1)[0]
    basename = column_name(window)
    path = f"{node_prefix}/{basename}"
    cfg = _overview_config(fields)
    grids = {res: HealpixGrid(node_order, res, config=cfg, sharded=True) for res in resolutions}
    spec = GroupSpec(
        members={str(res): grids[res].shard_spec() for res in resolutions}, attributes={}
    )
    store = open_store(path, **store_kwargs)
    staged: dict = {}
    with zarr_config.set({"async.concurrency": 128}), vlen_dtype_warning_suppressed():
        sync(store.delete_dir(""))
        _delete_sidecar(node_prefix, _sidecar_name(basename), store_kwargs)
        spec.to_zarr(store, "", overwrite=True)
    for res in resolutions:
        words = np.asarray(generate_morton_children(int(shard_key), res), dtype=np.uint64)
        arr = open_array(store, path=f"{res}/morton", zarr_format=3, consolidated=False)
        arr[:] = words
        staged[f"{res}/morton"] = words
        for name, slab in folded[res].items():
            arr = open_array(store, path=f"{res}/{name}", zarr_format=3, consolidated=False)
            arr[:] = slab
            staged[f"{res}/{name}"] = slab
    root = zarr.open_group(store, path="", mode="r+", zarr_format=3)
    root.attrs.update(
        {
            ROLE_ATTR: COLUMN_ROLE,
            COLUMN_ATTR: {
                "spec": COLUMN_SPEC,
                "node": morton_decimal(int(shard_key)),
                "order": node_order,
                "source_cell_order": cell_order,
                "window": window if window is not None else SCHEDULE_NONE_TOKEN,
                "fields": {n: _field_provenance(m) for n, m in fields.items()},
                "groups": {
                    str(res): {
                        "regime": LEAF_REGIME,
                        "merges_from_raw": 1,
                        "n_cells": 4 ** (res - node_order),
                    }
                    for res in resolutions
                },
                "cells_with_data_order": resolutions[0],
                "generated_at": _utcnow(),
            },
        }
    )
    populated = _populated_mask(folded[resolutions[0]], fields)
    stamp_commit(
        store,
        cells_with_data=int(populated.sum()),
        granule_count=int(granule_count),
        window=window,
        time_range=time_range if window is not None else None,
    )
    _write_sidecar(
        store, path, shard_key, staged, int(populated.sum()), granule_count, window, store_kwargs
    )
    return basename


def _sidecar_name(basename: str) -> str:
    """The column's D20 sidecar basename: its own stem + ``.stats.json``."""
    return basename.removesuffix(".zarr") + ".stats.json"


def _delete_sidecar(prefix: str, name: str, store_kwargs: dict) -> None:
    """Drop a prior run's column sidecar, fail-open (absent beats stale)."""
    try:
        import obstore

        from zagg.store import open_object_store

        obstore.delete(open_object_store(prefix, **store_kwargs), name)
    except Exception as e:
        logger.debug(f"column stats sidecar clear skipped at {prefix}/{name}: {e}")


def _write_sidecar(
    store, path, shard_key, staged, cells_with_data, granule_count, window, store_kwargs
) -> None:
    """The column's D20 stats sidecar: ``{stem}.stats.json``, after the stamp.

    The overview writer's O11 recipe (issue #342, spec §5): the content
    hashes computed from the staged arrays just written, in a
    :func:`zagg.telemetry.build_record` row keyed by the shard. The name is
    derived from the column's own stem — ``telemetry.sidecar_key``'s label
    grammar (rightly) rejects the dotted ``.pyramid`` stem, and the rule is
    the same ``{stem}.stats.json`` one. Fail-open (D9 telemetry posture):
    §5.3 reads absence as unverifiable, never tampered — so EVERY import a
    sidecar needs sits inside the ``try`` (``_write_overview``'s posture): an
    ImportError here is a telemetry-class failure and must not fail a column
    that is already committed. The caller's open ``store`` is reused rather
    than re-derived — one fewer store construction and root-metadata read.
    """
    try:
        import json

        import obstore
        import zarr

        from zagg.content_hash import content_hashes_record, hash_arrays
        from zagg.store import open_object_store
        from zagg.telemetry import build_record

        group = zarr.open_group(store, path="", mode="r", zarr_format=3)
        record = build_record(
            shard_key=int(shard_key),
            metadata={
                "cells_with_data": int(cells_with_data),
                "granule_count": int(granule_count),
                "content_hashes": content_hashes_record(hash_arrays(group, staged=staged)),
            },
            window=window,
        )
        prefix, _, name = path.rstrip("/").rpartition("/")
        obstore.put(
            open_object_store(prefix, **store_kwargs),
            _sidecar_name(name),
            json.dumps(record).encode(),
        )
    except Exception as e:
        logger.warning(f"column stats sidecar failed (fail-open, issue #383): {e}")
