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

This module owns the fold core (pure functions over in-memory slabs); the
column writer and the worker integration are the later phases of issue #383.
Nothing here reads or writes a store.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


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
    would return.

    ``fields`` is filtered to the composable classes first
    (:func:`composable_fields`), which is what makes the ``(n_cells,)`` extent
    check sound: those two classes admit nothing but cell-resolution scalars
    and unlocated ragged payloads, so a staged slab of any other extent really
    is a sink that disagrees with the grid — and folding it would write a
    wrong column, so it raises.
    """
    slabs: dict = {}
    for name, meta in composable_fields(fields).items():
        slab = staged.get(f"{group_path}/{name}")
        if slab is None:
            if meta["class"] == "exact":
                from zagg.sweep_overview import _fill_scalar

                dtype = np.dtype(meta.get("dtype") or "float32")
                slab = np.full(n_cells, _fill_scalar(meta.get("fill_value", "NaN"), dtype), dtype)
            else:
                slab = np.full(n_cells, b"", dtype=object)
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
    — a D24 ``none`` field has no coarser fold and never becomes a group.
    """
    from zagg.sweep_overview import decode_digest, fold_dense, fold_digests

    cell_order = int(cell_order)
    fields = composable_fields(fields)
    out: dict = {}
    for res in resolutions:
        res = int(res)
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
