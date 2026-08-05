"""Pyramid level grammar: the ``(node, cells)`` declaration model (issue #382).

The ``zagg-pyramid/2`` revision of the manifest pyramid declaration, ratified
on the issue #381 design record (points (4)–(5)): a pyramid is an ordered
list of **level entries** ``{node: N, cells: [...]}`` — ``node`` the hive
prefix where the level's artifacts live (one artifact per ``(node, window)``),
``cells`` the reader-facing resolutions stored there (slab length
``4^(cells - node)``; ``cells == node`` is the degenerate 1-cell
whole-footprint group, the writer's universal partial — there is no
``partial/`` grammar). Today's constant-depth grammar (``zagg-pyramid/1``,
``orders`` + ``spacing``) is the special case
``cells = node + (child_order - parent_order)`` and stays readable; its
constants and its sweep live in :mod:`zagg.sweep_overview`.

Semantics pinned on the issue:

- **Whole-list replacement** — an explicit ``overviews:`` block replaces the
  derived default wholesale; there is no per-node merge (the
  explicit-beats-derived rule).
- **Internal members are derived, never spelled** — spelling one (e.g.
  ``cells: [13, 11]`` where 11 is derivable from a coarser declaration)
  is legal and PROMOTES it to reader-facing contract.
- Refusals are loud and named; nothing widens silently.

This module owns the grammar only: parsing/normalization, validation, the
derived default, and the per-field D24 composability declarations both spec
revisions record. Writers (issue #383) and the staged sweep (issue #384)
consume the normalized form; nothing here writes a store.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

#: Version string of the manifest ``pyramid`` block carrying grouped
#: ``(node, cells)`` levels (issue #382). The ``/1`` marker lives in
#: :mod:`zagg.sweep_overview`, which owns sweeping that revision.
PYRAMID_SPEC_V2 = "zagg-pyramid/2"


def normalize_overviews(raw) -> list[dict]:
    """``output.pyramid.overviews`` in the normalized grouped form.

    A non-empty ordered list of ``{node: N, cells: [...]}`` entries; a scalar
    ``cells`` is sugar for the one-element list. Shape errors raise here;
    the ordering/range rules live in :func:`validate_overviews` (they need the
    grid orders). The manifest records exactly this normalized form.
    """
    if not isinstance(raw, list) or not raw:
        raise ValueError(
            f"output.pyramid.overviews must be a non-empty list of {{node, cells}} entries "
            f"(got {raw!r}); `output.pyramid: false` is how a store declares no pyramid"
        )
    levels = []
    for i, entry in enumerate(raw):
        if not isinstance(entry, dict) or set(entry) != {"node", "cells"}:
            raise ValueError(
                f"output.pyramid.overviews[{i}] must be a mapping with exactly "
                f"'node' and 'cells' (got {entry!r})"
            )
        node, cells = entry["node"], entry["cells"]
        # YAML `true`/`yes` is a bool, and bool IS an int in Python — without
        # the explicit rejection a typo'd `node: true` would launder into a
        # real order 1 declaration (the repo convention, cf. config.py).
        if not isinstance(node, int) or isinstance(node, bool) or node < 0:
            raise ValueError(
                f"output.pyramid.overviews[{i}].node must be an int >= 0 (got {node!r})"
            )
        if isinstance(cells, int) and not isinstance(cells, bool):
            cells = [cells]
        if (
            not isinstance(cells, list)
            or not cells
            or not all(isinstance(c, int) and not isinstance(c, bool) for c in cells)
        ):
            raise ValueError(
                f"output.pyramid.overviews[{i}].cells must be an int or a non-empty "
                f"list of ints (got {entry['cells']!r})"
            )
        levels.append({"node": int(node), "cells": [int(c) for c in cells]})
    return levels


def validate_overviews(levels: list, *, parent_order: int, child_order: int) -> None:
    """Refuse an out-of-contract level schedule — loudly, never by widening.

    The issue #382 rules, over the normalized form:

    - entries by strictly descending ``node``, each in ``[0, parent_order]``
      (the shard node itself is legal — its levels become the leaf-written
      column of #381 point (2));
    - within an entry, ``cells`` strictly descending, each ``>= node``
      (equality: the 1-cell whole-footprint group) and ``< child_order``
      (a member at the base data's own order IS the base data — binding at
      the leaf node, implied coarser by the descent rule);
    - across consecutive entries, resolutions strictly descend — the next
      entry's finest member must be strictly coarser than the previous
      entry's coarsest **reader-facing** one. Two exemptions: the declared
      gather (the SAME ``cells`` list at a coarser ``node`` — pure
      concatenation, #381 point (3)), and a trailing ``cells == node``
      member, which is the 1-cell whole-footprint group of #381 point (2)
      — the writer's universal partial, whose entire purpose is serving the
      coarser folds declared below it, so it never sets the descent floor.
      An entry declaring ONLY its node-order member therefore constrains
      nothing that follows.
    """
    parent_order, child_order = int(parent_order), int(child_order)
    for i, entry in enumerate(levels):
        node, cells = entry["node"], entry["cells"]
        if not 0 <= node <= parent_order:
            raise ValueError(
                f"output.pyramid.overviews[{i}].node = {node} is outside [0, parent_order = "
                f"{parent_order}] — level artifacts live at hive prefixes, none finer "
                f"than the shard order"
            )
        if any(b >= a for a, b in zip(cells, cells[1:])):
            raise ValueError(f"output.pyramid.overviews[{i}].cells {cells} must strictly descend")
        if cells[-1] < node:
            raise ValueError(
                f"output.pyramid.overviews[{i}].cells {cells} reach coarser than their own "
                f"node {node} — a node stores resolutions >= its own order (equality is "
                f"the 1-cell whole-footprint group)"
            )
        if cells[0] >= child_order:
            raise ValueError(
                f"output.pyramid.overviews[{i}].cells {cells} reach the base data's own "
                f"cell order ({child_order}) — the base level is the store itself, "
                f"not a declarable overview member"
            )
    for i, (prev, entry) in enumerate(zip(levels, levels[1:]), start=1):
        if entry["node"] >= prev["node"]:
            raise ValueError(
                f"output.pyramid.overviews nodes must strictly descend: overviews[{i}].node = "
                f"{entry['node']} does not descend from overviews[{i - 1}].node = {prev['node']}"
            )
        if entry["cells"] == prev["cells"]:
            continue  # the declared gather: same cells, coarser node
        # The previous entry's floor is its coarsest READER-FACING member: a
        # trailing `cells == node` member is the 1-cell whole-footprint group
        # (#381 point (2)), the writer's universal partial that exists to serve
        # the coarser folds below it, so it is exempt. An entry that declares
        # only that member constrains nothing.
        reader = [c for c in prev["cells"] if c != prev["node"]]
        if not reader:
            continue
        if entry["cells"][0] >= reader[-1]:
            raise ValueError(
                f"output.pyramid.overviews[{i}].cells {entry['cells']} do not descend from "
                f"overviews[{i - 1}].cells {prev['cells']} — across entries resolutions "
                f"strictly descend below the previous entry's coarsest reader-facing "
                f"member ({reader[-1]}), except a declared gather (the SAME cells at a "
                f"coarser node)"
            )


def default_overviews(parent_order: int, chunk_order: int, *, child_order: int) -> list[dict]:
    """The derived default schedule, bound to the grid block (#381 point (5)).

    With ``d = chunk_order - parent_order``: ``{node: parent_order, cells:
    [chunk_order]}`` plus ``{node: k, cells: [k + d]}`` for
    ``k = parent_order - 2`` stepping ``-2`` down to ``parent_order % 2`` —
    node 0 on even shard orders, node 1 on odd (step by 2 until you run out;
    espg ruling on the PR #389 thread, matching the ``/1`` default's
    ``range(shard - 2, -1, -2)`` reach). For the 19/13/9 reference geometry:
    (9,[13]) (7,[11]) (5,[9]) (3,[7]) (1,[5]). An explicit ``overviews:``
    block replaces this default WHOLESALE.

    ``chunk_order`` is the grid's RESOLVED chunk order — pass
    ``grid.chunk_order``, never the raw ``output.grid.chunk_inner`` knob: an
    unset knob does not mean K == 1, since :func:`zagg.grids.get_grid`
    derives ``child_order - 6`` for a sharded fullsphere grid (issue #259).
    Passing the raw ``None``-shaped knob as ``parent_order`` would silently
    hand back an all-whole-footprint schedule for exactly the flagship
    geometry. When K == 1 the resolved chunk order equals ``parent_order``,
    giving ``d = 0`` — every level the degenerate whole-footprint group,
    still legal.

    The result is checked against :func:`validate_overviews` before it is
    returned: the derived default is what every store that never spells
    ``overviews:`` gets, so it must be valid by construction for every geometry
    the grid admits. ``HealpixGrid`` accepts ``chunk_order == child_order``,
    whose base entry would BE the base data — that geometry refuses loudly
    here rather than shipping an invalid schedule into a manifest.
    """
    parent_order, chunk_order = int(parent_order), int(chunk_order)
    d = chunk_order - parent_order
    levels = [{"node": parent_order, "cells": [parent_order + d]}]
    levels += [{"node": k, "cells": [k + d]} for k in range(parent_order - 2, -1, -2)]
    validate_overviews(levels, parent_order=parent_order, child_order=child_order)
    return levels


def declared_fields(config) -> tuple[dict, list]:
    """Per-field D24 composability declarations for the manifest pyramid block.

    The ``fields`` map is identical under both spec revisions: ``exact``
    entries carry the fold method + nan policy + dense dtype/fill, and
    ``approximate`` entries the k-way t-digest law + ragged element typing;
    ``none`` fields are declared with their **class only** — the recorded
    absence that gives readers zero-open filtering (option A, the ruled D24
    default) — and returned in the excluded list for the caller's loud
    template-time warning (:func:`warn_excluded`).
    """
    from zagg.config import get_agg_fields
    from zagg.semantics import EXACT_MERGE_LAWS, _fold_function_name, composability_classes
    from zagg.sweep_overview import EXACT_NAN_POLICY, TDIGEST_LAW

    agg = get_agg_fields(config)
    fields: dict = {}
    excluded: list = []
    for name, cls in composability_classes(config).items():
        meta = agg[name]
        if cls == "exact":
            fields[name] = {
                "class": "exact",
                "method": EXACT_MERGE_LAWS[_fold_function_name(meta.get("function")) or ""],
                "nan_policy": EXACT_NAN_POLICY,
                "dtype": meta.get("dtype", "float32"),
                "fill_value": _json_fill(meta.get("fill_value", "NaN")),
            }
        elif cls == "approximate":
            inner = meta.get("inner_shape") or (2,)
            fields[name] = {
                "class": "approximate",
                "method": TDIGEST_LAW,
                "dtype": meta.get("dtype", "float32"),
                "inner_shape": [int(inner)] if isinstance(inner, int) else [int(x) for x in inner],
                "delta": int((meta.get("params") or {}).get("delta", 512)),
            }
        else:
            fields[name] = {"class": "none"}
            excluded.append(name)
    return fields, excluded


def warn_excluded(excluded: list) -> None:
    """The loud template-time D24 warning for ``none``-class fields."""
    logger.warning(
        f"pyramid: fields {excluded} are non-composable (D24 class 'none') and will "
        f"exist ONLY at native resolution — excluded from every overview level (the "
        f"per-field-exclusion default; declare a derived summary to opt in, issue #201)"
    )


def _json_fill(fill_value):
    """A JSON-safe fill token (zarr v3 uses the string ``"NaN"``)."""
    if isinstance(fill_value, float) and np.isnan(fill_value):
        return "NaN"
    return fill_value


def overview_block_v2(knob: dict, levels: list, fold: tuple, fields: dict, excluded: list) -> dict:
    """The ``zagg-pyramid/2`` manifest block for an explicit ``overviews:`` knob.

    ``levels`` is the NORMALIZED grouped form (scalars expanded) — the
    manifest is the reader-facing contract, so no sugar survives into it.
    The list records at BLOCK level (``pyramid.overviews``, espg shape
    ruling on the PR #389 thread): it is the store-wide product declaration,
    consumed by fleet writers, stage sweeps, forecasts, and external
    readers, while the singular ``pyramid.overview`` family dict keeps one
    sweep leg's execution regime exactly as under ``/1`` (``all_time``,
    the issue #376 ``fold_source``/``exact_levels`` pair — ``exact_levels``
    only under the cascade — ``fields``, ``summarize``, and the sweep's
    ``materialized`` actuals). Entries carry exactly ``node`` and ``cells``
    at template time; the staged sweep (issues #383/#384) later nests
    per-``(node, cells)`` materialization actuals inside the entry it swept
    — it never rewrites the declaration. A declared store is NOT yet
    sweepable by this zagg: declaring is free, sweeping is the operational
    decision (#381 point (11)), and
    :func:`zagg.sweep_overview.sweep_overviews` refuses ``/2`` loudly.
    """
    if excluded:
        warn_excluded(excluded)
    fold_source, exact_levels = fold
    overview: dict = {
        "all_time": bool(knob.get("all_time", False)),
        "fold_source": fold_source,
    }
    if fold_source == "cascade":
        overview["exact_levels"] = exact_levels
    overview["fields"] = fields
    if knob.get("summarize"):
        overview["summarize"] = {str(k): dict(v) for k, v in knob["summarize"].items()}
    return {
        "spec": PYRAMID_SPEC_V2,
        "overviews": [dict(e) for e in levels],
        "overview": overview,
    }
