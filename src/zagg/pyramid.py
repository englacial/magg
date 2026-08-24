"""Pyramid overview grammar: leaf resolutions + the fixed ladder (issue #382).

The ``zagg-pyramid/2`` revision of the manifest pyramid declaration, after
the espg grammar-collapse ruling on the PR #389 thread: the ONLY thing a
config declares is ``output.pyramid.overviews`` — the **leaf cell
resolutions** (scalar sugar for one; strictly descending ints; each strictly
between ``parent_order`` and ``child_order``; omitted, the default is the
grid's resolved chunk order). All declared resolutions materialize at the
leaf (shard) node. Everything ABOVE the shard is LAW, not declaration — the
**fixed every-order ladder**: with ``d = base - parent_order`` (``base`` the
coarsest leaf resolution), every order ``k`` from ``parent_order - 1`` down
to 0 inclusive carries exactly one member at resolution ``k + d``. Two
numbers determine the whole above-shard pyramid; there is no per-node
spelling, no gather declaration, no member promotion, and no ``partial/``
grammar. ``output.pyramid: false`` remains the off switch.

The manifest records the **fully expanded** ``(node, cells)`` list at block
level (``pyramid.overviews`` — readers never re-derive the ladder); slab
length per member is ``4^(cells - node)``. Today's constant-depth grammar
(``zagg-pyramid/1``, ``orders`` + ``spacing``) is the special case
``cells = [node + (child_order - parent_order)]`` and stays readable; its
constants and its sweep live in :mod:`zagg.sweep_overview`.

Refusals are loud and named; nothing widens silently. This module owns the
grammar only: parsing/normalization, validation, the ladder expansion, and
the per-field D24 composability declarations both spec revisions record.
Writers (issue #383) and the staged sweep (issue #384) consume the expanded
form; nothing here writes a store.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

#: Version string of the manifest ``pyramid`` block carrying grouped
#: ``(node, cells)`` levels (issue #382). The ``/1`` marker lives in
#: :mod:`zagg.sweep_overview`, which owns sweeping that revision.
PYRAMID_SPEC_V2 = "zagg-pyramid/2"


def normalize_overviews(raw) -> list[int]:
    """``output.pyramid.overviews`` normalized: the leaf cell resolutions.

    An int (scalar sugar for one resolution) or a non-empty list of ints.
    Shape errors raise here; the ordering/range rules live in
    :func:`validate_overviews` (they need the grid orders). The old
    ``{node, cells}`` pair spelling was collapsed away (espg ruling, PR #389
    thread) and refuses by name.
    """
    # YAML `true`/`yes` is a bool, and bool IS an int in Python — without the
    # explicit rejection a typo'd `overviews: true` would launder into a real
    # resolution-1 declaration (the repo convention, cf. config.py).
    if isinstance(raw, int) and not isinstance(raw, bool):
        raw = [raw]
    if isinstance(raw, list) and any(isinstance(e, dict) for e in raw):
        raise ValueError(
            "output.pyramid.overviews takes leaf cell resolutions only — the "
            "{node, cells} pair spelling was collapsed away (everything above the "
            "shard is the fixed ladder; espg ruling on the PR #389 thread)"
        )
    if (
        not isinstance(raw, list)
        or not raw
        or not all(isinstance(r, int) and not isinstance(r, bool) for r in raw)
    ):
        raise ValueError(
            f"output.pyramid.overviews must be leaf cell resolutions — an int or a "
            f"non-empty list of ints (got {raw!r}); `output.pyramid: false` is how a "
            f"store declares no pyramid"
        )
    return [int(r) for r in raw]


def validate_overviews(resolutions: list, *, parent_order: int, child_order: int) -> None:
    """Refuse an out-of-contract leaf resolution list — loudly, never by widening.

    The collapsed issue #382 rules: strictly descending, and every resolution
    strictly between ``parent_order`` and ``child_order`` — a member at the
    shard's own order is the writer-side whole-footprint aggregate (never
    declared), and a member at the base data's own order IS the base data.
    """
    parent_order, child_order = int(parent_order), int(child_order)
    if any(b >= a for a, b in zip(resolutions, resolutions[1:])):
        raise ValueError(
            f"output.pyramid.overviews {resolutions} must strictly descend "
            f"(finest leaf resolution first)"
        )
    for r in resolutions:
        if not parent_order < r < child_order:
            raise ValueError(
                f"output.pyramid.overviews resolution {r} is not strictly between "
                f"parent_order ({parent_order}) and child_order ({child_order}) — leaf "
                f"overview members live strictly inside the shard's resolution window "
                f"(the base data itself is order {child_order}; the shard-order "
                f"aggregate is writer-side, never declared)"
            )


def expand_overviews(resolutions: list, *, parent_order: int) -> list[dict]:
    """The fully expanded ``(node, cells)`` list the manifest records.

    One leaf entry — every declared resolution materializes at the shard
    node — plus the **fixed every-order ladder** (espg ruling, PR #389
    thread, option D): with ``d = base - parent_order`` (``base`` the
    coarsest leaf resolution), ``{node: k, cells: [k + d]}`` for every ``k``
    from ``parent_order - 1`` down to 0 inclusive. Every store roots at
    order 0; two numbers determine everything above the shard, so readers
    never re-derive — the manifest list is the contract. Valid by
    construction GIVEN a :func:`validate_overviews`-clean input; callers
    that skip validation (a grid-less retrofit config) rely on
    ``declare_pyramid`` validating against the manifest before any write.
    """
    parent_order = int(parent_order)
    resolutions = [int(r) for r in resolutions]
    d = resolutions[-1] - parent_order
    levels = [{"node": parent_order, "cells": list(resolutions)}]
    levels += [{"node": k, "cells": [k + d]} for k in range(parent_order - 1, -1, -1)]
    return levels


def default_overviews(parent_order: int, chunk_order: int, *, child_order: int) -> list[dict]:
    """The omitted-knob default: one leaf resolution at the chunk order.

    ``output.pyramid.overviews`` omitted means ``[chunk_order]`` (espg
    ruling, PR #389 thread) — the fully expanded list is then the leaf entry
    ``{node: parent_order, cells: [chunk_order]}`` plus the fixed ladder of
    :func:`expand_overviews`. For the 19/13/9 reference geometry: (9,[13])
    then (8,[12]) (7,[11]) … (0,[4]).

    ``chunk_order`` is the grid's RESOLVED chunk order — pass
    ``grid.chunk_order``, never the raw ``output.grid.chunk_inner`` knob: an
    unset knob does not mean K == 1, since :func:`zagg.grids.get_grid`
    derives ``child_order - 6`` for a sharded fullsphere grid (issue #259).
    The default is validated before expansion, so the two geometries with no
    strictly-interior chunk order refuse loudly rather than shipping an
    invalid schedule: ``chunk_order == parent_order`` (K == 1) and
    ``chunk_order == child_order`` (the base entry would BE the base data) —
    both are declarable only explicitly, or off.
    """
    validate_overviews([int(chunk_order)], parent_order=parent_order, child_order=child_order)
    return expand_overviews([int(chunk_order)], parent_order=parent_order)


def declared_fields(config) -> tuple[dict, list]:
    """Per-field D24 composability declarations for the manifest pyramid block.

    The ``fields`` map is identical under both spec revisions: ``exact``
    entries carry the fold method + nan policy + dense dtype/fill,
    ``approximate`` entries the k-way t-digest law + ragged element typing,
    and ``packed`` entries (issue #515) the composition k-way law + the §3.3
    ``of`` linkage the fold's ``n`` inputs come from; ``none`` fields are
    declared with their **class only** — the recorded absence that gives
    readers zero-open filtering (option A, the ruled D24 default) — and
    returned in the excluded list for the caller's loud template-time warning
    (:func:`warn_excluded`).

    A ``packed`` field is demoted to ``none`` (and excluded) when the digest
    its ``attrs.composition.of`` names is not itself declared ``approximate``
    here, or when that digest declares non-default §2.0 ``weights``: the law
    divides by that digest's per-cell weight at every level (spec §3.3/§3.4),
    so composition folds only where its divisor does *and* only where that
    divisor is a count — declaring a fold whose ``n`` no overview carries, or
    whose ``n`` is a flux sum, would be the same dishonesty the class exists
    to end.
    """
    from zagg.config import get_agg_fields
    from zagg.semantics import EXACT_MERGE_LAWS, _fold_function_name, composability_classes
    from zagg.sweep_overview import (
        COMPOSITION_LAW,
        EXACT_NAN_POLICY,
        TDIGEST_LAW,
        overview_fold_delta,
    )

    agg = get_agg_fields(config)
    fields: dict = {}
    excluded: list = []
    classes = composability_classes(config)
    for name, cls in classes.items():
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
            delta = int((meta.get("params") or {}).get("delta", 512))
            fields[name] = {
                "class": "approximate",
                "method": TDIGEST_LAW,
                "dtype": meta.get("dtype", "float32"),
                "inner_shape": [int(inner)] if isinstance(inner, int) else [int(x) for x in inner],
                "delta": delta,
                # The split pyramid-fold budget (issue #424), recorded RESOLVED
                # so the manifest is self-describing: the sweep folds at this
                # value. Pre-#424 manifests lack the key — the reader-side
                # capped fallback in overview_fold_delta reproduces their
                # historical fold-at-leaf-δ behavior exactly (all carried
                # δ ≤ 512).
                "overview_delta": overview_fold_delta(
                    {"delta": delta, "overview_delta": meta.get("overview_delta")}
                ),
            }
            # The §9 located companion (ruling 4 on issue #410): a located field
            # folds through the pyramid, so every overview level carries the
            # ``{field}_locations`` sibling and the sweep folds it with the
            # digest. Recorded because the manifest is the ONLY thing the
            # overview writer reconstructs a field from
            # (``sweep_overview._overview_config``) — without it the overview
            # template would emit no sibling and the fold would have nowhere to
            # write. Keyed only when set, so an unlocated field's manifest entry
            # is byte-identical to pre-#410.
            if meta.get("location") is not None:
                fields[name]["location"] = str(meta["location"])
            # The §8.3 temporal companion, on the same footing (espg-ruled
            # 2026-08-17, amending ruling 3): per-centroid at every level, so
            # the overview template needs the sibling and the fold needs to know
            # to thread the channel. Recorded as the SHAPE, which is all a level
            # above the leaf can act on — the leaf's ingest column is not
            # re-read by any fold. Keyed only when set.
            if meta.get("temporal") is not None:
                fields[name]["temporal"] = str(meta["temporal"])
            # The §2.0 weights declaration, keyed only when non-default so
            # existing manifests stay byte-identical; the sweep's fold gate
            # compares it against the stored arrays (issue #424). Its
            # calibration provenance rides along, because the manifest is the
            # ONLY thing the overview writer reconstructs a field from
            # (``sweep_overview._overview_config``) and §2.0 makes ``gain``
            # REQUIRED on every flux-declared array — without it the overview
            # would be stamped flux with its calibration unrecoverable.
            if meta.get("weights") not in (None, "counts"):
                fields[name]["weights"] = meta["weights"]
                gain = (meta.get("attrs") or {}).get("gain")
                if gain is not None:
                    fields[name]["gain"] = gain
        elif cls == "packed":
            block = (meta.get("attrs") or {}).get("composition") or {}
            of = block.get("of")
            # The divisor must ALSO be a §2.0 counts digest: the law divides by
            # that digest's summed centroid weights (``payload_weight`` rounds
            # them to an int) and ``N_signal`` is a photon COUNT, so a
            # flux-weighted digest is the wrong ``n`` — wrong-but-plausible
            # lanes, which is the failure mode the issue #424 weights gate
            # exists to stop and which the packed fold arms never reach (they
            # never call ``check_weights_match``). Demoted here instead.
            of_weights = (agg.get(of) or {}).get("weights") if of else None
            if classes.get(of) != "approximate" or of_weights not in (None, "counts"):
                fields[name] = {"class": "none"}
                excluded.append(name)
                continue
            fields[name] = {
                "class": "packed",
                "method": COMPOSITION_LAW,
                "dtype": meta.get("dtype", "uint64"),
                "fill_value": _json_fill(meta.get("fill_value", 0)),
                # Load-bearing exactly as ``location`` is below: the overview
                # writer reconstructs a level's arrays from this entry alone,
                # and the fold pairs each contributor's word with THIS
                # digest's per-cell weight (spec §3.3/§3.4).
                "of": str(of),
            }
            # The committed cut rides along so the overview array's §3.3 attrs
            # block can carry it (keyed only when declared, like the digest
            # entries' optional keys).
            if block.get("threshold") is not None:
                fields[name]["threshold"] = block["threshold"]
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

    ``levels`` is the FULLY EXPANDED ``(node, cells)`` list from
    :func:`expand_overviews` (the leaf entry plus the fixed every-order
    ladder to 0) — the manifest is the reader-facing contract, so readers
    never re-derive the ladder and no config sugar survives into it.
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
