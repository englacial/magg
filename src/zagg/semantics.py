"""Semantic-core canonicalization + hash (issue #299, D19).

A product's identity splits in two (D19, ``docs/design/sparse_coverage.md``):
the **name** addresses it (the ``{store_root}/{name}/`` product root), and the
**``semantic_hash``** verifies it — sha256 over the canonicalized
**output-defining subset only**. Two configs with the same semantic core
produce the same values in every cell they share; everything else is
packaging.

Included (the semantic core):

- the ``aggregation`` block — functions, params, dtypes, fills, ragged kinds,
  declared coordinates, ``chunk_precompute`` — minus the ``handoff`` carrier
  choice (arrow vs pandas is a worker-internal transport);
- the ``data_source`` **semantics** — which groups/variables/coordinates are
  read and how observations are filtered (``filters``/``quality_filter``,
  photon ``base_level``/``levels``, raster ``bands``/``nodata``/
  ``collections``/``static_data``) — minus the read machinery (``reader``,
  ``driver``, ``read_plan``, ``anonymous``);
- the grid **type + indexing scheme** (D19: cell order is a resolution axis
  (D24), parent/shard order and chunking are packaging — hashing the whole
  template would have made o8 and o9 runs different products and blocked
  mixed-order processing);
- the **pipeline type** (``spatial`` | ``temporal`` | ``event``; absent
  normalizes to ``spatial`` — espg-ruled on the PR #316 review: a temporal
  engine over the same aggregation block is a different product; D19's
  ratified list omitted it only because the temporal path wasn't in frame);
- the raster **time-coordinate encoding** (``output.time_encoding``, spec §8 /
  issue #443), keyed only when non-default: toc words and legacy microseconds
  are different stored values meaning different things, exactly as
  ``weights: "flux"`` is.

Excluded as packaging: all orders (``parent_order``/``child_order``/
``chunk_inner``), ``sharded``, store layout/path, ``emit_cell_ids`` (the
issue #304 transition hatch), worker sizing, streaming mode, read knobs,
catalog/bounds (run inputs, recorded per-run — catalog identity lives in the
D20 sidecar, never the product identity), and the per-variable
``overview_delta`` (issue #424 — the pyramid-fold budget shapes overview
artifacts only, and the overview family is already packaging). A variable's
``weights: "counts"`` normalizes away as the spec §2.0 absent-key default;
``weights: "flux"`` is output-defining and hashes.

Canonical form: the core dict serialized as sorted-key, compact,
ASCII-escaped JSON — so YAML comments, whitespace, and key order can never
change the hash (§8.3 canonicalization obligations). The hash is the **full
sha256 hex digest** (git-style: the full digest is what is compared; the
12-hex :func:`semantic_fingerprint` is the display/CLI shorthand — 48 bits,
comfortable for the only collision domain that matters, display within one
store's product listing).

This module formalizes the #89 signature seam (``grid.spatial_signature`` /
``config.output_field_signature`` are dict fingerprints; this is the
content-addressed form) — see the issue #299 thread for the design record.
"""

from __future__ import annotations

import hashlib
import json

from zagg.config import PipelineConfig, get_pipeline_type
from zagg.time_axis import DEFAULT_TIME_ENCODING

#: ``data_source`` keys that are read machinery, not output semantics (D19).
#: Changing any of these must never change the ``semantic_hash``.
DATA_SOURCE_PACKAGING_KEYS = ("reader", "driver", "read_plan", "anonymous")

#: ``aggregation`` keys that are packaging: the per-cell carrier choice
#: (issue #132) transports identical values either way.
AGGREGATION_PACKAGING_KEYS = ("handoff",)

#: Per-variable aggregation keys that are packaging (issue #424):
#: ``overview_delta`` shapes the pyramid/overview fold budget only — the
#: overview family is packaging already (``output.pyramid`` never enters the
#: core), so the split budget must not move a base product's identity either.
VARIABLE_PACKAGING_KEYS = ("overview_delta",)

#: Display length of :func:`semantic_fingerprint` (12 hex = 48 bits; the
#: birthday bound puts same-store collision odds around 1e-8 at 1e4 products
#: — recorded rationale on the issue #299 thread).
FINGERPRINT_HEX = 12

#: Non-healpix grid keys that spatially define the product (F1, issue #299).
#: For rect/other grids the cell geometry is fixed by CRS + resolution +
#: bounds, so two such products differing in any of these are different
#: products (D24's resolution-axis exclusion is a HEALPix/morton composability
#: argument that does not extend to rect — over-discriminating is safe, a
#: semantic collision is not). HEALPix stays type + indexing scheme only.
GRID_SPATIAL_KEYS = ("crs", "resolution", "bounds")


#: D24 exact merge laws: aggregator name -> fold law. These are the reducers
#: whose per-cell outputs compose EXACTLY across cell orders — count/sum by
#: addition, min/max by extremum — so an up-tree fold of leaf values equals a
#: direct aggregation at the coarser order, byte for byte (§8.3). Names are
#: matched after :func:`_fold_function_name` normalization, so ``min``,
#: ``np.min`` and ``numpy.min`` all key the same law.
#:
#: The plain and nan-aware variants share a law ON PURPOSE, and the exactness
#: claim is against the **nan-skipping** reduction for both (review finding,
#: issue #201): a leaf's stored NaN is the same bytes whether it is the fill
#: sentinel or a NaN datum, so nothing downstream can tell them apart —
#: :func:`zagg.sweep_overview.fold_dense` skips both, and the overview records
#: :data:`zagg.sweep_overview.EXACT_NAN_POLICY` per field so a reader knows
#: which reduction it actually got. A NaN-propagating ``min``/``max``/``sum``
#: is unrecoverable at this seam, not merely unimplemented.
EXACT_MERGE_LAWS = {
    "len": "sum",
    "count": "sum",
    "sum": "sum",
    "nansum": "sum",
    "min": "min",
    "nanmin": "min",
    "max": "max",
    "nanmax": "max",
}

#: The three D24 composability classes, weakest first. ``exact`` folds byte-
#: equal; ``approximate`` merges natively but order-dependently (t-digests —
#: ``np.isclose`` equality, the merge-vs-spill epistemic class); ``none`` has
#: no merge law and exists only at native resolution (per-field exclusion —
#: the ruled D24 default, issue #201).
COMPOSABILITY_CLASSES = ("none", "approximate", "exact")


def _fold_function_name(name) -> str | None:
    """Normalize an aggregator ``function`` name for merge-law lookup.

    ``np.``/``numpy.`` prefixes strip (they resolve to the same callables in
    :func:`zagg.config.resolve_function`); other dotted paths pass through
    unchanged (their laws are keyed by full path, e.g. the t-digest builders).
    """
    if not isinstance(name, str):
        return None
    for prefix in ("np.", "numpy."):
        if name.startswith(prefix):
            return name.removeprefix(prefix)
    return name


def field_composability(meta: dict) -> str:
    """The D24 composability class of one aggregation field's metadata.

    Derived from the field's aggregator via the merge-law flags the #217
    mergeable-reducer machinery already established (the same set
    ``validate_streaming`` accepts, widened by the exact sum/min/max laws):

    - ``exact`` — scalar ``function`` in :data:`EXACT_MERGE_LAWS`;
    - ``approximate`` — an unlocated ragged t-digest field with the standard
      ``(2,)`` centroid inner shape (merge is order-dependent; ``np.isclose``
      equality class, cf. D24);
    - ``none`` — everything else: expressions, vector fields, chunk-resolution
      companions, located ragged (no streaming merge law for the location
      channel yet), and any scalar reducer without an exact law (mean, std,
      median, quantiles, ...).
    """
    from zagg.config import get_output_signature

    sig = get_output_signature(meta)
    if meta.get("expression") is not None or sig["resolution"] != "cell":
        return "none"
    function = _fold_function_name(meta.get("function"))
    if sig["kind"] == "ragged":
        from zagg.processing.streaming import _TDIGEST_FUNCTIONS

        if (
            sig["location"] is None
            and meta.get("function") in _TDIGEST_FUNCTIONS
            and tuple(sig["inner_shape"]) == (2,)
        ):
            return "approximate"
        return "none"
    if sig["kind"] == "scalar" and function in EXACT_MERGE_LAWS:
        return "exact"
    return "none"


def composability_classes(config: PipelineConfig) -> dict[str, str]:
    """``{field_name: composability class}`` for every aggregation field (D24)."""
    from zagg.config import get_agg_fields

    return {name: field_composability(meta) for name, meta in get_agg_fields(config).items()}


def _without(mapping: dict, keys: tuple[str, ...]) -> dict:
    return {k: v for k, v in (mapping or {}).items() if k not in keys}


def _normalize_variables(aggregation: dict) -> dict:
    """Drop per-variable packaging keys and default-valued declarations (#424).

    Two normalizations, both hash-stability obligations:

    - :data:`VARIABLE_PACKAGING_KEYS` (``overview_delta``) drop outright —
      overview artifacts are packaging, so declaring the split fold budget
      hashes identically to omitting it;
    - ``weights: "counts"`` drops because it is the spec §2.0 **absent-key
      default** — the explicit and absent spellings mean the same bytes, so
      they must be the same product. ``weights: "flux"`` stays: the stored
      weight column means something else, which is exactly output-defining.
    """
    variables = (aggregation or {}).get("variables")
    if not variables:
        return aggregation
    out = {}
    for name, meta in variables.items():
        if isinstance(meta, dict):
            meta = {k: v for k, v in meta.items() if k not in VARIABLE_PACKAGING_KEYS}
            if meta.get("weights") == "counts":
                meta = {k: v for k, v in meta.items() if k != "weights"}
        out[name] = meta
    return {**aggregation, "variables": out}


def _prune_nulls(obj):
    """Recursively drop ``None``-valued keys from every dict in ``obj``.

    §8.3 canonicalization: a YAML explicit-null (``key:``) must hash identically
    to an absent key, at every depth — not just the top level. Applied to the
    whole core so a nested ``None`` (e.g. ``quality_filter.value:``) drops out.
    Lists are recursed but never pruned by value: list entries are positional,
    so a ``None`` element is content, not an absent key.
    """
    if isinstance(obj, dict):
        return {k: _prune_nulls(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, list):
        return [_prune_nulls(v) for v in obj]
    return obj


def semantic_core(config: PipelineConfig) -> dict:
    """The output-defining subset of ``config`` (D19), as a plain dict.

    Deterministic given the config's semantics: two configs differing only in
    packaging knobs (orders, chunking, worker size, read machinery, carrier)
    map to the same core. ``None``-valued keys are pruned recursively so a
    YAML explicit-null hashes identically to an absent key (§8.3). The returned
    structure is JSON-serializable plain data (the YAML loader guarantees it).
    """
    grid_cfg = (config.output or {}).get("grid", {}) or {}
    grid_type = grid_cfg.get("type", "healpix")
    grid: dict = {"type": grid_type}
    if grid_type == "healpix":
        # The one indexing scheme zagg writes (the morton store convention
        # rides D16 attrs; the underlying cell tiling is HEALPix NESTED).
        grid["indexing_scheme"] = "nested"
    else:
        # Rect/other: fold in the spatially-defining params when present (F1).
        for key in GRID_SPATIAL_KEYS:
            if key in grid_cfg:
                grid[key] = grid_cfg[key]
    core: dict = {
        "aggregation": _normalize_variables(
            _without(config.aggregation, AGGREGATION_PACKAGING_KEYS)
        ),
        "data_source": _without(config.data_source, DATA_SOURCE_PACKAGING_KEYS),
        "grid": grid,
        # pipeline.type is output-defining (espg-ruled on the PR #316
        # review, 2026-07-21): a temporal/event engine over an identical
        # aggregation block is a DIFFERENT product. Absent normalizes to
        # the "spatial" default on both sides (get_pipeline_type's default),
        # the same discipline as the manifest's path_grouping absent=>1, so
        # every existing config hashes stably.
        "pipeline": {"type": get_pipeline_type(config)},
    }
    # The time coordinate's encoding (spec §8, issue #443) is output-defining
    # for the same reason `weights: "flux"` is: the stored axis MEANS
    # something else. Keyed only when non-default, so every config written
    # before §8 — and the explicit `microseconds` spelling of the absent-key
    # default — hashes byte-identically to today.
    encoding = (config.output or {}).get("time_encoding")
    if encoding not in (None, DEFAULT_TIME_ENCODING):
        core["time_encoding"] = encoding
    return _prune_nulls(core)


def canonical_semantic_json(config: PipelineConfig) -> str:
    """The canonical serialized form the hash is computed over.

    Sorted keys, compact separators, ASCII-escaped: syntactic YAML edits
    (comments, whitespace, key order) cannot reach this string.
    """
    return json.dumps(
        semantic_core(config), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )


def semantic_hash(config: PipelineConfig) -> str:
    """Full sha256 hex digest of the canonical semantic core (64 hex chars).

    The frozen manifest key (D19): reusing a product name with a different
    semantic hash refuses up front, exactly as an orders mismatch does. Always
    compare the FULL digest; display via :func:`semantic_fingerprint`.
    """
    return hashlib.sha256(canonical_semantic_json(config).encode()).hexdigest()


def semantic_fingerprint(digest: str) -> str:
    """12-hex display shorthand of a full ``semantic_hash`` digest."""
    if len(digest) < FINGERPRINT_HEX:
        raise ValueError(f"not a semantic hash digest: {digest!r}")
    return digest[:FINGERPRINT_HEX]


__all__ = [
    "AGGREGATION_PACKAGING_KEYS",
    "COMPOSABILITY_CLASSES",
    "DATA_SOURCE_PACKAGING_KEYS",
    "VARIABLE_PACKAGING_KEYS",
    "EXACT_MERGE_LAWS",
    "FINGERPRINT_HEX",
    "GRID_SPATIAL_KEYS",
    "canonical_semantic_json",
    "composability_classes",
    "field_composability",
    "semantic_core",
    "semantic_fingerprint",
    "semantic_hash",
]
