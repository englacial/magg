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
  ``collections``/``static_data``) — minus the read machinery, the credential
  mechanism and fan-out sizing (``reader``, ``driver``, ``read_plan``,
  ``anonymous``, ``credentials_provider``,
  ``shard_workers``/``granule_workers``);
- the grid **type + indexing scheme** (D19: cell order is a resolution axis
  (D24), parent/shard order and chunking are packaging — hashing the whole
  template would have made o8 and o9 runs different products and blocked
  mixed-order processing);
- the **pipeline type** (``spatial`` | ``temporal`` | ``event``; absent
  normalizes to ``spatial`` — espg-ruled on the PR #316 review: a temporal
  engine over the same aggregation block is a different product; D19's
  ratified list omitted it only because the temporal path wasn't in frame);
- the **leaf-shaping ``output`` knobs** (the issue #415 hash epoch, espg-ruled
  as PR #397 question (8)(c)): every knob that changes what a leaf *contains*
  or which artifacts sit *beside* it — :data:`OUTPUT_LEAF_SHAPING_KEYS` and
  :data:`GRID_LEAF_SHAPING_KEYS`. Before the epoch the whole ``output`` block
  was outside the core, so the issue #388 skip gate's identity pair could not
  see a leaf-shape change at all and read a config-changed rerun as ``equal``.
  Every knob here is folded in **resolved through its accessor**, never as
  spelled, so an explicit default hashes identically to an absent key (§8.3,
  the ``pipeline.type`` discipline);
- the raster **time-coordinate encoding** (``output.time_encoding``, spec §8 /
  issue #443), keyed only when non-default: toc words and legacy microseconds
  are different stored values meaning different things, exactly as
  ``weights: "flux"`` is.

Excluded as packaging: all orders (``parent_order``/``child_order``/
``chunk_inner`` — the D24 resolution axis, which the epoch deliberately did
NOT touch: hashing them would still make o8 and o9 runs different products and
block mixed-order processing), store layout/path/name, ``coverage_moc`` and
``sweep`` (run triggers over regenerable D9 caches), ``consolidate_metadata``
(a derived finalize blob), the whole ``pyramid`` block (D11 keeps it out of
the manifest's frozen keys because the §7 sweep populates it; the leaf column
it declares is verified by READING the artifact instead — see
:data:`OUTPUT_LEAF_SHAPING_KEYS`), worker sizing, streaming mode, read knobs,
the credential mechanism (``credentials_provider``/``anonymous`` select how
source bytes are fetched, never what is computed — espg-ruled 2026-08-17),
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

#: ``data_source`` keys that are read machinery or worker sizing, not output
#: semantics (D19 names the two as separate packaging categories).
#: Changing any of these must never change the ``semantic_hash``.
#:
#: The granule fan-out width joined at the issue #415 hash epoch, under BOTH
#: its spellings — canonical ``shard_workers`` (issue #232) and the legacy
#: ``granule_workers`` (issue #180) the config still honors. D19's ratified
#: exclusion list already named "worker size" as packaging; the keys were
#: simply never listed here, which made the seams' fallback hash
#: **clamp-sensitive** (PR #397 question (7)): both dispatchers send a
#: per-cell ``data_source`` whose ``granule_workers`` is clamped to
#: ``min(K, n_granules)`` (:func:`zagg.runner._clamped_data_source`, issue
#: #184), so a small shard's worker-side hash differed from the run-level
#: hash — and ``semantic_hash`` being in :data:`zagg.telemetry._EQ_OR_NONE_KEYS`
#: then collapsed the identity half to ``None`` in any rollup mixing clamped
#: and unclamped shards.
#:
#: The clamp writes ``granule_workers`` whatever the config spelled, so that
#: key alone closes the reported drift; ``shard_workers`` rides along because
#: the two ARE one knob (``zagg.processing.worker._granule_workers`` resolves
#: the canonical name first) and hashing the canonical spelling of an excluded
#: knob would be incoherent. Issue #415 named only the legacy alias, so the
#: canonical one is flagged for ruling on the epoch PR rather than assumed.
#:
#: ``credentials_provider`` (issue #213 Phase 4/6) joined at the same epoch,
#: **espg-ruled 2026-08-17**: it selects HOW source bytes are fetched — which
#: registry name mints the DAAC credentials the dispatcher attaches to every
#: event — and never WHAT is computed from them, the same D19 class as
#: ``reader``/``driver``/``read_plan``. Three reasons it is ruled *here*
#: rather than left pinned as-is (issue #449):
#:
#: * the same granules fetched with ``lpdaac`` creds, ``gesdisc`` creds, or an
#:   ``anonymous`` open produce byte-identical leaves — a hash that moves with
#:   the provider is a false product split. It is the same class as the read
#:   knobs and as ``anonymous``, already excluded (a *class*, not one knob
#:   under two names: ``anonymous`` is read only by the raster source-store
#:   kwargs, ``credentials_provider`` only by the point and temporal paths);
#: * the failure asymmetry is loud in the excluded direction and silent in the
#:   included one: the wrong credential fails the *fetch* — a 403 at read time,
#:   never a quiet wrong value — so nothing depends on the hash to catch it,
#:   while hashing it silently splits one product in two;
#: * a credential migration must never rehash unchanged data. The MERRA-2 /
#:   gesdisc path is the live case: moving an existing store from one provider
#:   registration to another (or adding the key to a config that ran without
#:   it) would otherwise refuse every leaf as ``semantic-mismatch`` and rewrite
#:   the store to produce the same bytes.
#:
#: Ruled at the epoch deliberately so the first GEDI store's identity is born
#: without an auth knob in it: no store carries a provider-bearing hash yet
#: (the ``lpdaac`` line lands with the GEDI template, PR #450), so this costs
#: nothing now and cannot be taken back cheaply later.
#:
#: Unruled, and therefore **still hashed**: ``read_workers`` (the third
#: fan-out width beside the two spellings above), ``source_region`` (the
#: raster source-store kwarg that sits in the same dict literal as
#: ``anonymous``) and ``write_buffer``
#: (the live-slab bound on the streamed raster sink) all read as the same
#: fetch-mechanism class as the entries here. Excluding a key is a ruling, not
#: an assumption — the same footing ``shard_workers`` was flagged on above —
#: so they are raised on the epoch PR rather than added, and every one of them
#: still moves ``semantic_hash`` today.
DATA_SOURCE_PACKAGING_KEYS = (
    "reader",
    "driver",
    "read_plan",
    "anonymous",
    "credentials_provider",
    "shard_workers",
    "granule_workers",
)

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

#: ``output.grid`` keys that shape the LEAF, folded into the core's ``grid``
#: at the issue #415 hash epoch (PR #397 question (8)(c)).
#:
#: - ``sharded`` (issue #108) — the knob question (8) checked: flipping it
#:   over an existing store changes the leaf's object set (one ShardingCodec
#:   object per array vs K regular chunk objects) while the granule set holds,
#:   so pre-epoch the skip gate read ``equal`` and the store kept its old
#:   layout forever. The orders it interacts with (``chunk_inner``, and hence
#:   K) stay excluded — D24's resolution axis is untouched, so this is
#:   deliberately the one packing knob in the core.
#:
#:   Two consequences of hashing the DECLARATION rather than the effective
#:   layout, both known and both deliberate (resolving the effective layout
#:   needs K, and K needs the orders D24 keeps out):
#:
#:   * at ``K == 1`` the grid silently no-ops sharding (issue #215), so
#:     ``sharded: true`` and ``sharded: false`` write identical leaves and
#:     still hash apart — an over-discrimination, the safe direction (F1's
#:     recorded posture: "over-discriminating is safe, a semantic collision
#:     is not");
#:   * ``chunk_inner`` changes K, hence the leaf's object set, and still
#:     hashes equal — the object-layout hole is narrowed here, not closed.
#: - ``emit_cell_ids`` (issue #304) — the D16 transition hatch writes an
#:   ADDITIONAL ``cell_ids`` array into every leaf, which is a leaf-content
#:   difference by the same test, not a display preference. **Inferred from
#:   the criterion, not named by issue #415** (and previously excluded here
#:   by name), so it is flagged for ruling on the epoch PR alongside
#:   ``shard_workers``. Note the hatch is scheduled for removal: after that,
#:   a store built with it ON has a hash no legal config reproduces.
GRID_LEAF_SHAPING_KEYS = ("sharded", "emit_cell_ids")

#: ``output`` keys outside ``grid`` that shape the LEAF (issue #415 epoch).
#:
#: - ``aoi_mask`` (issue #101) — masks cells outside the AOI, so the leaf's
#:   VALUES differ. The plainest semantic knob that was outside the core.
#: - ``windowing`` (issue #246, D13) — partitions observations by time into
#:   per-window leaves, so a windowed and an unwindowed store over the same
#:   granules shared a hash before the epoch. What this buys is **product
#:   identity and the D20 sidecar's identity half**, not the manifest guard:
#:   ``temporal`` is already a frozen manifest key (:data:`zagg.hive.
#:   _FROZEN_MANIFEST_KEYS`) and a schedule change also lands the unit on a
#:   ``{window}.zarr`` path with no sidecar (D23), so pre-dispatch refusal and
#:   the leaf gate were already covered — the hash is what a multi-product
#:   root and a fleet leaf's recorded identity compare. Folded in its
#:   NORMALIZED form (:func:`zagg.config.get_windowing`), so ``epoch``
#:   spellings and the issue #355 point-window sugar canonicalize.
#:
#: ``output.pyramid`` is deliberately NOT here, though it does shape the leaf
#: (issue #383's column artifact), for three reasons that all point the same
#: way: the artifact half is already covered by reading the artifact —
#: :func:`zagg.hive.leaf_column_expectation`, the specification's §4.6 "read
#: the columns, not the manifest" posture, which is what closed question (8)'s
#: concrete regression; the block is excluded from the manifest's frozen keys
#: by D11 because the §7 sweep populates it; and the retrofit path
#: (:func:`zagg.sweep_overview.declare_pyramid`, ``--declare-pyramid``) exists
#: precisely to add a pyramid declaration to the config that built a store,
#: which its own semantic guard would then refuse. Hashing it would break a
#: supported workflow to re-cover ground already covered.
OUTPUT_LEAF_SHAPING_KEYS = ("aoi_mask", "windowing")


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


def _windowing_leaf_shape(config: PipelineConfig):
    """The normalized ``output.windowing`` declaration, or ``None`` (D13).

    Normalized rather than as-spelled so ``epoch`` spellings and the issue
    #355 point-window sugar canonicalize (§8.3). Total by contract: an
    out-of-grammar block is ``_validate_windowing``'s refusal to raise, not
    the hash's — :func:`semantic_core` must not start raising on a config that
    hashed before the epoch (the Lambda worker builds its config without
    ``validate_config``), so such a block simply hashes as spelled.

    Totality is split by fault class rather than caught around the whole call:
    a non-mapping block is refused on SHAPE here, and a mapping missing a
    required key (or carrying an unparsable ``epoch``/window bound) raises
    ``KeyError``/``ValueError`` out of the normalizer, which is a GRAMMAR
    fault ``_validate_windowing`` also refuses by name. Anything else —
    ``TypeError``, ``AttributeError`` — would be a fault INSIDE the normalizer
    and is deliberately left to propagate rather than launder into a digest.
    """
    from zagg.config import get_windowing

    block = (config.output or {}).get("windowing")
    if not isinstance(block, dict):
        return block
    try:
        return get_windowing(config)
    except (KeyError, ValueError):
        return block


def semantic_core(config: PipelineConfig) -> dict:
    """The output-defining subset of ``config`` (D19), as a plain dict.

    Deterministic given the config's semantics: two configs differing only in
    packaging knobs (orders, chunking, worker size, read machinery, carrier)
    map to the same core. ``None``-valued keys are pruned recursively so a
    YAML explicit-null hashes identically to an absent key (§8.3). The returned
    structure is JSON-serializable plain data (the YAML loader guarantees it).
    """
    from zagg.config import get_aoi_mask, get_emit_cell_ids, get_sharded

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
    # GRID_LEAF_SHAPING_KEYS, resolved (issue #415): `sharded`'s default is
    # grid-family-shaped exactly as `grids.from_config` resolves it — HEALPix
    # output defaults to sharded (issue #215), rect does not — so an explicit
    # `sharded: true` on a HEALPix config hashes identically to omitting it.
    grid["sharded"] = get_sharded(config, default=grid_type == "healpix")
    grid["emit_cell_ids"] = get_emit_cell_ids(config)
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
        # OUTPUT_LEAF_SHAPING_KEYS (issue #415): the knobs that decide what a
        # leaf CONTAINS or which artifacts sit beside it. Resolved, never as
        # spelled — an unwindowed store prunes `windowing` away entirely, so
        # only stores that actually declare a schedule pay for the key.
        "output": {
            "aoi_mask": get_aoi_mask(config),
            "windowing": _windowing_leaf_shape(config),
        },
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
    "GRID_LEAF_SHAPING_KEYS",
    "GRID_SPATIAL_KEYS",
    "OUTPUT_LEAF_SHAPING_KEYS",
    "canonical_semantic_json",
    "composability_classes",
    "field_composability",
    "semantic_core",
    "semantic_fingerprint",
    "semantic_hash",
]
