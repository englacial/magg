"""Output-side block validators: store layout, windowing, output kinds (issue #330).

Split out of the single-file ``zagg.config`` (issue #330 phase 4); the public
surface is unchanged and re-exported from :mod:`zagg.config`. The validators
:func:`zagg.config.validate_config` delegates to for the ``output`` half of a
config, plus the pyramid declaration and the NaN-fill round-trip warning.
"""

from __future__ import annotations

import logging

import numpy as np

from zagg.config.accessors import get_agg_fields, get_pyramid, get_store_layout
from zagg.config.base import (
    PipelineConfig,
    _is_numeric,
    _segment_variable_names,
    _validate_expression_columns,
)
from zagg.config.expressions import resolve_function

logger = logging.getLogger(__name__)


def _validate_store_layout_keys(config: PipelineConfig) -> None:
    """Shared ``output.store_layout`` / ``coverage_moc`` cross-checks.

    Optional store layout (issue #199 phase 2): "hive" (one leaf zarr per
    shard under a morton digit tree — ``docs/design/sparse_coverage.md``
    D1-D6) or "flat" (the single shared store). The default is grid-aware
    (issue #253, ``get_store_layout``): HEALPix -> hive, rectilinear -> flat —
    so the hive gates below resolve through the default, not the raw key.
    Hive ids are morton decimal strings, so the layout is HEALPix-only;
    metadata consolidation assumes the single shared store, so it is rejected
    with hive (explicit or defaulted) rather than silently mis-writing. Called
    from both the point-pipeline and raster validation branches (issue #247:
    raster + hive is legal, so the checks live in one place; the issue #239
    stopgap rejections are gone).

    The end-of-run root coverage MOC (issue #200 phase 3) is boolean when
    present, default ON for the hive layout (O9); it writes
    ``{store}/coverage.moc``, a hive-root object, so an EXPLICIT true on a
    non-healpix grid or a flat-layout store is a config mistake, not a no-op —
    rejected pointedly. Absent simply means off there (``get_coverage_moc``
    resolves the default).
    """
    grid = config.output.get("grid")
    store_layout = config.output.get("store_layout")
    if store_layout is not None and store_layout not in ("flat", "hive"):
        raise ValueError(f"output.store_layout must be 'flat' or 'hive' (got {store_layout!r})")
    # D19 product name (issue #299): validated at load so a bad name fails
    # before any store I/O. The grammar lives in hive.py (one source).
    name = config.output.get("product_name")
    if name is not None:
        from zagg.hive import validate_product_name

        validate_product_name(name)
    if get_store_layout(config) == "hive":
        if (grid or {}).get("type", "healpix") != "healpix":
            raise ValueError(
                "output.store_layout: hive requires a healpix grid (hive node names "
                f"are morton decimal digits; grid type is {(grid or {}).get('type')!r})"
            )
        if config.output.get("consolidate_metadata"):
            raise ValueError(
                "output.store_layout: hive has no store-root zarr hierarchy to "
                "consolidate (D5/D12) — drop output.consolidate_metadata"
            )
    coverage_moc = config.output.get("coverage_moc")
    if coverage_moc is not None and not isinstance(coverage_moc, bool):
        raise ValueError(f"output.coverage_moc must be a boolean (got {coverage_moc!r})")
    if coverage_moc:
        if (grid or {}).get("type", "healpix") != "healpix":
            raise ValueError(
                "output.coverage_moc requires a healpix grid (the root coverage.moc "
                "is a morton MOC; drop the flag for rectilinear output)"
            )
        if get_store_layout(config) != "hive":
            raise ValueError(
                "output.coverage_moc requires output.store_layout: hive (the root "
                "coverage.moc lives at the hive store root; flat stores have no "
                "hive root to bootstrap from)"
            )
    # End-of-run rollup sweep trigger (issue #300), same posture as
    # coverage_moc: boolean when present, default ON for hive (get_sweep
    # resolves it), explicit true on a non-hive store is a config mistake.
    sweep = config.output.get("sweep")
    if sweep is not None and not isinstance(sweep, bool):
        raise ValueError(f"output.sweep must be a boolean (got {sweep!r})")
    if sweep and get_store_layout(config) != "hive":
        raise ValueError(
            "output.sweep requires output.store_layout: hive (the rollup sweep "
            "folds hive-tree leaf artifacts; flat stores have no digit tree)"
        )
    # Overview pyramid declaration (issue #201): explicit blocks are grammar-
    # checked here; the D24 none-field warning fires at manifest build time
    # (template time for the store), not per config validation. The NaN-fill
    # reduction warning below IS per config validation (espg ruling on the
    # PR #344 nan-policy finding): it concerns the store encoding itself.
    _validate_pyramid(config)
    _warn_nan_ambiguous_reductions(config)


def _validate_windowing(config: PipelineConfig) -> None:
    """Validate the ``output.windowing`` block (issue #246, morton-hive/2).

    ``{schedule, time_field, epoch, scale?, units?, windows?}`` — the nested
    form ratified on the #246 thread. Absent/null is schedule ``none``
    (today's behavior). A present block requires the hive store layout on a
    healpix grid (windowed leaf names are a morton-hive convention, like
    ``coverage_moc``). ``quarterly`` is grammar-reserved on the mortie spec
    page but NOT implemented — rejected with a pointed message. The explicit
    windows list must be well-formed: frozen label grammar, half-open
    ``start < end``, unique labels, non-overlapping ranges. ``time_field``
    must name a declared BASE-RATE ``data_source`` variable (a flat
    ``variables`` entry or the base level's ``variables``) — a coordinate or
    a non-base (segment-rate) level column is rejected, since the stamp
    ``time_range`` is pooled from read variable columns and window membership
    is decided per observation. On the RASTER path (issue #247) membership is
    the acquisition's STAC ``datetime`` instead: ``time_field`` is optional
    (fixed to ``datetime``) and the ``epoch``/``scale``/``units`` conversion
    knobs are rejected.
    """
    from zagg import windows as _windows

    block = config.output.get("windowing")
    if block is None:
        return
    if not isinstance(block, dict):
        raise ValueError(
            f"output.windowing must be a mapping "
            f"{{schedule, time_field, epoch, ...}} (got {block!r})"
        )
    try:
        schedule = _windows.check_schedule(block.get("schedule", "none"))
    except ValueError as e:
        raise ValueError(f"output.windowing.schedule: {e}") from e
    if schedule == "none":
        # An inert ``schedule: none`` block is equivalent to an absent block
        # (``get_windowing`` returns ``None``, no windowed output), so it must
        # NOT require the hive/healpix layout — this early return precedes the
        # layout guards. A stray ``windows`` key is still rejected, mirroring
        # the generative-schedule check below (validation symmetry).
        if block.get("windows") is not None:
            raise ValueError(
                "output.windowing.windows only applies to schedule: explicit "
                "(schedule: none produces no windowed output)"
            )
        return
    grid = config.output.get("grid") or {}
    if (grid.get("type", "healpix")) != "healpix":
        raise ValueError(
            "output.windowing requires a healpix grid (windowed leaves are a "
            f"morton-hive convention; grid type is {grid.get('type')!r})"
        )
    if get_store_layout(config) != "hive":
        raise ValueError(
            "output.windowing requires output.store_layout: hive (window leaves "
            "are hive leaf zarrs; the flat shared store has no leaves to window)"
        )
    time_field = block.get("time_field")
    if (config.data_source or {}).get("reader") == "raster":
        # Raster window membership is the acquisition's STAC ``datetime``,
        # decided at dispatch (issue #247, ratified): there is no
        # per-observation timestamp column, so ``time_field`` is optional and
        # fixed — the manifest records the resolved field, and a property-name
        # knob (e.g. ``start_datetime`` for interval-typed items) can be added
        # later without breaking any existing manifest.
        if time_field is not None and time_field != "datetime":
            raise ValueError(
                f"output.windowing.time_field {time_field!r} is not configurable "
                "on the raster path: window membership is the acquisition's STAC "
                "datetime (drop the key, or set it to 'datetime')"
            )
        # The epoch/scale/units knobs describe a dataset-unit time_field
        # conversion; STAC datetimes are already ISO-8601 UTC instants, so
        # there is nothing to configure — reject rather than record a
        # misleading manifest temporal block (get_windowing resolves the
        # fixed encoding).
        knobs = [k for k in ("epoch", "scale", "units") if block.get(k) is not None]
        if knobs:
            raise ValueError(
                f"output.windowing.{knobs[0]} does not apply to raster "
                "pipelines: STAC datetimes are ISO-8601 UTC instants (drop the "
                "key)"
            )
        _validate_windowing_windows(block, schedule)
        return
    if not isinstance(time_field, str) or not time_field:
        raise ValueError(
            "output.windowing.time_field is required: the per-observation "
            "timestamp column that decides window membership"
        )
    # ``time_field`` must be a BASE-RATE variable column, i.e. a
    # ``data_source.variables`` entry (the base level reads its columns from
    # there too — ``_validate_levels`` forbids a base-level ``variables``
    # mapping). The stamp ``time_range`` (a windowed leaf's headline truth) is
    # pooled from the read VARIABLE columns (``col_arrays`` in the worker),
    # never from ``coordinates`` — so a coordinate ``time_field`` would filter
    # yet silently drop the stamp, and a lat/lon coordinate is not a timestamp
    # anyway. A non-base (segment-rate) level column is rejected too: window
    # membership would be decided per whole segment, not per observation,
    # which is not supported this round (see ``window_time_filters``).
    ds = config.data_source or {}
    declared = set(ds.get("variables") or {})
    if time_field not in declared:
        if time_field in set(ds.get("coordinates") or {}):
            raise ValueError(
                f"output.windowing.time_field {time_field!r} is a data_source "
                f"coordinate; the stamp time_range is pooled from read variable "
                f"columns and a coordinate is not read as a timestamp. Declare it "
                f"under data_source.variables (or the base level's variables)."
            )
        if time_field in _segment_variable_names(ds):
            raise ValueError(
                f"output.windowing.time_field {time_field!r} is declared on a "
                f"non-base (segment-rate) level; segment-rate window membership is "
                f"not supported yet (whole segments would be kept or dropped). "
                f"Declare it on the base level or data_source.variables."
            )
        raise ValueError(
            f"output.windowing.time_field {time_field!r} is not a declared "
            f"base-rate data_source column (one of {sorted(declared)}); the "
            f"worker filters per observation on variable columns it reads"
        )
    epoch = block.get("epoch")
    if epoch is None:
        raise ValueError(
            "output.windowing.epoch is required: the dataset's zero time as an "
            "ISO-8601 UTC instant (e.g. '2018-01-01T00:00:00Z' for ICESat-2 "
            "delta_time)"
        )
    try:
        _windows.parse_utc(epoch)
    except ValueError as e:
        raise ValueError(f"output.windowing.epoch: {e}") from e
    scale = block.get("scale") or "utc"
    if scale not in _windows.EPOCH_SCALES:
        raise ValueError(
            f"output.windowing.scale must be one of {_windows.EPOCH_SCALES} (got {scale!r})"
        )
    units = block.get("units") or "seconds"
    if units not in _windows.UNIT_SECONDS:
        raise ValueError(
            f"output.windowing.units must be one of {tuple(_windows.UNIT_SECONDS)} (got {units!r})"
        )
    _validate_windowing_windows(block, schedule)


def _validate_windowing_windows(block: dict, schedule: str) -> None:
    """Validate the ``windows`` list half of ``output.windowing`` (issue #246).

    Shared by the point-pipeline and raster branches of
    :func:`_validate_windowing`: frozen label grammar, half-open
    ``start < end``, unique labels, non-overlapping ranges — required for
    ``schedule: explicit``, rejected otherwise.
    """
    from zagg import windows as _windows

    declared_windows = block.get("windows")
    if schedule != "explicit":
        if declared_windows is not None:
            raise ValueError(
                f"output.windowing.windows only applies to schedule: explicit "
                f"(the {schedule} schedule derives its windows from labels)"
            )
        return
    if not isinstance(declared_windows, list) or not declared_windows:
        raise ValueError(
            "output.windowing.windows is required for schedule: explicit — a "
            "non-empty list of {label, start, end} entries"
        )
    seen: dict = {}
    for entry in declared_windows:
        if not isinstance(entry, dict) or not {"label", "start", "end"} <= set(entry):
            raise ValueError(
                f"each explicit window must be a {{label, start, end}} mapping (got {entry!r})"
            )
        try:
            label = _windows.validate_label(entry["label"], "explicit")
            start, end = _windows.parse_utc(entry["start"]), _windows.parse_utc(entry["end"])
        except ValueError as e:
            raise ValueError(f"output.windowing.windows: {e}") from e
        if label == _windows.SCHEDULE_NONE_TOKEN:
            # The opaque grammar admits it, but the token names the unwindowed
            # / all-time artifact everywhere (leaf_name_v3, the D23 overview
            # basename and its envelope key), so an explicit window by this
            # name would ALIAS the all-time fold — the per-window overview gets
            # overwritten and never regenerates (review finding, issue #201).
            raise ValueError(
                f"explicit window label {label!r} is the reserved schedule:none token "
                f"(SCHEDULE_NONE_TOKEN, D23) — it would alias the all-time artifact; "
                f"declare a different explicit label"
            )
        if not start < end:
            raise ValueError(
                f"explicit window {label!r} is not half-open: start "
                f"{entry['start']!r} must precede end {entry['end']!r}"
            )
        if label in seen:
            raise ValueError(f"explicit window label {label!r} is declared twice")
        seen[label] = (start, end)
    ordered = sorted(seen.items(), key=lambda kv: kv[1][0])
    for (a, (_sa, ea)), (b, (sb, _eb)) in zip(ordered, ordered[1:]):
        if ea > sb:
            raise ValueError(
                f"explicit windows {a!r} and {b!r} overlap; windows must be "
                f"disjoint half-open ranges"
            )


def _validate_chunk_precompute(aggregation: dict, ds_vars: set[str]) -> None:
    """Validate the ``aggregation.chunk_precompute`` block (issue #30, item 1).

    Each named entry is a chunk-level scalar computed ONCE per chunk (shard) over
    the shard's pooled column data, before the per-cell loop; its name then enters
    the per-cell expression namespace. Validation mirrors ``aggregation.variables``:
    each entry must declare exactly one of ``function``/``expression``, and any
    ``source`` / expression / param column references must exist in
    ``data_source.variables``. ``dtype`` (optional) must be a valid numpy dtype.

    A precompute name must not collide with a ``data_source.variables`` column or
    a reserved namespace name (``leaf_id``): the per-cell namespace is built as
    ``{**cell_data, **chunk_scalars}`` in :func:`zagg.processing.process_shard`, so
    a colliding name would shadow the real column *array* with a 0-d *scalar* and
    corrupt every cell. Such names are rejected here.

    Inter-precompute references are NOT supported: an entry's expression is
    evaluated only over the pooled columns (:func:`zagg.processing._eval_chunk_precompute`
    iterates the entries independently, with no defined order), so one entry cannot
    reference another (e.g. ``chunk_gain`` cannot read ``chunk_offset``). A name
    that references another precompute entry is rejected as an unknown column.

    The block is optional; a config without it is unchanged.

    Parameters
    ----------
    aggregation : dict
        The config's ``aggregation`` mapping.
    ds_vars : set[str]
        Available ``data_source.variables`` column names.

    Raises
    ------
    ValueError
        On any invalid ``chunk_precompute`` declaration.
    """
    precompute = aggregation.get("chunk_precompute")
    if precompute is None:
        return
    if not isinstance(precompute, dict):
        raise ValueError("aggregation.chunk_precompute must be a mapping of name -> entry")
    reserved = ds_vars | {"leaf_id"}
    for name, meta in precompute.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("chunk_precompute entry names must be non-empty strings")
        if name in reserved:
            raise ValueError(
                f"chunk_precompute '{name}': name collides with a "
                f"data_source.variables column or the reserved 'leaf_id'; the "
                f"per-cell namespace merge would shadow the real column with a "
                f"chunk scalar. Rename the precompute entry."
            )
        if not isinstance(meta, dict):
            raise ValueError(f"chunk_precompute '{name}': entry must be a mapping")

        has_func = "function" in meta
        has_expr = "expression" in meta
        if has_func and has_expr:
            raise ValueError(
                f"chunk_precompute '{name}': 'function' and 'expression' are mutually exclusive"
            )
        if not has_func and not has_expr:
            raise ValueError(f"chunk_precompute '{name}': must specify 'function' or 'expression'")

        source = meta.get("source")
        if source is not None and source not in ds_vars:
            raise ValueError(
                f"chunk_precompute '{name}': source '{source}' not in data_source.variables"
            )

        if has_func:
            resolve_function(meta["function"])  # raises ValueError on failure

        for pval in meta.get("params", {}).values():
            if not isinstance(pval, str):
                continue  # numeric literal
            if pval in ds_vars or _is_numeric(pval):
                continue
            if any(v in pval for v in ds_vars):
                continue
            raise ValueError(
                f"chunk_precompute '{name}': param value '{pval}' references "
                f"unknown column (available: {ds_vars})"
            )

        if has_expr:
            _validate_expression_columns(name, meta["expression"], ds_vars)

        if "dtype" in meta:
            try:
                np.dtype(meta["dtype"])
            except (TypeError, ValueError) as e:
                raise ValueError(
                    f"chunk_precompute '{name}': dtype {meta['dtype']!r} is not a valid "
                    f"numpy dtype ({e})"
                ) from e


# Recognized per-field output kinds. ``ragged`` is the Tier-2 carrier
# for variable-length per-cell outputs; see issue #48.
OUTPUT_KINDS = ("scalar", "vector", "ragged")

# Recognized per-field output resolutions (issue #30 item 2). ``cell`` (default)
# stores one value per aggregation cell; ``chunk`` stores one value per chunk in a
# companion array shaped at the chunk grid, indexed by ``grid.block_index``.
OUTPUT_RESOLUTIONS = ("cell", "chunk")


def _validate_output_kind(name: str, meta: dict) -> None:
    """Validate a variable's non-scalar output declaration.

    A field may declare ``kind`` (``scalar`` default, ``vector``, or ``ragged``)
    and a shape key (``trailing_shape`` for ``vector``, ``inner_shape`` for
    ``ragged``). ``scalar`` fields need neither and stay the default path.
    ``vector`` and ``ragged`` fields may be driven by either ``function`` or
    ``expression``; ``len``/``count`` are rejected for both (they short-circuit
    to a scalar count). See issue #29 (vector) and issue #48 (ragged).

    A field may also declare ``resolution`` (``cell`` default, or ``chunk``).
    A ``resolution: chunk`` field (issue #30 item 2) is written ONCE per chunk
    into a companion array shaped at the chunk grid (``main.shape //
    chunk_shape``), indexed by ``grid.block_index(shard_key)`` — the compact
    storage for a chunk-uniform value (e.g. a ``chunk_precompute`` anchor).
    ``scalar``, ``vector``, and ``ragged`` kinds may all be ``resolution: chunk``
    (issue #82): a ``scalar`` companion is a plain chunk-grid array, a ``vector``
    companion appends the field's ``trailing_shape`` to the chunk grid (chunked
    whole), and a ``ragged`` companion holds one
    variable-length payload per chunk, written by ``write_ragged_to_zarr`` (phase
    4c). The shape keys are validated below exactly as for cell resolution — the
    chunk axis just replaces the cell axis.

    Parameters
    ----------
    name : str
        Variable name (for error messages).
    meta : dict
        The variable's aggregation metadata.

    Raises
    ------
    ValueError
        On any invalid output-kind declaration.
    """
    kind = meta.get("kind", "scalar")
    if kind not in OUTPUT_KINDS:
        allowed = ", ".join(OUTPUT_KINDS)
        raise ValueError(
            f"Variable '{name}': output kind '{kind}' is not supported (allowed: {allowed})"
        )

    # ``location`` (issue #87) is the ragged location channel; other kinds have
    # no companion ragged array to carry it.
    if "location" in meta and kind != "ragged":
        raise ValueError(
            f"Variable '{name}': 'location' is only valid for kind 'ragged', not '{kind}'"
        )

    # resolution (cell default, or chunk). A chunk-resolution field stores one
    # value per chunk in a companion array (issue #30 item 2). ``scalar`` and
    # ``vector`` chunk companions are wired (issue #82): a scalar companion is a
    # plain chunk-grid array, a vector companion appends the field's
    # ``trailing_shape`` to the chunk grid (chunked whole). The kind-specific shape
    # keys are validated by the per-kind branches below regardless of resolution.
    resolution = meta.get("resolution", "cell")
    if resolution not in OUTPUT_RESOLUTIONS:
        allowed = ", ".join(OUTPUT_RESOLUTIONS)
        raise ValueError(
            f"Variable '{name}': resolution '{resolution}' is not supported (allowed: {allowed})"
        )
    # ``ragged`` at chunk resolution (issue #82) stores one variable-length
    # payload per chunk instead of per cell. It rides the same vlen writer as
    # cell-resolution ragged (``write_ragged_to_zarr``), which collapses
    # the populated cells to the single chunk payload under the same chunk-uniform
    # contract as scalar/vector chunk companions (raise if populated cells
    # disagree). No special shape key is needed beyond ``inner_shape``.

    # dtype, when declared, must name a real numpy dtype (applies to all kinds).
    if "dtype" in meta:
        try:
            np.dtype(meta["dtype"])
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"Variable '{name}': dtype {meta['dtype']!r} is not a valid numpy dtype ({e})"
            ) from e

    has_trailing = "trailing_shape" in meta

    if kind == "scalar":
        if has_trailing:
            raise ValueError(
                f"Variable '{name}': 'trailing_shape' is only valid for kind 'vector', not 'scalar'"
            )
        return

    if kind == "vector":
        # trailing_shape is required and must be positive ints.
        if not has_trailing:
            raise ValueError(f"Variable '{name}': kind 'vector' requires 'trailing_shape'")
        _validate_trailing_shape(name, meta["trailing_shape"])

        # ``len``/``count`` short-circuit to a scalar obs count in
        # ``calculate_cell_statistics``; pairing them with kind 'vector' would
        # silently emit a scalar, so reject the nonsensical combination.
        if meta.get("function") in ("len", "count"):
            raise ValueError(
                f"Variable '{name}': function {meta['function']!r} produces a scalar "
                f"count and cannot be combined with kind 'vector'"
            )
        return

    # kind == "ragged": inner_shape is required; trailing_shape is rejected.
    if has_trailing:
        raise ValueError(
            f"Variable '{name}': 'trailing_shape' is only valid for 'vector', not 'ragged'"
        )
    if "inner_shape" not in meta:
        raise ValueError(f"Variable '{name}': kind 'ragged' requires 'inner_shape'")
    _validate_trailing_shape(name, meta["inner_shape"], key_name="inner_shape")

    # Same restriction as vector: ``len``/``count`` produce a scalar count.
    if meta.get("function") in ("len", "count"):
        raise ValueError(
            f"Variable '{name}': function {meta['function']!r} produces a scalar "
            f"count and cannot be combined with kind 'ragged'"
        )

    # Optional location channel (issue #87): the reducer also receives the named
    # per-observation morton column (``locations=`` kwarg) and returns a
    # ``(payload, locations)`` pair, stored as a uint64 companion vlen array. Only
    # a ``function`` reducer can accept the kwarg, and the chunk-uniform collapse
    # of ``resolution: chunk`` has no location fold — reject both combinations.
    location = meta.get("location")
    if location is not None:
        if not isinstance(location, str):
            raise ValueError(
                f"Variable '{name}': 'location' must be a column name string (got {location!r})"
            )
        if "function" not in meta:
            raise ValueError(
                f"Variable '{name}': 'location' requires a 'function' reducer "
                f"(an expression cannot receive the locations column)"
            )
        if resolution == "chunk":
            raise ValueError(
                f"Variable '{name}': 'location' is not supported with "
                f"'resolution: chunk' (locations are folded per cell)"
            )


def _validate_trailing_shape(name: str, trailing_shape, key_name: str = "trailing_shape") -> None:
    """Check a shape field (trailing_shape or inner_shape) is a tuple of positive ints."""
    if isinstance(trailing_shape, int):
        dims: tuple = (trailing_shape,)
    elif isinstance(trailing_shape, (list, tuple)):
        dims = tuple(trailing_shape)
    else:
        raise ValueError(
            f"Variable '{name}': '{key_name}' must be an int or a "
            f"sequence of ints (got {trailing_shape!r})"
        )
    if not dims:
        raise ValueError(f"Variable '{name}': '{key_name}' must have at least one dimension")
    for dim in dims:
        if not isinstance(dim, int) or isinstance(dim, bool) or dim < 1:
            raise ValueError(
                f"Variable '{name}': '{key_name}' entries must be positive integers (got {dim!r})"
            )


def _validate_pyramid(config: PipelineConfig) -> None:
    """Validate the ``output.pyramid`` knob's grammar (issue #201).

    Only an EXPLICIT block is checked (absent/false are always legal); an
    explicit block requires the hive store layout, like ``sweep`` — the
    pyramid materializes through the hive manifest.
    """
    raw = config.output.get("pyramid")
    if raw is None or raw is False:
        return
    knob = get_pyramid(config) or {}  # raises on a non-mapping
    if get_store_layout(config) != "hive":
        raise ValueError(
            "output.pyramid requires output.store_layout: hive (overviews live at "
            "hive-tree ancestor nodes; flat stores have no digit tree)"
        )
    unknown = set(knob) - {"spacing", "orders", "all_time", "summarize"}
    if unknown:
        raise ValueError(f"output.pyramid has unknown keys {sorted(unknown)}")
    spacing = knob.get("spacing")
    if spacing is not None and (not isinstance(spacing, int) or spacing < 1):
        raise ValueError(f"output.pyramid.spacing must be an int >= 1 (got {spacing!r})")
    parent_order = int((config.output.get("grid") or {}).get("parent_order", 0))
    orders = knob.get("orders")
    if orders is not None:
        ok = isinstance(orders, list) and all(isinstance(k, int) for k in orders)
        if not ok or any(not (0 <= k < parent_order) for k in orders):
            raise ValueError(
                f"output.pyramid.orders must be a list of ancestor orders in "
                f"[0, parent_order) = [0, {parent_order}) (got {orders!r})"
            )
    all_time = knob.get("all_time")
    if all_time is not None and not isinstance(all_time, bool):
        raise ValueError(f"output.pyramid.all_time must be a boolean (got {all_time!r})")
    summarize = knob.get("summarize")
    if summarize is None:
        return
    if not isinstance(summarize, dict):
        raise ValueError(f"output.pyramid.summarize must be a mapping (got {summarize!r})")
    fields = get_agg_fields(config)
    for source, decl in summarize.items():
        if source not in fields:
            raise ValueError(
                f"output.pyramid.summarize names unknown field {source!r} "
                f"(declared: {sorted(fields)})"
            )
        target = (decl or {}).get("as")
        if not isinstance(target, str) or not target:
            raise ValueError(f"output.pyramid.summarize.{source} requires 'as': <new field name>")
        if target in fields:
            raise ValueError(
                f"output.pyramid.summarize.{source}.as = {target!r} collides with a "
                f"declared field: derived summaries must use a DIFFERENT name so overview "
                f"schema never silently differs from source (D24)"
            )


#: Exact-law reductions whose plain (NaN-propagating) forms cannot round-trip
#: through a NaN fill: at read, a NaN datum and the fill are the same bytes.
_NAN_AMBIGUOUS_REDUCTIONS = ("min", "max", "sum")


def _warn_nan_ambiguous_reductions(config: PipelineConfig) -> None:
    """Warn when a plain ``min``/``max``/``sum`` pairs with a NaN fill (#201).

    Warning, not error — the pairing is legal (the shipped atl06 config uses
    it by design), but the store encoding has a consequence the author should
    have chosen knowingly: a NaN datum and the fill sentinel are the same
    bytes, so downstream the reduction behaves as its nan-skipping form (the
    pyramid block declares this as ``nan_policy: "skip"``) and a
    NaN-propagating result is unrecoverable. Declaring ``nanmin``/``nanmax``/
    ``nansum`` states the same semantics explicitly and silences this.
    """
    from zagg.semantics import _fold_function_name, field_composability

    for name, meta in get_agg_fields(config).items():
        fn = _fold_function_name(meta.get("function"))
        if fn not in _NAN_AMBIGUOUS_REDUCTIONS or not _is_nan_fill(meta):
            continue
        try:
            if field_composability(meta) != "exact":
                continue
        except Exception:
            continue  # malformed meta: the kind/shape validators own the error
        logger.warning(
            f"aggregation.variables.{name}: {meta.get('function')!r} with a NaN fill_value — "
            f"NaN data is indistinguishable from fill at read; the reduction behaves as "
            f'nan{fn} in overviews (declared in the pyramid block as nan_policy: "skip") '
            f"and any NaN-carrying result is unrecoverable — use nan{fn} explicitly to "
            f"silence this (issue #201)"
        )


def _is_nan_fill(meta: dict) -> bool:
    """Whether a field's fill sentinel is NaN (explicitly, or by float default)."""
    fill = meta.get("fill_value")
    if fill is None:
        try:
            return np.issubdtype(np.dtype(meta.get("dtype", "float32")), np.floating)
        except TypeError:
            return False
    if isinstance(fill, str):
        return fill.strip().lower() == "nan"
    return isinstance(fill, float) and np.isnan(fill)
