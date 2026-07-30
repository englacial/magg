"""Typed accessors over a pipeline config (issue #330).

Split out of the single-file ``zagg.config`` (issue #330 phase 4); the public
surface is unchanged and re-exported from :mod:`zagg.config`. Every ``get_*``
reader that turns a raw YAML block into the normalized value the rest of zagg
consumes — output signature, grid orders, store layout, windowing, AOI.
"""

from __future__ import annotations

import logging

from zagg.config.base import PipelineConfig
from zagg.config.expressions import filters_from_data_source

logger = logging.getLogger(__name__)


def get_raster_bands(config: PipelineConfig) -> dict:
    """Normalized ``data_source.bands``: field -> {asset, dtype, fill_value, attrs}.

    ``attrs`` carries CF ``scale_factor``/``add_offset`` when the band
    declares ``scale``/``offset`` — recorded on the array for readers, never
    applied to the stored DNs (issue #218: exact source values).
    """
    out = {}
    for name, meta in (config.data_source.get("bands") or {}).items():
        attrs = {}
        if "scale" in meta:
            attrs["scale_factor"] = meta["scale"]
        if "offset" in meta:
            attrs["add_offset"] = meta["offset"]
        out[name] = {
            "asset": meta["asset"],
            "dtype": meta["dtype"],
            "fill_value": meta.get("fill_value", 0),
            "attrs": attrs,
        }
    return out


def collection_options(config: PipelineConfig) -> dict[str, dict]:
    """Normalize ``data_source.collections`` to ``{name: options}``.

    Both config forms collapse to one shape: the list form (names only) yields
    empty option dicts; the mapping form yields each collection's options
    verbatim (``null`` becomes ``{}``). Consumed by the temporal reader /
    :func:`zagg.temporal.prepare_collection`.
    """
    colls = (config.data_source or {}).get("collections") or []
    if isinstance(colls, dict):
        return {name: dict(opts or {}) for name, opts in colls.items()}
    return {name: {} for name in colls}


def get_agg_fields(config: PipelineConfig) -> dict:
    """Return aggregation variable metadata keyed by variable name.

    Parameters
    ----------
    config : PipelineConfig

    Returns
    -------
    dict
        ``{name: {function/expression, source, params, dtype, fill_value, ...}}``.
        A field may also declare a non-scalar output (issue #29) via ``kind``
        (``scalar`` default, or ``vector``) and ``trailing_shape``; use
        :func:`get_output_signature` to read the normalized declaration.
    """
    return dict(config.aggregation.get("variables", {}))


def get_chunk_precompute(config: PipelineConfig) -> dict:
    """Return the ``aggregation.chunk_precompute`` entries keyed by name (issue #30).

    Each entry is a chunk-level scalar evaluated ONCE per chunk (shard) over the
    shard's pooled column data, before the per-cell loop; the resulting scalar is
    injected into the per-cell expression namespace. Returns an empty dict when no
    ``chunk_precompute`` block is present, so the existing per-cell path is a no-op.

    Parameters
    ----------
    config : PipelineConfig

    Returns
    -------
    dict
        ``{name: {function/expression, source, params, dtype}}``.
    """
    return dict(config.aggregation.get("chunk_precompute", {}))


def get_output_signature(meta: dict) -> dict:
    """Return the normalized non-scalar output signature for one agg field.

    This is the single read point for a field's output declaration (issues
    #29 and #48): its output ``kind``, the per-cell ``trailing_shape``,
    ``inner_shape``, and ``dtype``. Later phases (statistic eval, the
    per-shard container, and the grid ``signature()``) consume this rather
    than re-parsing the raw metadata.

    Parameters
    ----------
    meta : dict
        A single variable's aggregation metadata (a value of
        :func:`get_agg_fields`).

    Returns
    -------
    dict
        ``{"kind": str, "trailing_shape": tuple, "inner_shape": tuple, "dtype":
        str, "resolution": str}``.
        ``trailing_shape`` is ``()`` for scalar and ragged fields.
        ``inner_shape`` is ``()`` for scalar and vector fields; for ragged it
        holds the per-element shape (e.g. ``(2,)`` for a centroid pair).
        ``dtype`` is the declared dtype string, or ``None`` if unset.
        ``resolution`` is ``"cell"`` (default — one value per aggregation cell) or
        ``"chunk"`` (issue #30 item 2 — one value per chunk, stored in a companion
        array shaped at the chunk grid and indexed by ``grid.block_index``).
        ``location`` is the ragged location channel's column name (issue #87), or
        ``None`` when the field carries no location.
    """
    kind = meta.get("kind", "scalar")
    if kind == "vector":
        ts = meta["trailing_shape"]
        trailing_shape = (ts,) if isinstance(ts, int) else tuple(ts)
        inner_shape: tuple = ()
    elif kind == "ragged":
        trailing_shape = ()
        rs = meta["inner_shape"]
        inner_shape = (rs,) if isinstance(rs, int) else tuple(rs)
    else:
        trailing_shape = ()
        inner_shape = ()
    return {
        "kind": kind,
        "trailing_shape": trailing_shape,
        "inner_shape": inner_shape,
        "dtype": meta.get("dtype"),
        "resolution": meta.get("resolution", "cell"),
        # Ragged location channel (issue #87): the per-observation morton column
        # the reducer folds per centroid; ``None`` for unlocated fields.
        "location": meta.get("location"),
    }


def output_field_signature(config: PipelineConfig) -> list[dict]:
    """Return the Option-B output-field signature for a config (issue #29).

    A canonical, JSON-serializable list of ``{"name", "kind", "trailing_shape",
    "dtype"}`` for every aggregation variable, sorted by ``name``. Recorded in a
    grid's :meth:`signature` so a shard map can never be silently paired with a
    grid whose output schema (scalar vs vector, trailing shape, dtype) differs,
    and compared in ``nests_with`` so co-aggregated grids must share a field set.

    ``trailing_shape`` is rendered as a ``list`` (``()`` for scalar fields) so
    the structure round-trips through JSON unchanged.

    Parameters
    ----------
    config : PipelineConfig

    Returns
    -------
    list of dict
    """
    fields = []
    for name, meta in get_agg_fields(config).items():
        sig = get_output_signature(meta)
        entry = {
            "name": name,
            "kind": sig["kind"],
            "trailing_shape": list(sig["trailing_shape"]),
            "inner_shape": list(sig["inner_shape"]),
            "dtype": sig["dtype"],
        }
        # A location channel changes the store schema (a uint64 companion vlen
        # array — issue #87), so it belongs in the signature; keyed only when set
        # so existing shard-map signatures are byte-identical.
        if sig["location"] is not None:
            entry["location"] = sig["location"]
        fields.append(entry)
    return sorted(fields, key=lambda f: f["name"])


def get_coords(config: PipelineConfig) -> list[str]:
    """Return coordinate column names from the aggregation config.

    Parameters
    ----------
    config : PipelineConfig

    Returns
    -------
    list[str]
    """
    return list(config.aggregation.get("coordinates", {}).keys())


def get_data_vars(config: PipelineConfig) -> list[str]:
    """Return data variable column names from the aggregation config.

    Parameters
    ----------
    config : PipelineConfig

    Returns
    -------
    list[str]
    """
    return list(config.aggregation.get("variables", {}).keys())


def get_driver(config: PipelineConfig) -> str:
    """Return the data access driver from the config.

    Parameters
    ----------
    config : PipelineConfig

    Returns
    -------
    str
        ``"s3"`` or ``"https"``. Defaults to ``"s3"``.
    """
    return config.data_source.get("driver", "s3")


def get_handoff(config: PipelineConfig) -> str:
    """Return the per-cell aggregation carrier from the aggregation config (issue #132).

    The ``handoff`` knob lives on the ``aggregation`` block, default ``"arrow"``,
    and selects the per-cell read->concat->extract carrier: ``"arrow"`` (an
    ``arro3.core`` Table, faster + lighter on dense shards) or ``"pandas"`` (a
    DataFrame, which tolerates nullable columns natively). Both feed identical
    numpy arrays into the same reductions, so scalar outputs are byte-for-byte
    identical (issues #130/#131). A pipeline declares its carrier next to the rest
    of its aggregation settings rather than relying on a global default; the
    explicit ``handoff=`` kwarg still overrides this value.

    Parameters
    ----------
    config : PipelineConfig

    Returns
    -------
    str
        ``"arrow"`` (default) or ``"pandas"``.
    """
    return config.aggregation.get("handoff", "arrow")


def get_child_order(config: PipelineConfig) -> int:
    """Return child_order from the output grid config.

    Parameters
    ----------
    config : PipelineConfig

    Returns
    -------
    int

    Raises
    ------
    ValueError
        If child_order is not set in the config.
    """
    grid = config.output.get("grid", {})
    child_order = grid.get("child_order")
    if child_order is None:
        raise ValueError("output.grid.child_order is required")
    return int(child_order)


def get_parent_order(config: PipelineConfig) -> int:
    """Return parent_order (shard order) from the output grid config.

    Parameters
    ----------
    config : PipelineConfig

    Returns
    -------
    int

    Raises
    ------
    ValueError
        If parent_order is not set in the config.
    """
    grid = config.output.get("grid", {})
    parent_order = grid.get("parent_order")
    if parent_order is None:
        raise ValueError("output.grid.parent_order is required")
    return int(parent_order)


def get_layout(config: PipelineConfig) -> str:
    """Return the HEALPix storage layout from the output grid config.

    Parameters
    ----------
    config : PipelineConfig

    Returns
    -------
    str
        ``"fullsphere"`` (the only layout; the deprecated ``"dense"`` pack
        was removed — issue #88).
    """
    return config.output.get("grid", {}).get("layout", "fullsphere")


def get_sharded(config: PipelineConfig, default: bool = False) -> bool:
    """Return whether the output grid uses ShardingCodec storage (issue #108).

    The ``sharded`` knob lives on the grid/chunk block next to ``chunk_inner``
    (mirroring its accessor). When ``True`` the grid bundles a dispatch shard's K
    inner chunks into one zarr shard object instead of K independent regular chunk
    objects; a K==1 grid has nothing to bundle, so the grid silently no-ops it
    (issue #215). ``default`` is the value returned when the flag is omitted —
    ``False`` here, but ``from_config`` passes ``True`` for HEALPix output on
    both store layouts (issue #215 flat, issue #236 hive: a missing flag should
    not cost the ~K-fold object blow-up).
    """
    return bool(config.output.get("grid", {}).get("sharded", default))


def get_emit_cell_ids(config: PipelineConfig) -> bool:
    """Whether to keep writing the legacy ``cell_ids`` array (issue #304).

    Default ``False`` (the D16 flip): ``morton`` is the only stored cell
    coordinate; NESTED ids are derived at read (moczarr fabrication). ``True``
    is the transition escape hatch — the ``cell_ids`` array is written in
    ADDITION to ``morton`` (holding the NESTED encode) so legacy
    browser-direct consumers keep working until the gridlook morton decode
    lands; the dggs attrs are unaffected by the hatch.
    """
    return bool(config.output.get("grid", {}).get("emit_cell_ids", False))


def get_product_name(config: PipelineConfig) -> str | None:
    """The D19 product name (issue #299), or ``None`` for a bare store.

    When set, the run's effective store root becomes
    ``{output.store}/{product_name}/`` (``zagg.hive.effective_store_root``) —
    a NAMED product root in a multi-product store. Absent (the default),
    stores keep today's bare single-product layout byte-identically.
    """
    return config.output.get("product_name")


def get_store_path(config: PipelineConfig) -> str | None:
    """Return the store path from the output config, or None.

    Parameters
    ----------
    config : PipelineConfig

    Returns
    -------
    str or None
    """
    return config.output.get("store")


def get_store_layout(config: PipelineConfig) -> str:
    """Return the STORE layout — morton hive vs flat single store (issue #199).

    Distinct from ``output.grid.layout`` (the HEALPix *array* layout inside one
    store): ``output.store_layout`` selects how shard output is arranged under
    ``output.store``. ``"hive"`` writes one self-describing leaf zarr per shard
    at ``{store}/{sign+base}/{d1}/.../{full_id}.zarr`` with a root
    ``morton_hive.json`` manifest and a per-leaf commit stamp — see
    ``docs/design/sparse_coverage.md`` (D1-D6) and :mod:`zagg.hive`. ``"flat"``
    is the single shared zarr store.

    The default is grid-aware and pipeline-agnostic (issue #253): HEALPix
    grids default to ``"hive"`` (the ratified profile — for point aggregation
    AND the raster path, which writes hive leaves per issue #247); rectilinear
    grids default to ``"flat"`` (hive ids are morton digits — HEALPix-only).
    An explicit HEALPix ``"flat"`` remains valid for interop/debug but is
    deprecated — removal is gated on the sparse-DGGS read path (issue #251
    phase 3). A present-but-null key falls back to the default, like
    ``layout``.

    Returns
    -------
    str
        ``"flat"`` or ``"hive"``.
    """
    layout = config.output.get("store_layout")
    if layout is not None:
        return layout
    grid_type = (config.output.get("grid") or {}).get("type", "healpix")
    return "hive" if grid_type == "healpix" else "flat"


def get_coverage_moc(config: PipelineConfig) -> bool:
    """Whether the end-of-run root ``coverage.moc`` write is on (issue #200).

    Default ON for hive-layout stores (O9/espg: coverage MOCs are the default
    for healpix templates — and hive is healpix-only by validation);
    ``output.coverage_moc: false`` opts out. Flat-layout / non-healpix
    configs default off, and an explicit ``true`` there is rejected by
    ``validate_config``. A present-but-null key falls back to the default,
    like ``store_layout``.
    """
    flag = config.output.get("coverage_moc")
    if flag is None:
        return get_store_layout(config) == "hive"
    return bool(flag)


def get_sweep(config: PipelineConfig) -> bool:
    """Whether the end-of-run rollup sweep trigger is on (issue #300).

    Default ON for hive-layout stores (the D22 sweep folds hive-tree leaf
    artifacts into interior rollups; every artifact is a fail-open regenerable
    cache, D9); ``output.sweep: false`` opts out. Non-hive configs default
    off, and an explicit ``true`` there is rejected by ``validate_config``,
    mirroring ``coverage_moc``. A present-but-null key falls back to the
    default. The trigger transport is backend-shaped (D8): the local
    dispatchers run the sweep in-process; the Lambda dispatchers post a
    fire-and-forget ``mode="sweep"`` worker Event invoke.
    """
    flag = config.output.get("sweep")
    if flag is None:
        return get_store_layout(config) == "hive"
    return bool(flag)


def get_pyramid(config: PipelineConfig) -> dict | None:
    """The overview pyramid declaration knob, or ``None`` when disabled (#201).

    ``output.pyramid`` shapes the manifest's template-time pyramid block
    (``zagg.sweep_overview.build_pyramid_block``): absent/null returns ``{}``
    — the defaults apply (an every-2-orders overview schedule below the shard
    order, the espg-ratified display default; no all-time fold) — and
    ``false`` returns ``None``, declaring the overview family OFF for the
    store. An explicit mapping may carry ``spacing`` (int >= 1), ``orders``
    (explicit ancestor orders, winning over ``spacing``), ``all_time``
    (bool: also maintain the ``all.zarr`` cross-window fold on windowed
    stores), and ``summarize`` (the D24 opt-in derived-summary declarations,
    ``{source_field: {"as": name, ...}}`` — recorded in the pyramid block,
    never the semantic core).
    """
    raw = config.output.get("pyramid")
    if raw is False:
        return None
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"output.pyramid must be a mapping or false (got {raw!r})")
    return dict(raw)


def get_windowing(config: PipelineConfig) -> dict | None:
    """The normalized temporal windowing declaration, or ``None`` (issue #246).

    ``None`` — absent block, null block, or an explicit ``schedule: none`` —
    is today's unwindowed behavior (bare leaf names, ``morton-hive/1``).
    Otherwise a normalized dict with defaults resolved::

        {"schedule", "time_field", "epoch", "scale", "units", "windows"}

    ``epoch`` and explicit-window boundaries are canonicalized to ISO-8601
    UTC strings; ``windows`` is ``None`` except for ``schedule: explicit``.
    On the raster branch (``reader: raster``) ``time_field`` is the fixed STAC
    ``datetime`` and ``epoch``/``scale``/``units`` are hardcoded to the
    Unix-epoch UTC-seconds encoding any ISO instant normalizes to, rather than
    canonicalizing a declared ``epoch`` off the block (``_validate_windowing``
    rejects those conversion knobs there). The same dict feeds the manifest
    temporal block (:func:`zagg.hive.build_manifest`) and the dispatch fan-out,
    so the two can never disagree.
    """
    from zagg import windows as _windows

    block = config.output.get("windowing")
    if not block or block.get("schedule", "none") == "none":
        return None
    declared = None
    if block["schedule"] == "explicit":
        declared = [
            {
                "label": w["label"],
                "start": _windows.iso_utc(_windows.parse_utc(w["start"])),
                "end": _windows.iso_utc(_windows.parse_utc(w["end"])),
            }
            for w in block["windows"]
        ]
    if (config.data_source or {}).get("reader") == "raster":
        # Raster membership is the acquisition's STAC ``datetime`` (issue
        # #247, ratified): the manifest records the resolved field plus the
        # fixed encoding any ISO-8601 UTC instant normalizes to (UTC seconds
        # since the Unix epoch). _validate_windowing rejects the conversion
        # knobs on raster configs, so nothing here can disagree with it.
        return {
            "schedule": block["schedule"],
            "time_field": "datetime",
            "epoch": "1970-01-01T00:00:00+00:00",
            "scale": "utc",
            "units": "seconds",
            "windows": declared,
        }
    return {
        "schedule": block["schedule"],
        "time_field": block["time_field"],
        "epoch": _windows.iso_utc(_windows.parse_utc(block["epoch"])),
        "scale": block.get("scale") or "utc",
        "units": block.get("units") or "seconds",
        "windows": declared,
    }


def window_time_filters(config: PipelineConfig, start: float, end: float) -> list[dict]:
    """Structured filters implementing one window's ``[start, end)`` (issue #246).

    The observation-level window filter is exactly a pair of structured
    predicates on the declared ``time_field`` — ``ge start`` / ``lt end`` in
    DATASET units (converted once at dispatch) — so it rides the existing,
    pushdown-eligible filter machinery (issue #43) on every backend instead
    of a bespoke row filter. ``_validate_windowing`` restricts ``time_field``
    to a base-rate ``data_source.variables`` entry (the base level reads its
    columns from there too), so the predicate always filters at base rate
    (``level=None``, per observation); a coordinate or non-base/segment-rate
    ``time_field`` is rejected at validation and never reaches here. Appended
    to :func:`filters_from_data_source`'s normalized output by the hive write
    path, preserving any declared filters.

    A ``time_field`` declared in the ``{path, column}`` mapping form (issue
    #321) carries its column selector into both predicates — a structured
    filter takes the same ``column`` key, so a time column packed inside a 2-D
    dataset filters at base rate like any other.
    """
    windowing = get_windowing(config)
    if windowing is None:
        raise ValueError("window_time_filters requires a windowed config (output.windowing)")
    field = windowing["time_field"]
    entry = ((config.data_source or {}).get("variables") or {}).get(field)
    path, column = _variable_entry(entry)
    if not (isinstance(path, str) and path):
        raise ValueError(
            f"windowing time_field {field!r} has no base-rate dataset path in "
            f"data_source.variables — validate_config should have rejected "
            f"this configuration"
        )
    # ``column`` is emitted only for the mapping form, so the string form's
    # filter pair stays byte-identical to the pre-#321 shape.
    col = {"column": column} if column is not None else {}
    return [
        {"level": None, "dataset": path, **col, "op": "ge", "value": float(start)},
        {"level": None, "dataset": path, **col, "op": "lt", "value": float(end)},
    ]


def _variable_entry(entry) -> tuple[str | None, int | None]:
    """``(path, column)`` of one ``data_source.variables`` entry, mapping form or not.

    The read path's :func:`zagg.processing.read._variable_specs` normalizes the
    whole block; this is the single-entry form for config-side consumers that
    hold one name (currently the windowing time_field).
    """
    if isinstance(entry, str):
        return entry, None
    if isinstance(entry, dict):
        col = entry.get("column")
        return entry.get("path"), None if col is None else int(col)
    return None, None


def windowed_cell_config(config: PipelineConfig, window: dict) -> tuple[PipelineConfig, dict]:
    """One work unit's config with its window filter injected (issue #246).

    Returns ``(config, windowing)``: a per-unit config copy whose normalized
    filter list gains the :func:`window_time_filters` pair for ``window``'s
    dataset-unit ``[start, end)`` (declared filters preserved — the explicit
    list wins over ``quality_filter`` sugar, so normalize-then-append), plus
    the normalized windowing declaration. Raises if the config declares no
    ``output.windowing`` — a dispatched window without one is dispatcher/
    config drift, never guessed around.
    """
    from dataclasses import replace

    windowing = get_windowing(config)
    if windowing is None:
        raise ValueError(
            "a window was dispatched but the config declares no output.windowing "
            "block — dispatcher/config drift, refusing to guess the time_field"
        )
    ds = dict(config.data_source)
    ds["filters"] = filters_from_data_source(ds) + window_time_filters(
        config, window["start"], window["end"]
    )
    return replace(config, data_source=ds), windowing


def get_aoi_mask(config: PipelineConfig) -> bool:
    """Whether the optional strict-AOI cell mask is enabled (issue #101).

    ``output.aoi_mask: true`` packages a per-cell boolean ``aoi_mask`` array
    aligned to the output cell grid, ``True`` where the cell is inside the AOI.
    Defaults to ``False`` — when off, no array is emitted and outputs are
    byte-identical to a run without the feature.

    Parameters
    ----------
    config : PipelineConfig

    Returns
    -------
    bool
    """
    return bool(config.output.get("aoi_mask", False))


def get_consolidate_metadata(config: PipelineConfig) -> bool:
    """Whether the finalize step consolidates zarr metadata (issue #191).

    ``output.consolidate_metadata: true`` runs ``zarr.consolidate_metadata`` as a
    finalize step after every cell completes, producing a consolidated-metadata
    blob. Defaults to ``False``: no zagg reader opens with ``use_consolidated`` —
    readers navigate to specific paths in a few GETs — and consolidation is an
    optional zarr-v3 extension that costs ~70 s of serial metadata GETs per run.
    When off, the finalize invoke is skipped entirely and stores read fine (v3
    readers do lazy metadata reads natively).

    Parameters
    ----------
    config : PipelineConfig

    Returns
    -------
    bool
    """
    return bool(config.output.get("consolidate_metadata", False))


def get_output_endpoint_url(config: PipelineConfig) -> str | None:
    """Return the output S3 endpoint URL from the output config, or None.

    Non-secret S3-compatible endpoint (e.g. R2, MinIO). Credentials are never
    stored in config; they are supplied at runtime.

    Parameters
    ----------
    config : PipelineConfig

    Returns
    -------
    str or None
    """
    return config.output.get("endpoint_url")


def get_output_region(config: PipelineConfig) -> str | None:
    """Return the output S3 region from the output config, or None.

    Parameters
    ----------
    config : PipelineConfig

    Returns
    -------
    str or None
    """
    return config.output.get("region")
