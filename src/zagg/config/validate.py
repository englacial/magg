"""``validate_config`` and the cross-block schema checks (issue #330).

Split out of the single-file ``zagg.config`` (issue #330 phase 4); the public
surface is unchanged and re-exported from :mod:`zagg.config`. The single
submission-time entry point plus the validators that span blocks — variables,
filters, levels, index, composition attrs, worker size. Per-block validators
live in :mod:`zagg.config.blocks`.
"""

from __future__ import annotations

import inspect
import logging

from zagg.config.base import (
    _SET_OPS,
    FILTER_OPS,
    WORKER_MEMORIES,
    PipelineConfig,
    _is_numeric,
    _segment_variable_names,
    _validate_expression_columns,
    get_pipeline_type,
)
from zagg.config.expressions import filters_from_data_source, resolve_function
from zagg.config.validate_output import (
    _validate_chunk_precompute,
    _validate_output_kind,
    _validate_store_layout_keys,
    _validate_windowing,
)
from zagg.config.validate_source import _validate_raster_config, _validate_temporal_config

logger = logging.getLogger(__name__)


def validate_config(config: PipelineConfig) -> None:
    """Cross-validate a PipelineConfig.

    Parameters
    ----------
    config : PipelineConfig

    Raises
    ------
    ValueError
        On any validation failure.
    """
    # Pipeline kind drives which validation branch runs; spatial keeps the
    # full grid + aggregation cross-checks below. Temporal/event pipelines
    # validate their own (much smaller) spec shape and return early -- they
    # carry no output grid and run through the event-streaming engine
    # (``zagg.temporal.process_event``) rather than the point-cloud path.
    # Both pipeline kinds honor a provider-selected source-credential fetch
    # (issue #213 Phase 6), so the key's shape is checked before the branch.
    provider = (config.data_source or {}).get("credentials_provider")
    if provider is not None and not isinstance(provider, str):
        raise ValueError(
            "data_source.credentials_provider must be a credential-provider "
            f"registry name string, e.g. 'nsidc' or 'gesdisc' (got {provider!r})"
        )

    # The optional top-level worker block (issue #235) selects a
    # pre-provisioned Lambda size variant on any pipeline kind, so it is
    # validated before the kind branch, like credentials_provider above.
    _validate_worker(config)

    ptype = get_pipeline_type(config)
    if ptype != "spatial":
        _validate_temporal_config(config)
        return

    # Raster pipelines (issue #218) share the spatial grid checks but replace
    # the HDF5 read-side schema (groups/coordinates/variables) and the
    # aggregation section with a declarative band map: pull-NN yields exactly
    # one value per cell per timestep, so there is nothing to reduce.
    if (config.data_source or {}).get("reader") == "raster":
        _validate_raster_config(config)
        return

    # Required sections
    for section in ("data_source", "aggregation", "output"):
        val = getattr(config, section)
        if not val:
            raise ValueError(f"Missing required section: {section}")

    # Validate output.grid structure
    grid = config.output.get("grid")
    if grid is not None:
        if not isinstance(grid, dict):
            raise ValueError("output.grid must be a mapping (e.g. type: healpix, child_order: 12)")
        if "type" not in grid:
            raise ValueError("output.grid.type is required")
        if grid["type"] == "healpix" and "child_order" not in grid:
            raise ValueError("output.grid.child_order is required for healpix grid")
        if grid["type"] == "healpix" and "parent_order" not in grid:
            raise ValueError("output.grid.parent_order is required for healpix grid")
        if grid["type"] == "rectilinear":
            for field in ("crs", "resolution", "bounds"):
                if field not in grid:
                    raise ValueError(f"output.grid.{field} is required for rectilinear grid")
            if len(grid["bounds"]) != 4:
                raise ValueError("output.grid.bounds must be [xmin, ymin, xmax, ymax]")
        layout = grid.get("layout")
        if layout is not None and layout != "fullsphere":
            raise ValueError(
                f"output.grid.layout must be 'fullsphere' (got {layout!r}; "
                "the deprecated 'dense' layout was removed — issue #88)"
            )
        # The issue #135 knob is retired (issue #304 phase 3, D16 — #135
        # inverts): morton is the stored cell coordinate and NESTED ids are
        # DERIVED at read (moczarr fabrication). A config still carrying the
        # key gets a pointed error, never a silent ignore.
        if grid.get("cell_ids_encoding") is not None:
            raise ValueError(
                "output.grid.cell_ids_encoding was retired (issue #304): morton is "
                "the stored cell coordinate and NESTED ids are derived at read; "
                "to keep writing the legacy NESTED cell_ids array for transition "
                "stores set output.grid.emit_cell_ids: true"
            )
        # Transition escape hatch (issue #304 phase 2): emit the legacy NESTED
        # cell_ids array ALONGSIDE the morton coordinate. Default off — morton
        # is the only stored cell coordinate (D16); the hatch exists for
        # transition stores (browser-direct demos) until the gridlook morton
        # path lands. HEALPix-only: rectilinear grids have no cell_ids
        # coordinate, so the flag would silently do nothing there.
        emit = grid.get("emit_cell_ids")
        if emit is not None:
            if not isinstance(emit, bool):
                raise ValueError(f"output.grid.emit_cell_ids must be a boolean (got {emit!r})")
            if emit and grid["type"] != "healpix":
                raise ValueError(
                    "output.grid.emit_cell_ids only applies to healpix grids "
                    f"(grid type is {grid['type']!r})"
                )
        # The legacy output.grid.indexing_scheme key is descriptive only (the
        # shipped configs carry "nested"); it never selected an encoding.
        # Reject any other value so a user reaching for it fails loudly.
        legacy_scheme = grid.get("indexing_scheme")
        if legacy_scheme is not None and legacy_scheme != "nested":
            raise ValueError(
                f"output.grid.indexing_scheme is descriptive and must be 'nested' "
                f"(got {legacy_scheme!r}); the store's declared coordinate is morton "
                f"(D16 — issue #304)"
            )

    # Validate the optional per-cell carrier (issue #132). Mirrors the worker's
    # ``{"pandas", "arrow"}`` guard (worker.py) so a typo in the aggregation YAML
    # fails at load, not deep in a worker.
    handoff = config.aggregation.get("handoff")
    if handoff is not None and handoff not in ("pandas", "arrow"):
        raise ValueError(f"aggregation.handoff must be 'pandas' or 'arrow' (got {handoff!r})")

    # Validate the optional read fan-out width (issue #170). Mirrors the read
    # module's guard (_read_workers) so a config typo is rejected at
    # submission -- inside the worker the same error would be swallowed into
    # per-group read_errors and surface as a silently-empty shard.
    read_workers = (config.data_source or {}).get("read_workers")
    if read_workers is not None and (
        isinstance(read_workers, bool) or not isinstance(read_workers, int) or read_workers < 1
    ):
        raise ValueError(f"data_source.read_workers must be an integer >= 1 (got {read_workers!r})")

    # Validate the optional granule fan-out width (issue #180). Mirrors the
    # worker's guard (_granule_workers) with the same rejection rationale as
    # read_workers above. ``shard_workers`` is the canonical cross-pipeline
    # name (issue #232 — "source units in flight per shard"); the legacy
    # ``granule_workers`` key stays honored (canonical wins when both set).
    for _key in ("shard_workers", "granule_workers"):
        _w = (config.data_source or {}).get(_key)
        if _w is not None and (isinstance(_w, bool) or not isinstance(_w, int) or _w < 1):
            raise ValueError(f"data_source.{_key} must be an integer >= 1 (got {_w!r})")
    # Optional strict-AOI cell mask (issue #101), default off. Must be a bool.
    aoi_mask = config.output.get("aoi_mask")
    if aoi_mask is not None and not isinstance(aoi_mask, bool):
        raise ValueError(f"output.aoi_mask must be a boolean (got {aoi_mask!r})")

    # Optional zarr metadata consolidation (issue #191), default off. Must be a
    # bool. No zagg reader depends on the consolidated blob, and building it is a
    # ~70 s serial-GET finalize tax, so it is opt-in.
    consolidate_metadata = config.output.get("consolidate_metadata")
    if consolidate_metadata is not None and not isinstance(consolidate_metadata, bool):
        raise ValueError(
            f"output.consolidate_metadata must be a boolean (got {consolidate_metadata!r})"
        )

    # Store layout + root coverage MOC — shared with the raster branch
    # (issue #247: raster + hive is legal, so the checks live in one place).
    _validate_store_layout_keys(config)

    # Temporal windowing block (issue #246, morton-hive/2): nested
    # output.windowing declares the window schedule + time encoding. Absent =
    # schedule none = today's behavior; the block itself is hive-only (windowed
    # leaves are a hive-layout convention), mirroring coverage_moc's posture.
    _validate_windowing(config)

    # Validate bounds structure (optional)
    if config.bounds is not None:
        allowed_keys = {"temporal", "spatial"}
        unknown = set(config.bounds.keys()) - allowed_keys
        if unknown:
            raise ValueError(f"Unknown bounds keys: {unknown} (allowed: {allowed_keys})")
        temporal = config.bounds.get("temporal")
        if temporal is not None:
            if "start_date" not in temporal or "end_date" not in temporal:
                raise ValueError("bounds.temporal requires start_date and end_date")

    # Validate base-level variable entries: plain path string, or the
    # column-selector mapping form (issue #321)
    _validate_ds_variables(config.data_source)

    # Validate the structured filter list (issue #43, Phase A)
    _validate_filters(config.data_source)

    # Validate hierarchical multi-level form (issue #43, Phase B)
    _validate_levels(config.data_source)

    # Cross-check: each filter's level field must name a key in levels (issue #43)
    _validate_filter_levels(config.data_source)

    # Virtual chunk-index backend block (issue #160)
    _validate_index(config.data_source)

    # Segment-level (non-base) ``variables`` mappings (issue #30) become real
    # per-photon columns in the pooled shard data once broadcast, so they are valid
    # column references everywhere a ``data_source.variables`` column is (agg
    # sources/expressions, chunk_precompute sources). Fold their names into ds_vars.
    ds_vars = set(config.data_source.get("variables", {}).keys()) | _segment_variable_names(
        config.data_source
    )
    agg_vars = config.aggregation.get("variables", {})

    # Validate the per-chunk precompute hook (issue #30, item 1). Each entry is
    # evaluated ONCE per chunk over the shard's pooled column data, before the
    # per-cell loop; its name becomes available in the per-cell expression
    # namespace. Validation mirrors ``aggregation.variables`` (exactly one of
    # function/expression, sources exist) but the entries are chunk-level scalars.
    _validate_chunk_precompute(config.aggregation, ds_vars)

    # Base-level ``expression`` filters evaluate over the read columns at read time
    # (before chunk_precompute), so their valid names are exactly ``ds_vars`` —
    # ``data_source.variables`` plus any broadcast segment-level variable (issue
    # #30). Validate their column references the same way agg/precompute
    # expressions are, so e.g. an ``{expression: "dem_h > ..."}`` filter is accepted.
    for f in filters_from_data_source(config.data_source):
        if "expression" in f:
            _validate_expression_columns(f"filter {f['expression']!r}", f["expression"], ds_vars)

    # Chunk-precompute names are injected into the per-cell expression namespace
    # (issue #30), so a per-cell ``expression`` (or its params) may reference them
    # like a column. Treat them as valid identifiers in the per-cell validation.
    precompute_names = set(config.aggregation.get("chunk_precompute", {}).keys())
    expr_vars = ds_vars | precompute_names

    for name, meta in agg_vars.items():
        has_func = "function" in meta
        has_expr = "expression" in meta

        # Mutual exclusivity
        if has_func and has_expr:
            raise ValueError(
                f"Variable '{name}': 'function' and 'expression' are mutually exclusive"
            )

        # Must have one (count via function:len is allowed)
        if not has_func and not has_expr:
            raise ValueError(f"Variable '{name}': must specify 'function' or 'expression'")

        # Validate source references
        source = meta.get("source")
        if source is not None and source not in ds_vars:
            raise ValueError(f"Variable '{name}': source '{source}' not in data_source.variables")

        # Validate function resolves
        if has_func:
            resolve_function(meta["function"])  # raises ValueError on failure

        # Validate params: bare column names, numeric literals, or expressions
        for pval in meta.get("params", {}).values():
            if not isinstance(pval, str):
                continue  # numeric literal
            if pval in expr_vars or _is_numeric(pval):
                continue  # column / chunk-precompute reference or number
            # Expression containing column names (e.g. "1.0 / s_li**2")
            if any(v in pval for v in expr_vars):
                continue
            raise ValueError(
                f"Variable '{name}': param value '{pval}' references "
                f"unknown column (available: {expr_vars})"
            )

        # Validate expression column references (chunk-precompute names included)
        if has_expr:
            _validate_expression_columns(name, meta["expression"], expr_vars)

        # Validate the output-kind declaration (kind + trailing_shape + dtype)
        _validate_output_kind(name, meta)

        # Located ragged fields (issue #87): the location column is the
        # per-observation morton the HEALPix read path supplies as ``leaf_id``
        # (or a declared data column). Rect grids have no per-obs morton, so a
        # located field on any non-HEALPix grid is a config error — raise here
        # rather than emit garbage downstream.
        location = meta.get("location")
        if location is not None:
            if location != "leaf_id" and location not in ds_vars:
                raise ValueError(
                    f"Variable '{name}': location '{location}' is not 'leaf_id' or a "
                    f"data_source variable (available: {sorted(ds_vars)})"
                )
            # An absent output.grid defaults to healpix everywhere else (the
            # grid factory's ``grid_cfg.get("type", "healpix")``), so mirror
            # that default here rather than falsely rejecting a valid config.
            grid_type = grid.get("type") if grid is not None else "healpix"
            if grid_type != "healpix":
                raise ValueError(
                    f"Variable '{name}': 'location' requires a healpix output grid "
                    f"(the per-observation morton column); got grid type {grid_type!r}"
                )
            # The location channel is injected as the reducer's ``locations=``
            # kwarg (aggregate.py), so a params entry of that name would collide
            # (TypeError deep in the per-cell loop) and a reducer that cannot
            # accept the kwarg would crash on every populated cell — reject both
            # at load time instead.
            if "locations" in meta.get("params", {}):
                raise ValueError(
                    f"Variable '{name}': params may not name 'locations' — it is "
                    f"reserved for the location channel (issue #87)"
                )
            if has_func:
                func = resolve_function(meta["function"])
                try:
                    func_params = inspect.signature(func).parameters
                except (TypeError, ValueError):
                    func_params = None  # ufuncs/builtins: not introspectable
                if func_params is not None and "locations" not in func_params:
                    has_var_kw = any(
                        p.kind == inspect.Parameter.VAR_KEYWORD for p in func_params.values()
                    )
                    if not has_var_kw:
                        raise ValueError(
                            f"Variable '{name}': function {meta['function']!r} does not "
                            f"accept a 'locations' keyword, which 'location' requires "
                            f"(use a located reducer, e.g. zagg.stats.tdigest.build_tdigest)"
                        )
            # The location channel is stored as a SIBLING vlen array named
            # ``{name}_locations`` (issue #209) — a name the user never wrote.
            # A field or coordinate already claiming it would silently lose in
            # the template members dict and interleave into the same object
            # slab at write time (data corruption), so reject at load
            # (review, PR #211).
            from zagg.grids.base import ragged_locations_name

            sibling = ragged_locations_name(name)
            if sibling in agg_vars or sibling in config.aggregation.get("coordinates", {}):
                raise ValueError(
                    f"Variable '{name}': its location channel is stored in a sibling "
                    f"array named '{sibling}', which collides with the declared "
                    f"field/coordinate of that name — rename one of them (issue #209)"
                )

        # Field-level store attrs (issue #321): free-form provenance the grid
        # template merges onto the field's array spec (e.g. the composition
        # field's lane order and signal threshold), so readers bind to
        # metadata instead of reconstructing config conventions.
        attrs = meta.get("attrs")
        if attrs is not None:
            if not isinstance(attrs, dict) or not all(isinstance(k, str) for k in attrs):
                raise ValueError(f"Variable '{name}': attrs must be a mapping with string keys")
            from zagg.grids.base import RAGGED_ELEMENT_ATTR

            if RAGGED_ELEMENT_ATTR in attrs:
                raise ValueError(
                    f"Variable '{name}': attrs key {RAGGED_ELEMENT_ATTR!r} is reserved "
                    f"for the ragged layout block (issue #209)"
                )
            import json

            try:
                json.dumps(attrs)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Variable '{name}': attrs must be JSON-serializable ({exc})"
                ) from exc

            _validate_composition_attrs(name, meta, attrs, agg_vars)


def _validate_composition_attrs(name: str, meta: dict, attrs: dict, agg_vars: dict) -> None:
    """Cross-check a ``composition`` attrs block against the field it describes.

    The whole field rests on one invariant: the ``N`` a reader divides the lanes
    by is the total weight of the digest named by ``of``, and the lanes were
    packed at the ``threshold`` recorded here (issue #321). Both halves are
    silent-wrong-answer failures — lanes stay in range, only the recovered
    counts are wrong — so they are checked at load:

    * ``of`` must name another ``aggregation.variables`` entry of
      ``kind: ragged`` (a typo or a later rename of the digest field would
      otherwise leave readers dereferencing a field that does not exist);
    * ``threshold`` must equal the field's own ``params.threshold`` (the value
      the reducer actually packs at).

    The third coupling — the digest fields' ``where`` predicates committing the
    same cut as ``threshold`` — is *not* validated here: ``where`` is an
    arbitrary expression, so any check would be a regex over Python source that
    both misses real drift (``> 1`` is the same cut as ``>= 2``) and rejects
    valid predicates. It is stated in :func:`zagg.stats.composition.pack_composition`'s
    docstring instead, where it travels with the function rather than with one
    config template.
    """
    block = attrs.get("composition")
    if not isinstance(block, dict):
        return
    of = block.get("of")
    if of is not None:
        if of == name:
            raise ValueError(
                f"Variable '{name}': attrs.composition.of names the composition field "
                f"itself; it must name the signal digest whose total weight is N_signal"
            )
        target = agg_vars.get(of)
        if target is None:
            raise ValueError(
                f"Variable '{name}': attrs.composition.of '{of}' is not a declared "
                f"aggregation.variables field (readers resolve N_signal through it)"
            )
        if target.get("kind") != "ragged":
            raise ValueError(
                f"Variable '{name}': attrs.composition.of '{of}' must be a "
                f"'kind: ragged' digest field (got kind "
                f"{target.get('kind', 'scalar')!r}) — N_signal is its total weight"
            )
    threshold = block.get("threshold")
    if threshold is not None:
        packed = (meta.get("params") or {}).get("threshold")
        if packed is not None and packed != threshold:
            raise ValueError(
                f"Variable '{name}': attrs.composition.threshold {threshold!r} disagrees "
                f"with params.threshold {packed!r} — the recorded threshold must be the "
                f"one the reducer packs at (issue #321)"
            )


def _validate_worker(config: PipelineConfig) -> None:
    """Validate the optional top-level ``worker:`` block (issue #235).

    ``{memory, extra_disk?}`` selects one of the pre-provisioned Lambda
    worker-size variants: ``memory`` (required, one of
    :data:`WORKER_MEMORIES`) picks the ``-<memory>`` function, and
    ``extra_disk: true`` (optional, default false) the ``-disk`` twin with
    ``/tmp`` sized memory + 2048 MB. Fails at load — a typo here would
    otherwise surface as a per-shard ResourceNotFound after the fan-out
    starts. Absent block is today's behavior (unsuffixed function).
    """
    worker = config.worker
    if worker is None:
        return
    if not isinstance(worker, dict):
        raise ValueError(f"worker must be a mapping {{memory, extra_disk?}} (got {worker!r})")
    unknown = set(worker) - {"memory", "extra_disk"}
    if unknown:
        raise ValueError(
            f"Unknown worker keys: {sorted(unknown)} (allowed: 'memory', 'extra_disk')"
        )
    memory = worker.get("memory")
    if isinstance(memory, bool) or memory not in WORKER_MEMORIES:
        raise ValueError(
            f"worker.memory must be one of {sorted(WORKER_MEMORIES)} MB — the "
            f"pre-provisioned variant sizes (issue #235) — (got {memory!r})"
        )
    extra_disk = worker.get("extra_disk")
    if extra_disk is not None and not isinstance(extra_disk, bool):
        raise ValueError(f"worker.extra_disk must be a boolean (got {extra_disk!r})")


def _validate_ds_variables(data_source: dict) -> None:
    """Validate ``data_source.variables`` entries (issue #321).

    Each value is either a path-template string (reads the whole dataset) or a
    mapping ``{path: <str>, column: <int >= 0>}`` selecting one column of a 2-D
    dataset — the variable analogue of the ``column`` key on structured
    filters, so several scalar columns can be declared against one shared
    dataset path (deduped to a single read).
    """
    variables = (data_source or {}).get("variables", {})
    if not isinstance(variables, dict):
        raise ValueError("data_source.variables must be a mapping of name -> path")
    for name, entry in variables.items():
        if isinstance(entry, str):
            continue
        if not isinstance(entry, dict):
            raise ValueError(
                f"data_source.variables['{name}'] must be a path string or a "
                f"{{path, column}} mapping (got {type(entry).__name__})"
            )
        extra = set(entry) - {"path", "column"}
        if extra:
            raise ValueError(
                f"data_source.variables['{name}']: unknown keys {sorted(extra)} "
                "(the mapping form takes exactly 'path' and 'column')"
            )
        if not isinstance(entry.get("path"), str):
            raise ValueError(f"data_source.variables['{name}'].path must be a string")
        column = entry.get("column")
        if not isinstance(column, int) or isinstance(column, bool) or column < 0:
            raise ValueError(
                f"data_source.variables['{name}'].column must be an integer >= 0 (got {column!r})"
            )


def _validate_filters(data_source: dict) -> None:
    """Validate the ``filters`` list and the flat ``quality_filter`` sugar.

    For the structured ``filters:`` list, raises ``ValueError`` on: unknown op,
    missing ``dataset``, ``in``/``not_in`` without a list ``values``, scalar ops
    without ``value``, non-int ``column``, a non-base-level ``expression`` filter,
    or wrong ``value`` type. ``column`` is required for the N-D flag case but
    cannot be checked against array rank here (no data); rank checks happen at
    read time.

    For the flat ``quality_filter`` sugar, only ``dataset`` and ``value`` are
    honored (``filters_from_data_source`` synthesizes ``op: eq, column: null``).
    Reject any extra keys at load time so a user-typoed ``op: gt`` or stray
    ``column:`` is not silently dropped on the floor — the structured ``filters:``
    list is the right form when those knobs are wanted.
    """
    qf = data_source.get("quality_filter")
    if qf is not None:
        allowed = {"dataset", "value"}
        unknown = set(qf) - allowed
        if unknown:
            raise ValueError(
                f"data_source.quality_filter only honors {sorted(allowed)} "
                f"(got extra keys {sorted(unknown)}); use the structured "
                "'filters:' list to set 'op', 'column', 'keep', etc."
            )
    filters = data_source.get("filters")
    if filters is None:
        return
    if not isinstance(filters, list):
        raise ValueError("data_source.filters must be a list")
    for i, f in enumerate(filters):
        if not isinstance(f, dict):
            raise ValueError(f"filter[{i}] must be a mapping")
        if "expression" in f:
            if "op" in f or "dataset" in f:
                raise ValueError(
                    f"filter[{i}]: 'expression' filters take no 'op'/'dataset' "
                    "(base-level aggregation-time escape hatch, no pushdown)"
                )
            if f.get("level") is not None:
                raise ValueError(
                    f"filter[{i}]: 'expression' filters are base-level only (level must be omitted)"
                )
            if not isinstance(f["expression"], str):
                raise ValueError(f"filter[{i}]: 'expression' must be a string")
            continue
        if "dataset" not in f:
            raise ValueError(f"filter[{i}]: structured filter requires 'dataset'")
        op = f.get("op")
        if op not in FILTER_OPS:
            raise ValueError(f"filter[{i}]: unknown op {op!r} (allowed: {sorted(FILTER_OPS)})")
        col = f.get("column")
        if col is not None and (not isinstance(col, int) or isinstance(col, bool)):
            raise ValueError(f"filter[{i}]: 'column' must be an integer index (got {col!r})")
        if op in _SET_OPS:
            if not isinstance(f.get("values"), list):
                raise ValueError(f"filter[{i}]: op {op!r} requires a 'values' list")
            for v in f["values"]:
                if not isinstance(v, (int, float)) or isinstance(v, bool):
                    raise ValueError(f"filter[{i}]: 'values' must be numeric (got {v!r})")
        else:
            if "value" not in f:
                raise ValueError(f"filter[{i}]: op {op!r} requires a scalar 'value'")
            if not isinstance(f["value"], (int, float)) or isinstance(f["value"], bool):
                raise ValueError(f"filter[{i}]: 'value' must be numeric (got {f['value']!r})")


def _validate_levels(data_source: dict) -> None:
    """Validate the hierarchical ``levels``/``base_level`` form (issue #43, Phase B).

    Rules:
    - ``base_level`` must name a key in ``levels``.
    - ``link.to`` in each level must name another key in ``levels``.
    - ``link.index_base`` must be a non-negative int when present.
    - ``link.reference_index`` must be ``None`` when present (reserved slot).
    - Only ``base_level`` may omit ``link`` (it has no coarser parent).
    - Flat single-level form (no ``levels`` key) is always valid.
    """
    levels = data_source.get("levels")
    if levels is None:
        return
    if not isinstance(levels, dict) or not levels:
        raise ValueError("data_source.levels must be a non-empty mapping")
    base_level = data_source.get("base_level")
    if base_level is None:
        raise ValueError("data_source.base_level is required when levels is present")
    if base_level not in levels:
        raise ValueError(
            f"data_source.base_level {base_level!r} is not a key in levels "
            f"(available: {sorted(levels)})"
        )
    base_vars = set(data_source.get("variables", {}))
    seg_var_names: set[str] = set()  # segment-variable names seen across non-base levels
    level_keys = set(levels)
    for name, lvl in levels.items():
        if not isinstance(lvl, dict):
            raise ValueError(f"levels.{name} must be a mapping")
        if "path" not in lvl:
            raise ValueError(f"levels.{name}: 'path' is required")
        # A non-base level may declare ``variables`` as a ``{name: path-template}``
        # mapping (issue #30): a readable segment-level variable broadcast to the
        # base rows at read time. Validate it like ``data_source.variables`` (string
        # names -> non-empty string path templates) and forbid the mapping form on
        # the base level (the base level uses ``data_source.variables``). The
        # documentation-only ``list[str]`` form stays valid on any level.
        lvl_vars = lvl.get("variables")
        if isinstance(lvl_vars, dict):
            if name == base_level:
                raise ValueError(
                    f"levels.{base_level}: the base level uses data_source.variables; "
                    f"a non-base level uses the 'variables' mapping for segment-level reads"
                )
            for var_name, tmpl in lvl_vars.items():
                if not isinstance(var_name, str) or not var_name:
                    raise ValueError(
                        f"levels.{name}.variables: variable names must be non-empty strings"
                    )
                if not isinstance(tmpl, str) or not tmpl:
                    raise ValueError(
                        f"levels.{name}.variables.{var_name}: path template must be a "
                        f"non-empty string (got {tmpl!r})"
                    )
                if var_name in base_vars:
                    raise ValueError(
                        f"levels.{name}.variables.{var_name}: collides with a "
                        f"data_source.variables column"
                    )
                # Two non-base levels declaring the same name would silently
                # overwrite each other when broadcast into one per-photon column
                # (the read keys by name); reject the ambiguity.
                if var_name in seg_var_names:
                    raise ValueError(
                        f"levels.{name}.variables.{var_name}: a segment-level "
                        f"variable named {var_name!r} is already declared on another level"
                    )
                seg_var_names.add(var_name)
        link = lvl.get("link")
        if link is None:
            if name != base_level:
                raise ValueError(
                    f"levels.{name}: non-base levels must have a 'link' "
                    f"(only {base_level!r} may omit it)"
                )
            continue
        if not isinstance(link, dict):
            raise ValueError(f"levels.{name}.link must be a mapping")
        for field_name in ("to", "index_beg", "count"):
            if field_name not in link:
                raise ValueError(f"levels.{name}.link: '{field_name}' is required")
        unknown = set(link) - {"to", "index_beg", "count", "index_base", "reference_index"}
        if unknown:
            raise ValueError(
                f"levels.{name}.link: unknown fields {sorted(unknown)} "
                f"(allowed: to, index_beg, count, index_base, reference_index)"
            )
        if link.get("to") == name:
            raise ValueError(f"level '{name}': link.to cannot reference the level itself")
        if link["to"] not in level_keys:
            raise ValueError(
                f"levels.{name}.link.to {link['to']!r} is not a key in levels "
                f"(available: {sorted(level_keys)})"
            )
        index_base = link.get("index_base", 0)
        if not isinstance(index_base, int) or isinstance(index_base, bool) or index_base < 0:
            raise ValueError(
                f"levels.{name}.link.index_base must be a non-negative int (got {index_base!r})"
            )
        ref = link.get("reference_index")
        if ref is not None:
            raise ValueError(
                f"levels.{name}.link.reference_index is reserved and must be null/omitted "
                f"(explicit index-array variant not yet implemented)"
            )


def _validate_filter_levels(data_source: dict) -> None:
    """Cross-check each filter's level field against the levels keys (issue #43).

    A filter with ``level: "nonexistent"`` would otherwise only fail at read time
    with an opaque ``KeyError``. Raises ``ValueError`` with a clear message when a
    filter's ``level`` names a key not present in ``levels``.
    """
    levels = data_source.get("levels")
    if levels is None:
        return
    level_keys = set(levels)
    filters = data_source.get("filters") or []
    for i, f in enumerate(filters):
        lvl = f.get("level")
        if lvl is not None and lvl not in level_keys:
            raise ValueError(
                f"filter[{i}]: level {lvl!r} is not a key in levels "
                f"(available: {sorted(level_keys)})"
            )


def _validate_index(data_source: dict) -> None:
    """Validate the optional ``data_source.index`` block (issue #160).

    Delegates to :func:`zagg.index.validate_index_config` (lazy import — the
    backend registry knows each backend's accepted keys, including entry-point
    backends this module cannot enumerate). Absent block → the default
    hierarchical path, nothing to check.
    """
    index_cfg = data_source.get("index")
    if index_cfg is None:
        return
    from zagg.index import validate_index_config

    validate_index_config(index_cfg, data_source)
