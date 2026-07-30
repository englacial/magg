"""Source-side block validators: temporal specs, raster, collections (issue #330).

Split out of the single-file ``zagg.config`` (issue #330 phase 4); the public
surface is unchanged and re-exported from :mod:`zagg.config`. The validators
:func:`zagg.config.validate_config` delegates to for the ``data_source`` half
of a config.
"""

from __future__ import annotations

from zagg.config.base import PipelineConfig
from zagg.config.validate_output import _validate_store_layout_keys, _validate_windowing

# Required per-variable keys for a temporal/event aggregation spec. ``mask``
# is optional (``specs_from_config`` defaults it to ``"ais"``); capability
# *names* (spatial_func / temporal_reducer / mask) are resolved by the registry
# at run time, not here -- a typo surfaces as a clean ``UnknownCapability`` from
# ``process_event`` rather than a load-time guess about which plugins are
# installed.
_TEMPORAL_SPEC_KEYS = ("variable", "collection", "spatial_func", "temporal_reducer")


def _validate_temporal_config(config: PipelineConfig) -> None:
    """Validate a temporal/event pipeline config (issue #12, Phase 5).

    Temporal pipelines carry no output grid; the only cross-check is that the
    ``aggregation.variables`` block names, per variable, the four keys
    :data:`_TEMPORAL_SPEC_KEYS` that :func:`zagg.temporal.specs_from_config`
    requires (the rest are optional flags with defaults). Raises ``ValueError``
    on a missing section or key.
    """
    if not config.aggregation:
        raise ValueError("Missing required section: aggregation")
    variables = config.aggregation.get("variables")
    if not variables:
        raise ValueError("temporal pipeline requires aggregation.variables")
    for name, meta in variables.items():
        missing = [k for k in _TEMPORAL_SPEC_KEYS if k not in meta]
        if missing:
            raise ValueError(
                f"temporal variable '{name}' is missing required key(s): "
                f"{', '.join(missing)} (need {', '.join(_TEMPORAL_SPEC_KEYS)})"
            )
        params = meta.get("params")
        if params is not None and not isinstance(params, dict):
            raise ValueError(
                f"temporal variable '{name}' params must be a mapping (got {params!r})"
            )
    _validate_collection_options(config)


def _validate_raster_config(config: PipelineConfig) -> None:
    """Validate a raster (pull-NN) pipeline config (issue #218).

    ``data_source.bands`` maps output field name -> ``{asset, dtype, ...}``:
    ``asset`` names the STAC asset key holding that band's GeoTIFF, ``dtype``
    is required (bands store exact source DNs — no float default), and
    optional ``fill_value`` (default 0), ``scale``/``offset`` (recorded as
    ``scale_factor``/``add_offset`` array attrs, never applied to the data).
    ``data_source.nodata`` (optional scalar) marks source nodata; a cell
    whose sampled pixel equals it in any band is left at fill.
    """
    for section in ("data_source", "output"):
        if not getattr(config, section):
            raise ValueError(f"Missing required section: {section}")
    if config.aggregation:
        raise ValueError(
            "raster pipelines take no aggregation section: pull-NN yields one "
            "value per cell per timestep (declare bands under data_source.bands)"
        )
    bands = config.data_source.get("bands")
    if not bands or not isinstance(bands, dict):
        raise ValueError("raster pipeline requires data_source.bands (field -> {asset, dtype})")
    for name, meta in bands.items():
        if not isinstance(meta, dict):
            raise ValueError(f"band '{name}' must be a mapping (got {meta!r})")
        for key in ("asset", "dtype"):
            if not isinstance(meta.get(key), str) or not meta.get(key):
                raise ValueError(f"band '{name}' requires a string '{key}'")
        for key in ("fill_value", "scale", "offset"):
            if key in meta and (
                not isinstance(meta[key], (int, float)) or isinstance(meta[key], bool)
            ):
                raise ValueError(f"band '{name}' {key} must be a number (got {meta[key]!r})")
    nodata = config.data_source.get("nodata")
    if nodata is not None and (not isinstance(nodata, (int, float)) or isinstance(nodata, bool)):
        raise ValueError(f"data_source.nodata must be a number (got {nodata!r})")
    # Optional per-shard fan-out cap (issues #231/#232): ``shard_workers`` is
    # the ONE cross-pipeline knob for "source units in flight per shard" —
    # here, acquisition groups (timesteps) sampling concurrently. Guarded like
    # read_workers so a config typo is rejected at submission, not swallowed
    # inside the worker's event loop. Default 4
    # (``processing.raster._shard_workers``); ``1`` is serial. NOTE: the
    # memory cost per unit is pipeline-dependent (a granule read buffer on the
    # spatial path vs a timestep of decoded COG tiles here) — see issue #228.
    shard_workers = config.data_source.get("shard_workers")
    if shard_workers is not None and (
        isinstance(shard_workers, bool) or not isinstance(shard_workers, int) or shard_workers < 1
    ):
        raise ValueError(
            f"data_source.shard_workers must be an integer >= 1 (got {shard_workers!r})"
        )
    # Streamed-write slab budget (PR #232 double-buffer): ``1`` (default) is
    # the strict serial bound; ``N`` overlaps up to N-1 slab writes with
    # sampling at a peak of N slabs alive (``processing.raster._write_buffer``).
    write_buffer = config.data_source.get("write_buffer")
    if write_buffer is not None and (
        isinstance(write_buffer, bool) or not isinstance(write_buffer, int) or write_buffer < 1
    ):
        raise ValueError(f"data_source.write_buffer must be an integer >= 1 (got {write_buffer!r})")
    grid = config.output.get("grid")
    if not isinstance(grid, dict) or grid.get("type") != "healpix":
        raise ValueError(
            "raster pipelines currently require output.grid.type: healpix "
            "(the rectilinear (time, y, x) template is future work — issue #218)"
        )
    for key in ("parent_order", "child_order"):
        if key not in grid:
            raise ValueError(f"output.grid.{key} is required for healpix grid")
    if grid.get("sharded"):
        # Permanent exclusion, not a deferral (espg-ratified on issue #247):
        # per-timestep slab streaming over a ShardingCodec object would
        # read-modify-write it once per timestep, and raster object count is
        # time-axis-dominated anyway — sharding the cell axis buys little.
        raise ValueError(
            "raster pipelines do not support sharded: true (per-timestep slab "
            "streaming would read-modify-write each ShardingCodec object; "
            "chunks stay (1, cells_per_chunk) — one object per timestep-chunk)"
        )
    # Store layout, root coverage MOC, and temporal windowing ride the SHARED
    # checks (issue #247: raster + hive is legal — the issue #239 stopgap
    # rejections are gone). _validate_windowing resolves the raster
    # time_field/encoding rules (membership is the acquisition's STAC
    # datetime; the conversion knobs do not apply).
    _validate_store_layout_keys(config)
    _validate_windowing(config)


_RESAMPLE_HOWS = ("sum", "mean")


def _validate_collection_options(config: PipelineConfig) -> None:
    """Validate ``data_source.collections`` (issue #213, Phase 3).

    The block accepts a list of collection names (no options) or a mapping of
    name -> per-collection reader options consumed by
    :func:`zagg.temporal.prepare_collection`: ``coord_round`` (non-negative
    int, decimals to round lat/lon coords to -- for source grids whose own
    coordinate arrays carry float dirt), ``variables`` (list of names),
    ``time_offset`` (a pandas-parseable offset string), ``resample``
    (``{freq, how: sum|mean, scale}``), and ``derived`` (name -> numpy
    expression string). ``time_offset`` and ``resample.freq`` are parsed with
    pandas here so an unparseable value fails at load rather than surfacing as
    a per-event worker error after the Lambda spend. Unknown option keys (e.g.
    ``doi``) pass through by design for catalog tooling metadata.
    """
    colls = (config.data_source or {}).get("collections")
    if colls is None or isinstance(colls, list):
        return
    if not isinstance(colls, dict):
        raise ValueError(
            "data_source.collections must be a list of names or a mapping of "
            f"name -> options (got {type(colls).__name__})"
        )
    for cname, opts in colls.items():
        if opts is None:
            continue
        if not isinstance(opts, dict):
            raise ValueError(
                f"data_source.collections[{cname!r}] must be a mapping of options (or null)"
            )
        variables = opts.get("variables")
        if variables is not None and (
            not isinstance(variables, list) or not all(isinstance(v, str) for v in variables)
        ):
            raise ValueError(
                f"data_source.collections[{cname!r}].variables must be a list of names"
            )
        coord_round = opts.get("coord_round")
        if coord_round is not None and (
            isinstance(coord_round, bool) or not isinstance(coord_round, int) or coord_round < 0
        ):
            raise ValueError(
                f"data_source.collections[{cname!r}].coord_round must be a "
                f"non-negative integer (got {coord_round!r})"
            )
        offset = opts.get("time_offset")
        if offset is not None:
            if not isinstance(offset, str):
                raise ValueError(
                    f"data_source.collections[{cname!r}].time_offset must be an offset "
                    f"string like '-30min' (got {offset!r})"
                )
            import pandas as pd

            try:
                pd.to_timedelta(offset)
            except (ValueError, TypeError) as exc:
                raise ValueError(
                    f"data_source.collections[{cname!r}].time_offset is not a valid "
                    f"pandas offset string (got {offset!r})"
                ) from exc
        resample = opts.get("resample")
        if resample is not None:
            if not isinstance(resample, dict) or "freq" not in resample:
                raise ValueError(
                    f"data_source.collections[{cname!r}].resample must be a mapping "
                    "with at least 'freq' (optional: how, scale)"
                )
            freq = resample["freq"]
            if not isinstance(freq, str):
                raise ValueError(
                    f"data_source.collections[{cname!r}].resample.freq must be a "
                    f"frequency string like '3h' (got {freq!r})"
                )
            import pandas as pd

            try:
                pd.tseries.frequencies.to_offset(freq)
            except (ValueError, TypeError) as exc:
                raise ValueError(
                    f"data_source.collections[{cname!r}].resample.freq is not a valid "
                    f"pandas frequency string (got {freq!r})"
                ) from exc
            how = resample.get("how", "sum")
            if how not in _RESAMPLE_HOWS:
                raise ValueError(
                    f"data_source.collections[{cname!r}].resample.how must be one of "
                    f"{_RESAMPLE_HOWS} (got {how!r})"
                )
            scale = resample.get("scale", 1)
            if isinstance(scale, bool) or not isinstance(scale, (int, float)):
                raise ValueError(
                    f"data_source.collections[{cname!r}].resample.scale must be numeric "
                    f"(got {scale!r})"
                )
        derived = opts.get("derived")
        if derived is not None and (
            not isinstance(derived, dict)
            or not all(isinstance(k, str) and isinstance(v, str) for k, v in derived.items())
        ):
            raise ValueError(
                f"data_source.collections[{cname!r}].derived must map variable names "
                "to expression strings"
            )
