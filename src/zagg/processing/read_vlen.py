"""Generic vlen (variable-length record) reader primitives (issue #425).

Three sensor-agnostic declarative primitives extend the reader grammar so
vlen-packed products (GEDI L1B ``rxwaveform``, LVIS, any future waveform
sensor) are pure config — no sensor module (ruling on issue #422):

- **ragged gather** — the base level is a flat sample array tiled by a coarse
  (record) level's origin-aware ``link`` (``index_beg``/``count``/
  ``index_base``). GEDI's ``rx_sample_start_index`` is origin-1, the exact
  ``index_base: 1`` precedent of ATL03 ``ph_index_beg``. Planned reads slice
  the flat array by the record link, so only the shard's records are decoded.
- **expand** — per-record scalars repeat by sample count to base rate.
  Coordinates declare it with ``data_source.coordinates.level: <record
  level>`` (the base level carries no native coordinates); per-record
  variables reuse the existing segment-level ``variables`` broadcast
  (:func:`zagg.processing.read._read_segment_broadcasts`).
- **coordinate synthesis** — a per-sample column computed as
  ``linspace(start, stop, count)`` within each record: a
  ``data_source.variables`` entry of the mapping form
  ``{synthesize: linspace, level: <record level>, start: <path>, stop:
  <path>}`` (GEDI per-sample elevation from the per-shot
  ``elevation_bin0``/``elevation_lastbin`` endpoints).

After expansion each sample is an ordinary point observation; everything
downstream of the returned columns (filters, aggregation, write) is untouched.
Split out of ``read.py`` per the §4 module cap (the route would push it past
1,200 lines); the shared helpers are imported from there.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from zagg.config import evaluate_filter_expression, filters_from_data_source
from zagg.processing.read import (
    _expand_mask_at_rows,
    _predicate_mask,
    _read_paths_pooled,
    _read_segment_broadcasts,
    _read_workers,
    _record_obs_read,
    _segment_level_variables,
    _select_column,
    _variable_specs,
    link_base_extent,
)
from zagg.read_plan import execute_read_plan, plan_read


def vlen_coordinates_level(data_source: dict) -> str | None:
    """The record level named by ``coordinates.level``, or ``None`` when flat.

    Presence of the key is the vlen-route dispatch predicate
    (:func:`zagg.processing.read._read_group`): the base level has no native
    coordinate datasets, so lat/lon expand from this coarse level via its
    ``link`` counts.
    """
    coords = data_source.get("coordinates")
    if not isinstance(coords, dict):
        return None
    return coords.get("level")


def synthesized_specs(variables: dict) -> dict[str, dict]:
    """The ``synthesize:``-form entries of ``data_source.variables``.

    Returns ``{name: entry}`` for every mapping entry carrying ``synthesize``;
    the plain path / ``{path, column}`` entries are read by the ordinary
    variable machinery (:func:`zagg.processing.read._variable_specs`).
    """
    return {
        name: entry
        for name, entry in variables.items()
        if isinstance(entry, dict) and "synthesize" in entry
    }


def path_form_variables(variables: dict) -> dict:
    """``data_source.variables`` minus the ``synthesize:`` entries.

    The subset the ordinary path machinery understands: a synthesized entry
    carries no ``path``, so :func:`zagg.processing.read._variable_specs`
    raises ``KeyError`` on one. Every consumer of that normalizer on a
    possibly-vlen source (the read below, the inline backend's write-back
    prebuild) filters through here first.
    """
    synth = synthesized_specs(variables)
    return {name: entry for name, entry in variables.items() if name not in synth}


def expand_link_indices(
    ibeg_arr: np.ndarray, cnt_arr: np.ndarray, index_base: int, n_base: int
) -> tuple[np.ndarray, np.ndarray]:
    """Per-base-row ``(record, within-record)`` index arrays from a link.

    The gather map of the vlen grammar: ``parent_idx[j]`` is the record owning
    base row ``j`` (``-1`` for a row no record tiles — a gap surfaces as an
    unassignable row, never as data borrowed from a neighbor), and
    ``within_idx[j]`` is the row's 0-based offset inside its record (what the
    linspace synthesis interpolates over). Placement is by ``index_beg`` under
    the same origin (``index_base``) and empty-record discipline as
    :func:`zagg.processing.read._broadcast_segment_to_base`.

    ``n_base`` is the caller's: :func:`_vlen_read_group` derives it from the
    link itself (:func:`zagg.processing.read.link_base_extent`, issue #452). On
    a strided product the ``-1`` rows are ROUTINE — the slack between a
    record's valid samples and the next record's window start (GEDI L1B: ~50%
    of the rows a plan decodes) — and are dropped downstream by the
    ``valid & mask_spatial`` test, never assigned a neighbor's record. That
    handling is load-bearing, not defensive: GEDI's slack is **zero-filled in
    the product** (``rx_sample_start_index`` steps by exactly 1,420 with
    ``rx_sample_count`` ~701, and ``rxwaveform[701:1420]`` of each window reads
    as literal zeros), so a gap row that leaked through would aggregate as a
    real zero-amplitude sample. On a contiguous product these rows do not
    occur. A record reaching past ``n_base`` still raises rather than wrapping.
    """
    parent_idx = np.full(n_base, -1, dtype=np.int64)
    within_idx = np.zeros(n_base, dtype=np.int64)
    for p in range(len(cnt_arr)):
        cnt = int(cnt_arr[p])
        # Empty records cover no samples; under index_base=1 their sentinel
        # start (0) would read as beg=-1 and raise below, so skip them (the
        # issue #116 convention the segment broadcast follows).
        if cnt == 0:
            continue
        beg = int(ibeg_arr[p]) - index_base
        if beg < 0:
            raise ValueError(
                f"index_beg_arr[{p}]={ibeg_arr[p]} is less than index_base={index_base}"
            )
        if beg + cnt > n_base:
            raise ValueError(
                f"record {p} range [{beg}:{beg + cnt}] exceeds base size {n_base}; "
                f"the record link does not tile the declared base extent"
            )
        parent_idx[beg : beg + cnt] = p
        within_idx[beg : beg + cnt] = np.arange(cnt)
    return parent_idx, within_idx


def _planned_gather_map(
    ibeg_arr: np.ndarray,
    cnt_arr: np.ndarray,
    index_base: int,
    n_base: int,
    base_slices: list[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """:func:`expand_link_indices` restricted to a plan's rows.

    Same placement, validation and gap semantics, but the arrays are sized to
    the planned rows (``sum(e - s)``) instead of the whole base extent — the
    planned arm exists so a shard touching a handful of records never
    materializes anything at sample rate (the issue #43 OOM posture), and at
    GEDI's ~10^8 samples per beam group a full-rate ``int64`` pair is ~1.6 GB
    against a 2 GB worker. Validation still runs over every record, so a
    granule whose link does not tile the base extent raises here exactly as it
    does on the full read.

    Returns ``(global_idx, parent_idx, within_idx)``: the planned base rows and
    their per-row record / within-record indices.
    """
    beg_all = np.asarray(ibeg_arr).astype(np.int64) - index_base
    cnt_all = np.asarray(cnt_arr).astype(np.int64)
    # Empty records cover no samples; their origin-1 sentinel start would read
    # as beg=-1, so they are skipped rather than validated (the issue #116
    # convention expand_link_indices follows).
    nonempty = cnt_all > 0
    bad = nonempty & (beg_all < 0)
    if bad.any():
        p = int(np.argmax(bad))
        raise ValueError(f"index_beg_arr[{p}]={ibeg_arr[p]} is less than index_base={index_base}")
    over = nonempty & (beg_all + cnt_all > n_base)
    if over.any():
        p = int(np.argmax(over))
        raise ValueError(
            f"record {p} range [{beg_all[p]}:{beg_all[p] + cnt_all[p]}] exceeds base size "
            f"{n_base}; the record link does not tile the declared base extent"
        )
    if not base_slices:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty.copy(), empty.copy()
    global_idx = np.concatenate([np.arange(s, e, dtype=np.int64) for s, e in base_slices])
    parent_idx = np.full(len(global_idx), -1, dtype=np.int64)
    within_idx = np.zeros(len(global_idx), dtype=np.int64)
    off = 0
    for s, e in base_slices:
        # Records overlapping this slice, walked in index order so a later
        # record shadows an earlier one exactly as the full-rate map does.
        for p in np.nonzero(nonempty & (beg_all < e) & (beg_all + cnt_all > s))[0]:
            beg = int(beg_all[p])
            lo = max(beg, s)
            hi = min(beg + int(cnt_all[p]), e)
            parent_idx[off + lo - s : off + hi - s] = p
            within_idx[off + lo - s : off + hi - s] = np.arange(lo - beg, hi - beg)
        off += e - s
    return global_idx, parent_idx, within_idx


def synthesize_linspace(
    start_vals: np.ndarray,
    stop_vals: np.ndarray,
    cnt_arr: np.ndarray,
    parent_idx: np.ndarray,
    within_idx: np.ndarray,
) -> np.ndarray:
    """Per-sample ``linspace(start, stop, count)`` values within each record.

    Row ``j`` of record ``p`` (count ``n_p``) gets
    ``start[p] + (stop[p] - start[p]) * j / (n_p - 1)`` — endpoints exact, a
    single-sample record gets ``start[p]``. Rows with no record
    (``parent_idx == -1``) get NaN; they carry no coordinates either and are
    dropped by the spatial mask.
    """
    parent_idx = np.asarray(parent_idx)
    valid = parent_idx >= 0
    out = np.full(len(parent_idx), np.nan, dtype=np.float64)
    p = parent_idx[valid]
    start = np.asarray(start_vals, dtype=np.float64)[p]
    stop = np.asarray(stop_vals, dtype=np.float64)[p]
    denom = np.maximum(np.asarray(cnt_arr, dtype=np.float64)[p] - 1.0, 1.0)
    out[valid] = start + (stop - start) * (np.asarray(within_idx)[valid] / denom)
    return out


def _validate_vlen_config(data_source: dict) -> tuple[str, dict]:
    """Completeness gate for the vlen route; returns ``(coord_level_key, level)``.

    The coordinates level must be a declared non-base level whose ``link``
    points directly at the base level — the same shape
    :func:`zagg.processing.read._planned_read_group` requires of
    ``read_plan.spatial_index``.
    """
    levels = data_source.get("levels")
    base_level = data_source.get("base_level")
    if not isinstance(levels, dict) or not levels or not base_level:
        raise ValueError("coordinates.level requires 'levels' and 'base_level'")
    key = data_source["coordinates"]["level"]
    lvl = levels.get(key)
    if lvl is None:
        raise ValueError(f"coordinates.level {key!r} is not a key in levels")
    link = lvl.get("link")
    if not isinstance(link, dict) or link.get("to") != base_level:
        raise ValueError(
            f"coordinates.level {key!r} must link directly to base level {base_level!r}"
        )
    return key, lvl


def _vlen_read_group(
    h5obj,
    group: str,
    data_source: dict,
    shard_key: int,
    grid,
    arrow: bool = False,
    read_fn=None,
    io_stats: dict | None = None,
    siblings: dict | None = None,
):
    """Read and spatially filter one vlen-packed HDF5 group.

    Same ``pandas.DataFrame`` / ``arro3.core.Table`` / ``None`` contract as
    :func:`zagg.processing.read._read_group`. The record level's coordinates
    and link arrays are read fully (record rate is ~3 orders lighter than
    sample rate on GEDI); when ``read_plan.spatial_index`` names the
    coordinates level the flat base datasets are sliced to the shard's records
    via :func:`zagg.read_plan.plan_read` (the ragged gather), otherwise they
    are read in full. Coordinates never read at base rate — they expand from
    the record level — and synthesized columns are computed, not read.

    ``siblings`` maps sibling-asset names to open h5coro handles (issue #425,
    the paired-asset L2A join); ``None`` when the granule carries no sibling
    assets. ``read_fn`` is the index-backend addressing seam and ``io_stats``
    the read counter, both as in ``_read_group``.
    """
    coord_key, coord_lvl = _validate_vlen_config(data_source)
    levels = data_source["levels"]
    base_level_key = data_source["base_level"]
    coordinates = data_source["coordinates"]
    variables = data_source["variables"]
    link = coord_lvl["link"]
    index_base = int(link.get("index_base", 0))

    workers = _read_workers(data_source) if read_fn is not None else 1

    def _read_full(paths: list[str]) -> dict:
        """Full-dataset reads through the active addressing seam."""
        if read_fn is None:
            return h5obj.readDatasets(list(dict.fromkeys(paths)))
        return _read_paths_pooled(
            [(p, None) for p in dict.fromkeys(paths)], lambda p, dt: read_fn(p), workers
        )

    # ---- Record-level coordinates + link arrays (small, full read).
    lat_path = coordinates["latitude"].format(group=group)
    lon_path = coordinates["longitude"].format(group=group)
    ibeg_path = link["index_beg"].format(group=group)
    cnt_path = link["count"].format(group=group)
    coarse = _read_full([lat_path, lon_path, ibeg_path, cnt_path])
    coarse_lats = np.asarray(coarse[lat_path])
    coarse_lons = np.asarray(coarse[lon_path])
    ibeg_arr = np.asarray(coarse[ibeg_path])
    cnt_arr = np.asarray(coarse[cnt_path])

    if len(coarse_lats) == 0:
        _record_obs_read(io_stats, 0)
        return None
    # The base extent the record link tiles (issue #452): the last base row any
    # non-empty record reaches. Contiguous products make that ``Σcount`` (#43's
    # assumption, ATL03's shape); STRIDED ones do not — GEDI L1B gives every
    # shot a fixed 1,420-sample window in ``rxwaveform`` and counts only the
    # valid samples, so ``Σcount`` understates the array by ~50%, ``plan_read``
    # clamps every run to ``base_end <= base_start``, and the group returns
    # zero rows with no error raised. Under striding the slack rows between
    # records are ROUTINE (143k of 310k planned rows on the reference shard):
    # they belong to no record, so the gather map marks them ``parent_idx ==
    # -1``, ``synthesize_linspace`` NaNs them, and ``valid & mask_spatial``
    # drops them before a single column is assembled. That drop is the
    # correctness guarantee, not a nicety: GEDI zero-fills the slack in the
    # product (the tail of each 1,420-sample window past ``rx_sample_count``
    # reads as literal zeros), so any row that leaked through would land in a
    # cell as a real zero-amplitude sample — ~46% of the fetched span.
    n_base = link_base_extent(ibeg_arr, cnt_arr, index_base)
    if n_base <= 0:
        _record_obs_read(io_stats, 0)
        return None

    # Match records to this shard with the same exact mortie test the sample
    # rows get below — expanded sample coordinates ARE the record coordinates,
    # so the record mask is exact, not a rep-point approximation.
    coarse_mask = grid.shards_of(grid.assign(coarse_lats, coarse_lons)) == shard_key

    rp = data_source.get("read_plan")
    plan = None
    if isinstance(rp, dict) and rp.get("spatial_index"):
        if rp["spatial_index"] != coord_key:
            raise ValueError(
                f"read_plan.spatial_index {rp['spatial_index']!r} must name the "
                f"coordinates level {coord_key!r} on a vlen data source"
            )
        plan = plan_read(
            coarse_lats,
            coarse_lons,
            ibeg_arr,
            cnt_arr,
            n_base,
            index_base=index_base,
            pad=int(rp.get("pad", 1)),
            full_read_threshold=float(rp.get("full_read_threshold", 0.9)),
            coarse_mask=coarse_mask,
        )
        if not plan.parent_runs:
            _record_obs_read(io_stats, 0)
            return None
        if plan.full_read:
            plan = None  # selectivity above threshold: read the flat datasets fully
    elif not coarse_mask.any():
        _record_obs_read(io_stats, 0)
        return None

    # ---- The gather map: (record, within-record) index per DECODED base row.
    # The planned arm builds it at plan rate, never at sample rate — the map is
    # the read's dominant allocation on a real granule. The full arm is
    # O(n_base) by necessity and needs no row selection at all, so ``global_idx``
    # stays ``None`` (the identity) rather than a full-rate ``arange``.
    if plan is not None:
        global_idx, parent_planned, within_planned = _planned_gather_map(
            ibeg_arr, cnt_arr, index_base, n_base, plan.base_slices
        )
    else:
        global_idx = None
        parent_planned, within_planned = expand_link_indices(ibeg_arr, cnt_arr, index_base, n_base)

    rows_cache: np.ndarray | None = None

    def _base_rows() -> np.ndarray:
        """Global base index of each row this read decoded.

        The plan's rows, or the identity map on the full arm (which is O(n_base)
        by necessity anyway). Built lazily and once: a read with no cross-level
        filter and no segment-level variable never needs it, and on the full arm
        it is a sample-rate array.
        """
        nonlocal rows_cache
        if rows_cache is None:
            rows_cache = global_idx if global_idx is not None else np.arange(n_base, dtype=np.int64)
        return rows_cache

    # Rows DECODED by this group (issue #374): the planned slices (or the full
    # base extent), counted before the shard mask and filters below.
    _record_obs_read(io_stats, len(parent_planned))
    if len(parent_planned) == 0:
        return None

    # ---- Expand coordinates; exact per-sample shard mask. A gap row (no
    # record) has no coordinates: clamp its gather index so the fancy index
    # cannot wrap to the last record, and force it out via the valid mask.
    valid = parent_planned >= 0
    parent_safe = np.where(valid, parent_planned, 0)
    lats = coarse_lats[parent_safe]
    lons = coarse_lons[parent_safe]
    leaf_ids = grid.assign(lats, lons)
    mask_spatial = valid & (grid.shards_of(leaf_ids) == shard_key)
    if not mask_spatial.any():
        return None

    # ---- Read base-rate variables + base-level filter datasets (the ragged
    # gather: planned slices when a plan is active, full reads otherwise).
    filters = filters_from_data_source(data_source)
    base_structured = [
        f
        for f in filters
        if "expression" not in f
        and f.get("asset") is None
        and (f.get("level") is None or f.get("level") == base_level_key)
    ]
    coarse_structured = [
        f
        for f in filters
        if "expression" not in f
        and f.get("asset") is None
        and f.get("level") is not None
        and f.get("level") != base_level_key
    ]
    asset_filters = [f for f in filters if "expression" not in f and f.get("asset") is not None]
    expressions = [f for f in filters if "expression" in f]

    path_specs = _variable_specs(path_form_variables(variables))
    var_paths = {col: tmpl.format(group=group) for col, (tmpl, _) in path_specs.items()}
    filter_paths = {id(f): f["dataset"].format(group=group) for f in base_structured}
    base_paths = list(dict.fromkeys(list(var_paths.values()) + list(filter_paths.values())))
    if plan is not None:
        _base_read_fn = read_fn
        if _base_read_fn is None:

            def _base_read_fn(path, hyperslice=None):
                if hyperslice is None:
                    return h5obj.readDatasets([path])[path]
                return h5obj.readDatasets([{"dataset": path, "hyperslice": hyperslice}])[path]

        arrays_by_path = _read_paths_pooled(
            [(p, None) for p in base_paths],
            lambda p, dt: execute_read_plan(plan, _base_read_fn, p, dt),
            workers,
        )
    else:
        # Full arm: the flat datasets may be LONGER than the link's extent — a
        # strided product pads the tail of its last record's window (GEDI:
        # 1,420 samples allocated, ``count`` valid). Those rows belong to no
        # record, so trim them here and keep every base-rate array the same
        # length as the gather map (issue #452). A dataset SHORTER than the
        # extent is a real inconsistency (the link over-runs the array) and is
        # reported with the gather map's wording.
        arrays_by_path = {}
        for path, arr in _read_full(base_paths).items():
            arr = np.asarray(arr)
            if len(arr) < n_base:
                raise ValueError(
                    f"dataset {path!r} has {len(arr)} rows, fewer than the record "
                    f"link's base extent {n_base}; the record link does not tile "
                    f"the declared base extent"
                )
            arrays_by_path[path] = arr[:n_base]

    # ---- Filters. Base-level structured predicates over the gathered rows;
    # record-level predicates expand to base rate via each level's link (the
    # existing cross-level machinery); sibling-asset predicates join per
    # record on the declared key, then expand the same way.
    keep_mask: np.ndarray | None = None
    for f in base_structured:
        flag = np.asarray(arrays_by_path[filter_paths[id(f)]])[mask_spatial]
        fmask = _predicate_mask(flag, f)
        keep_mask = fmask if keep_mask is None else (keep_mask & fmask)

    # Both cross-level arms expand at PLANNED rate (issue #452): each record-rate
    # verdict is gathered onto the rows this read decoded, never painted across
    # the whole base extent and sliced afterwards (196 MB per filter on a GEDI
    # beam group). A row owned by no record is False, exactly as the base-rate
    # form leaves it.
    cross_planned: np.ndarray | None = None
    for f in coarse_structured:
        cf_lvl = levels[f["level"]]
        cf_link = cf_lvl["link"]
        cf_paths = [
            f["dataset"].format(group=group),
            cf_link["index_beg"].format(group=group),
            cf_link["count"].format(group=group),
        ]
        cf_data = _read_full(cf_paths)
        coarse_fmask = _predicate_mask(np.asarray(cf_data[cf_paths[0]]), f)
        # A non-coordinate level owns its own link, so its per-row owner comes
        # from a searchsorted over that link's starts rather than from the
        # coordinates level's gather map.
        expanded = _expand_mask_at_rows(
            coarse_fmask,
            np.asarray(cf_data[cf_paths[1]]),
            np.asarray(cf_data[cf_paths[2]]),
            int(cf_link.get("index_base", 0)),
            _base_rows(),
        )
        cross_planned = expanded if cross_planned is None else (cross_planned & expanded)

    if asset_filters:
        record_mask = _sibling_record_mask(
            h5obj, group, data_source, asset_filters, len(coarse_lats), siblings, _read_full
        )
        # The sibling join is keyed to the coordinates level, whose per-row owner
        # the gather map already carries.
        expanded = record_mask[parent_safe] & valid
        cross_planned = expanded if cross_planned is None else (cross_planned & expanded)

    if cross_planned is not None:
        cross_masked = cross_planned[mask_spatial]
        keep_mask = cross_masked if keep_mask is None else (keep_mask & cross_masked)
    if keep_mask is not None and not keep_mask.any():
        return None

    # ---- Synthesized columns (computed, never read) + record-level variable
    # broadcasts (the existing expand primitive), aligned to the gathered rows.
    synth_cols: dict[str, np.ndarray] = {}
    for name, entry in synthesized_specs(variables).items():
        sp = _read_full([entry["start"].format(group=group), entry["stop"].format(group=group)])
        synth_cols[name] = synthesize_linspace(
            np.asarray(sp[entry["start"].format(group=group)]),
            np.asarray(sp[entry["stop"].format(group=group)]),
            cnt_arr,
            parent_planned,
            within_planned,
        )

    # Record-level variable broadcasts, gathered straight onto this read's rows
    # (issue #452): the base-rate form is one sample-rate array per companion
    # variable — ~7 GB for GEDI's six against a 2 GB worker. ``_base_rows`` is
    # only built when there is something to broadcast.
    seg_broadcasts = (
        _read_segment_broadcasts(
            h5obj, group, data_source, levels, n_base, read_fn=read_fn, base_rows=_base_rows()
        )
        if _segment_level_variables(data_source)
        else {}
    )

    # ---- Assemble columns: gathered variables, synthesized, broadcasts, leaf.
    data_dict: dict[str, np.ndarray] = {}
    for col_name, path in var_paths.items():
        values = _select_column(np.asarray(arrays_by_path[path]), path_specs[col_name][1], col_name)
        values = values[mask_spatial]
        if keep_mask is not None:
            values = values[keep_mask]
        data_dict[col_name] = values
    for col_name, values in synth_cols.items():
        values = values[mask_spatial]
        if keep_mask is not None:
            values = values[keep_mask]
        data_dict[col_name] = values
    for col_name, planned_values in seg_broadcasts.items():
        values = planned_values[mask_spatial]
        if keep_mask is not None:
            values = values[keep_mask]
        data_dict[col_name] = values
    leaf_after = leaf_ids[mask_spatial]
    data_dict["leaf_id"] = leaf_after[keep_mask] if keep_mask is not None else leaf_after

    # Base-level expression filters over the assembled columns, as in the
    # dense routes (aggregation-time escape hatch, no pushdown).
    expr_names = list(variables) + list(seg_broadcasts)
    for f in expressions:
        cols = {c: data_dict[c] for c in expr_names if c in data_dict}
        try:
            emask = evaluate_filter_expression(f["expression"], cols)
        except NameError as e:
            raise NameError(
                f"expression filter {f['expression']!r} references an undefined name: {e}"
            ) from e
        if emask.shape != data_dict["leaf_id"].shape:
            raise ValueError(
                f"expression filter {f['expression']!r} must yield a per-row "
                f"boolean mask (got shape {emask.shape})"
            )
        if not emask.any():
            return None
        data_dict = {k: v[emask] for k, v in data_dict.items()}

    if arrow:
        from arro3.core import Table

        return Table.from_pydict(data_dict)
    return pd.DataFrame(data_dict)


def _sibling_record_mask(
    h5obj,
    group: str,
    data_source: dict,
    asset_filters: list[dict],
    n_records: int,
    siblings: dict | None,
    read_full,
) -> np.ndarray:
    """Per-record keep-mask from sibling-asset predicates (the L2A join).

    Each declared sibling asset (``data_source.assets``) carries a ``join``
    block naming the record-level key dataset on both sides (GEDI:
    ``shot_number`` on L1B and L2A). The sibling's key and filter datasets are
    read fully (per-record rate, small), records are matched by key value, and
    each predicate evaluates on the matched sibling rows. A record with no
    sibling match fails every predicate on that asset — an unfilterable record
    must not pass a quality gate silently (the pairless-granule posture of the
    shardmap build, applied per record).
    """
    assets_cfg = data_source.get("assets")
    if not isinstance(assets_cfg, dict):
        raise ValueError("asset filters require a data_source.assets block")
    mask = np.ones(n_records, dtype=bool)
    by_asset: dict[str, list[dict]] = {}
    for f in asset_filters:
        by_asset.setdefault(f["asset"], []).append(f)
    for asset, filters in by_asset.items():
        cfg = assets_cfg.get(asset)
        if cfg is None:
            raise ValueError(f"filter asset {asset!r} is not declared in data_source.assets")
        sibling = (siblings or {}).get(asset)
        if sibling is None:
            raise ValueError(
                f"granule carries no open sibling handle for asset {asset!r}; "
                f"was the shard map built with the paired-asset join?"
            )
        join = cfg["join"]
        left_path = join["left"].format(group=group)
        right_path = join["right"].format(group=group)
        left_keys = np.asarray(read_full([left_path])[left_path])
        sib_paths = [right_path] + [f["dataset"].format(group=group) for f in filters]
        sib_data = sibling.readDatasets(list(dict.fromkeys(sib_paths)))
        right_keys = np.asarray(sib_data[right_path])
        if len(left_keys) != n_records:
            raise ValueError(
                f"assets.{asset}.join.left has {len(left_keys)} records; the "
                f"coordinates level has {n_records}"
            )
        # Key-value join: position of each left key in the sibling's rows.
        order = np.argsort(right_keys, kind="stable")
        sorted_keys = right_keys[order]
        pos = np.searchsorted(sorted_keys, left_keys, side="left")
        pos_clip = np.minimum(pos, len(sorted_keys) - 1) if len(sorted_keys) else pos
        matched = (
            (sorted_keys[pos_clip] == left_keys)
            if len(sorted_keys)
            else np.zeros(n_records, dtype=bool)
        )
        sib_idx = order[pos_clip] if len(sorted_keys) else np.zeros(n_records, dtype=np.int64)
        asset_mask = matched.copy()
        for f in filters:
            flag = np.asarray(sib_data[f["dataset"].format(group=group)])[sib_idx]
            asset_mask &= _predicate_mask(flag, f)
        mask &= asset_mask
    return mask
