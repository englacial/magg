"""Raster (GeoTIFF/COG) read path: pull-NN sampling at grid cell centers.

Issue #218. The decode engine is **async-tiff** (espg-ratified on the issue):
byte-range tile reads through an obspec store, Rust-side decode, typed numpy
buffers. zagg owns the mapping — ``grid.sample()`` turns cell centers into
source-pixel indices, and this module fetches exactly the COG tiles those
indices touch and gathers per-cell values. The engine never touches the
output side (HEALPix emission stays pure mortie/pyproj), and GDAL is never
involved.

Sync facade over async-tiff's async API: worker call sites are synchronous
(one shard per Lambda invoke), so :func:`sample_asset` runs its own event
loop; the per-asset tile fan-out inside it is concurrent.

Georeferencing is read from the GeoTIFF IFD itself (geo keys + pixel scale +
tiepoint), so a granule entry needs only the asset href — no STAC ``proj:*``
carriage through the shardmap.
"""

from __future__ import annotations

import asyncio
import time
import warnings

import numpy as np

# Facade re-exports (issue #330 phase 2). Plain imports are the names this
# module's own code calls; the ``x as x`` redundant aliases are the explicit
# re-export marker for the rest of the pre-split surface, privates included
# (``tests/test_raster.py`` imports ``_geo_from_ifd``/``_store_and_path``,
# ``tests/test_raster_pipeline.py`` imports ``_shard_cell_range``).
# READ compat only: a ``monkeypatch.setattr`` against a name here rebinds the
# facade, not the submodule global its callers read — patch the owning
# submodule instead.
from zagg.processing.raster.decode import _DTYPES as _DTYPES
from zagg.processing.raster.decode import _S3_VHOST as _S3_VHOST
from zagg.processing.raster.decode import _STORE_CACHE as _STORE_CACHE
from zagg.processing.raster.decode import _STORE_LOCK as _STORE_LOCK
from zagg.processing.raster.decode import _build_store as _build_store
from zagg.processing.raster.decode import (
    _chord2,
    _iso_us,
    _run_sync,
    _us_iso,
    new_stage_stats,
    raster_time_index,
    sample_asset,
    sample_asset_async,
    sample_item_async,
)
from zagg.processing.raster.decode import _geo_from_ifd as _geo_from_ifd
from zagg.processing.raster.decode import _raster_center_lonlat as _raster_center_lonlat
from zagg.processing.raster.decode import _sample_one as _sample_one
from zagg.processing.raster.decode import _store_and_path as _store_and_path
from zagg.processing.raster.template import _TIME_ATTRS as _TIME_ATTRS
from zagg.processing.raster.template import _check_raster_grid as _check_raster_grid
from zagg.processing.raster.template import _raster_array_spec as _raster_array_spec
from zagg.processing.raster.template import _raster_members as _raster_members
from zagg.processing.raster.template import (
    emit_raster_leaf_template,
    emit_raster_template,
    raster_group_spec,
    raster_leaf_spec,
)
from zagg.processing.raster.write import _shard_cell_range as _shard_cell_range
from zagg.processing.raster.write import (
    write_raster_coords,
    write_raster_leaf_slab,
    write_raster_slab,
)

# Default cap on how many acquisition groups sample concurrently per shard
# (issue #231). One knob per pipeline family (issue #232): ``shard_workers`` is
# "source units in flight per shard" — granules on the spatial path, acquisition
# groups (timesteps) here — mirroring the spatial default of 4: every
# in-flight group holds one timestep's decoded COG tiles + per-band gather
# buffers, so the cap bounds peak sampling memory to ~K timesteps instead of
# all T at once, while still overlapping S3 fetches at fine orders.
_DEFAULT_SHARD_WORKERS = 4


def _shard_workers(config) -> int:
    """``data_source.shard_workers``: acquisition groups in flight per shard.

    Bounds the :class:`asyncio.Semaphore` over timesteps in
    :func:`process_raster_shard` (issue #231). Default 4; ``1`` samples one
    timestep at a time. Re-checked here with the same int>=1 / bool-trap guard
    ``validate_config`` applies at submission, so a hand-rolled worker payload
    fails loudly rather than passing a bad width to ``Semaphore``.
    """
    k = (config.data_source or {}).get("shard_workers", _DEFAULT_SHARD_WORKERS)
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError(f"data_source.shard_workers must be an integer >= 1 (got {k!r})")
    return k


# Streamed-write slab budget (PR #232 review): ``1`` is the strict serial
# bound — a completed slab is written and freed before the next group drains.
# ``N`` allows N-1 writes in flight on worker threads while the next slab
# builds, so peak output memory holds <= N slabs; write latency then overlaps
# sampling instead of serializing against it.
_DEFAULT_WRITE_BUFFER = 1


def _write_buffer(config) -> int:
    """``data_source.write_buffer``: max slabs alive under a streamed sink.

    Only meaningful when ``process_raster_shard`` runs with ``on_slab``;
    dict-mode accumulation ignores it. Same int>=1 / bool-trap guard as the
    sibling worker knobs.
    """
    n = (config.data_source or {}).get("write_buffer", _DEFAULT_WRITE_BUFFER)
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError(f"data_source.write_buffer must be an integer >= 1 (got {n!r})")
    return n


def process_raster_shard(
    grid,
    shard_key: int,
    granules: list,
    config,
    time_index: dict,
    *,
    region: str | None = None,
    anonymous: bool = True,
    on_slab=None,
    stage_stats: dict | None = None,
    occupied_out: list | None = None,
):
    """Process one shard of a raster pipeline: every acquisition group -> slab.

    For each timestep, every covering item is sampled (pull-NN over the
    shard's cells); where several items cover a cell — the MGRS overlap — the
    cell takes the value from the item whose tile center is nearest
    (espg-ratified ownership rule on issue #218; subsumes same-zone dedupe).

    Invariant: ``time_index`` MUST be built from the same manifest the shards
    were dispatched from (see :func:`raster_time_index`). A granule whose
    acquisition-group key is absent raises :class:`ValueError` naming the key,
    rather than failing with an opaque ``KeyError`` deep in the gather.

    Parameters
    ----------
    on_slab : callable, optional
        ``on_slab(t_idx, slab)`` sink invoked once per timestep as its
        acquisition group completes (issue #231). When given, the slab is
        handed off and dropped immediately, so peak output memory holds
        ``data_source.write_buffer`` slabs (default 1: written + freed before
        the next group drains; ``N`` runs up to ``N-1`` sink calls on worker
        threads so write latency overlaps sampling — the PR #232
        double-buffer); the returned ``slabs`` is then empty. When ``None``
        (the default) every slab accumulates into ``slabs`` and is returned,
        as before, and ``write_buffer`` is ignored.
    stage_stats : dict, optional
        Per-invoke stage accumulator from :func:`new_stage_stats` (issue
        #249): the sample path adds per-stage seconds (``open`` / ``geometry``
        / ``fetch`` / ``decode`` / ``gather``) and counts (``assets`` /
        ``tiles`` / ``geom_hits``) in place. Stage seconds are work volume,
        not a wall decomposition — concurrent samples overlap, so their sum
        can exceed this call's wall (see :func:`new_stage_stats`). ``None``
        (the default) makes no timing calls — the sample path is unchanged.
    occupied_out : list, optional
        When given, receives one uint64 array of the shard's OCCUPIED cell
        words — cells valid in at least one timestep, i.e. the spatial union
        across the acquisitions sampled (the D14 coverage input; per-timestep
        validity stays data-plane nodata, D9). Mirrors ``process_shard``'s
        seam of the same name. ``None`` (the default) allocates nothing.

    Returns
    -------
    (slabs, metadata)
        ``slabs``: ``{t_idx: {field: values}}`` — one dense per-cell array per
        band per timestep, fill where no valid source (empty when ``on_slab``
        streamed them). ``metadata``: counts (``granule_count``, ``skipped``,
        ``timesteps``, ``shard_key``).
    """
    from zagg.config import get_raster_bands

    cells = grid.children(int(shard_key))
    bands = get_raster_bands(config)
    nodata = config.data_source.get("nodata")

    groups: dict = {}
    skipped = 0
    for e in granules:
        if not e.get("assets"):
            skipped += 1
            continue
        groups.setdefault(e.get("time_key") or e["datetime"], []).append(e)

    for key in groups:
        if key not in time_index:
            raise ValueError(
                f"acquisition group key {key!r} is absent from the passed "
                "time_index; time_index must be built from the same manifest "
                "the shards were dispatched from — see raster_time_index"
            )

    # One event loop per shard, but bound how many acquisition groups sample
    # concurrently: an ``asyncio.Semaphore(K)`` over the timesteps caps peak
    # memory at ~K in-flight timesteps of decoded COG tiles + per-band gather
    # buffers instead of all T at once (issue #231). Within a bounded group
    # every item still fans out concurrently — adjacent MGRS tiles of one
    # datatake are seconds apart, so the wall-clock overlap survives at fine
    # orders.
    k = _shard_workers(config)
    wb = _write_buffer(config)
    # Pull-NN geometry memo (issue #244), scoped to THIS invoke: the (rows,
    # cols, valid) mapping embeds the shard's cells, so per-invoke scoping
    # makes cross-shard collisions impossible by construction. A full-year
    # Sentinel-2 shard has exactly two distinct source grids (10 m bands,
    # 20 m scl) — this turns 425 geometry computations into 2.
    geom_cache: dict = {}
    # Occupied-cell union (issue #247): OR of per-timestep validity across the
    # shard's acquisition groups — the D14 coverage input. Accumulation is an
    # in-place index-assign on the event loop (no await between read and
    # write, and no name rebinding into the coroutine scope), atomic by the
    # same argument as the geom_cache store; allocated only when a sink was
    # passed, so the default path is unchanged.
    occupied_acc = np.zeros(len(cells), dtype=bool) if occupied_out is not None else None
    # Read-volume counters (issue #297): always-on inputs for the stats
    # record — compressed bytes fetched, pixels decoded (whole tiles), and
    # cell samples gathered. Their decoded/sampled ratio reads as the extract's
    # read-time over-provision only when the output grid is coarser than the
    # source; a finer grid (more cells than source pixels) inverts it below 1.
    # Stored raw (associative sums), never as a ratio, per the
    # mergeable-by-construction schema rule.
    io_stats = {"bytes_read": 0, "px_decoded": 0, "px_sampled": 0}

    async def _run_all():
        sem = asyncio.Semaphore(k)

        async def _sample_group(key, items):
            async with sem:
                sampled = await asyncio.gather(
                    *[
                        sample_item_async(
                            grid,
                            cells,
                            e["assets"],
                            bands,
                            nodata=nodata,
                            region=region,
                            anonymous=anonymous,
                            geom_cache=geom_cache,
                            stage_stats=stage_stats,
                            io_stats=io_stats,
                        )
                        for e in items
                    ]
                )
            return time_index[key], sampled

        lonlat = None  # computed once, only if some timestep has overlapping items
        slabs: dict = {}
        # Streamed-sink hand-off. At the default ``write_buffer`` of 1 the
        # sink runs synchronously in the loop: a completed slab is written +
        # freed before the next group drains (the strict issue #231 bound).
        # At N>1 (the PR #232 double-buffer) up to N-1 sink calls run on
        # worker threads while the next slab builds — <= N slabs alive, write
        # latency overlapped with sampling. A sink error surfaces at most one
        # group late, at the next hand-off (or the final drain below).
        pending: list = []

        async def _emit(t, slab):
            if wb <= 1:
                on_slab(t, slab)
                return
            while len(pending) >= wb - 1:
                await pending.pop(0)
            pending.append(asyncio.create_task(asyncio.to_thread(on_slab, t, slab)))

        # Drain groups as they finish (as_completed): build each timestep's slab
        # and hand it to the sink — the output side stays ~write_buffer
        # timesteps (issues #231/#232).
        try:
            for fut in asyncio.as_completed(
                [_sample_group(key, items) for key, items in groups.items()]
            ):
                t, sampled = await fut
                if len(sampled) == 1:
                    values, valid, _center = sampled[0]
                else:
                    if lonlat is None:
                        lonlat = grid.cell_lonlat(cells)
                    values, valid = _combine_by_ownership(sampled, lonlat, bands)
                if occupied_acc is not None:
                    occupied_acc[valid] = True
                slab = {}
                for f, v in values.items():
                    out = v.copy()  # keep the asset dtype (np.where would promote)
                    out[~valid] = bands[f]["fill_value"]
                    slab[f] = out
                if on_slab is not None:
                    await _emit(t, slab)
                else:
                    slabs[t] = slab
        except BaseException:
            # Reap in-flight writes before propagating the primary error, so
            # no task is left un-awaited (their own errors are secondary here).
            await asyncio.gather(*pending, return_exceptions=True)
            raise
        if pending:
            await asyncio.gather(*pending)  # propagate any trailing write error
        return slabs

    slabs = _run_sync(_run_all())
    if occupied_out is not None:
        occupied_out.append(np.asarray(cells, dtype=np.uint64)[occupied_acc])
    metadata = {
        "shard_key": int(shard_key),
        "granule_count": len(granules),
        "skipped": skipped,
        "timesteps": len(groups),
        # Read-volume counters (issue #297) for the stats record.
        "raster_bytes_read": io_stats["bytes_read"],
        "raster_px_decoded": io_stats["px_decoded"],
        "raster_px_sampled": io_stats["px_sampled"],
    }
    return slabs, metadata


def _combine_by_ownership(sampled, lonlat, bands):
    """Nearest-tile-center combine across one timestep's overlapping items."""
    lons, lats = lonlat
    dists = np.stack(
        [_chord2(lons, lats, *center) for _v, _m, center in sampled]
    )  # (n_items, n_cells)
    valid_stack = np.stack([m for _v, m, _c in sampled])
    dists[~valid_stack] = np.inf
    owner = np.argmin(dists, axis=0)
    any_valid = valid_stack.any(axis=0)
    values = {}
    for f in bands:
        stack = np.stack([v[f] for v, _m, _c in sampled])
        values[f] = stack[owner, np.arange(stack.shape[1])]
    return values, any_valid


def process_and_write_raster_hive(
    shard_key,
    granules,
    grid,
    store_root: str,
    config,
    *,
    store_kwargs: dict,
    window: dict | None = None,
    profile: bool = False,
    region: str | None = None,
    anonymous: bool = True,
    stage_stats: dict | None = None,
):
    """Process one raster shard into its own hive leaf store (issue #247).

    The raster analog of :func:`zagg.hive.process_and_write_hive` — the
    SHARED per-(shard, window) write path for both dispatchers, so leaf
    templating, slab placement, coverage, and stamp ordering cannot drift
    between backends. The unit's output is a self-describing leaf zarr at
    :func:`zagg.hive.shard_leaf_path` (windowed name when ``window`` is
    given, the bare schedule-``none`` leaf otherwise, D13), whose time axis
    is the unit's OWN acquisition groups (:func:`raster_time_index` over the
    dispatched subset — deterministic, so both dispatchers produce identical
    leaves). The leaf template is emitted lazily on the first slab
    (``overwrite=True``): a no-data unit never creates the ``.zarr/`` prefix,
    a torn worker leaves an UNSTAMPED prefix (debris, D4), and a re-run
    replaces the leaf wholesale (the D13 append/idempotency story).

    ``window`` is the dispatch unit's ``{"label", ...}`` payload. Membership
    was decided AT DISPATCH — the acquisition group's STAC ``datetime``, the
    ratified issue #247 rule — so unlike the aggregation path no
    observation-level filter is injected; the window selects the leaf name,
    arms the D14 popcount (``encoding: "full"`` — gated off on ``None``
    exactly as aggregation gates it for schedule-none stores), and adds the
    D15 stamp truth: the window label plus the ACTUAL ISO-UTC ``[min, max]``
    of the unit's acquisition datetimes (also returned as
    ``metadata["time_range"]`` for the dispatcher's root-summary union).

    The stamp is the leaf's FINAL write: dense slabs (streamed) -> coverage
    sidecar (edge shards only; interior shards stamp ``"full"`` with no
    sidecar PUT) -> stamp. ``cells_with_data`` counts the occupied-cell
    union; ``granule_count`` the unit's acquisitions (asset-carrying
    entries). Phase timings are always collected (issue #297):
    ``metadata["phase_timings"] = {"sample", "write"}`` with the leaf
    write-out (template + slabs + sidecar + stamp) as ``write``; the
    per-stage ``stages`` block (issue #249) stays gated on ``profile`` /
    a passed ``stage_stats`` (the local dispatcher's debug-logging flavor).
    """
    from zagg.hive import (
        COVERAGE_SIDECAR,
        build_coverage,
        encode_coverage_bitmap,
        shard_leaf_path,
        stamp_commit,
        write_coverage_sidecar,
    )
    from zagg.store import open_store

    t_start = time.time()
    if profile and stage_stats is None:
        stage_stats = new_stage_stats()
    label = window["label"] if window else None
    leaf_path = shard_leaf_path(store_root, int(shard_key), window=label)
    # The leaf's own time axis, from the dispatched subset. Every group key in
    # the subset is in this index by construction, so the worker never trips
    # the foreign-manifest guard.
    time_index, times_us = raster_time_index([granules])
    box: dict = {}
    write_s = 0.0

    def _leaf():
        if "store" not in box:
            store = open_store(leaf_path, **store_kwargs)
            # overwrite=True: an existing prefix is debris from a torn run
            # (D4) or a prior committed leaf being redone (D13 re-run) — both
            # replaced wholesale; per-leaf state never blocks a retry. The
            # overwrite enumeration warns about the prior attempt's coverage
            # sidecar — the one foreign key we put there ourselves — so that
            # specific warning is expected and suppressed (the
            # process_and_write_hive posture); anything else stays loud.
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=f"Object at {COVERAGE_SIDECAR}")
                emit_raster_leaf_template(
                    store, grid, config, int(shard_key), times_us, overwrite=True
                )
            box["store"] = store
        return box["store"]

    def _write_slab(t_idx, slab):
        nonlocal write_s
        _t0 = time.time()
        write_raster_leaf_slab(_leaf(), grid, t_idx, slab)
        write_s += time.time() - _t0

    occupied: list = []
    _slabs, meta = process_raster_shard(
        grid,
        int(shard_key),
        granules,
        config,
        time_index,
        region=region,
        anonymous=anonymous,
        on_slab=_write_slab,
        stage_stats=stage_stats,
        occupied_out=occupied,
    )
    # Stamp ONLY a leaf that wrote slabs: a unit that streamed nothing has no
    # prefix, and a worker error raised out above, leaving debris (D4). Write
    # order is pinned: dense slabs -> coverage sidecar -> stamp.
    meta["cells_with_data"] = 0
    # Accurate leaf-written signal for the stats-sidecar gate (issue #297): set
    # iff a slab streamed (``"store" in box``), so both dispatchers gate the
    # sidecar PUT on leaf existence rather than the ``timesteps`` proxy (a unit
    # with acquisitions but no occupied cell writes no leaf). ``phase_timings``
    # cannot serve as the gate — it rides only under ``profile``.
    meta["leaf_written"] = "store" in box
    if "store" in box:
        _t0 = time.time()
        words = occupied[0] if occupied and occupied[0].size else None
        # D14 popcount: a fully-occupied subtree stamps encoding "full" — no
        # sidecar PUT. Gated on a windowed unit (/2 stores) so schedule-none
        # output mirrors aggregation's gate (hive.process_and_write_hive).
        depth = int(grid.child_order) - int(grid.parent_order)
        full = window is not None and words is not None and np.unique(words).size == 4**depth
        bitmap = None
        if words is not None and not full and depth > 0:
            bitmap = encode_coverage_bitmap(int(shard_key), words, grid.child_order)
            write_coverage_sidecar(leaf_path, bitmap, **store_kwargs)
        # D15 truth: the actual acquisition extent written, as ISO-UTC — the
        # min/max STAC datetime over the unit's asset-carrying entries (item
        # instants, not group coordinates, so adjacent-tile spreads count).
        time_range = None
        if window is not None:
            instants = [_iso_us(e["datetime"]) for e in granules if e.get("assets")]
            if instants:
                time_range = [_us_iso(min(instants)), _us_iso(max(instants))]
                meta["time_range"] = time_range
        meta["cells_with_data"] = int(words.size) if words is not None else 0
        stamp_commit(
            box["store"],
            cells_with_data=meta["cells_with_data"],
            granule_count=meta["granule_count"] - meta["skipped"],
            coverage=build_coverage(
                int(shard_key), words, grid.child_order, bitmap=bitmap, full=full
            ),
            window=label,
            time_range=time_range,
        )
        write_s += time.time() - _t0
    # Phase split (issues #100/#249; always-on collection since issue #297 —
    # the stats sidecar needs complete timings by default): only a unit that
    # actually wrote carries it, so a no-data unit stays write-less and
    # sample/write always decompose this call's wall. The per-stage ``stages``
    # block stays verbosity, gated on profiling/debug (a passed stage_stats).
    if "store" in box:
        meta["phase_timings"] = {
            "sample": (time.time() - t_start) - write_s,
            "write": write_s,
        }
        if stage_stats is not None:
            meta["phase_timings"]["stages"] = stage_stats
    return meta


__all__ = [
    "new_stage_stats",
    "sample_asset",
    "sample_asset_async",
    "sample_item_async",
    "raster_time_index",
    "process_raster_shard",
    "process_and_write_raster_hive",
    "raster_group_spec",
    "raster_leaf_spec",
    "emit_raster_template",
    "emit_raster_leaf_template",
    "write_raster_slab",
    "write_raster_leaf_slab",
    "write_raster_coords",
]
