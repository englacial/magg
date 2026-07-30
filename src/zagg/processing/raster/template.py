"""Raster (time, cell) template and band-map specs (issue #330).

Split out of the single-file ``zagg.processing.raster`` (issue #330 phase 2);
the public surface is unchanged and re-exported from
:mod:`zagg.processing.raster`. Holds the flat store template, the hive leaf
spec, and the band -> array mapping both derive from.
"""

from __future__ import annotations

import numpy as np

# ── lean (time, cell) template + slab writer (issue #218) ────────────────────
#
# Pull-NN emits one dense slab per (timestep x shard) by construction, so the
# raster output path is a plain Zarr region assignment — it BYPASSES the
# aggregation write machinery (carriers / reductions / ragged) rather than
# threading a time dimension through it. Appends are the standard Zarr
# resize-then-write-slab pattern and are single-writer (the runner owns the
# resize, as it owns template emission).

_TIME_ATTRS = {"units": "microseconds since 1970-01-01T00:00:00", "calendar": "proleptic_gregorian"}


def _raster_array_spec(shape, chunks, dims, dtype, fill, attrs=None):
    """ArraySpec shared by the flat template and the hive leaf spec."""
    from pydantic_zarr.experimental.v3 import ArraySpec, NamedConfig

    return ArraySpec(
        attributes=attrs or {},
        shape=shape,
        dimension_names=dims,
        data_type=dtype,
        chunk_grid=NamedConfig(name="regular", configuration={"chunk_shape": chunks}),
        chunk_key_encoding=NamedConfig(name="default", configuration={"separator": "/"}),
        codecs=(
            NamedConfig(name="bytes", configuration={"endian": "little"}),
            NamedConfig(name="zstd", configuration={"level": 3, "checksum": False}),
        ),
        storage_transformers=(),
        fill_value=fill,
    )


def _check_raster_grid(grid) -> None:
    """Shared template guards: no sharded output, 1-D cell axis only."""
    if getattr(grid, "sharded", False):
        # Permanent exclusion (espg-ratified on issue #247), mirroring the
        # validate_config message: per-timestep slab streaming would
        # read-modify-write each ShardingCodec object.
        raise ValueError(
            "raster templates do not support sharded output (per-timestep slab "
            "streaming would read-modify-write each ShardingCodec object)"
        )
    if len(grid.array_shape) != 1:
        raise ValueError(
            "raster templates currently require a 1-D cell axis (HEALPix); "
            "the rectilinear (time, y, x) variant is future work (issue #218)"
        )


def _raster_members(grid, config, n_time: int, n_cells: int) -> dict:
    """The ``time``/``morton``/band ArraySpec members for one raster store.

    ``morton`` (packed u64 words) is the sole stored cell coordinate — the
    D16 flip applies to the raster path too (espg-ruled on the PR #314
    review: one default cell coordinate everywhere). The legacy NESTED
    ``cell_ids`` array rides only the same ``emit_cell_ids`` transition
    hatch as the spatial path — never a separate schedule.
    """
    from zagg.config import get_raster_bands

    members = {
        "time": _raster_array_spec(
            (n_time,), (max(n_time, 1),), ("time",), "int64", 0, dict(_TIME_ATTRS)
        ),
        "morton": _raster_array_spec((n_cells,), (grid.cells_per_chunk,), ("cells",), "uint64", 0),
    }
    if grid.emit_cell_ids:
        members["cell_ids"] = _raster_array_spec(
            (n_cells,), (grid.cells_per_chunk,), ("cells",), "uint64", 0
        )
    for name, meta in get_raster_bands(config).items():
        members[name] = _raster_array_spec(
            (n_time, n_cells),
            (1, grid.cells_per_chunk),
            ("time", "cells"),
            meta["dtype"],
            meta["fill_value"],
            meta["attrs"] or {},
        )
    return members


def raster_group_spec(grid, config, n_time: int):
    """pydantic-zarr GroupSpec for the raster ``(time, cells)`` template.

    Per band: shape ``(n_time, n_pixels)``, chunks ``(1, cells_per_chunk)`` —
    one storage object per (timestep, chunk), so per-date rewrites are exact.
    Plus ``time`` (int64 microseconds, CF attrs) and ``morton`` (packed u64
    words, written per shard by :func:`write_raster_coords`). The group
    carries the same morton-declared dggs attrs block as the spatial path
    (issues #304/#305): one reader contract for every store.
    """
    from pydantic_zarr.experimental.v3 import GroupSpec

    _check_raster_grid(grid)
    n_pixels = int(np.prod(grid.array_shape))
    return GroupSpec(
        members=_raster_members(grid, config, n_time, n_pixels),
        attributes=grid._dggs_attrs(),
    )


def emit_raster_template(store, grid, config, times_us: np.ndarray, *, overwrite: bool = False):
    """Write the raster template and its ``time`` coordinate values."""
    from zarr import config as zarr_config
    from zarr import open_array
    from zarr.errors import ArrayNotFoundError, ContainsGroupError

    times_us = np.asarray(times_us, dtype=np.int64)
    spec = raster_group_spec(grid, config, int(len(times_us)))
    time_path = f"{grid.group_path}/time"
    with zarr_config.set({"async.concurrency": 128}):
        if not overwrite:
            # ``to_zarr(overwrite=False)`` only refuses a template whose SPEC
            # differs (a changed timestep COUNT -> different ``time`` shape ->
            # ContainsGroupError). A store already holding a same-length but
            # different-valued time axis slips past it, and the unconditional
            # ``arr[:]`` below would silently rewrite the coordinate the
            # workers slab-write against. Refuse that too, so overwrite=False
            # uniformly won't clobber a differing template (issue #264).
            try:
                existing = open_array(store, path=time_path, zarr_format=3, consolidated=False)
            except ArrayNotFoundError:
                existing = None
            if existing is not None and not np.array_equal(existing[:], times_us):
                raise ContainsGroupError(store, grid.group_path)
        spec.to_zarr(store, grid.group_path, overwrite=overwrite)
        arr = open_array(store, path=time_path, zarr_format=3, consolidated=False)
        arr[:] = times_us
    return store


def raster_leaf_spec(grid, config, n_time: int):
    """GroupSpec for ONE shard's hive leaf zarr (issue #247, D3/D13).

    The raster analog of ``HealpixGrid.shard_spec``: the same member set as
    :func:`raster_group_spec` — ``time``/``morton`` plus one ``(time,
    cells)`` array per band, same dtypes/fills/chunking — with the cells axis
    sized to a single shard (``cells_per_shard``) and the time axis to the
    LEAF's own acquisitions (``n_time`` = the groups intersecting this shard
    × window, known at dispatch from the catalog). Wrapped in a ROOT group
    (members under ``grid.group_path``, mirroring ``emit_shard_template``) so
    the D4 commit stamp is one attrs update on an object that exists anyway.
    """
    from pydantic_zarr.experimental.v3 import GroupSpec

    _check_raster_grid(grid)
    inner = GroupSpec(
        members=_raster_members(grid, config, n_time, grid.cells_per_shard),
        # The same morton-declared dggs attrs as the spatial leaf (issue
        # #304 — one reader contract), on the inner group like
        # HealpixGrid._group_spec.
        attributes=grid._dggs_attrs(),
    )
    return GroupSpec(members={grid.group_path: inner}, attributes={})


def emit_raster_leaf_template(
    store, grid, config, shard_key: int, times_us: np.ndarray, *, overwrite: bool = False
):
    """Write one leaf's template plus its ``time`` and ``morton`` coords.

    Unlike the flat path (template at fan-out time, coords per shard after
    the slabs), a leaf's coordinates are fully known at template time — the
    time axis is the leaf's own acquisition groups and ``morton`` is the
    shard's children (packed words; the legacy ``cell_ids`` only under the
    ``emit_cell_ids`` hatch) — so both are written here, once. Called lazily on the
    first slab (mirroring ``process_and_write_hive``'s lazy ``_leaf``) with
    ``overwrite=True`` so a no-data shard never creates the ``.zarr/`` prefix
    and a retry replaces debris wholesale (D4).
    """
    from zarr import config as zarr_config
    from zarr import open_array

    spec = raster_leaf_spec(grid, config, int(len(times_us)))
    children = np.asarray(grid.children(int(shard_key)), dtype=np.uint64)
    with zarr_config.set({"async.concurrency": 128}):
        spec.to_zarr(store, "", overwrite=overwrite)
        arr = open_array(store, path=f"{grid.group_path}/time", zarr_format=3, consolidated=False)
        arr[:] = np.asarray(times_us, dtype=np.int64)
        arr = open_array(store, path=f"{grid.group_path}/morton", zarr_format=3, consolidated=False)
        arr[:] = children
        if grid.emit_cell_ids:
            arr = open_array(
                store, path=f"{grid.group_path}/cell_ids", zarr_format=3, consolidated=False
            )
            arr[:] = np.asarray(grid.encode_cell_ids(children), dtype=np.uint64)
    return store
