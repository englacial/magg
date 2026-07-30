"""Raster slab and coordinate writers (issue #330).

Split out of the single-file ``zagg.processing.raster`` (issue #330 phase 2);
the public surface is unchanged and re-exported from
:mod:`zagg.processing.raster`. Plain Zarr region assignment — the raster output
path bypasses the aggregation write machinery by construction.
"""

from __future__ import annotations

import numpy as np


def write_raster_leaf_slab(store, grid, t_idx: int, slab: dict):
    """Write one timestep's slab at LEAF-LOCAL indices: ``array[t, :] = values``.

    The leaf's arrays span exactly one shard, so the cell axis needs no
    block offset (contrast :func:`write_raster_slab`); ``t_idx`` is the
    leaf-local timestep from the leaf's own time index. Chunk-aligned by
    construction (whole rows of ``(1, cells_per_chunk)`` chunks).
    """
    from zarr import config as zarr_config
    from zarr import open_array

    with zarr_config.set({"async.concurrency": 128}):
        for name, values in slab.items():
            arr = open_array(
                store, path=f"{grid.group_path}/{name}", zarr_format=3, consolidated=False
            )
            arr[int(t_idx), :] = np.asarray(values, dtype=arr.dtype)
    return store


def _shard_cell_range(grid, shard_key: int) -> tuple[int, int]:
    """The shard's contiguous cell-axis extent ``[start, stop)``.

    ``block_index`` is the parent's nested id — the shard's block on a
    ``cells_per_shard``-strided axis.
    """
    start = int(grid.block_index(int(shard_key))[0]) * grid.cells_per_shard
    return start, start + grid.cells_per_shard


def write_raster_slab(store, grid, shard_key: int, t_idx: int, slab: dict):
    """Write one timestep x shard slab: ``array[t, start:stop] = values``.

    Chunk-aligned by construction (``start`` is a multiple of
    ``cells_per_chunk``), so no read-modify-write.
    """
    from zarr import config as zarr_config
    from zarr import open_array

    start, stop = _shard_cell_range(grid, shard_key)
    with zarr_config.set({"async.concurrency": 128}):
        for name, values in slab.items():
            arr = open_array(
                store, path=f"{grid.group_path}/{name}", zarr_format=3, consolidated=False
            )
            arr[int(t_idx), start:stop] = np.asarray(values, dtype=arr.dtype)
    return store


def write_raster_coords(store, grid, shard_key: int):
    """Write the shard's ``morton`` coordinate block (once per shard).

    Packed u64 words — the sole stored cell coordinate (D16, issue #304);
    the legacy NESTED ``cell_ids`` block rides only the ``emit_cell_ids``
    transition hatch, exactly like the spatial path.
    """
    from zarr import config as zarr_config
    from zarr import open_array

    start, stop = _shard_cell_range(grid, shard_key)
    children = np.asarray(grid.children(int(shard_key)), dtype=np.uint64)
    with zarr_config.set({"async.concurrency": 128}):
        arr = open_array(store, path=f"{grid.group_path}/morton", zarr_format=3, consolidated=False)
        arr[start:stop] = children
        if grid.emit_cell_ids:
            arr = open_array(
                store, path=f"{grid.group_path}/cell_ids", zarr_format=3, consolidated=False
            )
            arr[start:stop] = np.asarray(grid.encode_cell_ids(children), dtype=np.uint64)
    return store
