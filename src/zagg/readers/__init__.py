"""Client-side read helpers for zagg products (issue #79).

These sit just outside the core write path: they reconstruct dense, fixed-size
arrays a downstream client can consume from the gridded Zarr products zagg
writes.  Pure-numpy + zarr (both already core deps); no new dependency.
"""

from __future__ import annotations

from zagg.readers._layout import rank_to_rowcol, rowcol_to_rank
from zagg.readers.tdigest_tensor import (
    cell_index,
    chunk_z_range,
    has_exact_occupancy,
    rasterize_cell,
    read_cell,
    read_locations,
    read_raw_values,
    read_tensors,
)

# The raster ``(time, cells)`` family's time coordinate (spec §8, issue
# #443). Re-exported rather than reimplemented: the encode and decode halves
# of one wire contract belong in one module (``zagg.time_axis``), and the
# writer imports the same constants.
from zagg.time_axis import decode_time_axis, read_time_axis, time_axis_overlaps

__all__ = [
    "cell_index",
    "chunk_z_range",
    "decode_time_axis",
    "has_exact_occupancy",
    "rank_to_rowcol",
    "rasterize_cell",
    "read_cell",
    "read_locations",
    "read_raw_values",
    "read_tensors",
    "read_time_axis",
    "rowcol_to_rank",
    "time_axis_overlaps",
]
