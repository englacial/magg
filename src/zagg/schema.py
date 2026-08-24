"""Compatibility wrappers for HEALPix Zarr template emission.

The actual implementation lives on ``zagg.grids.HealpixGrid``. These functions
preserve the pre-refactor public API (fullsphere only; the deprecated
``n_parent_cells`` dense pack was removed — issue #88).
"""

from __future__ import annotations

from pydantic_zarr.experimental.v3 import GroupSpec
from typing_extensions import NotRequired, TypedDict
from zarr.abc.store import Store

from zagg.config import PipelineConfig
from zagg.grids.healpix import HEALPIX_BASE_CELLS, HealpixGrid


class ProcessingMetadata(TypedDict):
    shard_key: int
    cells_with_data: int
    total_obs: int
    granule_count: int
    # Base-rate rows DECODED before filtering/assignment (issue #374), summed
    # over the shard's granules — always >= ``total_obs``, whose difference is
    # the read-plan padding, boundary-straddling segments, and filter rejects.
    # ``NotRequired``: present only when a read route measured it, so absence
    # reads as unmeasured (a stubbed read seam, or a worker predating the
    # field) rather than a genuine zero.
    total_obs_read: NotRequired[int]
    files_processed: int
    duration_s: float
    error: str | None
    # Peak resident memory (RSS) of the worker process in MB. Stamped by the
    # Lambda handler from ``resource.getrusage`` after the write phase (issue
    # #120); absent on the local runner path, hence ``NotRequired``.
    max_memory_mb: NotRequired[float]
    # Per-phase wall timings (read/index/aggregate/write), present only when the
    # worker is dispatched with ``profile=True`` (issue #100).
    phase_timings: NotRequired[dict[str, float]]
    # Count of per-group reads that raised during the read loop (issue #116).
    # Present (and non-zero) only when at least one group read failed; a raised
    # read is always a real error, so this surfaces a shard whose "no data"
    # result is actually a read failure rather than a legitimately-empty read.
    read_errors: NotRequired[int]
    # Count of GRANULE-scope read failures (issue #341): the granule itself
    # failed (H5Coro open, credentials, URL rewrite) or its fold did, so no
    # group of it was read. Warn-and-continue like ``read_errors``, but a
    # different diagnosis, hence a separate counter. Present only when non-zero.
    granule_errors: NotRequired[int]
    # Count of read failures that are DEFINITELY credentials-shaped (issue
    # #449): a botocore 401/403, or a denial token in the exception text. A
    # subset of ``read_errors``/``granule_errors``, split out because the
    # diagnosis inverts — a denied read is a CONFIGURATION fault (wrong
    # ``data_source.credentials_provider`` for the DAAC hosting the product),
    # so retrying the shard just burns another invoke. Present only when
    # non-zero, INCLUDING when the shard still produced data (a partially
    # denied read is a partial product — the run summary warns on that case).
    # The ambiguous empty-body shape is deliberately not counted here; it is a
    # hint in ``error`` instead (fold review — see ``is_empty_body_failure``).
    auth_errors: NotRequired[int]
    # First N distinct read exception messages, truncated (issue #341): the
    # bounded diagnosis payload for ``read_errors``/``granule_errors`` — present
    # exactly when either is. Messages only; tracebacks stay in the worker log.
    read_error_exemplars: NotRequired[list[str]]
    # Container telemetry (issue #171), stamped by the Lambda handler's
    # dispatcher into every per-unit envelope (all status branches) so the
    # runner can surface the warm-container RSS ratchet (#169). Absent on the
    # local runner path. ``container_cold``: first invocation on this sandbox.
    # ``container_generation``: invocations this sandbox has served (all
    # modes). ``rss_start_mb``: process RSS at handler entry -- the ratchet
    # signal (None off Linux). ``sandbox_id``: CloudWatch log-stream name,
    # unique per sandbox. ``container_init_ts``: module-import epoch seconds.
    # The per-invocation *peak* is the existing ``max_memory_mb`` (issue #141).
    container_cold: NotRequired[bool]
    container_generation: NotRequired[int]
    rss_start_mb: NotRequired[float | None]
    sandbox_id: NotRequired[str | None]
    container_init_ts: NotRequired[float]


def xdggs_spec(
    parent_order: int,
    child_order: int,
    config: PipelineConfig | None = None,
) -> GroupSpec:
    """Return the full-sphere HEALPix GroupSpec (back-compat wrapper)."""
    return HealpixGrid(
        parent_order=parent_order,
        child_order=child_order,
        layout="fullsphere",
        config=config,
    ).spec()


def xdggs_zarr_template(
    store: Store,
    parent_order: int,
    child_order: int,
    n_parent_cells: int | None = None,
    overwrite: bool = False,
    config: PipelineConfig | None = None,
) -> Store:
    """Write a full-sphere HEALPix Zarr template to ``store``.

    The array has shape ``(12 · 4^child_order,)``.

    Parameters
    ----------
    store : Store
        Zarr-compatible store.
    parent_order : int
        Parent (shard) HEALPix order.
    child_order : int
        Leaf HEALPix order. Must be ``>= parent_order``.
    n_parent_cells : int, optional
        Removed. Passing a value raises — the dense-pack layout it selected
        was removed (issue #88).
    overwrite : bool, optional
        Overwrite an existing array or group at the path.
    config : PipelineConfig, optional
        Pipeline configuration. Falls back to ``default_config("atl06")``.
    """
    if n_parent_cells is not None:
        raise ValueError(
            "xdggs_zarr_template(n_parent_cells=...) selected the dense-pack "
            "layout, which was removed (issue #88); omit n_parent_cells for "
            "the fullsphere template"
        )
    grid = HealpixGrid(
        parent_order=parent_order,
        child_order=child_order,
        layout="fullsphere",
        config=config,
    )
    return grid.emit_template(store, overwrite=overwrite)


__all__ = [
    "HEALPIX_BASE_CELLS",
    "ProcessingMetadata",
    "xdggs_spec",
    "xdggs_zarr_template",
]
