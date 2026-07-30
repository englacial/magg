"""Raster decode engine: async-tiff COG sampling at cell centers (issue #330).

Split out of the single-file ``zagg.processing.raster`` (issue #330 phase 2);
the public surface is unchanged and re-exported from
:mod:`zagg.processing.raster`. This is the read half — the process-wide obspec
store cache, IFD georeferencing, the per-asset tile fan-out, and the
acquisition-time index that groups items into timesteps. It never touches the
output side.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import os
import re
import threading
import time
from urllib.parse import urlparse

import numpy as np

# TIFF SampleFormat x BitsPerSample -> numpy dtype, for sizing the fill return
# when a shard has no valid cells (no tile fetched, so no decoded buffer to
# take a dtype from). 1=unsigned int, 2=signed int, 3=IEEE float.
_DTYPES = {
    (1, 8): np.uint8,
    (1, 16): np.uint16,
    (1, 32): np.uint32,
    (2, 8): np.int8,
    (2, 16): np.int16,
    (2, 32): np.int32,
    (3, 32): np.float32,
    (3, 64): np.float64,
}

_S3_VHOST = re.compile(r"(?P<bucket>.+)\.s3[.-](?P<region>[a-z0-9-]+)\.amazonaws\.com$")


def _geo_from_ifd(ifd) -> tuple[int, tuple[float, float, float, float, float, float]]:
    """``(epsg, affine)`` from a GeoTIFF IFD, STAC ``proj:transform`` order.

    Supports the ModelPixelScale + ModelTiepoint form (what COGs write); the
    full ModelTransformation matrix form raises rather than misgeoreference.
    """
    gkd = ifd.geo_key_directory
    if gkd is None:
        raise ValueError("not a GeoTIFF: no GeoKeyDirectory")
    epsg = gkd.projected_type or gkd.geographic_type
    if not epsg:
        raise ValueError("GeoKeyDirectory carries no projected or geographic EPSG code")
    scale, tie = ifd.model_pixel_scale, ifd.model_tiepoint
    if not scale or not tie:
        raise ValueError(
            "only ModelPixelScale+ModelTiepoint GeoTIFFs are supported "
            "(no pixel scale / tiepoint tags found)"
        )
    sx, sy = float(scale[0]), float(scale[1])
    i, j, _, x, y, _ = (float(v) for v in tie[:6])
    return int(epsg), (sx, 0.0, x - i * sx, 0.0, -sy, y + j * sy)


# Store cache (issue #244): one obspec store per (kind, bucket-or-host-or-dir,
# region, anonymous) per PROCESS. A fresh S3Store per asset-sample cost ~300 ms
# of client/TLS setup and made every tile GET ride a cold connection (425
# clients per full-year o9 invoke — the measured breakdown on the issue).
# Module lifetime == sandbox lifetime (espg-ratified): warm Lambda invocations
# keep their connection pools, matching the issue #171 sandbox-lifetime
# pattern. Lock-guarded because the running-loop fallback in ``sample_asset``
# and hand-rolled callers can construct from other threads; construction runs
# under the lock deliberately (single-flight per key).
#
# Credential lifetime (issue #244 review): every key deployed today is
# anonymous (``sentinel2_l2a.yaml`` sets ``anonymous: true``; runner.py:676
# and ``sample_asset*`` default it True), so the cached ``S3Store`` carries
# ``skip_signature=True`` and signs nothing — credentials never enter the
# picture and warm-caching a store cannot go stale. If a *signed* store
# (``anonymous=False``) were ever cached, sandbox-lifetime caching would only
# be safe because of how async-tiff's Rust ``object_store``-backed ``S3Store``
# resolves credentials, and that splits by environment:
#   - On AWS Lambda the execution-role credentials arrive as static env vars
#     valid for the whole sandbox lifetime and do NOT rotate mid-sandbox, so a
#     sandbox-lifetime store cannot outlive its creds — safe by construction.
#   - Off-Lambda (EC2 instance profile / IMDS, SSO), object_store resolves
#     through a caching credential *provider* that refreshes on expiry per
#     request rather than freezing a token at construction, so a warm store
#     keeps working across rotation.
# (The Lambda case is directly grounded; the off-Lambda refresh behavior
# reflects object_store's documented provider design — ``S3Store`` exposes a
# ``credential_provider`` slot — and was not source-verified against the
# installed binary wheel.) Caveat: no explicit-credential store exists here
# (the raster path is anonymous); do NOT add one to the cache without
# revisiting this, since a statically-supplied token would be frozen at
# construction and eventually go stale on a warm worker.
_STORE_CACHE: dict = {}
_STORE_LOCK = threading.Lock()


def _store_and_path(href: str, *, region: str | None = None, anonymous: bool = True):
    """obspec store + in-store path for an asset href.

    Handles ``s3://bucket/key``, virtual-hosted S3 HTTPS
    (``https://bucket.s3.region.amazonaws.com/key`` -- what Earth Search
    asset hrefs look like), plain HTTPS, and local paths. Stores are cached
    per ``(kind, location, region, anonymous)`` for the life of the process
    (issue #244) — the returned store is shared, never per-call.
    """
    u = urlparse(href)
    if u.scheme == "s3":
        key = ("s3", u.netloc, region, anonymous)
        path = u.path.lstrip("/")
    elif u.scheme in ("http", "https"):
        m = _S3_VHOST.match(u.netloc)
        if m:
            key = ("s3", m["bucket"], region or m["region"], anonymous)
            path = u.path.lstrip("/")
        else:
            key = ("http", f"{u.scheme}://{u.netloc}", None, None)
            path = u.path.lstrip("/")
    else:
        d, name = os.path.split(href)
        key = ("local", d or ".", None, None)
        path = name
    with _STORE_LOCK:
        store = _STORE_CACHE.get(key)
        if store is None:
            store = _build_store(key)
            _STORE_CACHE[key] = store
    return store, path


def _build_store(key):
    """Construct the obspec store for a cache key (see ``_STORE_CACHE``)."""
    from async_tiff.store import HTTPStore, LocalStore, S3Store

    kind, loc, region, anonymous = key
    if kind == "s3":
        kw: dict = {}
        if anonymous:
            kw["skip_signature"] = True
        if region:
            kw["region"] = region
        return S3Store(loc, **kw)
    if kind == "http":
        return HTTPStore(loc)
    return LocalStore(loc)


def new_stage_stats() -> dict:
    """Fresh per-invoke stage accumulator for the sample path (issue #249).

    The floats are wall-clock seconds (``time.time()`` deltas, the issue #100
    convention) of **stage work volume**: each asset-sample times its own
    stages independently and the K x bands concurrent samples of an invoke
    overlap on one event loop, so a stage total is the sum of per-sample
    elapsed walls *including* time suspended while sibling samples ran. The
    sums attribute where the samples' time went (which stage differs between
    a fast and a slow invoke) — they are NOT a wall decomposition and can
    exceed the invoke's wall clock. That is deliberate: the ``write_buffer >
    1`` sample/write remainder on PR #232 already showed wall splits go
    approximate under overlap.

    Keys — seconds: ``open`` (store lookup + TIFF header round trips + geo/
    dtype parse), ``geometry`` (pull-NN mapping; a ``geom_cache`` hit records
    ~0), ``fetch`` (tile GETs), ``decode``, ``gather`` (tile-index derivation
    + numpy scatter/gather).
    Counts: ``assets`` (asset-samples), ``tiles`` (tiles fetched),
    ``geom_hits`` (mappings served from ``geom_cache``).
    """
    return {
        "open": 0.0,
        "geometry": 0.0,
        "fetch": 0.0,
        "decode": 0.0,
        "gather": 0.0,
        "assets": 0,
        "tiles": 0,
        "geom_hits": 0,
    }


async def sample_asset_async(
    grid,
    cells,
    href: str,
    *,
    region: str | None = None,
    anonymous: bool = True,
    fill=0,
):
    """Pull-NN sample one raster asset at the centers of ``cells``.

    Fetches only the tiles the sampled pixels touch (concurrently), decodes,
    and gathers one value per cell.

    Returns
    -------
    (values, valid)
        ``values`` in the asset's dtype (``fill`` where invalid); ``valid``
        True where the cell center lands on a source pixel. Nodata masking is
        the caller's concern (e.g. Sentinel-2 encodes nodata as DN 0).

    ``fill`` must be representable in the asset's dtype; a non-fitting sentinel
    (e.g. ``-1`` or ``NaN`` into a ``uint16`` band) raises ``ValueError`` up
    front rather than failing opaquely inside the gather.
    """
    values, valid, _center = await _sample_one(
        grid, cells, href, region=region, anonymous=anonymous, fill=fill
    )
    return values, valid


async def _sample_one(
    grid,
    cells,
    href: str,
    *,
    region: str | None = None,
    anonymous: bool = True,
    fill=0,
    geom_cache: dict | None = None,
    stage_stats: dict | None = None,
    io_stats: dict | None = None,
):
    """:func:`sample_asset_async` body, also returning the raster's center
    ``(lon, lat)`` — the ownership rule's tile-center input (#218).

    ``geom_cache`` (issue #244) memoizes the pull-NN mapping ``(rows, cols,
    valid)`` per ``(epsg, transform, shape)``: the mapping is invariant across
    every timestep and band that shares a source grid, so a shard invoke
    computes it once per distinct grid (~175 ms at o9) instead of once per
    asset-sample. ``None`` (the default, and the public ``sample_asset*``
    path) computes per call, unchanged.

    ``stage_stats`` (issue #249) accumulates per-stage seconds + counts in
    place — see :func:`new_stage_stats` for the keys and the work-volume (not
    wall-decomposition) semantics. ``None`` (the default, and the public
    ``sample_asset*`` path) makes no timing calls at all — the hot path is
    unchanged. Accumulation is plain ``+=`` on the event loop with no await
    between read and write, so it is atomic by the same argument as the
    ``geom_cache`` store below — no locks; the ``write_buffer`` sink threads
    never touch this dict.

    ``io_stats`` (issue #297) accumulates the read-volume counters for the
    per-shard stats record, in place and by the same on-loop ``+=`` argument:
    ``bytes_read`` (compressed tile bytes fetched), ``px_decoded`` (pixels in
    the decoded tiles — whole tiles are read to sample a few cells), and
    ``px_sampled`` (cell samples actually gathered). Unlike ``stage_stats``
    this is ALWAYS-ON in the shard workers (the counters are a ``len()`` and
    two multiplies per asset — no timing calls); ``None`` (the public
    ``sample_asset*`` path) counts nothing.
    """
    from async_tiff import TIFF

    prof = stage_stats is not None
    _t0 = time.time() if prof else None
    store, path = _store_and_path(href, region=region, anonymous=anonymous)
    tiff = await TIFF.open(path, store=store)
    ifd = tiff.ifds[0]
    if len(ifd.bits_per_sample) != 1:
        raise ValueError(
            "single-band rasters only; Sentinel-2 distributes one COG per band "
            f"(found samples-per-pixel = {len(ifd.bits_per_sample)})"
        )
    dtype = _DTYPES[(int(ifd.sample_format[0]), int(ifd.bits_per_sample[0]))]
    if not np.can_cast(np.min_scalar_type(fill), dtype):
        raise ValueError(
            f"fill={fill!r} is not representable in the asset dtype {np.dtype(dtype).name}"
        )
    epsg, transform = _geo_from_ifd(ifd)
    shape = (ifd.image_height, ifd.image_width)
    center = _raster_center_lonlat(epsg, transform, shape)
    if prof:
        stage_stats["open"] += time.time() - _t0
        stage_stats["assets"] += 1
        _t0 = time.time()
    geom_key = (epsg, transform, shape)
    geom = geom_cache.get(geom_key) if geom_cache is not None else None
    if prof and geom is not None:
        stage_stats["geom_hits"] += 1
    if geom is None:
        # INVARIANT (issue #244 thread): no ``await`` between this check and
        # the store below. asyncio interleaves only at await points, so the
        # compute-and-store is atomic on the loop and each source grid is
        # computed exactly once per invoke — no locks needed. If this compute
        # ever moves off-loop (``to_thread``), add per-key async locks.
        # INVARIANT (issue #244 thread): the key ``(epsg, transform, shape)``
        # is complete only because ``cells`` and ``grid`` are constants of the
        # invoke — ``cells = grid.children(shard_key)`` is computed once (see
        # ``process_raster_shard``) and threaded unchanged into every
        # ``_sample_one``, and ``geom_cache`` is allocated fresh per
        # ``process_raster_shard`` call. A future refactor that varies
        # ``cells`` (or ``grid``) per item/group within one invoke MUST fold
        # them into the key or drop the cache, else it returns a stale mapping.
        geom = grid.sample(cells, f"EPSG:{epsg}", transform, shape)
        if geom_cache is not None:
            geom_cache[geom_key] = geom
    rows, cols, valid = geom
    if prof:
        stage_stats["geometry"] += time.time() - _t0
        _t0 = time.time()

    th, tw = ifd.tile_height, ifd.tile_width
    vr, vc = rows[valid], cols[valid]
    tr, tc = vr // th, vc // tw

    if io_stats is not None:
        io_stats["px_sampled"] += int(vr.size)

    if vr.size == 0:
        if prof:
            stage_stats["gather"] += time.time() - _t0
        return np.full(rows.shape, fill, dtype=dtype), valid, center

    pairs = np.unique(np.stack([tr, tc], axis=1), axis=0)
    if prof:
        stage_stats["gather"] += time.time() - _t0
        _t0 = time.time()
    tiles = await asyncio.gather(*[tiff.fetch_tile(int(c), int(r), 0) for r, c in pairs])
    if prof:
        stage_stats["fetch"] += time.time() - _t0
        stage_stats["tiles"] += len(pairs)
        _t0 = time.time()
    if io_stats is not None:
        # compressed_bytes is one Buffer (chunky) or a list of Buffers (planar);
        # normalize so bytes_read counts bytes, not buffers, if the single-band
        # guard above (bits_per_sample != 1) is ever relaxed for multi-band COGs.
        # A lone Buffer answers len() with its byte count; only the planar
        # list/tuple needs per-buffer summing.
        io_stats["bytes_read"] += sum(
            sum(len(x) for x in cb) if isinstance(cb, (list, tuple)) else len(cb)
            for cb in (t.compressed_bytes for t in tiles)
        )
        io_stats["px_decoded"] += len(pairs) * th * tw
    decoded = await asyncio.gather(*[t.decode() for t in tiles])
    if prof:
        stage_stats["decode"] += time.time() - _t0
        _t0 = time.time()

    gathered = np.full(vr.shape, fill, dtype=dtype)
    for (trow, tcol), dec in zip(pairs, decoded):
        arr = np.asarray(dec)[:, :, 0]
        m = (tr == trow) & (tc == tcol)
        gathered[m] = arr[vr[m] - trow * th, vc[m] - tcol * tw]

    values = np.full(rows.shape, fill, dtype=dtype)
    values[valid] = gathered
    if prof:
        stage_stats["gather"] += time.time() - _t0
    return values, valid, center


def _raster_center_lonlat(epsg: int, transform, shape) -> tuple[float, float]:
    """The raster's center pixel as WGS84 ``(lon, lat)`` — its "tile center"."""
    from pyproj import CRS, Transformer

    a, b, c, d, e, f = (float(t) for t in transform[:6])
    col, row = shape[1] / 2.0, shape[0] / 2.0
    x, y = a * col + b * row + c, d * col + e * row + f
    tx = Transformer.from_crs(
        CRS.from_user_input(f"EPSG:{epsg}"), CRS.from_epsg(4326), always_xy=True
    )
    lon, lat = tx.transform(x, y)
    return float(lon), float(lat)


def sample_asset(grid, cells, href: str, **kwargs):
    """Sync facade over :func:`sample_asset_async` (worker call sites are sync).

    Safe to call under an already-running event loop (Jupyter/Binder): when one
    is detected the coroutine runs to completion on a one-shot worker thread and
    the result is returned synchronously. Async callers should prefer awaiting
    :func:`sample_asset_async` directly.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(sample_asset_async(grid, cells, href, **kwargs))

    def _run():
        return asyncio.run(sample_asset_async(grid, cells, href, **kwargs))

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(_run).result()


def _run_sync(coro):
    """Run a coroutine from sync code, safe under an already-running loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(asyncio.run, coro).result()


# ── item sampling + acquisition grouping (issue #218) ────────────────────────


async def sample_item_async(
    grid,
    cells,
    assets: dict,
    bands: dict,
    *,
    nodata=None,
    region: str | None = None,
    anonymous: bool = True,
    geom_cache: dict | None = None,
    stage_stats: dict | None = None,
    io_stats: dict | None = None,
):
    """Sample every configured band of one STAC item, concurrently.

    Parameters
    ----------
    assets : dict
        ``{asset_key: href}`` — the granule entry's per-band asset map.
    bands : dict
        Normalized band config (:func:`zagg.config.get_raster_bands`):
        field -> ``{asset, dtype, fill_value, attrs}``.
    nodata : scalar, optional
        Source nodata DN; a cell whose sampled pixel equals it in ANY band is
        marked invalid. This is a single *scene-wide* sentinel, not per-band:
        for Sentinel-2 a DN of 0 means the pixel is outside the scene footprint
        in every band, so a cell that is valid in ``red`` but reads ``scl == 0``
        is dropped intentionally (footprint masking). Only co-declare bands that
        share this sentinel's "no data" meaning — a band that legitimately
        carries a 0 with different semantics would drop otherwise-valid cells.

    Returns
    -------
    (values, valid, center)
        ``values`` ``{field: ndarray}`` (asset dtype, fill where invalid),
        ``valid`` the combined per-cell mask, ``center`` the item's raster
        center ``(lon, lat)`` for the nearest-tile-center ownership rule.
    """
    missing = [meta["asset"] for meta in bands.values() if meta["asset"] not in assets]
    if missing:
        raise ValueError(
            f"granule entry is missing configured asset(s) {sorted(missing)}; "
            f"available: {sorted(assets)}"
        )
    fields = list(bands)
    results = await asyncio.gather(
        *[
            _sample_one(
                grid,
                cells,
                assets[bands[f]["asset"]],
                region=region,
                anonymous=anonymous,
                fill=bands[f]["fill_value"],
                geom_cache=geom_cache,
                stage_stats=stage_stats,
                io_stats=io_stats,
            )
            for f in fields
        ]
    )
    values = {f: r[0] for f, r in zip(fields, results)}
    valid = results[0][1].copy()
    for _v, mask, _c in results[1:]:
        valid &= mask
    if nodata is not None:
        for f in fields:
            valid &= values[f] != nodata
    return values, valid, results[0][2]


def _iso_us(iso: str) -> int:
    """ISO datetime -> int microseconds since the epoch (UTC)."""
    from datetime import datetime, timezone

    dt = datetime.fromisoformat(iso)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1_000_000)


def _us_iso(us: int) -> str:
    """Microseconds since the epoch -> canonical ISO-8601 UTC (seconds precision).

    The inverse of :func:`_iso_us` at the stamp's seconds precision — the D15
    ``time_range`` rendering (:func:`zagg.windows.iso_utc` convention).
    """
    from datetime import datetime, timezone

    return datetime.fromtimestamp(int(us) // 1_000_000, tz=timezone.utc).isoformat(
        timespec="seconds"
    )


def raster_time_index(granules) -> tuple[dict, np.ndarray]:
    """Global timestep index from ShardMap granule lists.

    A timestep is an *acquisition group* — entries sharing a ``time_key``
    (e.g. the Sentinel-2 datatake id; adjacent MGRS tiles of one datatake are
    items seconds apart) — falling back to the entry datetime when no key was
    configured. The group's coordinate value is its earliest datetime.

    Parameters
    ----------
    granules : list of list of dict
        ``ShardMap.granules`` (raster entries carry ``assets`` + ``datetime``).

    Returns
    -------
    (time_index, times_us)
        ``{group_key: t_idx}`` and the int64 microseconds-since-epoch time
        coordinate, both in ascending time order.
    """
    earliest: dict = {}
    for shard_entries in granules:
        for e in shard_entries:
            if not e.get("assets"):
                continue
            if not e.get("datetime"):
                raise ValueError(f"raster granule entry {e.get('id')!r} carries no datetime")
            key = e.get("time_key") or e["datetime"]
            us = _iso_us(e["datetime"])
            if key not in earliest or us < earliest[key]:
                earliest[key] = us
    ordered = sorted(earliest, key=lambda k: (earliest[k], k))
    time_index = {k: i for i, k in enumerate(ordered)}
    times_us = np.array([earliest[k] for k in ordered], dtype=np.int64)
    return time_index, times_us


def _chord2(lons, lats, lon0: float, lat0: float) -> np.ndarray:
    """Squared unit-sphere chord distance from points to ``(lon0, lat0)``.

    Monotone in great-circle distance — all the ownership argmin needs — and
    comparable across UTM zones (unlike per-zone projected distances).
    """
    lam, phi = np.radians(np.asarray(lons, dtype=float)), np.radians(np.asarray(lats, dtype=float))
    lam0, phi0 = np.radians(lon0), np.radians(lat0)
    x = np.cos(phi) * np.cos(lam) - np.cos(phi0) * np.cos(lam0)
    y = np.cos(phi) * np.sin(lam) - np.cos(phi0) * np.sin(lam0)
    z = np.sin(phi) - np.sin(phi0)
    return x * x + y * y + z * z
