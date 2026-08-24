"""Build a HEALPix shard map + cost ceiling for an arbitrary AOI polygon.

Generalizes ``data/conus/build_conus_shardmap.py`` (issue #202) to any AOI
GeoJSON: same offline pipeline, same guards, plus the pre-invoke Lambda cost
ceiling that ``zagg.notebook.max_cost_preview`` reports before fan-out.

Pipeline
--------
1. Load the full local ATL03 v007 catalog (``Catalog.from_geoparquet``).
2. **bbox prefilter** to the AOI bounding box. The catalog ``bbox`` column is
   latitude-exact and longitude-conservative (data/atl03_v007/README.md), so
   the overlap test is a superset -- it cannot drop a real intersector.
3. **temporal filter** to ``[start, end]`` via each granule's ``datetime``.
4. Build the shard map at ``--order`` (``backend="mortie"``, swath footprint).
5. **Leak sanity check**: every covered shard's cell centre lies inside the AOI
   bbox + one-cell margin (mortie #103 regression guard, fixed in 0.9.0).
6. Emit the shard map JSON, the per-shard granule-count parquet, and a stats
   JSON carrying the max-cost block for the requested worker size.

Run: ``uv run python data/build_aoi_shardmap.py --polygon <aoi.geojson> --order 9``
(needs the local catalog; no network, no AWS).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

DEFAULT_CATALOG = "data/atl03_v007/atl03_v007_full.parquet"
DEFAULT_CONFIG = "src/zagg/configs/atl03_tdigest_healpix_hive.yaml"


def _polygon_parts(geojson_path: str):
    """Return (polygon_parts, bbox, properties) from an AOI GeoJSON.

    ``polygon_parts`` is ``[(lats, lons), ...]`` -- one exterior ring per part
    of every feature, the form ``HealpixGrid.coverage`` consumes. ``bbox`` is
    ``(lon_min, lat_min, lon_max, lat_max)`` over all features. ``properties``
    is the first feature's ``properties`` (decoration for the stats JSON: the
    provenance string and any ``area_km2*`` keys).

    A FeatureCollection, a bare Feature and a raw geometry are all accepted;
    the file is read once here so the caller cannot re-parse it under a
    different assumption about which of the three it is.
    """
    from shapely.geometry import shape
    from shapely.ops import unary_union

    fc = json.loads(Path(geojson_path).read_text())
    feats = fc["features"] if fc.get("type") == "FeatureCollection" else [fc]
    geom = unary_union([shape(f.get("geometry", f)).buffer(0) for f in feats])
    polys = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
    parts = []
    for p in polys:
        x, y = p.exterior.coords.xy
        parts.append((np.asarray(y), np.asarray(x)))
    minx, miny, maxx, maxy = geom.bounds
    props = dict(feats[0].get("properties") or {}) if feats else {}
    return parts, (float(minx), float(miny), float(maxx), float(maxy)), props


def _part_bboxes(parts, pad_deg: float):
    """Per-part ``(lon0, lat0, lon1, lat1)`` boxes, each padded by ``pad_deg``."""
    boxes = []
    for lats, lons in parts:
        boxes.append(
            (
                float(lons.min()) - pad_deg,
                float(lats.min()) - pad_deg,
                float(lons.max()) + pad_deg,
                float(lats.max()) + pad_deg,
            )
        )
    return boxes


def _bbox_temporal_prefilter(catalog, boxes, start: str, end: str, region: str, bbox):
    """Subset the catalog to granules overlapping any part box, within ``[start, end]``.

    The cut is per-part (OR over ``boxes``), not over the AOI's global bounding
    box: a multipart AOI like the NEON flight boxes spans Hawaii to Alaska, so
    the global box would keep most of the catalog and hand the exact
    intersection ~100x the granules it needs. Each part box is padded by
    ``--pad-deg`` to absorb geodesic edge sag (granule polygon edges are
    great-circle arcs that reach past their vertices -- data/atl03_v007/README.md),
    keeping the cut a superset of the true intersectors.
    """
    import pyarrow.compute as pc

    from zagg.catalog.sources import Catalog

    table = catalog.table
    bbcol = table.column("bbox")
    g_lon_min = pc.struct_field(bbcol, "xmin").to_numpy(zero_copy_only=False)
    g_lat_min = pc.struct_field(bbcol, "ymin").to_numpy(zero_copy_only=False)
    g_lon_max = pc.struct_field(bbcol, "xmax").to_numpy(zero_copy_only=False)
    g_lat_max = pc.struct_field(bbcol, "ymax").to_numpy(zero_copy_only=False)
    overlap = np.zeros(len(g_lon_min), dtype=bool)
    for lon0, lat0, lon1, lat1 in boxes:
        overlap |= (
            (g_lon_min <= lon1) & (g_lon_max >= lon0) & (g_lat_min <= lat1) & (g_lat_max >= lat0)
        )
    dt = table.column("datetime").to_numpy(zero_copy_only=False).astype("datetime64[us]")
    in_time = (dt >= np.datetime64(start)) & (dt <= np.datetime64(end + "T23:59:59"))
    mask = overlap & in_time
    sub = table.take(np.flatnonzero(mask))
    meta = dict(catalog.metadata or {})
    meta.update(
        collection=meta.get("collection", "ATL03_007"),
        bbox=list(bbox),
        start_date=start,
        end_date=end,
        region=region,
    )
    return Catalog(sub, meta), int(mask.sum()), int(len(mask))


def _leak_check(grid, shard_keys, boxes, margin_deg: float = 1.0) -> dict:
    """Assert every shard cell centre sits within ``margin_deg`` of some AOI part.

    Tested per part rather than against the AOI's global bounding box: for a
    scattered multipart AOI the global box spans everything between the parts,
    where a fill leak would go undetected.
    """
    from mortie import mort2polygon

    lats, lons = [], []
    for k in shard_keys:
        verts = mort2polygon(int(k), step=4)
        lats.append(float(np.mean([v[0] for v in verts])))
        lons.append(float(np.mean([v[1] for v in verts])))
    lats = np.asarray(lats)
    lons = np.asarray(lons)
    near_some_part = np.zeros(len(lats), dtype=bool)
    for lon0, lat0, lon1, lat1 in boxes:
        near_some_part |= (
            (lats >= lat0 - margin_deg)
            & (lats <= lat1 + margin_deg)
            & (lons >= lon0 - margin_deg)
            & (lons <= lon1 + margin_deg)
        )
    out_of_box = ~near_some_part
    n_leak = int(out_of_box.sum())
    if n_leak:
        bad = [
            (grid.shard_label(int(shard_keys[i])), round(lats[i], 3), round(lons[i], 3))
            for i in np.flatnonzero(out_of_box)[:10]
        ]
        raise AssertionError(
            f"leak check FAILED: {n_leak} shard cell centre(s) further than "
            f"{margin_deg} deg from every AOI part (e.g. {bad}) -- possible "
            f"mortie #103 regression"
        )
    return {
        "passed": True,
        "margin_deg": margin_deg,
        "cell_lat_min": round(float(lats.min()), 4),
        "cell_lat_max": round(float(lats.max()), 4),
        "cell_lon_min": round(float(lons.min()), 4),
        "cell_lon_max": round(float(lons.max()), 4),
    }


def _distribution(counts: list[int]) -> dict:
    arr = np.asarray(sorted(counts))
    all_edges = [1, 2, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000]
    hist_edges = [e for e in all_edges if e <= int(arr.max())]
    bins = [*hist_edges, int(arr.max()) + 1] if len(hist_edges) >= 1 else [0, int(arr.max()) + 1]
    hist, _ = np.histogram(arr, bins=bins)
    return {
        "n_shards": int(len(arr)),
        "min": int(arr.min()),
        "median": float(np.median(arr)),
        "mean": round(float(arr.mean()), 2),
        "p90": float(np.percentile(arr, 90)),
        "p99": float(np.percentile(arr, 99)),
        "max": int(arr.max()),
        "total_pairs": int(arr.sum()),
        "histogram_edges": hist_edges,
        "histogram_counts": [int(x) for x in hist],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--polygon", required=True, help="AOI GeoJSON (FeatureCollection)")
    ap.add_argument("--name", required=True, help="Region name (stats + output prefix)")
    ap.add_argument("--out-dir", required=True, help="Directory for the emitted artifacts")
    ap.add_argument("--catalog", default=DEFAULT_CATALOG)
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--start", default="2018-10-13")
    ap.add_argument("--end", default="2026-03-15")
    ap.add_argument("--order", type=int, default=9, help="HEALPix parent_order (shard order)")
    ap.add_argument(
        "--worker-memory",
        type=int,
        default=2048,
        help="Worker memory (MB): one of the pre-provisioned sizes 2048/4096/8192",
    )
    ap.add_argument("--extra-disk", action="store_true", help="Use the extra-/tmp worker variant")
    ap.add_argument(
        "--pad-deg",
        type=float,
        default=0.0,
        help="Degrees of padding on each part's prefilter box (geodesic sag margin)",
    )
    args = ap.parse_args(argv)

    import pyarrow as pa
    import pyarrow.parquet as pq

    from zagg.catalog.shardmap import ShardMap
    from zagg.catalog.sources import Catalog
    from zagg.config import WORKER_MEMORIES, load_config
    from zagg.grids import from_config
    from zagg.notebook import max_cost_preview

    # ``config.worker`` is set below the loader, so ``_validate_worker`` never
    # sees it: check the size here or an unlisted one silently prices a worker
    # variant no run can dispatch to (issue #235).
    if args.worker_memory not in WORKER_MEMORIES:
        ap.error(
            f"--worker-memory must be one of {sorted(WORKER_MEMORIES)} MB "
            f"(the pre-provisioned variant sizes, issue #235)"
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.name}_o{args.order}"

    print(f"loading catalog {args.catalog} ...", flush=True)
    catalog = Catalog.from_geoparquet(args.catalog)
    parts, bbox, props = _polygon_parts(args.polygon)
    print(f"AOI bbox={tuple(round(b, 3) for b in bbox)}, {len(parts)} polygon part(s)", flush=True)

    boxes = _part_bboxes(parts, args.pad_deg)
    sub, n_kept, n_total = _bbox_temporal_prefilter(
        catalog, boxes, args.start, args.end, args.name, bbox
    )
    print(f"prefilter: {n_kept:,}/{n_total:,} granules survive bbox+temporal cut", flush=True)

    config = load_config(args.config)
    config.output.setdefault("grid", {})["parent_order"] = args.order
    config.worker = {"memory": args.worker_memory, "extra_disk": bool(args.extra_disk)}
    grid = from_config(config)

    t0 = time.perf_counter()
    sm = ShardMap.build(sub, grid, region=parts, backend="mortie", footprint="swath")
    wall = time.perf_counter() - t0
    print(f"shard map built in {wall:.1f}s: {sm.metadata['total_shards']:,} shards", flush=True)

    counts = [len(g) for g in sm.granules]
    leak = _leak_check(grid, sm.shard_keys, boxes)
    print(f"leak check: {leak['passed']}", flush=True)

    shardmap_path = out_dir / f"shardmap_{stem}.json"
    sm.to_json(str(shardmap_path))

    labels = [grid.shard_label(int(k)) for k in sm.shard_keys]
    pq.write_table(
        pa.table(
            {
                "shard_key": pa.array([int(k) for k in sm.shard_keys], type=pa.uint64()),
                "shard_label": pa.array(labels),
                "n_granules": pa.array(counts, type=pa.int32()),
            }
        ),
        out_dir / f"{stem}_shard_granule_counts.parquet",
    )

    # The cost ceiling exactly as the run logs it before fan-out (issue #298):
    # unit accounting via the dispatch path, priced at the configured worker.
    preview = max_cost_preview(config, catalog=str(shardmap_path))

    dist = _distribution(counts)
    # Distinct granules assigned to >=1 shard, as ShardMap.build records it.
    n_intersecting = int(sm.metadata["granules_assigned"])
    stats = {
        "region": args.name,
        "polygon_reference": {
            "path": args.polygon,
            "bbox_lonlat": [round(b, 6) for b in bbox],
            "n_parts": len(parts),
            "provenance": props.get("provenance"),
            **{k: v for k, v in props.items() if k.startswith("area_km2")},
            "prefilter_pad_deg": args.pad_deg,
        },
        "temporal": {"start": args.start, "end": args.end},
        "grid": {
            "type": "healpix",
            "parent_order": args.order,
            "child_order": int(grid.child_order),
            "mortie_moc_order": sm.metadata.get("mortie_order"),
            "backend": sm.metadata.get("backend"),
            "footprint": sm.metadata.get("footprint"),
        },
        "catalog": {
            "path": args.catalog,
            "total_granules": n_total,
            "granules_after_prefilter": n_kept,
            "granules_intersecting_aoi": n_intersecting,
        },
        "summary": {
            "total_shards": sm.metadata["total_shards"],
            "total_granules_intersecting": n_intersecting,
            "total_pairs": sm.metadata["total_pairs"],
            "build_wall_s": round(wall, 1),
        },
        "max_cost": {
            **preview,
            "config": args.config,
            "worker": config.worker,
        },
        "granules_per_shard": dist,
        "leak_check": leak,
    }
    (out_dir / f"{stem}_stats.json").write_text(json.dumps(stats, indent=2))

    print(f"wrote {shardmap_path} + {stem}_stats.json", flush=True)
    print(
        f"SUMMARY {stem}: shards={dist['n_shards']:,} granules={n_intersecting:,} "
        f"pairs={dist['total_pairs']:,} max/shard={dist['max']:,} "
        f"median/shard={dist['median']}",
        flush=True,
    )
    print(
        f"MAX COST {stem}: ${preview['max_cost_usd']:,.2f} "
        f"({preview['n_units']:,} units x {preview['memory_gb']:g} GB x "
        f"{preview['timeout_s']}s, {preview['arch']})",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
