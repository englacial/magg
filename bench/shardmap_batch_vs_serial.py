"""Batch-vs-serial mortie intersection on REAL catalogs (issue #396, PR #400).

Measures the phase-1 rewire of ``_intersect_mortie`` -- one batched
``mortie.polygons_to_morton_mocs`` call per block of rings, replacing one
``morton_coverage_moc`` call per granule -- against the per-granule loop it
replaced, reporting **wall and peak RSS** for both.

Why this file exists at all: the first cut of these numbers came from synthetic
footprints, and the synthetic fixture was badly miscalibrated. A synthetic
quadrilateral covers ~220 MOC cells at order 13; a real ATL03 CMR footprint
covers ~10,000 (~47x more), and the whole memory profile of the batch path
scales with that number. The synthetic figure is what produced a block size
tuned an order of magnitude too large. Real catalogs are in the tree -- use
them. (``benchmarks/mortie_order_sweep.py`` remains synthetic and says so; its
real-catalog successor for the *order* question is ``bench/neon_order_sweep.py``,
whose conventions this script follows.)

Cases, smallest first. The first two are committed fixtures, so they run
anywhere in the repo; the last two need the 305 MB full-mission ATL03 clone
(``data/atl03_v007/``, not committed) and skip cleanly when it is absent, as the
other real-catalog benches do.

  neon        2,089 granules   committed  cat_neon.parquet  x AOP_NEON
  88s        35,639 granules   committed  cat_88s.parquet   x antarctic_88s
  california  4,354 granules   clone      bbox cut of the clone x California
  full      555,867 granules   clone      the whole clone   x California

``full`` is the issue's headline baseline (the 310.7 s single-pass mortie row);
it is slow by construction and opt-in via ``--cases full``.

Peak RSS needs process isolation -- ``ru_maxrss`` is a monotone high-water mark,
so serial and batch cannot be measured in one process. Each measurement
therefore re-executes this file as a child (``--measure``) that prints one JSON
line. The child reports its high-water mark both *before* the intersection (the
catalog-load plateau) and after, so the intersection's own peak is the
difference rather than the load's floor.

Run::

    uv run python bench/shardmap_batch_vs_serial.py
    uv run python bench/shardmap_batch_vs_serial.py --cases neon,88s
    uv run python bench/shardmap_batch_vs_serial.py --knee 88s
    uv run python bench/shardmap_batch_vs_serial.py --cases full   # slow, needs the clone
    uv run python bench/shardmap_batch_vs_serial.py --cases full --orders 13   # ~30 min
"""

from __future__ import annotations

import argparse
import json
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = "/Users/espg/software/zagg"
BENCH = f"{REPO}/tests/data/benchmark"
FULL = f"{REPO}/data/atl03_v007/atl03_v007_full.parquet"
CFG9 = f"{BENCH}/configs/atl03_tdigest_healpix_o9.yaml"

# ``atl03_tdigest_healpix_o9.yaml`` is the shipped production grid: parent_order
# 9 shards, child_order 19 leaves, chunk_inner 13 -- so ``ShardMap.build``'s
# default MOC order resolves to 13 (#92). Both orders are swept: 13 is what a
# default build actually runs, 9 is the legal floor the issue's baselines used.
ORDERS = (9, 13)

CASES = {
    "neon": {"catalog": f"{BENCH}/catalogs/cat_neon.parquet", "aoi": f"{BENCH}/AOP_NEON.geojson"},
    "88s": {
        "catalog": f"{BENCH}/catalogs/cat_88s.parquet",
        "aoi": f"{BENCH}/antarctic_88s.geojson",
    },
    "california": {"catalog": FULL, "aoi": "california", "bbox_cut": True},
    "full": {"catalog": FULL, "aoi": "california", "orders": (9,)},
}


def _rss_mb() -> float:
    """Process high-water RSS in MB (``ru_maxrss`` is bytes on macOS, KB on Linux)."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 1e6 if sys.platform == "darwin" else peak / 1e3


def _intersect_serial(records, grid, all_shards, order):
    """The pre-#396 per-granule loop -- the thing the batch path replaced.

    Kept in step with ``tests/test_shardmap.py::_intersect_mortie_serial``, which
    is the authoritative copy: that one is asserted equal to the shipping path on
    every run, so a drift between them shows up as a test failure, not as a
    quietly wrong benchmark.
    """
    from mortie import moc_to_order, morton_coverage_moc

    out: dict = {}
    parent_order = grid.parent_order
    for i, rec in enumerate(records):
        try:
            moc = np.asarray(morton_coverage_moc(rec["lats"], rec["lons"], order=order))
        except Exception:
            continue
        if moc.size == 0:
            continue
        try:
            shards = np.unique(moc_to_order(moc, parent_order))
        except Exception:
            continue
        for s in shards.tolist():
            s = int(s)
            if s in all_shards:
                out.setdefault(s, []).append(i)
    return {k: list(dict.fromkeys(v)) for k, v in out.items()}


def _load_case(name):
    """Return ``(records, grid, all_shards)`` for one case, as ``build`` would."""
    from zagg.catalog import load_polygon
    from zagg.catalog.shardmap import _region_parts
    from zagg.catalog.sources import Catalog
    from zagg.config import load_config
    from zagg.data import demo_aoi
    from zagg.grids import from_config

    spec = CASES[name]
    grid = from_config(load_config(CFG9))
    aoi = spec["aoi"]
    parts = load_polygon(demo_aoi(aoi) if not aoi.endswith(".geojson") else aoi)
    cat = Catalog.from_geoparquet(spec["catalog"])
    if spec.get("bbox_cut"):
        # The demo/01_query.ipynb path: cut the global clone to the AOI's
        # shard-complete bbox first, then let the exact intersection prune.
        cat = cat.filter_bbox([grid.coverage_bbox(parts)])
    records = cat.granule_records()
    all_shards = {int(s) for s in grid.coverage(_region_parts(parts, cat.metadata))}
    return records, grid, all_shards


def _measure_child(name, path, order, block):
    """Child mode: run one (case, path) measurement and print a JSON line."""
    from zagg.catalog import shardmap

    if block is not None:
        shardmap._MOC_BATCH_RINGS = block
    records, grid, all_shards = _load_case(name)
    rss_load = _rss_mb()
    fn = _intersect_serial if path == "serial" else shardmap._intersect_mortie
    t0 = time.perf_counter()
    out = fn(records, grid, all_shards, order=order)
    wall = time.perf_counter() - t0
    print(
        json.dumps(
            {
                "granules": len(records),
                "shards_aoi": len(all_shards),
                "shards_hit": len(out),
                "pairs": sum(len(v) for v in out.values()),
                "digest": hash(tuple(sorted((k, tuple(v)) for k, v in out.items()))),
                "wall_s": round(wall, 2),
                "rss_load_mb": round(rss_load, 1),
                "rss_peak_mb": round(_rss_mb(), 1),
            }
        )
    )


def _measure(name, path, order, block=None):
    argv = [sys.executable, __file__, "--measure", name, "--path", path, "--order", str(order)]
    if block is not None:
        argv += ["--block", str(block)]
    proc = subprocess.run(argv, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{name}/{path}/o{order} failed:\n{proc.stderr[-2000:]}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _available(name) -> bool:
    return Path(CASES[name]["catalog"]).exists()


def run_cases(names, orders=None):
    print(
        f"{'case':>11} {'order':>5} {'granules':>9} {'shards':>7} {'pairs':>9} "
        f"{'serial_s':>9} {'batch_s':>8} {'x':>5} {'ser_MB':>7} {'bat_MB':>7}",
        flush=True,
    )
    for name in names:
        if not _available(name):
            print(f"{name:>11}  -- skipped: {CASES[name]['catalog']} not present", flush=True)
            continue
        for order in orders or CASES[name].get("orders", ORDERS):
            ser = _measure(name, "serial", order)
            bat = _measure(name, "batch", order)
            assert ser["digest"] == bat["digest"], f"{name}@o{order}: batch != serial"
            print(
                f"{name:>11} {order:>5} {ser['granules']:>9,} {ser['shards_hit']:>7,} "
                f"{ser['pairs']:>9,} {ser['wall_s']:>9.2f} {bat['wall_s']:>8.2f} "
                f"{ser['wall_s'] / max(bat['wall_s'], 1e-9):>4.1f}x "
                f"{ser['rss_peak_mb'] - ser['rss_load_mb']:>7.0f} "
                f"{bat['rss_peak_mb'] - bat['rss_load_mb']:>7.0f}",
                flush=True,
            )


def run_knee(name, order=13):
    """Re-validate ``_MOC_BATCH_RINGS`` on real footprints, not synthetic ones."""
    if not _available(name):
        print(f"knee: skipped, {CASES[name]['catalog']} not present", flush=True)
        return
    print(f"\nblock-size knee -- {name} @order{order} (real footprints)", flush=True)
    print(f"{'block':>7} {'wall_s':>8} {'peak_MB':>8}", flush=True)
    for block in (32, 64, 128, 256, 512, 1024, 2048):
        m = _measure(name, "batch", order, block=block)
        print(
            f"{block:>7} {m['wall_s']:>8.2f} {m['rss_peak_mb'] - m['rss_load_mb']:>8.0f}",
            flush=True,
        )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases", default="neon,88s,california")
    ap.add_argument("--orders", default=None, help="override the swept MOC orders, e.g. 13")
    ap.add_argument("--knee", default=None, help="case to sweep _MOC_BATCH_RINGS on")
    ap.add_argument("--measure", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--path", choices=("serial", "batch"), default="batch")
    ap.add_argument("--order", type=int, default=13)
    ap.add_argument("--block", type=int, default=None)
    args = ap.parse_args(argv)

    if args.measure:
        _measure_child(args.measure, args.path, args.order, args.block)
        return 0
    orders = tuple(int(o) for o in args.orders.split(",")) if args.orders else None
    run_cases([c for c in args.cases.split(",") if c], orders)
    if args.knee:
        run_knee(args.knee)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
