"""DELIBERATELY re-pin the benchmark shard-map fixtures (issue #444).

The pins under ``tests/data/benchmark/`` — the committed
``shardmaps/sm_*.json`` maps plus their ``targets.json`` ``shard_key`` /
``n_granules`` entries — move only on purpose. A run of this script IS that
purpose: a convention or grammar change made the old words wrong (the authalic
latitude flip, issue #438 / PR #441, is the case it was written from) and the
pins are being restated against it. Nothing here detects drift. The accident
detector is and stays ``tests/test_benchmark_shardmap.py::
test_pinned_shardmap_no_drift``, which rebuilds the same maps and fails loudly
when a pin moves on its own; this script is that guard's deliberate
counterpart, and it IMPORTS the guard's recipe helpers (manifest
AOI/temporal/CMR resolution, the config lookup, the ``nested_in`` containment
rule) so the two cannot drift apart.

One run, per shard-map entry named on the command line:

1. rebuilds the map through the guard's recipe — the entry's config for the
   grid, its resolved AOI/temporal/CMR (per-entry override over the top-level
   manifest default), the committed map's ``metadata.backend``, and either the
   committed ``catalog_parquet`` snapshot when the entry carries one (offline)
   or a live CMR fetch when it does not;
2. selects the pin with ``bench_metrics.select_densest_shard`` — for a
   ``nested_in`` entry over the finer shards inside the pinned parent shard
   only, never the global densest;
3. prunes the written map to the pinned shard when the committed map is pruned
   (``metadata.pruned`` — the 88S ring maps, whose full form is hundreds of MB
   of JSON), carrying that note over verbatim: it is editorial prose, not a
   derived quantity;
4. writes the map and updates the entry's ``shard_key`` / ``n_granules`` in
   ``targets.json``, leaving every other byte of that hand-formatted file
   alone. The entry's ``note`` is prose and is NOT rewritten — restate it in
   the same commit.

``--check`` stops after (3) and reports how the rebuild differs from the
committed bytes instead of writing anything.
``tests/test_benchmark_shardmap.py::test_offline_pin_reproduces_committed_map``
is that path as an acceptance test over the two offline entries. The NEON trio
cannot be checked offline: an ATL03 footprint quad blankets the whole NEON box,
so a local full-catalog snapshot over-includes and inflates the pins
(``tests/data/benchmark/README.md``, "Reproducing / re-pinning the NEON maps")
— they need the live fetch, which is why only the ring entries carry
``catalog_parquet``.

Run from a zagg checkout::

    uv run python tools/repin_benchmark_shardmaps.py --check healpix_o9_88s
    uv run python tools/repin_benchmark_shardmaps.py healpix_o9_88s healpix_o10_88s

The ``sm_rect_*`` entries declare the ``spherely`` backend, the non-PyPI
exact-S2 fork (README): without it installed ``ShardMap.build`` raises rather
than quietly rebuilding them on the mortie backend.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BENCH = REPO / "tests" / "data" / "benchmark"
TARGETS = BENCH / "targets.json"
# The drift guard owns the rebuild recipe; importing it (rather than restating
# it) is what keeps this driver on the guard's rails. ``bench_metrics``, beside
# it under ``.github/scripts``, owns the densest-shard pin rule.
sys.path.insert(0, str(REPO / "tests"))
sys.path.insert(0, str(REPO / ".github" / "scripts"))

import bench_metrics  # noqa: E402
import test_benchmark_shardmap as drift  # noqa: E402

from zagg.catalog import load_polygon, polygon_to_bbox  # noqa: E402
from zagg.catalog.shardmap import ShardMap  # noqa: E402
from zagg.catalog.sources import Catalog, CMRSource, Query  # noqa: E402
from zagg.config import load_config  # noqa: E402
from zagg.grids import from_config  # noqa: E402

#: Metadata keys a faithful rebuild may still move, reported by ``--check`` as
#: such rather than as a changed pin: the build's own wall clock, and the MOC
#: cover order, whose unpinned default became the shard order in PR #447 (the
#: committed maps predate it; the assignment it produces is unchanged).
VOLATILE_META = ("build_wall_s", "mortie_order")


def entry(sm_key: str) -> dict:
    """One shard-map entry, read from ``targets.json`` as it stands NOW.

    Deliberately not ``drift.MANIFEST`` (loaded once at import): re-pinning a
    parent and its ``nested_in`` child in one run has to extract the child
    against the parent's freshly written pin, not the one this process started
    with. The top-level ``aoi``/``temporal``/``cmr`` defaults the guard
    resolves against are never rewritten here, so those stay the guard's.
    """
    return json.loads(TARGETS.read_text())["shardmaps"][sm_key]


def committed(sm_key: str) -> dict:
    """The committed shard map an entry points at, as JSON."""
    return json.loads((BENCH / entry(sm_key)["path"]).read_text())


def rebuild(sm_key: str) -> ShardMap:
    """The full rebuilt map for one ``targets.json`` shard-map entry."""
    sm_meta = entry(sm_key)
    backend = committed(sm_key)["metadata"]["backend"]
    grid = from_config(load_config(str(drift._config_for_shardmap(sm_key))))
    aoi, temporal, cmr = drift.resolve_aoi_temporal_cmr(sm_meta)
    parts = load_polygon(str(BENCH / aoi["file"]))
    if sm_meta.get("catalog_parquet"):
        catalog = Catalog.from_geoparquet(str(BENCH / sm_meta["catalog_parquet"]))
    else:
        catalog = CMRSource().fetch(
            Query(
                cmr["short_name"],
                cmr["version"],
                temporal["start"],
                temporal["end"],
                region=polygon_to_bbox(parts),
                provider=cmr["provider"],
            )
        )
    return ShardMap.build(catalog, grid, region=parts, backend=backend, footprint=cmr["footprint"])


def repin(sm_key: str) -> tuple[ShardMap, int, int]:
    """``(map to write, shard_key, n_granules)`` for one entry.

    The written map is the rebuild pruned to the pinned shard when the
    committed map is pruned, and the whole rebuild otherwise. Either way
    ``metadata`` is the FULL build's (``total_shards`` / ``total_pairs`` count
    the ring, not the surviving shard), matching the committed maps.
    """
    sm_meta = entry(sm_key)
    rebuilt = rebuild(sm_key)
    shard_keys, granules = rebuilt.shard_keys, rebuilt.granules
    nested_in = sm_meta.get("nested_in")
    if nested_in:
        # The nested pin (issue #148) is the densest finer shard INSIDE the
        # pinned parent, so one parent extraction pass covers both orders.
        parent_key = int(entry(nested_in)["shard_key"])
        parent_grid = from_config(load_config(str(drift._config_for_shardmap(nested_in))))
        keep = [
            i
            for i, k in enumerate(shard_keys)
            if drift._containing_shard(parent_grid, int(k)) == parent_key
        ]
        shard_keys = [shard_keys[i] for i in keep]
        granules = [granules[i] for i in keep]
    key, n = bench_metrics.select_densest_shard({"shard_keys": shard_keys, "granules": granules})
    note = committed(sm_key)["metadata"].get("pruned")
    if note is None:
        return rebuilt, key, n
    i = [j for j, k in enumerate(shard_keys) if int(k) == key][0]
    pruned = ShardMap(
        rebuilt.grid_signature,
        [shard_keys[i]],
        [granules[i]],
        {**rebuilt.metadata, "pruned": note},
    )
    return pruned, key, n


def differences(sm_key: str, mapped: ShardMap) -> list[str]:
    """How a rebuilt map differs from the committed one, key by key."""
    old_map = committed(sm_key)
    out = []
    for key in ("grid_signature", "shard_keys", "granules"):
        if getattr(mapped, key) != old_map[key]:
            out.append(f"{key} differs")
    old, new = old_map["metadata"], mapped.metadata
    for key in sorted(set(old) | set(new)):
        if old.get(key) != new.get(key):
            volatile = " (volatile)" if key in VOLATILE_META else ""
            out.append(f"metadata.{key}: {old.get(key)!r} -> {new.get(key)!r}{volatile}")
    return out


def update_targets(text: str, sm_key: str, key: int, n: int) -> str:
    """Restate one shard-map entry's pin literals in ``targets.json`` text.

    Surgical rather than a load/dump round trip: the manifest is hand-formatted
    (compact inline ``worker`` objects), so re-serializing would churn lines
    this re-pin does not touch. The entry's prose ``note`` is left alone.
    """
    decoder = json.JSONDecoder()
    maps_at = text.index("{", text.index('"shardmaps"'))
    _, maps_end = decoder.raw_decode(text, maps_at)
    entry_at = text.index("{", text.index(f'"{sm_key}"', maps_at, maps_end))
    _, entry_end = decoder.raw_decode(text, entry_at)
    entry = text[entry_at:entry_end]
    for field, value in (("shard_key", key), ("n_granules", n)):
        entry, hits = re.subn(rf'("{field}":\s*)\d+', rf"\g<1>{value}", entry, count=1)
        if hits != 1:
            raise ValueError(f"targets.json entry {sm_key!r} has no {field} literal to restate")
    return text[:entry_at] + entry + text[entry_end:]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "shardmaps",
        nargs="+",
        help="shard-map entry names from targets.json (e.g. healpix_o9_88s). "
        "Re-pin only what you mean to commit — each map is a deliberate move.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="rebuild and report differences against the committed bytes; write nothing.",
    )
    args = parser.parse_args(argv)
    known = json.loads(TARGETS.read_text())["shardmaps"]
    unknown = sorted(set(args.shardmaps) - set(known))
    if unknown:
        parser.error(f"unknown shard map(s) {unknown} (known: {sorted(known)})")
    # A nested entry extracts against its parent's pin, so re-pin parents first
    # even when the command line names them the other way round.
    for sm_key in sorted(args.shardmaps, key=lambda k: bool(known[k].get("nested_in"))):
        mapped, key, n = repin(sm_key)
        print(f"{sm_key}: pin {key} at {n} granules")
        for line in differences(sm_key, mapped) or ["identical to the committed map"]:
            print(f"  {line}")
        if args.check:
            continue
        path = entry(sm_key)["path"]
        mapped.to_json(str(BENCH / path))
        TARGETS.write_text(update_targets(TARGETS.read_text(), sm_key, key, n))
        print(f"  wrote {path} + its targets.json pin")
        print("  restate the entry's note by hand — this driver does not write prose")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
