"""Shard-map drift check for the pinned benchmark targets (issue #110).

Rebuilds each pinned shard map from CMR (same AOI + temporal window + grid as
``targets.json``) and asserts the densest shard hasn't materially drifted, so a
silent change in CMR coverage that would move the benchmark target gets caught
loudly instead of surfacing as a phantom cost/runtime regression.

An entry carrying ``catalog_parquet`` (issue #148) rebuilds from that committed
stac-geoparquet snapshot instead of re-fetching CMR — build the catalog once,
save the parquet, reuse it — so heavy AOIs (the 88S ring is a ~20 min fetch of
35,639 granules) don't re-download weekly; the rebuild then guards the
shardmap-build + pin deterministically.

This needs the network (CMR) and, for the rectilinear maps, the exact-S2
``spherely`` backend, so it is decoupled from the unit suite: it runs only when
``ZAGG_BENCHMARK_DRIFT=1`` is set (the `benchmark-drift` workflow does this on a
native x86_64 runner where the spherely wheel installs). The check is
**tie-tolerant** -- several shards tie (or near-tie) for densest in this AOI, and
the lowest-key tiebreak is deterministic but fragile to a +/-0 count nudge, so we
compare the densest *granule count* (within +/-1), not the exact shard key.

The NEON maps are pinned over the full-mission window ``2018-10-13 ..
2026-03-15`` (issue #202 re-pin). That window makes the tie tolerance load-bearing
for o11: its densest shard sits at 50 granules with a *close cluster of three
shards at 49*, so a one-granule CMR nudge can reselect the densest key while the
count stays put -- which the +/-1 count comparison (not a key comparison) absorbs
without a false drift alarm. The live matrix's AOI-mask arm reuses ``healpix_o9``
directly (issue #202): the strict-AOI mask adds a per-cell column that doesn't move
granules, so it is built on the fly at dispatch
(``run_benchmark._shardmap_with_mask``) rather than committed as a separate map --
``healpix_o9``'s pin guards both arms. The 88S stress pins keep their own temporal window
(issue #148) -- the re-pin does not touch them.
"""

import json
import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
BENCH = REPO / "tests" / "data" / "benchmark"
sys.path.insert(0, str(REPO / ".github" / "scripts"))
import bench_metrics  # noqa: E402

MANIFEST = json.loads((BENCH / "targets.json").read_text())


def resolve_aoi_temporal_cmr(sm_meta: dict) -> tuple[dict, dict, dict]:
    """Resolve a shard map's ``aoi``/``temporal``/``cmr`` (issue #121).

    A per-entry override wins; an absent key falls back to the top-level manifest
    default. Existing single-AOI (NEON) shard maps carry no override, so they
    resolve byte-identically to the top-level ``aoi``/``temporal``/``cmr``.
    """
    return (
        sm_meta.get("aoi", MANIFEST["aoi"]),
        sm_meta.get("temporal", MANIFEST["temporal"]),
        sm_meta.get("cmr", MANIFEST["cmr"]),
    )


#: The network gate is the drift check's own, not the module's (it was
#: module-level while the drift check was the only test here): the issue #444
#: re-pin-driver tests below rebuild from the committed catalogs and need no CMR.
needs_cmr = pytest.mark.skipif(
    os.environ.get("ZAGG_BENCHMARK_DRIFT") != "1",
    reason="set ZAGG_BENCHMARK_DRIFT=1 to run the CMR shard-map drift check",
)


def _containing_shard(parent_grid, shard_key: int) -> int:
    """The parent-grid shard containing a finer shard (via its center point).

    HEALPix cells nest, so a finer cell's center maps unambiguously into its
    containing coarser cell; routing through ``assign``/``shards_of`` keeps this
    on the same mortie machinery the shard maps themselves are built with.
    """
    import numpy as np
    from mortie import mort2geo

    lat, lon = mort2geo(np.array([shard_key], dtype=np.uint64))
    leaf = parent_grid.assign(np.atleast_1d(lat), np.atleast_1d(lon))
    return int(parent_grid.shards_of(leaf)[0])


def _config_for_shardmap(sm_key: str) -> Path:
    """Any target's config that uses this shard map (config sets the grid).

    Searches the committed matrix first, then ``provisional_targets`` (issue
    #130 block) — the 88S stress shard maps (issue #148) are referenced only by
    provisional targets, and the drift check still needs their grid config.
    """
    provisional = {
        k: v for k, v in MANIFEST.get("provisional_targets", {}).items() if k != "_comment"
    }
    for target in list(MANIFEST["targets"].values()) + list(provisional.values()):
        if target["shardmap"] == sm_key:
            return BENCH / target["config"]
    raise AssertionError(f"no target references shardmap '{sm_key}'")


def rebuild_shardmap(sm_key: str, sm_meta: dict):
    """Rebuild one shard map from its ``targets.json`` entry — THE recipe.

    Entry config → grid; the entry's resolved AOI/temporal/CMR (a per-entry
    override, issue #121, over the top-level manifest default — NEON entries
    carry none); the committed map's ``metadata.backend``; and either the
    committed ``catalog_parquet`` snapshot when the entry carries one or a live
    CMR fetch when it does not.

    Build-once catalog (issue #148): an entry carrying ``catalog_parquet``
    rebuilds from the committed stac-geoparquet snapshot instead of re-fetching
    CMR — the rebuild is then deterministic and offline, per the catalog design
    (fetch once, save the parquet, reuse). Regenerate the snapshot only to
    deliberately re-pin.

    Shared with the issue #444 re-pin driver rather than restated there: the
    drift guard and its deliberate counterpart must build the same way, and a
    second copy of this recipe is exactly what would let them drift apart.
    """
    from zagg.catalog import load_polygon, polygon_to_bbox
    from zagg.catalog.shardmap import ShardMap
    from zagg.catalog.sources import Catalog, CMRSource, Query
    from zagg.config import load_config
    from zagg.grids import from_config

    backend = json.loads((BENCH / sm_meta["path"]).read_text())["metadata"]["backend"]
    grid = from_config(load_config(str(_config_for_shardmap(sm_key))))
    aoi, temporal, cmr = resolve_aoi_temporal_cmr(sm_meta)
    # aoi.file is relative to the manifest dir, like the config/shardmap paths.
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


def select_pin(rebuilt, sm_meta: dict, parent_key: int | None = None) -> tuple[int, int]:
    """The ``(shard_key, n_granules)`` pin for a rebuilt map — THE pin rule.

    A nested pin (issue #148: the 88S o10 stress shard is the densest o10 shard
    INSIDE the pinned o9 stress shard, so one o9 extraction pass covers both
    orders) is selected over the shards nested in the parent's pin, never the
    global densest — otherwise a correct rebuild would read as drift.

    ``parent_key`` lets the issue #444 driver extract against a parent pin it
    has just rewritten on disk; the guard passes none and reads the manifest as
    loaded at import. Shared with that driver for the same reason as
    :func:`rebuild_shardmap`.
    """
    from zagg.config import load_config
    from zagg.grids import from_config

    shard_keys, granules = rebuilt.shard_keys, rebuilt.granules
    nested_in = sm_meta.get("nested_in")
    if nested_in:
        if parent_key is None:
            parent_key = int(MANIFEST["shardmaps"][nested_in]["shard_key"])
        parent_grid = from_config(load_config(str(_config_for_shardmap(nested_in))))
        keep = [
            i
            for i, k in enumerate(shard_keys)
            if int(_containing_shard(parent_grid, int(k))) == parent_key
        ]
        shard_keys = [shard_keys[i] for i in keep]
        granules = [granules[i] for i in keep]
    return bench_metrics.select_densest_shard({"shard_keys": shard_keys, "granules": granules})


@pytest.mark.slow
@needs_cmr
@pytest.mark.parametrize("sm_key", list(MANIFEST["shardmaps"]))
def test_pinned_shardmap_no_drift(sm_key):
    sm_meta = MANIFEST["shardmaps"][sm_key]
    committed = json.loads((BENCH / sm_meta["path"]).read_text())
    if committed["metadata"]["backend"] == "spherely":
        pytest.importorskip("spherely")

    key, n = select_pin(rebuild_shardmap(sm_key, sm_meta), sm_meta)
    pinned_n = sm_meta["n_granules"]
    # Tie-tolerant: the densest *count* is the stable quantity; an equally-dense
    # reselection (different key, same count) is fine -- a count drift is not.
    assert abs(n - pinned_n) <= 1, (
        f"{sm_key}: densest granule count drifted {pinned_n} -> {n} "
        f"(rebuilt densest shard {key}). Re-pin the shard map + targets.json."
    )


# -- the deliberate re-pin driver (issue #444) --------------------------------
#
# ``tools/repin_benchmark_shardmaps.py`` is the counterpart of the drift check
# above: the guard detects an accidental move, the driver makes a deliberate
# one. It imports the guard's recipe helpers, so these tests pin the parts the
# guard does not exercise -- the pruning, the pin write-back, and the claim the
# driver exists to support: that it reproduces the PR #441 artifacts from the
# committed catalogs.

OFFLINE_PINS = [k for k, v in MANIFEST["shardmaps"].items() if v.get("catalog_parquet")]


def _driver():
    """The re-pin driver, imported from ``tools/`` (not an installed module)."""
    sys.path.insert(0, str(REPO / "tools"))
    import repin_benchmark_shardmaps

    return repin_benchmark_shardmaps


def _without_volatile(text: str, volatile) -> str:
    """A written map's text minus the metadata lines a rebuild legitimately moves."""
    return "".join(
        line
        for line in text.splitlines(keepends=True)
        if not any(f'"{k}"' in line for k in volatile)
    )


@pytest.mark.slow
@pytest.mark.parametrize("sm_key", OFFLINE_PINS)
def test_offline_pin_reproduces_committed_map(sm_key, tmp_path):
    """The driver reproduces the PR #441 artifacts from the committed catalogs.

    The acceptance test issue #444 asks for, and the reason it can only cover
    the ``catalog_parquet`` (88S ring) entries: the NEON trio rebuilds from CMR
    by design -- an ATL03 footprint quad blankets the whole NEON box, so a local
    full-catalog snapshot over-includes (``tests/data/benchmark/README.md``).

    Byte-for-byte over the whole written manifest -- every granule record,
    ``shard_keys``, ``grid_signature``, and the metadata the build derives --
    except the two keys a faithful rebuild still moves, which are asserted
    separately below.
    """
    driver = _driver()
    mapped, key, n = driver.repin(sm_key)
    sm_meta = MANIFEST["shardmaps"][sm_key]
    assert (key, n) == (sm_meta["shard_key"], sm_meta["n_granules"])

    written = tmp_path / "rebuilt.json"
    mapped.to_json(str(written))
    assert _without_volatile(written.read_text(), driver.VOLATILE_META) == _without_volatile(
        (BENCH / sm_meta["path"]).read_text(), driver.VOLATILE_META
    )

    from zagg.config import load_config
    from zagg.grids import from_config

    # The one live divergence, and why it is not a pin move: PR #447 made the
    # unpinned HEALPix ``swath`` cover order the SHARD order, where the
    # committed maps recorded the chunk order they were built at. The
    # assignment is unchanged -- which is what the byte comparison above just
    # showed, over the same catalog.
    grid = from_config(load_config(str(_config_for_shardmap(sm_key))))
    assert mapped.metadata["mortie_order"] == grid.parent_order


def test_repin_updates_only_the_pin_literals_in_targets():
    # The write-back is surgical because targets.json is hand-formatted (compact
    # inline ``worker`` objects survive a re-pin); the entry's prose ``note`` is
    # the re-pinner's to restate, not the driver's to rewrite.
    driver = _driver()
    text = (BENCH / "targets.json").read_text()
    out = driver.update_targets(text, "healpix_o9", 4242, 7)

    entry = json.loads(out)["shardmaps"]["healpix_o9"]
    assert (entry["shard_key"], entry["n_granules"]) == (4242, 7)
    assert entry["note"] == MANIFEST["shardmaps"]["healpix_o9"]["note"]
    changed = [(a, b) for a, b in zip(text.splitlines(), out.splitlines(), strict=True) if a != b]
    assert len(changed) == 2, changed


def test_repin_refuses_an_unknown_shardmap(capsys):
    driver = _driver()
    with pytest.raises(SystemExit):
        driver.main(["--check", "healpix_o42"])
    assert "unknown shard map(s) ['healpix_o42']" in capsys.readouterr().err
