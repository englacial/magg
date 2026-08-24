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
from dataclasses import replace
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / ".github" / "scripts"))
# The rebuild recipe and the pin rule live in ``bench_metrics`` (shared with the
# issue #444 re-pin driver, ``tools/repin_benchmark_shardmaps.py`` -- neither a
# ``tools/`` script nor CI should depend on a test module for core logic); this
# guard CALLS them, so the accident detector and the deliberate counterpart
# cannot build or pin differently.
from bench_metrics import (  # noqa: E402
    BENCH,
    MANIFEST,
    _config_for_shardmap,
    rebuild_shardmap,
    select_pin,
)

#: The network gate is the drift check's own, not the module's (it was
#: module-level while the drift check was the only test here): the issue #444
#: re-pin-driver tests below rebuild from the committed catalogs and need no CMR.
needs_cmr = pytest.mark.skipif(
    os.environ.get("ZAGG_BENCHMARK_DRIFT") != "1",
    reason="set ZAGG_BENCHMARK_DRIFT=1 to run the CMR shard-map drift check",
)


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
# one. Both CALL ``bench_metrics.rebuild_shardmap`` and
# ``bench_metrics.select_pin`` (the shared recipe the drift check runs
# through), so the rebuild and the pin rule are already covered by the drift
# check; these tests pin the parts they do not reach -- the pruning, the pin
# write-back, the re-pin ordering, and the claim the driver exists to support:
# that it reproduces the PR #441 artifacts from the committed catalogs.

OFFLINE_PINS = [k for k, v in MANIFEST["shardmaps"].items() if v.get("catalog_parquet")]

#: The metadata keys the byte comparison below excuses, pinned HERE rather than
#: imported from the driver. The driver's ``EXCUSED_META`` has a second job --
#: labelling ``--check`` output -- so someone quieting a noisy check by
#: appending a key to it would silently widen this acceptance test's blind spot,
#: with nothing failing. That is precisely the relaxation issue #444's
#: "byte-identically" exists to prevent, so widening it has to be a deliberate
#: edit here too.
EXCUSED_META = ("build_wall_s", "mortie_order")


# ``tools/`` is not an installed package, so it goes on the path ONCE here --
# matching the ``bench_metrics`` pattern above -- rather than on every
# ``_driver()`` call, which stacked one identical entry per test.
sys.path.insert(0, str(REPO / "tools"))


def _driver():
    """The re-pin driver, imported from ``tools/`` (not an installed module).

    The import stays lazy so a broken driver fails the driver tests, not this
    whole module's collection (the drift check above does not need it).
    """
    import repin_benchmark_shardmaps

    return repin_benchmark_shardmaps


def _without_volatile(text: str, volatile) -> str:
    """A written map's text minus the metadata lines a rebuild legitimately moves.

    Line-oriented on purpose: what issue #444 asks for is "byte-identically",
    which a parsed comparison would soften to structural equality. The cost is
    a silent dependency on ``ShardMap.to_json`` pretty-printing one metadata key
    per line -- were it ever to emit compact JSON, the whole map would become a
    single line carrying ``"build_wall_s"``, both sides would filter down to
    ``""``, and the acceptance test would pass while comparing NOTHING.

    So the filter checks its own work: exactly one line per excused key, no
    more (a key string surfacing inside a granule record) and no less.
    """
    lines = text.splitlines(keepends=True)
    kept = [line for line in lines if not any(f'"{k}"' in line for k in volatile)]
    assert len(lines) - len(kept) == len(volatile), (
        f"expected one line per excused metadata key, dropped {len(lines) - len(kept)} of "
        f"{len(lines)} -- has ShardMap.to_json stopped pretty-printing?"
    )
    return "".join(kept)


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

    # The driver may not widen this test's exemption on its own (see EXCUSED_META).
    assert tuple(driver.EXCUSED_META) == EXCUSED_META, (
        "the driver's excused-metadata set moved -- restate it here deliberately"
    )

    written = tmp_path / "rebuilt.json"
    mapped.to_json(str(written))
    assert _without_volatile(written.read_text(), EXCUSED_META) == _without_volatile(
        (BENCH / sm_meta["path"]).read_text(), EXCUSED_META
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
    # ...and the OTHER side, which nothing else constrains. ``mortie_order`` is
    # deterministic on both sides, so its exemption is a stale-fixture excuse
    # with an expiry, not a standing licence: pinning the committed value makes
    # this test fail at the next deliberate re-pin, when ``STALE_META`` must be
    # emptied rather than left to mask a genuine regression.
    assert json.loads((BENCH / sm_meta["path"]).read_text())["metadata"]["mortie_order"] == 13
    assert driver.STALE_META == ("mortie_order",)


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


def test_repin_prune_slices_the_aoi_mask_with_the_shard_keys():
    """The prune is the one place the driver hand-rebuilds a ``ShardMap``.

    ``aoi_mask`` is parallel to ``shard_keys`` (issue #101), so it has to be
    sliced with them. Latent today -- no committed benchmark map carries one,
    the strict-AOI arm building its mask at dispatch instead
    (``run_benchmark._shardmap_with_mask``, issue #202) -- but a field dropped
    here would be written out as a maskless map without a word.
    """
    from zagg.catalog.shardmap import ShardMap

    driver = _driver()
    rebuilt = ShardMap(
        {"grid": "healpix"},
        [10, 11],
        [[{"granule": "a"}], [{"granule": "b"}]],
        {"total_shards": 2},
        aoi_mask=[[1, 2], [3, 4]],
    )
    pruned = driver.prune_to_pin(rebuilt, 11, "pruned to the pinned shard")

    assert (pruned.shard_keys, pruned.granules) == ([11], [[{"granule": "b"}]])
    assert pruned.aoi_mask == [[3, 4]]
    # metadata stays the FULL build's, plus the carried note
    assert pruned.metadata == {"total_shards": 2, "pruned": "pruned to the pinned shard"}
    assert driver.prune_to_pin(replace(rebuilt, aoi_mask=None), 10, "note").aoi_mask is None


def test_repin_targets_write_back_is_anchored_on_the_key():
    """The entry is located by KEY, not by any occurrence of its name.

    An entry name also appears as a ``"nested_in"`` VALUE (``healpix_o10_88s``
    names ``healpix_o9_88s`` that way today). Today's manifest writes the parent
    first, so an unanchored search happens to land right; it stops doing so the
    moment a child precedes its parent, which the nested-pin design invites (an
    o11 nested in an o10). Here the child comes first and carries an inner
    object, so the unanchored form splices that object instead — silently, since
    it too has the two literals to restate.
    """
    driver = _driver()
    text = json.dumps(
        {
            "shardmaps": {
                "child": {
                    "nested_in": "parent",
                    "provisional": {"shard_key": 1, "n_granules": 2},
                    "shard_key": 3,
                    "n_granules": 4,
                },
                "parent": {"shard_key": 5, "n_granules": 6},
            }
        },
        indent=2,
    )
    maps = json.loads(driver.update_targets(text, "parent", 4242, 7))["shardmaps"]

    assert maps["parent"] == {"shard_key": 4242, "n_granules": 7}
    assert maps["child"] == json.loads(text)["shardmaps"]["child"]


def test_repin_orders_by_nesting_depth():
    """Depth, not a parent/child boolean -- a grandchild must follow its parent."""
    driver = _driver()
    known = {"a": {}, "b": {"nested_in": "a"}, "c": {"nested_in": "b"}}

    assert [driver.nesting_depth(known, k) for k in ("a", "b", "c")] == [0, 1, 2]
    assert sorted("cab", key=lambda k: driver.nesting_depth(known, k)) == ["a", "b", "c"]
    with pytest.raises(ValueError, match="cycle"):
        driver.nesting_depth({"x": {"nested_in": "y"}, "y": {"nested_in": "x"}}, "x")


def test_repin_writes_parents_before_children(monkeypatch, tmp_path):
    """``main()``'s write path, and the ordering claim it rests on.

    A ``nested_in`` child extracts against its parent's pin as it stands on
    disk (issue #148), so a child re-pinned FIRST would be extracted against
    the parent's stale shard and commit a wrong fixture -- one the drift
    guard's +/-1 count tolerance need not catch. ``repin`` is stubbed, so this
    pins the ordering and the write-back without two shard-map builds.
    """
    driver = _driver()
    bench = tmp_path / "benchmark"
    (bench / "shardmaps").mkdir(parents=True)
    (bench / "targets.json").write_text((BENCH / "targets.json").read_text())
    monkeypatch.setattr(driver, "BENCH", bench)
    monkeypatch.setattr(driver, "TARGETS", bench / "targets.json")

    class _Stub:
        metadata: dict = {}

        def to_json(self, path):
            Path(path).write_text("{}")

    seen = []

    def fake_repin(sm_key):
        # record the parent pin VISIBLE ON DISK as this entry is re-pinned
        seen.append((sm_key, int(driver.entry("healpix_o9_88s")["shard_key"])))
        return _Stub(), 4242 if sm_key == "healpix_o9_88s" else 99, 7

    monkeypatch.setattr(driver, "repin", fake_repin)
    monkeypatch.setattr(driver, "differences", lambda sm_key, mapped: [])

    assert driver.main(["healpix_o10_88s", "healpix_o9_88s"]) == 0

    assert [k for k, _ in seen] == ["healpix_o9_88s", "healpix_o10_88s"]
    # the child was extracted against the parent's NEW pin, not the run's start state
    assert seen[1][1] == 4242
    maps = json.loads((bench / "targets.json").read_text())["shardmaps"]
    assert (maps["healpix_o9_88s"]["shard_key"], maps["healpix_o9_88s"]["n_granules"]) == (4242, 7)
    assert (maps["healpix_o10_88s"]["shard_key"], maps["healpix_o10_88s"]["n_granules"]) == (99, 7)
    # ...and the prose the driver refuses to write is still the committed prose
    assert maps["healpix_o10_88s"]["note"] == MANIFEST["shardmaps"]["healpix_o10_88s"]["note"]


def test_repin_refuses_an_unknown_shardmap(capsys):
    driver = _driver()
    with pytest.raises(SystemExit):
        driver.main(["--check", "healpix_o42"])
    assert "unknown shard map(s) ['healpix_o42']" in capsys.readouterr().err
