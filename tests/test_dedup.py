"""has_run dedup statuses (issue #299 phase 4, D19).

The acceptance criteria from the issue thread: "hit" only when identity AND
catalog match; a catalog-grown shard reports "stale", never "hit"; debris and
absent leaves are plain misses.
"""

import copy

import zarr

from zagg import hive
from zagg.config import default_config
from zagg.dedup import classify_leaf_identity, has_run, shard_status
from zagg.grids import HealpixGrid
from zagg.grids.morton import morton_word
from zagg.semantics import semantic_hash
from zagg.store import open_store
from zagg.telemetry import build_record, granules_sha256, sidecar_key, write_sidecar

WORD = morton_word("1121121")  # order-6 shard key
GRANULES = ["s3://b/g1.h5", "s3://b/g2.h5"]


def _cfg():
    return default_config("atl06")


def _grid(cfg):
    return HealpixGrid(parent_order=6, child_order=8, layout="fullsphere", config=cfg)


def _write_leaf(root, cfg, *, stamp=True, sidecar=True, sidecar_hash="match", granules=GRANULES):
    """Emit + optionally stamp a leaf, optionally with a stats sidecar."""
    leaf = hive.shard_leaf_path(root, WORD)
    store = open_store(leaf)
    _grid(cfg).emit_shard_template(store, overwrite=True)
    if stamp:
        group = zarr.open_group(store, path="", mode="a", zarr_format=3)
        group.attrs[hive.COMMIT_ATTR] = {"cells_with_data": 1}
    if sidecar:
        recorded = {
            "match": semantic_hash(cfg),
            "other": "0" * 64,
            "absent": None,
        }[sidecar_hash]
        record = build_record(
            shard_key=WORD,
            metadata={"total_obs": 2, "cells_with_data": 1, "duration_s": 0.1},
            granule_ids=granules,
            semantic_hash=recorded,
        )
        write_sidecar(leaf, record)
    return leaf


class TestShardStatus:
    def test_absent_leaf_is_miss(self, tmp_path):
        cfg = _cfg()
        status = shard_status(str(tmp_path), WORD, semantic_hash=semantic_hash(cfg))
        assert status == {"status": "miss"}

    def test_unstamped_leaf_is_miss(self, tmp_path):
        # Debris (D4): a template without the commit stamp is invisible.
        cfg = _cfg()
        _write_leaf(str(tmp_path), cfg, stamp=False, sidecar=False)
        status = shard_status(str(tmp_path), WORD, semantic_hash=semantic_hash(cfg))
        assert status["status"] == "miss"

    def test_stamped_without_sidecar_is_stale(self, tmp_path):
        cfg = _cfg()
        _write_leaf(str(tmp_path), cfg, sidecar=False)
        status = shard_status(str(tmp_path), WORD, semantic_hash=semantic_hash(cfg))
        assert status["status"] == "stale"
        assert "sidecar" in status["reason"]

    def test_full_match_is_hit(self, tmp_path):
        cfg = _cfg()
        _write_leaf(str(tmp_path), cfg)
        status = shard_status(
            str(tmp_path), WORD, semantic_hash=semantic_hash(cfg), granule_ids=GRANULES
        )
        assert status["status"] == "hit"
        assert status["semantic_hash_match"] is True
        assert status["catalog_match"] is True

    def test_catalog_growth_is_stale_never_hit(self, tmp_path):
        # The headline acceptance criterion: ATL03 is a living collection —
        # a grown catalog must recompute, not skip.
        cfg = _cfg()
        _write_leaf(str(tmp_path), cfg)
        status = shard_status(
            str(tmp_path),
            WORD,
            semantic_hash=semantic_hash(cfg),
            granule_ids=[*GRANULES, "s3://b/g3.h5"],
        )
        assert status["status"] == "stale"
        assert status["catalog_match"] is False
        assert status["semantic_hash_match"] is True
        # F8: the recorded vs current digests are surfaced so a systematic
        # id-space mismatch is distinguishable from a genuine catalog change.
        assert status["granules_sha256_recorded"] != status["granules_sha256_current"]

    def test_semantic_mismatch_is_stale(self, tmp_path):
        cfg = _cfg()
        _write_leaf(str(tmp_path), cfg)
        other = copy.deepcopy(cfg)
        other.aggregation["variables"]["count"]["dtype"] = "int64"
        status = shard_status(
            str(tmp_path), WORD, semantic_hash=semantic_hash(other), granule_ids=GRANULES
        )
        assert status["status"] == "stale"
        assert status["semantic_hash_match"] is False

    def test_pre299_sidecar_is_stale(self, tmp_path):
        # A pre-#299 sidecar records no semantic_hash: unverifiable identity
        # degrades to recompute, never to a false hit.
        cfg = _cfg()
        _write_leaf(str(tmp_path), cfg, sidecar_hash="absent")
        status = shard_status(
            str(tmp_path), WORD, semantic_hash=semantic_hash(cfg), granule_ids=GRANULES
        )
        assert status["status"] == "stale"


class TestHasRun:
    def test_mapping_input_checks_catalog(self, tmp_path):
        cfg = _cfg()
        _write_leaf(str(tmp_path), cfg)
        out = has_run(str(tmp_path), cfg, {WORD: GRANULES})
        assert out[WORD]["status"] == "hit"
        grown = has_run(str(tmp_path), cfg, {WORD: [*GRANULES, "s3://b/new.h5"]})
        assert grown[WORD]["status"] == "stale"

    def test_iterable_input_skips_catalog_check(self, tmp_path):
        cfg = _cfg()
        _write_leaf(str(tmp_path), cfg)
        out = has_run(str(tmp_path), cfg, [WORD])
        assert out[WORD]["status"] == "hit"
        assert out[WORD]["catalog_match"] is None

    def test_spec_defaults_from_manifest(self, tmp_path):
        # The sidecar key grammar is spec-keyed (#307); has_run reads the
        # manifest once for it. A manifest-less root uses the legacy names.
        cfg = _cfg()
        root = str(tmp_path)
        hive.ensure_manifest(root, hive.build_manifest(_grid(cfg)))
        _write_leaf(root, cfg)
        assert has_run(root, cfg, [WORD])[WORD]["status"] == "hit"

    def test_missing_shards_reported(self, tmp_path):
        cfg = _cfg()
        _write_leaf(str(tmp_path), cfg)
        other = morton_word("2431123")
        out = has_run(str(tmp_path), cfg, [WORD, other])
        assert out[WORD]["status"] == "hit"
        assert out[other]["status"] == "miss"

    def test_windowed_leaf_status(self, tmp_path):
        # A windowed store (issue #246): the (shard, window) leaf carries its
        # own legacy `stats_{window}.json` sidecar (spec=None), and has_run must
        # key on the window so a hit for one window is a miss for another.
        cfg = _cfg()
        root = str(tmp_path)
        leaf = hive.shard_leaf_path(root, WORD, window="2025")
        assert sidecar_key(leaf.rstrip("/").rsplit("/", 1)[-1]) == "stats_2025.json"
        store = open_store(leaf)
        _grid(cfg).emit_shard_template(store, overwrite=True)
        group = zarr.open_group(store, path="", mode="a", zarr_format=3)
        group.attrs[hive.COMMIT_ATTR] = {"cells_with_data": 1}
        write_sidecar(
            leaf,
            build_record(
                shard_key=WORD,
                metadata={"total_obs": 2, "cells_with_data": 1, "duration_s": 0.1},
                granule_ids=GRANULES,
                semantic_hash=semantic_hash(cfg),
            ),
        )
        hit = has_run(root, cfg, {WORD: GRANULES}, window="2025")
        assert hit[WORD]["status"] == "hit"
        # A different window has no leaf: a plain miss (windows are disjoint).
        miss = has_run(root, cfg, {WORD: GRANULES}, window="2024")
        assert miss[WORD]["status"] == "miss"


class TestClassifyLeafIdentity:
    """Worker-side identity readings (issue #388): equal / expansion /
    contraction / mixed, with every ambiguity degrading to rewrite."""

    SEM = "a" * 64
    IDS = ["s3://b/g1.h5", "s3://b/g2.h5", "s3://b/g3.h5"]

    def _recorded(self, ids=None, semantic=SEM, with_ids=True):
        ids = self.IDS if ids is None else ids
        rec = {"semantic_hash": semantic, "granules_sha256": granules_sha256(ids)}
        if with_ids:
            rec["granule_ids"] = sorted(ids)
        return rec

    def test_equal_skips_on_the_hash_fast_path(self):
        # Order must not matter: the hash is over the sorted ids.
        got = classify_leaf_identity(
            self._recorded(), semantic_hash=self.SEM, planned_ids=self.IDS[::-1]
        )
        assert got == {"action": "skip", "classification": "equal", "missing": []}

    def test_equal_without_recorded_ids_still_skips(self):
        # The fast path never needs the id list, so pre-#388 sidecars skip too.
        got = classify_leaf_identity(
            self._recorded(with_ids=False), semantic_hash=self.SEM, planned_ids=self.IDS
        )
        assert got["action"] == "skip"

    def test_expansion_rewrites(self):
        got = classify_leaf_identity(
            self._recorded(), semantic_hash=self.SEM, planned_ids=self.IDS + ["s3://b/g4.h5"]
        )
        assert got == {"action": "rewrite", "classification": "expansion", "missing": []}

    def test_pure_contraction_refuses_and_names_ids(self):
        got = classify_leaf_identity(
            self._recorded(), semantic_hash=self.SEM, planned_ids=self.IDS[:2]
        )
        assert got["action"] == "refuse"
        assert got["classification"] == "contraction"
        assert got["missing"] == ["s3://b/g3.h5"]

    def test_mixed_add_and_drop_refuses(self):
        # The ruled predicate is recorded - planned != {} — NOT strict subset:
        # the planned set can even be LARGER while data drops (the upstream
        # purge behind a fresh catalog query, the espg contraction ruling).
        planned = self.IDS[:2] + ["s3://b/new1.h5", "s3://b/new2.h5"]
        got = classify_leaf_identity(self._recorded(), semantic_hash=self.SEM, planned_ids=planned)
        assert got["action"] == "refuse"
        assert got["classification"] == "mixed"
        assert got["missing"] == ["s3://b/g3.h5"]

    def test_contraction_beats_semantic_mismatch(self):
        # Dropping inputs refuses even when the semantic hash also changed —
        # the guard is about data loss, not intent drift.
        got = classify_leaf_identity(
            self._recorded(semantic="b" * 64), semantic_hash=self.SEM, planned_ids=self.IDS[:1]
        )
        assert got["action"] == "refuse"
        assert got["missing"] == sorted(self.IDS[1:])

    def test_no_sidecar_rewrites(self):
        got = classify_leaf_identity(None, semantic_hash=self.SEM, planned_ids=self.IDS)
        assert got == {"action": "rewrite", "classification": "no-sidecar", "missing": []}

    def test_pre388_sidecar_on_mismatch_rewrites(self):
        # Hash mismatch + no recorded id list: undecidable, today's rewrite.
        got = classify_leaf_identity(
            self._recorded(with_ids=False), semantic_hash=self.SEM, planned_ids=self.IDS[:2]
        )
        assert got == {"action": "rewrite", "classification": "unrecorded-ids", "missing": []}

    def test_semantic_mismatch_with_equal_sets_rewrites_not_refuses(self):
        got = classify_leaf_identity(
            self._recorded(semantic="b" * 64), semantic_hash=self.SEM, planned_ids=self.IDS
        )
        assert got == {"action": "rewrite", "classification": "semantic-mismatch", "missing": []}

    def test_null_recorded_semantic_never_skips(self):
        # Fleet-written pre-#388 vector sidecars record semantic_hash null
        # (the Lambda handler passes none): never provably current.
        got = classify_leaf_identity(
            self._recorded(semantic=None), semantic_hash=self.SEM, planned_ids=self.IDS
        )
        assert got["action"] == "rewrite"

    def test_caller_without_semantic_hash_never_skips(self):
        got = classify_leaf_identity(self._recorded(), semantic_hash=None, planned_ids=self.IDS)
        assert got["action"] == "rewrite"

    def test_empty_planned_set_is_pure_contraction(self):
        got = classify_leaf_identity(self._recorded(), semantic_hash=self.SEM, planned_ids=[])
        assert got["action"] == "refuse"
        assert got["classification"] == "contraction"
        assert got["missing"] == sorted(self.IDS)

    def test_duplicate_drift_with_equal_sets_rewrites(self):
        # granules_sha256 keeps duplicates; the set diff dedups. Same set,
        # different multiset -> not current, not a contraction.
        rec = self._recorded(ids=self.IDS + [self.IDS[0]])
        got = classify_leaf_identity(rec, semantic_hash=self.SEM, planned_ids=self.IDS)
        assert got == {"action": "rewrite", "classification": "id-multiset-drift", "missing": []}
