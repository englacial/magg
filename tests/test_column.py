"""Leaf-worker pyramid columns (issue #383): fold core.

Phase 1 covers the pure fold core: the resolution set a ``zagg-pyramid/2``
declaration puts in a leaf-node column, the staged-sink adapter, and the
per-resolution folds — with the headline byte-parity check against the
sweep's own from-leaves fold (``sweep_overviews``) over the same committed
leaf, which is the issue #383 acceptance contract.
"""

import json

import numpy as np
import obstore
import pytest
import zarr

from zagg.column import column_resolutions, fold_column, leaf_slabs
from zagg.grids.morton import morton_word
from zagg.hive import MANIFEST_NAME, shard_leaf_path, stamp_commit
from zagg.stats.tdigest import build_tdigest, merge_tdigests_kway
from zagg.store import open_object_store, open_store
from zagg.sweep_overview import PYRAMID_SPEC, decode_digest, encode_digest, fold_dense

SHARD_ORDER = 2
CELL_ORDER = 4
LEAF_CELLS = 4 ** (CELL_ORDER - SHARD_ORDER)
DELTA = 64

#: The /2 declaration's per-field map (same shape under both revisions).
FIELDS = {
    "count": {"class": "exact", "method": "sum", "dtype": "int32", "fill_value": 0},
    "h_min": {"class": "exact", "method": "min", "dtype": "float32", "fill_value": "NaN"},
    "h_tdigest": {
        "class": "approximate",
        "method": "tdigest_kway",
        "dtype": "float32",
        "inner_shape": [2],
        "delta": DELTA,
    },
}


def _leaf_cfg():
    from zagg.config import PipelineConfig

    return PipelineConfig(
        aggregation={
            "coordinates": {"morton": {"dtype": "uint64", "fill_value": 0}},
            "variables": {
                "count": {"function": "len", "dtype": "int32", "fill_value": 0},
                "h_min": {"function": "min", "dtype": "float32"},
                "h_tdigest": {
                    "kind": "ragged",
                    "function": "zagg.stats.tdigest.build_tdigest",
                    "inner_shape": [2],
                    "dtype": "float32",
                    "fill_value": 0,
                },
            },
        }
    )


def _cell_slabs(cells: dict) -> dict:
    """The leaf's resident per-cell slabs (``{leaf row: observations}``)."""
    count = np.zeros(LEAF_CELLS, np.int32)
    h_min = np.full(LEAF_CELLS, np.nan, np.float32)
    digest = np.full(LEAF_CELLS, b"", dtype=object)
    for i, obs in cells.items():
        obs = np.asarray(obs, dtype=np.float64)
        count[i] = len(obs)
        h_min[i] = obs.min()
        digest[i] = encode_digest(build_tdigest(obs, delta=DELTA), "float32")
    return {"count": count, "h_min": h_min, "h_tdigest": digest}


def _make_leaf(root, decimal, cells):
    """One committed leaf on disk; returns its resident slabs (fold inputs)."""
    from mortie import generate_morton_children

    from zagg.grids.healpix import HealpixGrid

    grid = HealpixGrid(SHARD_ORDER, CELL_ORDER, config=_leaf_cfg())
    word = morton_word(decimal)
    store = open_store(shard_leaf_path(str(root), word))
    grid.emit_shard_template(store, overwrite=True)
    group = zarr.open_group(store, path=str(CELL_ORDER), mode="r+", zarr_format=3)
    group["morton"][:] = np.asarray(generate_morton_children(word, CELL_ORDER), dtype=np.uint64)
    slabs = _cell_slabs(cells)
    for name, slab in slabs.items():
        group[name][:] = slab
    stamp_commit(store, cells_with_data=len(cells), granule_count=1)
    return slabs


class TestColumnResolutions:
    def test_default_schedule_reaches_node_and_members(self):
        from zagg.pyramid import default_overviews

        # 19/13/9 reference geometry: base (9,[13]) + implied 11/9 members
        # from the (7,[11]) and (5,[9]) declarations + the node-order member.
        levels = default_overviews(9, 13, child_order=19)
        assert column_resolutions(levels, 9) == [13, 11, 9]

    def test_small_geometry_default(self):
        from zagg.pyramid import default_overviews

        levels = default_overviews(SHARD_ORDER, 3, child_order=CELL_ORDER)
        assert column_resolutions(levels, SHARD_ORDER) == [3, 2]

    def test_lone_base_entry_adds_the_node_member(self):
        assert column_resolutions([{"node": 2, "cells": [3]}], 2) == [3, 2]

    def test_spelled_node_member_dedupes(self):
        assert column_resolutions([{"node": 2, "cells": [3, 2]}], 2) == [3, 2]

    def test_coarser_members_are_not_column_groups(self):
        # (0,[1])'s resolution is coarser than the node: the leaf's
        # contribution to it is the node-order member itself.
        levels = [{"node": 2, "cells": [3]}, {"node": 0, "cells": [1]}]
        assert column_resolutions(levels, 2) == [3, 2]

    def test_no_leaf_node_entry_means_no_column(self):
        levels = [{"node": 1, "cells": [3]}, {"node": 0, "cells": [1]}]
        assert column_resolutions(levels, 2) == []
        assert column_resolutions([], 2) == []


class TestLeafSlabs:
    def test_staged_refs_pass_through(self):
        slabs = _cell_slabs({0: [1.0, 2.0]})
        staged = {f"{CELL_ORDER}/{n}": s for n, s in slabs.items()}
        out = leaf_slabs(staged, FIELDS, group_path=str(CELL_ORDER), n_cells=LEAF_CELLS)
        assert out["count"] is slabs["count"] and out["h_tdigest"] is slabs["h_tdigest"]

    def test_absent_fields_synthesize_fill(self):
        out = leaf_slabs({}, FIELDS, group_path=str(CELL_ORDER), n_cells=LEAF_CELLS)
        np.testing.assert_array_equal(out["count"], np.zeros(LEAF_CELLS, np.int32))
        assert np.isnan(out["h_min"]).all() and out["h_min"].dtype == np.float32
        assert all(p == b"" for p in out["h_tdigest"])

    def test_wrong_extent_refuses(self):
        staged = {f"{CELL_ORDER}/count": np.zeros(3, np.int32)}
        with pytest.raises(ValueError, match="cell extent"):
            leaf_slabs(staged, FIELDS, group_path=str(CELL_ORDER), n_cells=LEAF_CELLS)


class TestFoldColumn:
    CELLS = {0: [1.0, 2.0], 5: [10.0, 4.0], 6: [7.0], 15: [3.0, 8.0, 5.0]}

    def test_exact_fields_match_the_sweep_kernel(self):
        slabs = _cell_slabs(self.CELLS)
        folded = fold_column(slabs, FIELDS, cell_order=CELL_ORDER, resolutions=[3, 2])
        for res in (3, 2):
            factor = 4 ** (CELL_ORDER - res)
            np.testing.assert_array_equal(
                folded[res]["count"], fold_dense(slabs["count"], factor, "sum", 0)
            )
            np.testing.assert_array_equal(
                folded[res]["h_min"], fold_dense(slabs["h_min"], factor, "min", "NaN")
            )

    def test_exact_fields_match_direct_aggregation(self):
        # The D24 exact contract one hop further: byte-equal to aggregating
        # the observations directly at the coarser order (nan-skipping).
        slabs = _cell_slabs(self.CELLS)
        folded = fold_column(slabs, FIELDS, cell_order=CELL_ORDER, resolutions=[3])
        np.testing.assert_array_equal(folded[3]["count"][[0, 1, 3]], [2, 3, 3])
        assert folded[3]["h_min"][0] == np.float32(1.0)
        assert folded[3]["h_min"][1] == np.float32(4.0)
        assert np.isnan(folded[3]["h_min"][2])

    def test_digests_are_the_kway_fold_of_resident_cells(self):
        slabs = _cell_slabs(self.CELLS)
        folded = fold_column(slabs, FIELDS, cell_order=CELL_ORDER, resolutions=[3])
        # Row 1 pools cells 5 and 6 — a genuine multi-input k-way merge.
        oracle = merge_tdigests_kway(
            [decode_digest(slabs["h_tdigest"][i], "float32") for i in (5, 6)], delta=DELTA
        )
        assert bytes(folded[3]["h_tdigest"][1]) == encode_digest(oracle, "float32")
        # Row 0 has one contributor: passes through un-recompressed.
        assert bytes(folded[3]["h_tdigest"][0]) == bytes(slabs["h_tdigest"][0])
        # Empty rows keep the ragged fill.
        assert folded[3]["h_tdigest"][2] == b""

    def test_node_member_is_the_whole_footprint_aggregate(self):
        slabs = _cell_slabs(self.CELLS)
        folded = fold_column(slabs, FIELDS, cell_order=CELL_ORDER, resolutions=[SHARD_ORDER])
        group = folded[SHARD_ORDER]
        assert group["count"].shape == (1,) and group["count"][0] == 8
        assert group["h_min"][0] == np.float32(1.0)
        oracle = merge_tdigests_kway(
            [decode_digest(slabs["h_tdigest"][i], "float32") for i in sorted(self.CELLS)],
            delta=DELTA,
        )
        assert bytes(group["h_tdigest"][0]) == encode_digest(oracle, "float32")

    def test_fold_is_deterministic(self):
        slabs = _cell_slabs(self.CELLS)
        a = fold_column(slabs, FIELDS, cell_order=CELL_ORDER, resolutions=[3, 2])
        b = fold_column(slabs, FIELDS, cell_order=CELL_ORDER, resolutions=[3, 2])
        for res in (3, 2):
            for name in FIELDS:
                if FIELDS[name]["class"] == "exact":
                    np.testing.assert_array_equal(a[res][name], b[res][name])
                else:
                    assert [bytes(p) for p in a[res][name]] == [bytes(p) for p in b[res][name]]


class TestSweepParity:
    """The issue #383 headline: column groups == the sweep's from-leaves fold."""

    def test_column_matches_the_overview_sweep_bytes(self, tmp_path):
        from zagg.sweep import run_sweep

        # A /1 store whose order-1 overview holds cells at order 3 — the same
        # resolution a (2, [3]) column group folds — sourced from ONE leaf, so
        # the overview rows covering that leaf are exactly the leaf's fold.
        manifest = {
            "spec": "morton-hive/1",
            "dataset": {"short_name": "TEST", "version": "1"},
            "cell_order": CELL_ORDER,
            "shard_order": SHARD_ORDER,
            "split_schedule": [1] * SHARD_ORDER,
            "pyramid": {
                "spec": PYRAMID_SPEC,
                "overview": {
                    "spacing": 2,
                    "orders": [1],
                    "all_time": False,
                    "fields": dict(FIELDS),
                },
            },
            "generated_at": "2026-01-01T00:00:00+00:00",
        }
        obstore.put(open_object_store(str(tmp_path)), MANIFEST_NAME, json.dumps(manifest).encode())
        cells = {0: [1.0, 2.0], 5: [10.0, 4.0], 6: [7.0], 15: [3.0, 8.0, 5.0]}
        slabs = _make_leaf(tmp_path, "-311", cells)
        result = run_sweep(str(tmp_path), [(morton_word("-311"), None)], families=("overview",))
        assert result["families"]["overview"]["written"] == 1

        folded = fold_column(slabs, FIELDS, cell_order=CELL_ORDER, resolutions=[3])
        overview = zarr.open_group(
            open_store(f"{tmp_path}/-3/1/all.zarr"), path="3", mode="r", zarr_format=3
        )
        # Leaf -311 is child 0 of node -31: overview rows 0..3 are its fold.
        n = 4 ** (3 - SHARD_ORDER)
        for name, meta in FIELDS.items():
            stored = overview[name][:n]
            if meta["class"] == "exact":
                assert stored.dtype == folded[3][name].dtype
                np.testing.assert_array_equal(stored, folded[3][name])
            else:
                assert [bytes(p) for p in stored] == [bytes(p) for p in folded[3][name]]
