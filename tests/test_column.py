"""Leaf-worker pyramid columns (issue #383): fold core.

Phase 1 covers the pure fold core: the resolution set a ``zagg-pyramid/2``
declaration puts in a leaf-node column, the staged-sink adapter, and the
per-resolution folds — with the headline byte-parity check against the
sweep's own from-leaves fold (``sweep_overviews``) over the same committed
leaf, which is the issue #383 acceptance contract.
"""

import json
import pathlib

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


class TestNoneClassFields:
    """D24 ``class: "none"`` entries never become column content (option A)."""

    #: What `zagg.pyramid.declared_fields` records for a non-composable field:
    #: the class alone. `h_vec` stands for a vector field (staged dense at
    #: (n_cells, k)), `h_chunks` for a chunk-resolution companion (written per
    #: chunk-block, so never in the leaf's cell-slab sink).
    DECL = dict(FIELDS, h_vec={"class": "none"}, h_chunks={"class": "none"})

    def test_staged_vector_field_is_not_a_grid_disagreement(self):
        # Failure mode (1): the (n_cells, k) slab tripping the extent refusal
        # and killing the whole column fold, blaming a sink that is correct.
        slabs = _cell_slabs({0: [1.0, 2.0]})
        staged = {f"{CELL_ORDER}/{n}": s for n, s in slabs.items()}
        staged[f"{CELL_ORDER}/h_vec"] = np.zeros((LEAF_CELLS, 3), np.float32)
        out = leaf_slabs(staged, self.DECL, group_path=str(CELL_ORDER), n_cells=LEAF_CELLS)
        assert set(out) == set(FIELDS)

    def test_absent_companion_is_not_an_empty_ragged_group(self):
        # Failure mode (2): synthesizing b"" fill for a field that exists ONLY
        # at native resolution, then writing it as an all-empty column group.
        slabs = _cell_slabs({0: [1.0, 2.0]})
        staged = {f"{CELL_ORDER}/{n}": s for n, s in slabs.items()}
        out = leaf_slabs(staged, self.DECL, group_path=str(CELL_ORDER), n_cells=LEAF_CELLS)
        assert "h_chunks" not in out
        folded = fold_column(out, self.DECL, cell_order=CELL_ORDER, resolutions=[3, SHARD_ORDER])
        assert all(set(group) == set(FIELDS) for group in folded.values())

    def test_filtered_fold_matches_the_pre_filtered_one(self):
        slabs = _cell_slabs({0: [1.0, 2.0], 5: [10.0, 4.0]})
        a = fold_column(slabs, self.DECL, cell_order=CELL_ORDER, resolutions=[3])
        b = fold_column(slabs, FIELDS, cell_order=CELL_ORDER, resolutions=[3])
        np.testing.assert_array_equal(a[3]["count"], b[3]["count"])
        assert [bytes(p) for p in a[3]["h_tdigest"]] == [bytes(p) for p in b[3]["h_tdigest"]]


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

    def test_declared_delta_bounds_an_over_budget_merge(self):
        # Every CELLS digest is far under the δ budget, so no merge above ever
        # re-compresses and `delta` is unobservable. These two cells share
        # order-3 row 1 and their merge IS over budget, so the payload depends
        # on the declared δ — one of the three values that decide the bytes.
        slabs = _cell_slabs({4: np.arange(200.0), 5: np.arange(200.0, 400.0)})
        parts = [decode_digest(slabs["h_tdigest"][i], "float32") for i in (4, 5)]
        merged = merge_tdigests_kway(parts, delta=DELTA)
        assert len(merged) < sum(len(p) for p in parts)  # re-compression bit
        folded = fold_column(slabs, FIELDS, cell_order=CELL_ORDER, resolutions=[3])
        assert bytes(folded[3]["h_tdigest"][1]) == encode_digest(merged, "float32")
        # ... and a different δ is different bytes: the DECLARED one is used.
        other = merge_tdigests_kway(parts, delta=DELTA // 4)
        assert bytes(folded[3]["h_tdigest"][1]) != encode_digest(other, "float32")

    def test_resolution_finer_than_the_cells_refuses_by_name(self):
        # 4^(cell_order - res) would be fractional; the downstream divisibility
        # guards read 16 % 0.25 as clean and fail opaquely inside numpy.
        slabs = _cell_slabs(self.CELLS)
        with pytest.raises(ValueError, match="FINER than the leaf's cell order"):
            fold_column(slabs, FIELDS, cell_order=CELL_ORDER, resolutions=[CELL_ORDER + 1])

    def test_indivisible_ragged_slab_refuses_by_name(self):
        # The ragged guard, reached with only the approximate field declared
        # (the exact class gets the identically-worded refusal from fold_dense).
        ragged = {"h_tdigest": FIELDS["h_tdigest"]}
        slabs = {"h_tdigest": _cell_slabs(self.CELLS)["h_tdigest"][:6]}
        with pytest.raises(ValueError, match="cannot fold 6 cells 4-to-one for 'h_tdigest'"):
            fold_column(slabs, ragged, cell_order=CELL_ORDER, resolutions=[3])

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


def _assert_group_matches(overview, group: dict, n: int) -> None:
    """The leaf's ``n`` overview rows are byte-equal to its column group."""
    for name, meta in FIELDS.items():
        stored = overview[name][:n]
        if meta["class"] == "exact":
            assert stored.dtype == group[name].dtype
            np.testing.assert_array_equal(stored, group[name][:n])
        else:
            assert [bytes(p) for p in stored] == [bytes(p) for p in group[name][:n]]


class TestSweepParity:
    """The issue #383 headline: column groups == the sweep's from-leaves fold."""

    def test_column_matches_the_overview_sweep_bytes(self, tmp_path):
        from zagg.sweep import run_sweep

        # A /1 store sourced from ONE leaf, so the overview rows covering that
        # leaf are exactly the leaf's own fold. Both column group kinds are
        # covered: the order-1 overview holds cells at order 3 (the resolution
        # a (2, [3]) column group folds, `_fold_node`'s target >= shard arm),
        # and the order-0 overview holds cells at order 2 == the shard order —
        # the node-order member, where the leaf folds whole into one cell.
        # `fold_source: "leaves"` because from-leaves IS the parity contract.
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
                    "orders": [1, 0],
                    "fold_source": "leaves",
                    "all_time": False,
                    "fields": dict(FIELDS),
                },
            },
            "generated_at": "2026-01-01T00:00:00+00:00",
        }
        obstore.put(open_object_store(str(tmp_path)), MANIFEST_NAME, json.dumps(manifest).encode())
        # Cells 5 and 6 hold enough observations that their merge is over the
        # δ budget at both folded resolutions, so the declared delta decides
        # the parity bytes rather than being carried unobserved.
        cells = {0: [1.0, 2.0], 5: np.arange(200.0), 6: np.arange(200.0, 400.0), 15: [3.0, 8.0]}
        slabs = _make_leaf(tmp_path, "-311", cells)
        result = run_sweep(str(tmp_path), [(morton_word("-311"), None)], families=("overview",))
        assert result["families"]["overview"]["written"] == 2

        folded = fold_column(slabs, FIELDS, cell_order=CELL_ORDER, resolutions=[3, SHARD_ORDER])
        overview = zarr.open_group(
            open_store(f"{tmp_path}/-3/1/all.zarr"), path="3", mode="r", zarr_format=3
        )
        # Leaf -311 is child 0 of node -31: overview rows 0..3 are its fold.
        _assert_group_matches(overview, folded[3], 4 ** (3 - SHARD_ORDER))
        # The node-order member: leaf -311 is _rel_rank 0 under node -3, and
        # the order-0 fold's factor is the whole leaf (4^(shard_order - 0)).
        node_overview = zarr.open_group(
            open_store(f"{tmp_path}/-3/all.zarr"), path=str(SHARD_ORDER), mode="r", zarr_format=3
        )
        _assert_group_matches(node_overview, folded[SHARD_ORDER], 1)


def _root_meta_sans_timestamps(store) -> dict:
    """The column root's attrs with the wall-clock provenance stripped."""
    from zagg.column import COLUMN_ATTR
    from zagg.hive import COMMIT_ATTR

    attrs = dict(zarr.open_group(store, mode="r", zarr_format=3).attrs)
    attrs[COLUMN_ATTR] = {k: v for k, v in attrs[COLUMN_ATTR].items() if k != "generated_at"}
    attrs[COMMIT_ATTR] = {k: v for k, v in attrs[COMMIT_ATTR].items() if k != "written_at"}
    return attrs


class TestWriteColumn:
    """Phase 2: the column writer — D4 order, attrs, naming, sidecar."""

    CELLS = {0: [1.0, 2.0], 5: [10.0, 4.0], 6: [7.0], 15: [3.0, 8.0, 5.0]}

    def _write(self, root, *, window=None, time_range=None, cells=None):
        from zagg.column import fold_column, write_column

        slabs = _cell_slabs(cells if cells is not None else self.CELLS)
        folded = fold_column(slabs, FIELDS, cell_order=CELL_ORDER, resolutions=[3, SHARD_ORDER])
        basename = write_column(
            str(root),
            morton_word("-311"),
            folded,
            FIELDS,
            node_order=SHARD_ORDER,
            cell_order=CELL_ORDER,
            window=window,
            time_range=time_range,
            granule_count=3,
        )
        return basename, folded

    def test_layout_groups_and_morton(self, tmp_path):
        from mortie import generate_morton_children

        basename, folded = self._write(tmp_path)
        assert basename == "all.pyramid.zarr"
        word = morton_word("-311")
        store = open_store(f"{tmp_path}/-3/1/1/{basename}")
        for res, n in ((3, 4), (SHARD_ORDER, 1)):
            group = zarr.open_group(store, path=str(res), mode="r", zarr_format=3)
            np.testing.assert_array_equal(
                group["morton"][:], np.asarray(generate_morton_children(word, res), np.uint64)
            )
            for name, meta in FIELDS.items():
                if meta["class"] == "none":
                    assert name not in group
                elif meta["class"] == "exact":
                    np.testing.assert_array_equal(group[name][:], folded[res][name])
                else:
                    stored = [bytes(p) for p in group[name][:]]
                    assert stored == [bytes(p) for p in folded[res][name]]

    def test_role_and_provenance_attrs(self, tmp_path):
        from zagg.column import COLUMN_ATTR, COLUMN_ROLE, COLUMN_SPEC, LEAF_REGIME

        basename, _folded = self._write(tmp_path)
        root = zarr.open_group(open_store(f"{tmp_path}/-3/1/1/{basename}"), mode="r", zarr_format=3)
        assert root.attrs["role"] == COLUMN_ROLE
        attrs = dict(root.attrs[COLUMN_ATTR])
        assert attrs["spec"] == COLUMN_SPEC
        assert attrs["node"] == "-311" and attrs["order"] == SHARD_ORDER
        assert attrs["source_cell_order"] == CELL_ORDER and attrs["window"] == "all"
        assert set(attrs["fields"]) == {"count", "h_min", "h_tdigest"}
        assert attrs["fields"]["count"] == {"class": "exact", "method": "sum", "nan_policy": "skip"}
        # An approximate entry also carries what decided its centroid bytes:
        # a #370 k-way gather cannot recover δ from the leaf.
        assert attrs["fields"]["h_tdigest"] == {
            "class": "approximate",
            "method": "tdigest_kway",
            "delta": DELTA,
            "dtype": "float32",
            "inner_shape": [2],
        }
        assert attrs["groups"] == {
            "3": {"regime": LEAF_REGIME, "merges_from_raw": 1, "n_cells": 4},
            str(SHARD_ORDER): {"regime": LEAF_REGIME, "merges_from_raw": 1, "n_cells": 1},
        }
        # A column has N grids, so the stamp's count needs its denominator
        # named: the finest group, which is declaration-dependent.
        assert attrs["cells_with_data_order"] == 3

    def test_stamp_covers_the_column_and_is_last(self, tmp_path, monkeypatch):
        from zagg.column import fold_column, write_column
        from zagg.hive import read_commit

        basename, _folded = self._write(tmp_path)
        store = open_store(f"{tmp_path}/-3/1/1/{basename}")
        stamp = read_commit(store)
        # CELLS occupies leaf cells 0, 5, 6, 15 -> base-group rows 0, 1, 3:
        # the stamp counts populated cells of the FINEST group (3), like the
        # overview writer's populated mask.
        assert stamp is not None and stamp["cells_with_data"] == 3
        assert stamp["granule_count"] == 3

        # An interrupted writer (death before the stamp) leaves ignorable
        # debris: the artifact prefix exists, but read_commit sees no stamp.
        import zagg.column as column_mod

        def _boom(*a, **k):
            raise RuntimeError("interrupted before the stamp")

        monkeypatch.setattr("zagg.hive.stamp_commit", _boom)
        slabs = _cell_slabs(self.CELLS)
        folded = fold_column(slabs, FIELDS, cell_order=CELL_ORDER, resolutions=[3, SHARD_ORDER])
        with pytest.raises(RuntimeError, match="interrupted"):
            write_column(
                str(tmp_path),
                morton_word("-312"),
                folded,
                FIELDS,
                node_order=SHARD_ORDER,
                cell_order=CELL_ORDER,
            )
        debris = open_store(f"{tmp_path}/-3/1/2/{column_mod.column_name(None)}")
        group = zarr.open_group(debris, path="3", mode="r", zarr_format=3)  # arrays landed
        assert group["count"].shape == (4,)
        assert read_commit(debris) is None  # ...but the column is debris (D4)

    def test_windowed_naming_and_stamp(self, tmp_path):
        from zagg.hive import read_commit

        rng = ["2019-01-02T00:00:00+00:00", "2019-11-30T00:00:00+00:00"]
        basename, _folded = self._write(tmp_path, window="2019", time_range=rng)
        assert basename == "2019.pyramid.zarr"
        stamp = read_commit(open_store(f"{tmp_path}/-3/1/1/{basename}"))
        assert stamp["window"] == "2019" and stamp["time_range"] == rng

    def test_sidecar_lands_after_the_stamp_and_fails_open(self, tmp_path, monkeypatch):
        basename, _folded = self._write(tmp_path)
        sidecar = tmp_path / "-3" / "1" / "1" / "all.pyramid.stats.json"
        record = json.loads(sidecar.read_text())
        assert record["cells_with_data"] == 3 and record["n_granules"] == 3
        hashes = record["content_hashes"]["arrays"]
        assert set(hashes) >= {"3/morton", "3/count", "3/h_tdigest", f"{SHARD_ORDER}/morton"}

        # Fail-open: a hashing failure costs the sidecar, never the column —
        # and on a REWRITE the outcome is ABSENT, not the previous run's
        # record. A stale sidecar verifies as a mismatch (a false tamper
        # signal), the one outcome D20 exists to prevent; the wholesale clear
        # drops it before the template lands, so this rewrite is over a
        # sidecar that is still on disk.
        monkeypatch.setattr(
            "zagg.content_hash.hash_arrays",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        assert sidecar.exists()
        basename, _folded = self._write(tmp_path, cells={0: [1.0]})
        from zagg.hive import read_commit

        assert read_commit(open_store(f"{tmp_path}/-3/1/1/{basename}")) is not None
        assert not sidecar.exists()

    def test_attrs_window_round_trips_through_the_name(self, tmp_path):
        # The attrs record the unwindowed column's window as the reserved
        # SCHEDULE_NONE_TOKEN, and `leaf_name_v3` RAISES on that token — so
        # the basename derivation normalizes it back, exactly as
        # `_overview_basename` does, and attrs -> name round-trips.
        from zagg.column import COLUMN_ATTR, column_name
        from zagg.windows import SCHEDULE_NONE_TOKEN

        basename, _folded = self._write(tmp_path)
        root = zarr.open_group(open_store(f"{tmp_path}/-3/1/1/{basename}"), mode="r", zarr_format=3)
        assert root.attrs[COLUMN_ATTR]["window"] == SCHEDULE_NONE_TOKEN
        assert column_name(root.attrs[COLUMN_ATTR]["window"]) == basename
        assert column_name(SCHEDULE_NONE_TOKEN) == column_name(None) == "all.pyramid.zarr"

    def test_sidecar_import_failure_never_fails_the_column(self, tmp_path, monkeypatch):
        # Every sidecar-only import sits INSIDE the fail-open try: a
        # telemetry-class ImportError (a slimmed runtime, say) must cost the
        # sidecar, never a column whose stamp already landed.
        import builtins

        from zagg.hive import read_commit

        real_import = builtins.__import__

        def _boom(name, *args, **kwargs):
            if name == "zagg.telemetry":
                raise ImportError("no telemetry on this runtime")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _boom)
        basename, _folded = self._write(tmp_path)
        monkeypatch.undo()
        assert read_commit(open_store(f"{tmp_path}/-3/1/1/{basename}")) is not None
        assert not (tmp_path / "-3" / "1" / "1" / "all.pyramid.stats.json").exists()

    def test_idempotent_rewrite_same_array_bytes(self, tmp_path):
        """A re-run reproduces the DATA bytes; provenance timestamps move.

        What is pinned: every resolution group's array bytes, and the root
        attrs + commit stamp modulo their timestamps. What is NOT: the root
        ``zarr.json`` byte-for-byte, nor the sidecar — ``generated_at``, the
        stamp's ``written_at`` and the sidecar's ``timestamp`` are wall-clock
        provenance and legitimately differ (the same posture as
        ``_write_overview``). Comparing raw objects would make the outcome
        clock-dependent: identical inside one second, different a second later.
        """
        basename, first = self._write(tmp_path)
        store = open_store(f"{tmp_path}/-3/1/1/{basename}")
        before = {
            (res, name): ([bytes(p) for p in arr[:]] if arr.dtype == object else arr[:].tobytes())
            for res in (3, SHARD_ORDER)
            for name, arr in zarr.open_group(store, path=str(res), mode="r", zarr_format=3).arrays()
        }
        meta_before = _root_meta_sans_timestamps(store)
        basename2, _second = self._write(tmp_path)
        assert basename2 == basename
        after = {
            (res, name): ([bytes(p) for p in arr[:]] if arr.dtype == object else arr[:].tobytes())
            for res in (3, SHARD_ORDER)
            for name, arr in zarr.open_group(store, path=str(res), mode="r", zarr_format=3).arrays()
        }
        assert before == after
        # The root object is rewritten too: same attrs and same stamp CONTENT
        # (a regression that changed either would pass a data-bytes-only check).
        assert _root_meta_sans_timestamps(store) == meta_before

    def test_rewrite_spares_the_sibling_leaf_and_its_sidecar(self, tmp_path):
        # The wholesale clear (issue #341 semantics) is scoped to the column's
        # own prefix: the node dir now holds three siblings, and the leaf is
        # the one object a mis-scoped delete would eat.
        from zagg.hive import read_commit

        slabs = _make_leaf(tmp_path, "-311", self.CELLS)
        leaf_sidecar = tmp_path / "-3" / "1" / "1" / "-311.stats.json"
        leaf_sidecar.write_text("{}")
        basename, _folded = self._write(tmp_path)
        basename, _folded = self._write(tmp_path)  # ...and again, the rewrite

        leaf = open_store(shard_leaf_path(str(tmp_path), morton_word("-311")))
        assert read_commit(leaf) is not None
        group = zarr.open_group(leaf, path=str(CELL_ORDER), mode="r", zarr_format=3)
        np.testing.assert_array_equal(group["count"][:], slabs["count"])
        assert leaf_sidecar.read_text() == "{}"
        assert (tmp_path / "-3" / "1" / "1" / "all.pyramid.stats.json").exists()

    def test_clear_is_scoped_on_an_obstore_prefix_store(self, tmp_path, monkeypatch):
        # The string-prefix hazard only exists on the fleet backend, where
        # `open_store` builds `zarr.storage.ObjectStore(store=S3Store(prefix=…))`
        # and zarr's `ObjectStore.delete_dir` leaves the EMPTY prefix un-slashed
        # — scoping rests on obstore's prefix store re-adding the delimiter.
        # Mirrors tests/test_hive.py's
        # `test_overwrite_clear_is_scoped_on_an_obstore_prefix_store`, using
        # obstore's own LocalStore in prefix mode (the same composition).
        import obstore as _obstore
        from zarr.storage import ObjectStore

        import zagg.store as store_mod

        def _prefixed(path, **kwargs):
            pathlib.Path(path).mkdir(parents=True, exist_ok=True)
            return ObjectStore(store=_obstore.store.LocalStore(prefix=path))

        monkeypatch.setattr(store_mod, "open_store", _prefixed)
        basename, _folded = self._write(tmp_path)
        node = tmp_path / "-3" / "1" / "1"
        # The adversarial shape: a name-prefix sibling of the column, which a
        # missing delimiter would sweep up with it.
        twin = node / f"{basename}.status"
        twin.mkdir()
        (twin / "r.json").write_text("{}")
        stray = node / basename / "3" / "stray.json"
        stray.write_text("{}")  # in-column debris
        self._write(tmp_path)

        assert not stray.exists()  # the clear really ran
        assert (twin / "r.json").read_text() == "{}"
        assert (node / "all.pyramid.stats.json").exists()

    def test_narrowed_rewrite_drops_the_retired_group(self, tmp_path):
        # The docstring's "a prior torn write never survives": a rewrite under
        # a narrower declaration leaves no orphan group behind.
        from zagg.column import fold_column, write_column

        self._write(tmp_path)
        node = tmp_path / "-3" / "1" / "1" / "all.pyramid.zarr"
        assert (node / "3").exists()
        slabs = _cell_slabs(self.CELLS)
        write_column(
            str(tmp_path),
            morton_word("-311"),
            fold_column(slabs, FIELDS, cell_order=CELL_ORDER, resolutions=[SHARD_ORDER]),
            FIELDS,
            node_order=SHARD_ORDER,
            cell_order=CELL_ORDER,
        )
        assert not (node / "3").exists()
        assert (node / str(SHARD_ORDER)).exists()

    def test_missing_node_member_refuses_by_name(self, tmp_path):
        # The node-order member is the universal partial #384's gather may
        # assume (#381 point (2)); a column without it would stamp complete
        # and fold short, so the writer refuses before anything lands.
        from zagg.column import fold_column, write_column

        slabs = _cell_slabs(self.CELLS)
        folded = fold_column(slabs, FIELDS, cell_order=CELL_ORDER, resolutions=[3])
        with pytest.raises(ValueError, match="node-order member"):
            write_column(
                str(tmp_path),
                morton_word("-311"),
                folded,
                FIELDS,
                node_order=SHARD_ORDER,
                cell_order=CELL_ORDER,
            )
        assert not (tmp_path / "-3").exists()

    def test_empty_fold_refuses_by_name(self, tmp_path):
        # Same guard, the degenerate input: an empty `folded` used to reach
        # `folded[resolutions[0]]` and IndexError past the template write.
        from zagg.column import write_column

        with pytest.raises(ValueError, match="node-order member"):
            write_column(
                str(tmp_path),
                morton_word("-311"),
                {},
                FIELDS,
                node_order=SHARD_ORDER,
                cell_order=CELL_ORDER,
            )

    def test_none_class_fields_never_reach_the_template(self, tmp_path):
        from zagg.column import fold_column, write_column

        fields = dict(FIELDS, h_mean={"class": "none"})
        slabs = _cell_slabs(self.CELLS)
        folded = fold_column(slabs, fields, cell_order=CELL_ORDER, resolutions=[3, SHARD_ORDER])
        basename = write_column(
            str(tmp_path),
            morton_word("-311"),
            folded,
            fields,
            node_order=SHARD_ORDER,
            cell_order=CELL_ORDER,
        )
        group = zarr.open_group(
            open_store(f"{tmp_path}/-3/1/1/{basename}"), path="3", mode="r", zarr_format=3
        )
        assert "h_mean" not in group
        root = zarr.open_group(open_store(f"{tmp_path}/-3/1/1/{basename}"), mode="r", zarr_format=3)
        assert "h_mean" not in root.attrs["zagg_column"]["fields"]
