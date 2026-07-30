"""Overview sweep family (issue #201): D24 kernels, writer, pyramid block.

Phase A covers the composability-class helpers (the D24 derivation from the
aggregator merge-law set) and the per-field up-aggregation kernels: exact
folds byte-equal by construction, approximate (t-digest) folds via the
order-independent k-way merge. Phase B covers the overview writer — real
leaf zarrs folded into ``{window}.zarr``/``all.zarr`` overviews at declared
ancestor orders, with the D11 role/provenance attrs.
"""

import json

import numpy as np
import obstore
import pytest
import zarr

from zagg.grids.morton import morton_word
from zagg.hive import MANIFEST_NAME, read_commit, shard_leaf_path, stamp_commit
from zagg.semantics import (
    EXACT_MERGE_LAWS,
    composability_classes,
    field_composability,
)
from zagg.store import open_object_store, open_store
from zagg.sweep import run_sweep
from zagg.sweep_overview import (
    OVERVIEW_ATTR,
    PYRAMID_SPEC,
    ROLE_ATTR,
    combine_dense,
    decode_digest,
    encode_digest,
    fold_dense,
    fold_digests,
)

SHARD_ORDER = 2
CELL_ORDER = 4
LEAF_CELLS = 4 ** (CELL_ORDER - SHARD_ORDER)

#: The manifest pyramid declaration the writer consumes (template-time, D22).
FIELDS_DECL = {
    "count": {"class": "exact", "method": "sum", "dtype": "int32", "fill_value": 0},
    "h_min": {"class": "exact", "method": "min", "dtype": "float32", "fill_value": "NaN"},
    "h_tdigest": {
        "class": "approximate",
        "method": "tdigest_kway",
        "dtype": "float32",
        "inner_shape": [2],
        "delta": 64,
    },
    "h_mean": {"class": "none"},  # option A: absence declared in the block
}


def _write_manifest(
    root, *, orders=(1, 0), all_time=False, windowed=False, shard_order=SHARD_ORDER, fields=None
):
    manifest = {
        "spec": "morton-hive/2" if windowed else "morton-hive/1",
        "dataset": {"short_name": "TEST", "version": "1"},
        "cell_order": CELL_ORDER,
        "shard_order": shard_order,
        "split_schedule": [1] * shard_order,
        "pyramid": {
            "spec": PYRAMID_SPEC,
            "overview": {
                "spacing": 2,
                "orders": list(orders),
                "all_time": all_time,
                "fields": dict(FIELDS_DECL if fields is None else fields),
            },
        },
        "generated_at": "2026-01-01T00:00:00+00:00",
    }
    if windowed:
        manifest["temporal"] = {"schedule": "yearly", "time_field": "t"}
    obstore.put(open_object_store(str(root)), MANIFEST_NAME, json.dumps(manifest).encode())
    return manifest


def _leaf_cfg(*, with_h_min=True):
    from zagg.config import PipelineConfig

    variables = {
        "count": {"function": "len", "dtype": "int32", "fill_value": 0},
        "h_mean": {"function": "mean", "dtype": "float32"},
        "h_tdigest": {
            "kind": "ragged",
            "function": "zagg.stats.tdigest.build_tdigest",
            "inner_shape": [2],
            "dtype": "float32",
            "fill_value": 0,
        },
    }
    if with_h_min:
        variables["h_min"] = {"function": "min", "dtype": "float32"}
    return PipelineConfig(
        aggregation={
            "coordinates": {"morton": {"dtype": "uint64", "fill_value": 0}},
            "variables": variables,
        }
    )


def _make_leaf(
    root,
    decimal,
    cells,
    *,
    window=None,
    time_range=None,
    with_h_min=True,
    shard_order=SHARD_ORDER,
):
    """Write one committed leaf: ``cells`` maps leaf row -> observation values."""
    from mortie import generate_morton_children

    from zagg.grids.healpix import HealpixGrid
    from zagg.stats.tdigest import build_tdigest

    grid = HealpixGrid(shard_order, CELL_ORDER, config=_leaf_cfg(with_h_min=with_h_min))
    word = morton_word(decimal)
    store = open_store(shard_leaf_path(str(root), word, window=window))
    grid.emit_shard_template(store, overwrite=True)
    group = zarr.open_group(store, path=str(CELL_ORDER), mode="r+", zarr_format=3)
    n = 4 ** (CELL_ORDER - shard_order)
    group["morton"][:] = np.asarray(generate_morton_children(word, CELL_ORDER), dtype=np.uint64)
    count = np.zeros(n, np.int32)
    h_min = np.full(n, np.nan, np.float32)
    h_mean = np.full(n, np.nan, np.float32)
    digest = np.full(n, b"", dtype=object)
    for i, obs in cells.items():
        obs = np.asarray(obs, dtype=np.float64)
        count[i] = len(obs)
        h_min[i] = obs.min()
        h_mean[i] = obs.mean()
        digest[i] = encode_digest(build_tdigest(obs, delta=64), "float32")
    group["count"][:] = count
    if with_h_min:
        group["h_min"][:] = h_min
    group["h_mean"][:] = h_mean
    group["h_tdigest"][:] = digest
    stamp_commit(
        store,
        cells_with_data=len(cells),
        granule_count=1,
        window=window,
        time_range=time_range,
    )


def _overview_group(root, node_rel, basename, order):
    store = open_store(f"{root}/{node_rel}/{basename}")
    return zarr.open_group(store, path=str(order), mode="r", zarr_format=3)


def _overview_root(root, node_rel, basename):
    return zarr.open_group(open_store(f"{root}/{node_rel}/{basename}"), mode="r", zarr_format=3)


class TestComposabilityClasses:
    def test_default_atl06_config_classes(self):
        from zagg.config import default_config

        classes = composability_classes(default_config("atl06"))
        assert classes["count"] == "exact"
        assert classes["h_min"] == "exact"
        assert classes["h_max"] == "exact"
        # average / var / quantile have no exact fold law; the expression
        # field is never composable.
        for name in ("h_mean", "h_sigma", "h_variance", "h_q25", "h_q50", "h_q75"):
            assert classes[name] == "none"

    def test_tdigest_field_is_approximate(self):
        from zagg.config import default_config

        classes = composability_classes(default_config("atl03_tdigest_healpix"))
        assert classes["h_tdigest"] == "approximate"

    def test_located_tdigest_is_none(self):
        # The located channel has no streaming merge law yet -> excluded.
        meta = {
            "kind": "ragged",
            "function": "zagg.stats.tdigest.build_tdigest",
            "inner_shape": [2],
            "location": "leaf_id",
            "dtype": "float32",
        }
        assert field_composability(meta) == "none"

    @pytest.mark.parametrize("name", ["min", "np.min", "numpy.min", "nanmax", "np.nansum"])
    def test_prefix_normalization(self, name):
        assert field_composability({"function": name, "dtype": "float32"}) == "exact"

    def test_every_exact_law_is_known(self):
        assert set(EXACT_MERGE_LAWS.values()) == {"sum", "min", "max"}

    @pytest.mark.parametrize(
        "meta",
        [
            {"function": "mean"},
            {"function": "median"},
            {"expression": "np.sum(x)"},
            {"function": "min", "kind": "vector", "trailing_shape": [3]},
            {"function": "len", "resolution": "chunk"},
            {"function": "zagg.stats.tdigest.build_tdigest", "kind": "ragged", "inner_shape": [3]},
            {"function": "somepkg.custom", "kind": "ragged", "inner_shape": [2]},
        ],
    )
    def test_non_composable_metas(self, meta):
        assert field_composability(meta) == "none"

    def test_pairwise_tdigest_builder_is_approximate(self):
        meta = {
            "kind": "ragged",
            "function": "zagg.stats.tdigest.build_tdigest_pairwise",
            "inner_shape": [2],
            "dtype": "float32",
        }
        assert field_composability(meta) == "approximate"


class TestFoldDense:
    def test_sum_int_count(self):
        counts = np.array([1, 2, 0, 0, 3, 0, 4, 5], dtype=np.int32)
        out = fold_dense(counts, 4, "sum", 0)
        assert out.dtype == np.int32
        np.testing.assert_array_equal(out, [3, 12])

    def test_sum_float_nan_fill_skips_missing(self):
        vals = np.array([1.5, np.nan, 2.5, np.nan], dtype=np.float32)
        out = fold_dense(vals, 2, "sum", "NaN")
        np.testing.assert_array_equal(out, np.array([1.5, 2.5], dtype=np.float32))

    def test_all_missing_group_folds_to_fill(self):
        vals = np.array([np.nan, np.nan, 1.0, 2.0], dtype=np.float32)
        out = fold_dense(vals, 2, "min", "NaN")
        assert np.isnan(out[0]) and out[1] == 1.0

    def test_min_max_extrema(self):
        vals = np.array([3.0, np.nan, -1.0, 7.0], dtype=np.float32)
        np.testing.assert_array_equal(fold_dense(vals, 4, "min", "NaN"), [-1.0])
        np.testing.assert_array_equal(fold_dense(vals, 4, "max", "NaN"), [7.0])

    def test_exact_equals_direct(self):
        # The fold at factor 4 equals a direct order-coarser aggregation over
        # the same values (the §8.3 exactness claim, in kernel form). Exact
        # binary floats so equality is byte-for-byte.
        rng = np.random.default_rng(7)
        vals = (rng.integers(-64, 64, size=64) / 4.0).astype(np.float32)
        for law, direct in (("sum", np.sum), ("min", np.min), ("max", np.max)):
            out = fold_dense(vals, 4, law, "NaN")
            expect = np.array([direct(vals[i : i + 4]) for i in range(0, 64, 4)], np.float32)
            np.testing.assert_array_equal(out, expect)

    def test_fold_is_composable_two_hops(self):
        # fold(16x) == fold(4x) then fold(4x): the associativity that lets a
        # spacing-2 schedule read leaves directly at every declared order.
        vals = np.arange(32, dtype=np.float32)
        one_hop = fold_dense(vals, 16, "sum", "NaN")
        two_hop = fold_dense(fold_dense(vals, 4, "sum", "NaN"), 4, "sum", "NaN")
        np.testing.assert_array_equal(one_hop, two_hop)

    def test_int_extrema_identity(self):
        vals = np.array([5, 0, 0, 2], dtype=np.int32)  # fill 0 = missing
        np.testing.assert_array_equal(fold_dense(vals, 2, "max", 0), [5, 2])
        np.testing.assert_array_equal(fold_dense(vals, 2, "min", 0), [5, 2])

    def test_bad_factor_and_unknown_law_raise(self):
        with pytest.raises(ValueError, match="cannot fold"):
            fold_dense(np.zeros(3), 2, "sum", 0)
        with pytest.raises(ValueError, match="unknown exact fold law"):
            fold_dense(np.zeros(4), 2, "mean", 0)

    def test_combine_accumulates(self):
        a = np.array([1.0, np.nan], dtype=np.float32)
        b = np.array([2.0, np.nan], dtype=np.float32)
        out = combine_dense(a, b, "sum", "NaN")
        assert out[0] == 3.0 and np.isnan(out[1])


class TestFoldDigests:
    def _digest(self, values):
        from zagg.stats.tdigest import build_tdigest

        return build_tdigest(np.asarray(values, dtype=np.float64), delta=64)

    def test_roundtrip_encode_decode(self):
        d = self._digest([1.0, 2.0, 3.0])
        raw = encode_digest(d, "float32")
        np.testing.assert_array_equal(decode_digest(raw, "float32"), d.astype(np.float32))

    def test_empty_accumulation_is_empty_payload(self):
        assert fold_digests([], delta=64) == b""
        assert decode_digest(b"", "float32").shape == (0, 2)

    def test_single_digest_passes_through(self):
        d = self._digest([1.0, 5.0])
        assert fold_digests([d], delta=64) == encode_digest(d, "float32")

    def test_merge_matches_kway_oracle(self):
        from zagg.stats.tdigest import merge_tdigests_kway, quantile_from_tdigest

        a, b = self._digest(np.arange(100.0)), self._digest(np.arange(100.0, 200.0))
        raw = fold_digests([a, b], delta=64)
        merged = decode_digest(raw, "float32")
        oracle = merge_tdigests_kway([a, b], delta=64)
        np.testing.assert_array_equal(merged, oracle.astype(np.float32))
        # And the merged digest is the subtree's distribution (np.isclose class).
        assert np.isclose(quantile_from_tdigest(merged, 0.5), 99.5, rtol=0.05)

    def test_merge_is_permutation_stable(self):
        digests = [self._digest(np.arange(i, i + 50.0)) for i in (0, 30, 60)]
        forward = fold_digests(list(digests), delta=64)
        backward = fold_digests(list(reversed(digests)), delta=64)
        assert forward == backward  # the issue #279 order-independent law


class TestOverviewWriter:
    def test_folds_leaves_at_every_declared_order(self, tmp_path):
        from zagg.stats.tdigest import quantile_from_tdigest

        _write_manifest(tmp_path, orders=(1, 0))
        _make_leaf(tmp_path, "-311", {0: [1.0, 2.0], 5: [10.0]})
        _make_leaf(tmp_path, "-312", {0: [3.0]})
        refs = [(morton_word("-311"), None), (morton_word("-312"), None)]
        result = run_sweep(str(tmp_path), refs, families=("overview",))
        counts = result["families"]["overview"]
        # order 1: node -31; order 0: node -3 -> two overview zarrs.
        assert counts["written"] == 2 and counts["failed"] == 0

        # Order 1 (cells at order 3, 4 source cells per overview cell):
        # leaf -311 occupies overview rows 0-3, leaf -312 rows 4-7.
        g1 = _overview_group(tmp_path, "-3/1", "all.zarr", 3)
        np.testing.assert_array_equal(g1["count"][:8], [2, 1, 0, 0, 1, 0, 0, 0])
        assert g1["h_min"][0] == 1.0 and g1["h_min"][4] == 3.0
        merged = decode_digest(bytes(g1["h_tdigest"][:][0]), "float32")
        assert np.isclose(quantile_from_tdigest(merged, 0.5), 2.0, atol=0.6)
        # morton is the order-3 subtree of -31, ascending.
        assert g1["morton"].shape == (16,) and g1["morton"].dtype == np.uint64

        # Order 0 (cells at order 2 == the source shard order): each leaf
        # folds whole into its shard's cell — count 3 for -311, 1 for -312.
        g0 = _overview_group(tmp_path, "-3", "all.zarr", 2)
        np.testing.assert_array_equal(g0["count"][:2], [3, 1])
        assert g0["h_min"][0] == 1.0 and g0["h_min"][1] == 3.0

    def test_role_and_provenance_attrs(self, tmp_path):
        _write_manifest(tmp_path, orders=(1,))
        _make_leaf(tmp_path, "-311", {0: [1.0]})
        run_sweep(str(tmp_path), [(morton_word("-311"), None)], families=("overview",))
        root = _overview_root(tmp_path, "-3/1", "all.zarr")
        assert root.attrs[ROLE_ATTR] == "overview"
        info = root.attrs[OVERVIEW_ATTR]
        assert info["order"] == 1 and info["cell_order"] == 3
        assert info["source_shard_order"] == SHARD_ORDER
        assert info["source_cell_order"] == CELL_ORDER
        assert info["fields"]["count"] == {"class": "exact", "method": "sum"}
        assert info["fields"]["h_tdigest"]["method"] == "tdigest_kway"
        assert info["generation"]["n_leaves"] == 1
        # The overview is a stamped, committed zarr (D4 semantics apply).
        stamp = read_commit(open_store(str(tmp_path / "-3" / "1" / "all.zarr")))
        assert stamp is not None and stamp["granule_count"] == 1

    def test_none_field_is_absent_from_the_overview(self, tmp_path):
        # Option A (the ruled D24 default): h_mean exists at the leaves but is
        # excluded from every overview order; the pyramid block declares it.
        _write_manifest(tmp_path, orders=(1,))
        _make_leaf(tmp_path, "-311", {0: [1.0, 5.0]})
        run_sweep(str(tmp_path), [(morton_word("-311"), None)], families=("overview",))
        g = _overview_group(tmp_path, "-3/1", "all.zarr", 3)
        assert "h_mean" not in g
        assert "count" in g and "h_tdigest" in g

    def test_no_declaration_is_a_noop(self, tmp_path):
        manifest = _write_manifest(tmp_path, orders=(1,))
        manifest["pyramid"] = {"orders": [], "aggregation": {}}  # the legacy block
        obstore.put(open_object_store(str(tmp_path)), MANIFEST_NAME, json.dumps(manifest).encode())
        _make_leaf(tmp_path, "-311", {0: [1.0]})
        result = run_sweep(str(tmp_path), [(morton_word("-311"), None)], families=("overview",))
        assert result["families"]["overview"]["declared"] is False
        assert result["families"]["overview"]["written"] == 0
        assert not (tmp_path / "-3" / "1" / "all.zarr").exists()

    def test_unstamped_debris_leaf_is_invisible(self, tmp_path):
        _write_manifest(tmp_path, orders=(1,))
        _make_leaf(tmp_path, "-311", {0: [1.0]})
        # A torn worker's leaf: template only, no commit stamp (D4).
        from zagg.grids.healpix import HealpixGrid

        grid = HealpixGrid(SHARD_ORDER, CELL_ORDER, config=_leaf_cfg())
        debris = open_store(shard_leaf_path(str(tmp_path), morton_word("-312")))
        grid.emit_shard_template(debris, overwrite=True)
        refs = [(morton_word("-311"), None), (morton_word("-312"), None)]
        run_sweep(str(tmp_path), refs, families=("overview",))
        g = _overview_group(tmp_path, "-3/1", "all.zarr", 3)
        info = _overview_root(tmp_path, "-3/1", "all.zarr").attrs[OVERVIEW_ATTR]
        assert info["generation"]["n_leaves"] == 1
        np.testing.assert_array_equal(g["count"][4:8], [0, 0, 0, 0])

    def test_envelope_records_window_inventory(self, tmp_path):
        _write_manifest(tmp_path, orders=(1,))
        _make_leaf(tmp_path, "-311", {0: [1.0]})
        run_sweep(str(tmp_path), [(morton_word("-311"), None)], families=("overview",))
        envelope = json.loads((tmp_path / "-3" / "1" / "overview.rollup.json").read_text())
        assert envelope["family"] == "overview" and envelope["node"] == "-31"
        entry = envelope["windows"]["all"]
        assert entry["object"] == "all.zarr"
        assert entry["generation"]["n_leaves"] == 1
        assert entry["content_hash"]

    def test_windowed_per_window_and_all_time(self, tmp_path):
        _write_manifest(tmp_path, orders=(1,), windowed=True, all_time=True)
        _make_leaf(
            tmp_path,
            "-311",
            {0: [1.0, 2.0]},
            window="2019",
            time_range=["2019-02-01T00:00:00+00:00", "2019-11-01T00:00:00+00:00"],
        )
        _make_leaf(
            tmp_path,
            "-311",
            {0: [10.0]},
            window="2020",
            time_range=["2020-01-01T00:00:00+00:00", "2020-06-01T00:00:00+00:00"],
        )
        word = morton_word("-311")
        result = run_sweep(str(tmp_path), [(word, "2019"), (word, "2020")], families=("overview",))
        assert result["families"]["overview"]["written"] == 3  # 2019, 2020, all
        g2019 = _overview_group(tmp_path, "-3/1", "2019.zarr", 3)
        g2020 = _overview_group(tmp_path, "-3/1", "2020.zarr", 3)
        gall = _overview_group(tmp_path, "-3/1", "all.zarr", 3)
        assert g2019["count"][0] == 2 and g2020["count"][0] == 1
        assert gall["count"][0] == 3  # the all-time fold accumulates windows
        assert gall["h_min"][0] == 1.0
        # Stamps: per-window overviews carry their window; the all-time fold
        # carries the reserved token and the unioned extent (D15/D23).
        stamp = read_commit(open_store(str(tmp_path / "-3" / "1" / "all.zarr")))
        assert stamp["window"] == "all"
        assert stamp["time_range"] == [
            "2019-02-01T00:00:00+00:00",
            "2020-06-01T00:00:00+00:00",
        ]
        assert read_commit(open_store(str(tmp_path / "-3" / "1" / "2019.zarr")))["window"] == "2019"

    def test_sibling_contribution_via_root_moc(self, tmp_path):
        # Incremental run: the second sweep names ONLY the new leaf; the
        # untouched sibling still contributes through the root coverage.moc
        # the MOC family refreshed (run record + MOC, never a LIST).
        _write_manifest(tmp_path, orders=(0,))
        _make_leaf(tmp_path, "-311", {0: [1.0]})
        run_sweep(str(tmp_path), [(morton_word("-311"), None)], families=("moc", "overview"))
        _make_leaf(tmp_path, "-312", {0: [5.0]})
        run_sweep(str(tmp_path), [(morton_word("-312"), None)], families=("moc", "overview"))
        g = _overview_group(tmp_path, "-3", "all.zarr", 2)
        np.testing.assert_array_equal(g["count"][:2], [1, 1])
        assert g["h_min"][0] == 1.0 and g["h_min"][1] == 5.0

    def test_missing_field_reads_as_fill(self, tmp_path):
        # Schema evolution: a leaf written before h_min existed contributes
        # fill for it — the walk never dies on the absent array (issue #341).
        _write_manifest(tmp_path, orders=(1,))
        _make_leaf(tmp_path, "-311", {0: [1.0]}, with_h_min=False)
        result = run_sweep(str(tmp_path), [(morton_word("-311"), None)], families=("overview",))
        assert result["families"]["overview"]["written"] == 1
        g = _overview_group(tmp_path, "-3/1", "all.zarr", 3)
        assert g["count"][0] == 1
        assert np.isnan(g["h_min"][0])

    def test_orphan_objects_in_leaf_are_tolerated(self, tmp_path):
        # Issue #341 Bug A: schema evolution can leave orphan array prefixes
        # inside a leaf; the fold opens arrays BY NAME (no member walk), so
        # junk prefixes and foreign objects cannot crash it.
        _write_manifest(tmp_path, orders=(1,))
        _make_leaf(tmp_path, "-311", {0: [2.0]})
        leaf = tmp_path / "-3" / "1" / "1" / "-311.zarr"
        orphan = leaf / str(CELL_ORDER) / "orphan_field" / "c"
        orphan.mkdir(parents=True)
        (orphan / "0").write_bytes(b"\x00garbage")
        (leaf / "stray.status").write_text("worker status debris")
        result = run_sweep(str(tmp_path), [(morton_word("-311"), None)], families=("overview",))
        counts = result["families"]["overview"]
        assert counts["written"] == 1 and counts["failed"] == 0

    def test_unreadable_leaf_skips_loudly_not_fatally(self, tmp_path, caplog):
        _write_manifest(tmp_path, orders=(1,))
        _make_leaf(tmp_path, "-311", {0: [1.0]})
        # A committed leaf whose inner group vanished (torn schema surgery):
        # stamp present, arrays gone -> skip + log, the node still folds.
        _make_leaf(tmp_path, "-312", {0: [9.0]})
        import shutil

        shutil.rmtree(tmp_path / "-3" / "1" / "2" / "-312.zarr" / str(CELL_ORDER))
        refs = [(morton_word("-311"), None), (morton_word("-312"), None)]
        result = run_sweep(str(tmp_path), refs, families=("overview",))
        counts = result["families"]["overview"]
        assert counts["written"] == 1 and counts["failed"] == 1
        assert "skipping unreadable leaf" in caplog.text
        g = _overview_group(tmp_path, "-3/1", "all.zarr", 3)
        assert g["count"][0] == 1

    def test_coarser_than_shard_order_accumulates(self, tmp_path):
        # Overview cells COARSER than the shard order (deep schedule, shallow
        # depth): whole shards fold into one overview cell, accumulated
        # across leaves.
        _write_manifest(tmp_path, orders=(0,), shard_order=3)
        _make_leaf(tmp_path, "-3111", {0: [1.0]}, shard_order=3)
        _make_leaf(tmp_path, "-3112", {1: [4.0, 5.0]}, shard_order=3)
        refs = [(morton_word("-3111"), None), (morton_word("-3112"), None)]
        result = run_sweep(str(tmp_path), refs, families=("overview",))
        assert result["families"]["overview"]["written"] == 1
        # node -3, overview cells at order 1 (= 4 - 3 + 0): both shards live
        # under -31 -> row 0 accumulates 1 + 2 observations.
        g = _overview_group(tmp_path, "-3", "all.zarr", 1)
        np.testing.assert_array_equal(g["count"][:], [3, 0, 0, 0])
        assert g["h_min"][0] == 1.0
