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

    def test_nan_datum_is_skipped_not_propagated(self):
        # The DECLARED premise behind the §8.3 exactness claim (review finding,
        # issue #201, EXACT_NAN_POLICY): a stored NaN is the fill sentinel and a
        # NaN datum at once — same bytes — so the fold is nanmin/nansum, never
        # the NaN-propagating min/sum a direct coarse aggregation would return.
        from zagg.sweep_overview import EXACT_NAN_POLICY

        assert EXACT_NAN_POLICY == "skip"
        vals = np.array([1.0, np.nan, 3.0, 4.0], dtype=np.float32)
        np.testing.assert_array_equal(fold_dense(vals, 4, "min", "NaN"), [np.nanmin(vals)])
        np.testing.assert_array_equal(fold_dense(vals, 4, "sum", "NaN"), [np.nansum(vals)])
        assert np.isnan(np.min(vals)) and np.isnan(np.sum(vals))  # the divergence

    def test_nan_datum_skipped_even_under_a_numeric_fill(self):
        # Uniform policy: even where the fill is NOT NaN — so a NaN datum could
        # in principle be told apart — NaN still counts as missing.
        vals = np.array([2.0, np.nan], dtype=np.float32)
        np.testing.assert_array_equal(fold_dense(vals, 2, "min", -9999.0), [2.0])

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


class TestPyramidBlock:
    """The manifest declaration (Phase C): template time + config grammar."""

    def _cfg(self, **output):
        from zagg.config import default_config

        cfg = default_config("atl06")
        cfg.output.update(output)
        return cfg

    def test_default_declaration_from_atl06(self, caplog):
        from zagg.sweep_overview import build_pyramid_block

        block = build_pyramid_block(self._cfg(), shard_order=6)
        overview = block["overview"]
        assert block["spec"] == PYRAMID_SPEC
        assert overview["spacing"] == 2 and overview["orders"] == [4, 2, 0]
        assert overview["all_time"] is False
        assert overview["fields"]["count"] == {
            "class": "exact",
            "method": "sum",
            "nan_policy": "skip",
            "dtype": "int32",
            "fill_value": 0,
        }
        assert overview["fields"]["h_min"] == {
            "class": "exact",
            "method": "min",
            # The declared premise (issue #201 review): a stored NaN is the fill
            # sentinel and a NaN datum at once, so the fold skips both — this is
            # nanmin, not min. h_min's fill IS NaN in the shipped atl06 config.
            "nan_policy": "skip",
            "dtype": "float32",
            "fill_value": "NaN",  # JSON-safe token, not float nan
        }
        assert overview["fields"]["h_mean"] == {"class": "none"}
        # The D24 loud template-time warning names the excluded fields.
        assert "non-composable" in caplog.text and "h_mean" in caplog.text
        json.dumps(block)  # JSON-safe by construction

    def test_tdigest_field_declaration_carries_delta(self):
        from zagg.config import default_config
        from zagg.sweep_overview import build_pyramid_block

        cfg = default_config("atl03_tdigest_healpix")
        block = build_pyramid_block(cfg, shard_order=9)
        entry = block["overview"]["fields"]["h_tdigest"]
        assert entry["class"] == "approximate" and entry["method"] == "tdigest_kway"
        assert entry["inner_shape"] == [2] and entry["delta"] == 256

    def test_explicit_orders_and_all_time(self):
        from zagg.sweep_overview import build_pyramid_block

        cfg = self._cfg(pyramid={"orders": [5, 1], "all_time": True})
        overview = build_pyramid_block(cfg, shard_order=6)["overview"]
        assert overview["orders"] == [5, 1] and overview["all_time"] is True

    def test_spacing_knob(self):
        from zagg.sweep_overview import build_pyramid_block

        cfg = self._cfg(pyramid={"spacing": 3})
        assert build_pyramid_block(cfg, shard_order=9)["overview"]["orders"] == [6, 3, 0]

    def test_disabled_declares_off(self):
        from zagg.sweep_overview import build_pyramid_block

        block = build_pyramid_block(self._cfg(pyramid=False), shard_order=6)
        assert block["overview"] == {"orders": []}

    def test_build_manifest_carries_the_block(self):
        from zagg.grids.healpix import HealpixGrid
        from zagg.hive import build_manifest

        grid = HealpixGrid(6, 8, config=self._cfg())
        manifest = build_manifest(grid, dataset={"short_name": "ATL06", "version": "007"})
        assert manifest["pyramid"]["overview"]["orders"] == [4, 2, 0]
        json.dumps(manifest)

    def test_validate_rejects_bad_grammar(self):
        from zagg.config import validate_config

        with pytest.raises(ValueError, match="spacing must be an int >= 1"):
            validate_config(self._cfg(store_layout="hive", pyramid={"spacing": 0}))
        with pytest.raises(ValueError, match="orders must be a list of ancestor orders"):
            validate_config(self._cfg(store_layout="hive", pyramid={"orders": [6]}))
        with pytest.raises(ValueError, match="all_time must be a boolean"):
            validate_config(self._cfg(store_layout="hive", pyramid={"all_time": "yes"}))
        with pytest.raises(ValueError, match="unknown keys"):
            validate_config(self._cfg(store_layout="hive", pyramid={"levels": [1]}))
        with pytest.raises(ValueError, match="must be a mapping or false"):
            validate_config(self._cfg(store_layout="hive", pyramid=True))

    def test_validate_rejects_pyramid_on_flat(self):
        from zagg.config import validate_config

        with pytest.raises(ValueError, match="output.pyramid requires"):
            validate_config(
                self._cfg(
                    store_layout="flat", coverage_moc=False, sweep=False, pyramid={"spacing": 2}
                )
            )

    def test_validate_summarize_grammar(self):
        from zagg.config import validate_config

        ok = self._cfg(store_layout="hive", pyramid={"summarize": {"h_mean": {"as": "h_mean_d"}}})
        validate_config(ok)
        with pytest.raises(ValueError, match="unknown field"):
            validate_config(
                self._cfg(store_layout="hive", pyramid={"summarize": {"nope": {"as": "x"}}})
            )
        with pytest.raises(ValueError, match="collides with a declared field"):
            validate_config(
                self._cfg(store_layout="hive", pyramid={"summarize": {"h_mean": {"as": "count"}}})
            )
        with pytest.raises(ValueError, match="requires 'as'"):
            validate_config(self._cfg(store_layout="hive", pyramid={"summarize": {"h_mean": {}}}))

    def test_summarize_recorded_in_block(self):
        from zagg.sweep_overview import build_pyramid_block

        cfg = self._cfg(pyramid={"summarize": {"h_mean": {"as": "h_mean_digest"}}})
        overview = build_pyramid_block(cfg, shard_order=6)["overview"]
        assert overview["summarize"] == {"h_mean": {"as": "h_mean_digest"}}

    def test_default_pyramid_validates_clean(self):
        from zagg.config import validate_config

        validate_config(self._cfg(store_layout="hive"))  # absent knob is legal


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
        assert info["fields"]["count"] == {
            "class": "exact",
            "method": "sum",
            "nan_policy": "skip",
        }
        assert info["fields"]["h_tdigest"]["method"] == "tdigest_kway"
        assert "nan_policy" not in info["fields"]["h_tdigest"]
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

    def test_window_named_all_folds_into_the_all_time_overview(self, tmp_path, caplog):
        # Review finding, issue #201: a window labeled with the reserved token
        # resolves to the same basename AND envelope key as the all-time fold,
        # which used to overwrite it and then never regenerate it. Config
        # validation now rejects the label; a pre-guard manifest folds the
        # window into the all-time overview with a warning instead.
        _write_manifest(tmp_path, orders=(1,), windowed=True, all_time=True)
        _make_leaf(tmp_path, "-311", {0: [1.0, 2.0]}, window="2019")
        _make_leaf(tmp_path, "-311", {0: [10.0]}, window="all")
        word = morton_word("-311")
        result = run_sweep(str(tmp_path), [(word, "2019"), (word, "all")], families=("overview",))
        assert result["families"]["overview"]["written"] == 2  # 2019 + all-time
        assert "reserved all-time token" in caplog.text
        gall = _overview_group(tmp_path, "-3/1", "all.zarr", 3)
        assert gall["count"][0] == 3  # both windows' leaves are folded in
        envelope = json.loads((tmp_path / "-3" / "1" / "overview.rollup.json").read_text())
        assert sorted(envelope["windows"]) == ["2019", "all"]

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

    def test_summarize_declaration_warns_and_skips(self, tmp_path, caplog):
        _write_manifest(tmp_path, orders=(1,))
        manifest = json.loads((tmp_path / MANIFEST_NAME).read_text())
        manifest["pyramid"]["overview"]["summarize"] = {"h_mean": {"as": "h_mean_digest"}}
        obstore.put(open_object_store(str(tmp_path)), MANIFEST_NAME, json.dumps(manifest).encode())
        _make_leaf(tmp_path, "-311", {0: [1.0]})
        result = run_sweep(str(tmp_path), [(morton_word("-311"), None)], families=("overview",))
        assert result["families"]["overview"]["written"] == 1
        assert "derived summaries" in caplog.text and "issue #265" in caplog.text
        g = _overview_group(tmp_path, "-3/1", "all.zarr", 3)
        assert "h_mean_digest" not in g

    def test_sweep_populates_manifest_materialized(self, tmp_path):
        from zagg.hive import read_manifest

        _write_manifest(tmp_path, orders=(1, 0))
        _make_leaf(tmp_path, "-311", {0: [1.0]})
        result = run_sweep(str(tmp_path), [(morton_word("-311"), None)], families=("overview",))
        assert result["families"]["overview"]["manifest_updated"] is True
        block = read_manifest(str(tmp_path))["pyramid"]["overview"]
        assert block["materialized"]["orders"] == [0, 1]
        assert block["materialized"]["generated_at"]
        # The declaration itself is untouched (the sweep populates, never
        # rewrites — D11/D22 write-once discipline).
        assert block["orders"] == [1, 0] and block["fields"] == FIELDS_DECL

    def test_manifest_update_keeps_frozen_keys_resumable(self, tmp_path):
        # The pyramid block is excluded from the frozen resume keys, so the
        # sweep's materialized update can never brick an append precheck.
        from zagg.hive import _frozen_matches, read_manifest

        before = _write_manifest(tmp_path, orders=(1,))
        _make_leaf(tmp_path, "-311", {0: [1.0]})
        run_sweep(str(tmp_path), [(morton_word("-311"), None)], families=("overview",))
        after = read_manifest(str(tmp_path))
        assert after["pyramid"] != before["pyramid"]
        assert _frozen_matches(after, before)

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


class TestSection83Obligations:
    """The standing D22 test claims for the overview family (§8.3)."""

    #: Two shards' observations, keyed (leaf decimal, leaf row). Exact binary
    #: floats so exact-class equality asserts are byte-for-byte.
    OBS = {
        ("-311", 0): [1.5, 2.5, -3.0],
        ("-311", 3): [8.0],
        ("-311", 5): [0.25, 0.75],
        ("-311", 15): [4.0, 6.0],
        ("-312", 0): [3.0, -1.0],
        ("-312", 9): [2.0],
    }

    def _populate(self, root, orders=(1, 0)):
        _write_manifest(root, orders=orders)
        cells: dict = {}
        for (dec, row), obs in self.OBS.items():
            cells.setdefault(dec, {})[row] = obs
        for dec, rows in cells.items():
            _make_leaf(root, dec, rows)
        return [(morton_word(d), None) for d in sorted(cells)]

    def _direct(self, k):
        """Direct aggregation at overview order ``k``: pool raw observations
        per overview cell (the oracle the fold must match)."""
        span = 4 ** (CELL_ORDER - SHARD_ORDER - (SHARD_ORDER - k))
        pooled: dict[int, list] = {}
        for (dec, row), obs in self.OBS.items():
            shard_rank = {"-311": 0, "-312": 1}[dec]
            cell = shard_rank * span + row // 4 ** (SHARD_ORDER - k)
            pooled.setdefault(cell, []).extend(obs)
        return pooled

    @pytest.mark.parametrize("k", [1, 0])
    def test_rollup_equals_direct_at_every_declared_order(self, tmp_path, k):
        from zagg.stats.tdigest import build_tdigest, quantile_from_tdigest

        refs = self._populate(tmp_path)
        run_sweep(str(tmp_path), refs, families=("overview",))
        rel = {1: "-3/1", 0: "-3"}[k]
        g = _overview_group(tmp_path, rel, "all.zarr", CELL_ORDER - (SHARD_ORDER - k))
        count, h_min = g["count"][:], g["h_min"][:]
        digests = g["h_tdigest"][:]
        direct = self._direct(k)
        for cell in range(len(count)):
            if cell not in direct:
                assert count[cell] == 0 and np.isnan(h_min[cell])
                assert len(bytes(digests[cell])) == 0
                continue
            obs = np.asarray(direct[cell], dtype=np.float64)
            # Exact class: byte-equal to the direct aggregation (§8.3).
            assert count[cell] == len(obs)
            assert h_min[cell] == np.float32(obs.min())
            # Approximate class: np.isclose against the direct digest (D24).
            merged = decode_digest(bytes(digests[cell]), "float32")
            oracle = build_tdigest(obs, delta=64)
            for q in (0.25, 0.5, 0.75):
                assert np.isclose(
                    quantile_from_tdigest(merged, q),
                    quantile_from_tdigest(oracle, q),
                    rtol=0.05,
                    atol=0.25,
                )

    @pytest.mark.parametrize("k", [1, 0])
    def test_morton_labels_address_the_folded_cells(self, tmp_path, k):
        # Review finding, issue #201: the writer places slabs by base-4 rank
        # arithmetic but LABELS them with generate_morton_children(), two
        # orderings that must agree — and `_direct()` above cannot catch a
        # disagreement because it recomputes the writer's own formula. Resolve
        # every slot from the overview's OWN morton value instead: a leaf cell
        # belongs to the overview cell its morton decimal PREFIXES.
        from mortie import generate_morton_children

        from zagg.grids.morton import morton_decimal

        refs = self._populate(tmp_path, orders=(k,))
        run_sweep(str(tmp_path), refs, families=("overview",))
        target_order = CELL_ORDER - (SHARD_ORDER - k)
        node = {1: "-31", 0: "-3"}[k]
        base = node[: len(node) - k]
        g = _overview_group(tmp_path, {1: "-3/1", 0: "-3"}[k], "all.zarr", target_order)

        pooled: dict[str, list] = {}
        for (dec, row), obs in self.OBS.items():
            word = generate_morton_children(morton_word(dec), CELL_ORDER)[row]
            cell = morton_decimal(int(word))  # the order-4 leaf cell's id
            pooled.setdefault(cell[: len(base) + target_order], []).extend(obs)
        assert pooled  # every OBS cell resolved to an overview cell

        morton, count, h_min = g["morton"][:], g["count"][:], g["h_min"][:]
        assert len(set(morton.tolist())) == len(morton)  # labels are distinct
        seen = set()
        for j, word in enumerate(morton.tolist()):
            label = morton_decimal(int(word))
            assert label.startswith(node) and len(label) == len(base) + target_order
            if label in pooled:
                obs = np.asarray(pooled[label], dtype=np.float64)
                assert count[j] == len(obs)
                assert h_min[j] == np.float32(obs.min())
                seen.add(label)
            else:  # unclaimed slots stay at the fill
                assert count[j] == 0 and np.isnan(h_min[j])
        assert seen == set(pooled)  # no populated cell went unlabelled

    def test_second_pass_over_unchanged_tree_writes_nothing(self, tmp_path):
        refs = self._populate(tmp_path)
        first = run_sweep(str(tmp_path), refs, families=("moc", "overview"))
        assert first["families"]["overview"]["written"] == 2
        env_before = (tmp_path / "-3" / "1" / "overview.rollup.json").read_text()
        second = run_sweep(str(tmp_path), refs, families=("moc", "overview"))
        counts = second["families"]["overview"]
        assert counts["written"] == 0 and counts["current"] == 2
        assert "manifest_updated" not in counts  # no write, no manifest touch
        assert (tmp_path / "-3" / "1" / "overview.rollup.json").read_text() == env_before

    def test_same_second_leaf_rerun_rewrites_via_content_hash(self, tmp_path, monkeypatch):
        # Leaf stamps resolve to whole seconds, so a back-to-back re-run
        # carries an UNCHANGED generation stamp; freeze the clock to force the
        # collision and prove the content hash is the backstop that rewrites.
        import zagg.hive as hive_mod

        monkeypatch.setattr(hive_mod, "_utcnow", lambda: "2026-05-01T12:00:00+00:00")
        refs = self._populate(tmp_path, orders=(0,))
        run_sweep(str(tmp_path), refs, families=("moc", "overview"))
        stale = json.loads((tmp_path / "-3" / "overview.rollup.json").read_text())
        # Same wall-clock second, different content: leaf -311 changes.
        _make_leaf(tmp_path, "-311", {0: [100.0]})
        result = run_sweep(str(tmp_path), [(morton_word("-311"), None)], families=("overview",))
        assert result["families"]["overview"]["written"] == 1
        fresh = json.loads((tmp_path / "-3" / "overview.rollup.json").read_text())
        assert fresh["windows"]["all"]["generation"] == stale["windows"]["all"]["generation"]
        assert fresh["windows"]["all"]["content_hash"] != stale["windows"]["all"]["content_hash"]
        g = _overview_group(tmp_path, "-3", "all.zarr", 2)
        # The re-run leaf's new fold, with the untouched sibling retained
        # (candidates come from the root MOC, not just the dirty set).
        assert g["count"][0] == 1 and g["count"][1] == 3
        assert g["h_min"][0] == 100.0

    def test_deleting_every_overview_leaves_leaf_reads_green(self, tmp_path):
        import shutil

        from zagg.hive import read_commit as read_stamp

        refs = self._populate(tmp_path)
        run_sweep(str(tmp_path), refs, families=("moc", "overview"))
        hashes = {
            p: json.loads(p.read_text())["windows"]["all"]["content_hash"]
            for p in tmp_path.rglob("overview.rollup.json")
        }
        assert hashes
        for p in tmp_path.rglob("all.zarr"):
            shutil.rmtree(p)
        for p in list(tmp_path.rglob("overview.rollup.json")):
            p.unlink()
        # Leaf truth reads back untouched (nothing-load-bearing, D9)...
        for dec in ("-311", "-312"):
            leaf = open_store(shard_leaf_path(str(tmp_path), morton_word(dec)))
            assert read_stamp(leaf) is not None
            g = zarr.open_group(leaf, path=str(CELL_ORDER), mode="r", zarr_format=3)
            assert g["count"][0] >= 1
        # ...and one sweep regenerates byte-identical folds (same hashes).
        run_sweep(str(tmp_path), refs, families=("overview",))
        for p, digest in hashes.items():
            assert json.loads(p.read_text())["windows"]["all"]["content_hash"] == digest

    def test_deleting_only_the_zarrs_regenerates_them(self, tmp_path):
        # Review finding, issue #201: the envelope and the zarr are two objects,
        # so skip-if-current must probe the ARTIFACT — deleting only the zarrs
        # used to read back as `current` with nothing on disk (no D9 self-heal).
        import shutil

        refs = self._populate(tmp_path)
        run_sweep(str(tmp_path), refs, families=("moc", "overview"))
        hashes = {
            p: json.loads(p.read_text())["windows"]["all"]["content_hash"]
            for p in tmp_path.rglob("overview.rollup.json")
        }
        zarrs = sorted(tmp_path.rglob("all.zarr"))
        assert len(zarrs) == 2 and len(hashes) == 2
        for p in zarrs:
            shutil.rmtree(p)
        counts = run_sweep(str(tmp_path), refs, families=("overview",))["families"]["overview"]
        assert counts["written"] == 2 and counts["current"] == 0
        for p in zarrs:
            assert read_commit(open_store(str(p))) is not None
        # ...and the regenerated folds are byte-identical (same content hashes).
        for p, digest in hashes.items():
            assert json.loads(p.read_text())["windows"]["all"]["content_hash"] == digest

    def test_torn_overview_write_is_not_current(self, tmp_path):
        # The same probe covers a torn prior write: the zarr is there but the
        # commit stamp never landed, so the entry must not read as `current`.
        refs = self._populate(tmp_path, orders=(0,))
        run_sweep(str(tmp_path), refs, families=("moc", "overview"))
        (tmp_path / "-3" / "all.zarr" / "zarr.json").unlink()
        counts = run_sweep(str(tmp_path), refs, families=("overview",))["families"]["overview"]
        assert counts["written"] == 1 and counts["current"] == 0
        assert read_commit(open_store(str(tmp_path / "-3" / "all.zarr"))) is not None

    def test_repair_walk_ignores_overviews(self, tmp_path):
        # Review finding, issue #201: overview basenames collide with the leaf
        # grammar, so the D9 repair walk must classify them out by the D11 role
        # attr. `all.zarr` used to die outright on a shard_order-2 store
        # (_decimal_order("all") == 2 -> "malformed decimal Morton id 'all'").
        from zagg.coverage import refresh_root_coverage
        from zagg.grids.morton import morton_decimal
        from zagg.hive import root_coverage_words

        refs = self._populate(tmp_path, orders=(1, 0))
        run_sweep(str(tmp_path), refs, families=("moc", "overview"))
        assert sorted(p.name for p in tmp_path.rglob("all.zarr")) == ["all.zarr"] * 2
        env = refresh_root_coverage(str(tmp_path))
        assert [morton_decimal(int(w)) for w in root_coverage_words(env)] == ["-311", "-312"]

    def test_repair_walk_ignores_digit_labeled_overviews(self, tmp_path):
        # The windowed half of the same finding: a window label made of [1-4]
        # digits at length shard_order+1 parses as a VALID decimal, so the
        # rebuilt root MOC used to gain a phantom positive-base shard ("2411")
        # that then poisons every root-MOC consumer.
        from zagg.coverage import refresh_root_coverage
        from zagg.grids.morton import morton_decimal
        from zagg.hive import root_coverage_words

        _write_manifest(tmp_path, orders=(1,), windowed=True, shard_order=3)
        _make_leaf(tmp_path, "-3111", {0: [1.0]}, window="2411", shard_order=3)
        run_sweep(str(tmp_path), [(morton_word("-3111"), "2411")], families=("moc", "overview"))
        assert (tmp_path / "-3" / "1" / "2411.zarr").is_dir()
        env = refresh_root_coverage(str(tmp_path))
        assert [morton_decimal(int(w)) for w in root_coverage_words(env)] == ["-3111"]

    def test_role_never_inferred_from_position(self, tmp_path):
        # D11/D24: an overview is classified by its role attr, never by tree
        # depth — the leaf carries NO role, the overview always does.
        refs = self._populate(tmp_path, orders=(1,))
        run_sweep(str(tmp_path), refs, families=("overview",))
        leaf = zarr.open_group(
            open_store(shard_leaf_path(str(tmp_path), morton_word("-311"))), mode="r"
        )
        assert ROLE_ATTR not in leaf.attrs
        assert _overview_root(tmp_path, "-3/1", "all.zarr").attrs[ROLE_ATTR] == "overview"
