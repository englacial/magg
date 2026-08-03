"""Cross-block spill folds for located / strata / composition fields (issue #370).

Phase 1 — located ragged fields and ``build_tdigest_where`` strata survive a
block close: ``_fold_block`` builds per-block ``(digest, locations)`` partials
with the same reducers the pooled path uses (the ``leaf_id`` point words are
already spilled — they are a read-carrier column) and folds them under the
located ``merge_tdigests`` / ``merge_tdigests_kway`` overloads, so overflow
shards trade location resolution (merged centroids coarsen to common
ancestors — the located channel's defined behavior above weight 1) instead of
raising ``SpillOverflowError``.

Byte-equality tests pin ``shard_workers: 1`` (see ``test_spill.py``).
"""

import numpy as np
import pandas as pd
import pytest

from zagg.config import PipelineConfig
from zagg.grids import HealpixGrid
from zagg.processing import process_shard
from zagg.processing.spill import SpillAggregator
from zagg.processing.streaming import validate_spill_fold, validate_streaming
from zagg.stats.composition import counts_from_composition, unpack_composition
from zagg.stats.tdigest import quantile_from_tdigest

_CREDS = {"accessKeyId": "a", "secretAccessKey": "s", "sessionToken": "t"}

_SIGNAL = "h_ph > 0"
_NOISE = "~(h_ph > 0)"

_CONF_COLS = (
    "signal_conf_land",
    "signal_conf_ocean",
    "signal_conf_sea_ice",
    "signal_conf_land_ice",
    "signal_conf_inland_water",
)


def _composition_field(threshold=2):
    return {
        "function": "zagg.stats.composition.pack_composition",
        "source": "h_ph",
        "dtype": "uint64",
        "fill_value": 0,
        "params": {
            "conf_land": "signal_conf_land",
            "conf_ocean": "signal_conf_ocean",
            "conf_sea_ice": "signal_conf_sea_ice",
            "conf_land_ice": "signal_conf_land_ice",
            "conf_inland_water": "signal_conf_inland_water",
            "threshold": threshold,
        },
    }


def _variables(located=False, strata=False, pairwise=False, delta=16):
    fn = "zagg.stats.tdigest.build_tdigest" + ("_pairwise" if pairwise else "")
    base = {
        "kind": "ragged",
        "function": fn,
        "source": "h_ph",
        "inner_shape": [2],
        "params": {"delta": delta},
        "dtype": "float32",
        "fill_value": 0,
    }
    if located:
        base["location"] = "leaf_id"
    variables = {
        "count": {"function": "len", "source": "h_ph", "dtype": "int32", "fill_value": 0},
    }
    if strata:
        for name, where in (("h_sig", _SIGNAL), ("h_noise", _NOISE)):
            f = {**base, "function": "zagg.stats.tdigest.build_tdigest_where"}
            f["params"] = {**base["params"], "where": where}
            variables[name] = f
    else:
        variables["h_tdigest"] = base
    return variables


def _config(variables, streaming=None):
    agg = {"variables": variables}
    if streaming is not None:
        agg["streaming"] = streaming
    return PipelineConfig(
        data_source={
            "reader": "h5coro",
            "driver": "s3",
            "groups": ["gt1l"],
            "index": {"backend": "hierarchical"},
            "shard_workers": 1,
        },
        aggregation=agg,
    )


_SPILL = {"buffer_granules": 1, "mode": "spill"}


def _grid(cfg):
    return HealpixGrid(6, 8, layout="fullsphere", config=cfg)


def _shard_key():
    from mortie import geo2mort

    return int(geo2mort(-78.5, -132.0, order=6)[0])


def _point_leafs(grid, cell, n, rng):
    """``n`` order-29 point words strictly inside child cell ``cell``."""
    from mortie import mort2geo

    lat, lon = mort2geo(np.array([cell], dtype=np.uint64))
    out = np.empty(0, dtype=np.uint64)
    scale = 0.2
    while len(out) < n:
        lats = float(lat) + rng.uniform(-scale, scale, 4 * n)
        lons = float(lon) + rng.uniform(-scale, scale, 4 * n)
        leafs = np.asarray(grid.assign(lats, lons))
        out = np.concatenate([out, leafs[np.asarray(grid.cells_of(leafs)) == cell]])
        scale *= 0.5
    return out[:n]


def _granule_dfs(grid, shard_key, cell_idx_lists, obs_per_cell=60, seed=0, conf=False):
    """One DataFrame per granule; real order-29 point leafs per chosen cell.

    ``conf=True`` adds the five ``signal_conf_*`` columns (ATL03's ``-1..4``
    range) for composition tests.
    """
    rng = np.random.default_rng(seed)
    children = np.asarray(grid.children(shard_key), dtype=np.uint64)
    dfs = []
    for idxs in cell_idx_lists:
        n = obs_per_cell * len(idxs)
        h, leaf = [], []
        for ci in idxs:
            h.append(rng.normal(0.0, 10.0, obs_per_cell).astype(np.float32))
            leaf.append(_point_leafs(grid, int(children[ci]), obs_per_cell, rng))
        cols = {"h_ph": np.concatenate(h), "leaf_id": np.concatenate(leaf)}
        if conf:
            for name in _CONF_COLS:
                cols[name] = rng.integers(-1, 5, n).astype(np.int8)
        dfs.append(pd.DataFrame(cols))
    return dfs


def _run(monkeypatch, cfg, grid, shard_key, dfs):
    reads = iter(dfs)
    monkeypatch.setattr("zagg.processing._read_group", lambda *a, **k: next(reads))
    monkeypatch.setattr("zagg.processing.h5coro.H5Coro", lambda *a, **k: object())
    monkeypatch.setattr("zagg.processing._make_url_rewriter", lambda driver: lambda u: u)
    ragged: dict = {}
    df_out, meta = process_shard(
        grid,
        shard_key,
        [f"s3://b/g{i}.h5" for i in range(len(dfs))],
        s3_credentials=_CREDS,
        config=cfg,
        ragged_out=ragged,
    )
    return df_out, ragged, meta


def _force_tiny_blocks(monkeypatch):
    monkeypatch.setattr("zagg.processing.spill._default_block_bytes", lambda k, tmp_dir=None: 1)


_CELL_LISTS = [[0, 4, 8], [2, 4, 10], [1, 8, 9], [0, 10, 15], [4, 8, 10]]


def _contributors(dfs, grid, cell, mask_fn=None):
    """Every leaf word the granules contribute to child cell ``cell``."""
    words = []
    for df in dfs:
        in_cell = np.asarray(grid.cells_of(df["leaf_id"].values)) == cell
        keep = in_cell
        if mask_fn is not None:
            keep = in_cell & mask_fn(df)
        words.append(df["leaf_id"].values[keep])
    return np.concatenate(words)


def _assert_ancestor_or_equal(locs, contributors):
    """Each merged location lies between the contributor hull and a member.

    Membership per centroid is not observable from the output, so pin both
    bounds: every location is (a) the ancestor-or-equal of at least one
    contributor word (it sits on a member's path) and (b) a descendant-or-equal
    of the common ancestor of ALL contributors (it cannot escape the hull).
    """
    from mortie import clip2order, common_ancestor, orders_of

    contributors = np.asarray(contributors, dtype=np.uint64)
    hull = np.uint64(common_ancestor(contributors))
    hull_order = int(orders_of(np.array([hull], dtype=np.uint64))[0])
    loc_orders = np.asarray(orders_of(locs))
    for loc, order in zip(locs, loc_orders):
        assert np.any(clip2order(int(order), contributors) == loc), (
            f"location {loc} is no contributor's ancestor"
        )
        assert np.uint64(clip2order(hull_order, np.array([loc], dtype=np.uint64))[0]) == hull, (
            f"location {loc} escapes the contributor hull {hull}"
        )


class TestSpillFoldProbe:
    """The spill mergeability probe (validate_spill_fold) vs merge mode."""

    def test_located_config_is_mergeable(self):
        cfg = _config(_variables(located=True), streaming=_SPILL)
        agg = SpillAggregator(cfg, _grid(cfg), "pandas", 1)
        assert agg._mergeable
        assert agg._digest_fields["h_tdigest"].location == "leaf_id"
        agg.close()

    def test_strata_config_is_mergeable(self):
        cfg = _config(_variables(strata=True), streaming=_SPILL)
        agg = SpillAggregator(cfg, _grid(cfg), "pandas", 1)
        assert agg._mergeable
        assert agg._digest_fields["h_sig"].where == _SIGNAL
        agg.close()

    def test_located_strata_config_is_mergeable(self):
        cfg = _config(_variables(located=True, strata=True), streaming=_SPILL)
        agg = SpillAggregator(cfg, _grid(cfg), "pandas", 1)
        assert agg._mergeable
        agg.close()

    def test_expression_field_still_has_no_fold(self):
        variables = _variables()
        variables["h_spread"] = {"expression": "np.nanmax(h_ph) - np.nanmin(h_ph)"}
        with pytest.raises(ValueError, match="expression fields have no cross-block fold"):
            validate_spill_fold(_config(variables))

    def test_non_tdigest_ragged_still_has_no_fold(self):
        variables = _variables()
        variables["h_raw"] = {
            "function": "np.sort",
            "source": "h_ph",
            "kind": "ragged",
            "inner_shape": [1],
        }
        with pytest.raises(ValueError, match="h_raw.*fold law"):
            validate_spill_fold(_config(variables))

    def test_mode_merge_validation_unchanged(self):
        # The spill probe widening must not leak into mode: merge — its
        # per-flush fold degrades locations continuously and ignores where.
        with pytest.raises(ValueError, match="located ragged"):
            validate_streaming(_config(_variables(located=True)))
        with pytest.raises(ValueError, match="h_sig"):
            validate_streaming(_config(_variables(strata=True)))


class TestLocatedMultiBlock:
    """Located fields across forced block closes: digests exact vs unlocated,
    locations bounded by the contributor hull."""

    @pytest.mark.parametrize("pairwise", [False, True])
    def test_digest_bytes_match_unlocated_and_locations_bounded(self, monkeypatch, pairwise):
        _force_tiny_blocks(monkeypatch)
        key = _shard_key()
        out = {}
        dfs_cache = None
        for located in (False, True):
            cfg = _config(_variables(located=located, pairwise=pairwise), streaming=_SPILL)
            grid = _grid(cfg)
            if dfs_cache is None:
                dfs_cache = _granule_dfs(grid, key, _CELL_LISTS, seed=11)
            out[located] = _run(monkeypatch, cfg, grid, key, list(dfs_cache))
        (df_u, ragged_u, _), (df_l, ragged_l, _) = out[False], out[True]
        pd.testing.assert_frame_equal(df_u, df_l)
        vals_u, idx_u = ragged_u["h_tdigest"]
        vals_l, idx_l, locs_l = ragged_l["h_tdigest"]
        assert idx_u == idx_l
        # The digest channel is identical with or without locations (the
        # located build/merge overloads change only the companion channel).
        for a, b in zip(vals_u, vals_l, strict=True):
            np.testing.assert_array_equal(a, b)
        # Locations row-aligned with the payloads, uint64, hull-bounded.
        grid = _grid(_config(_variables(located=True), streaming=_SPILL))
        children = np.asarray(grid.children(key), dtype=np.uint64)
        assert len(locs_l) == len(vals_l)
        for cell_i, digest, locs in zip(idx_l, vals_l, locs_l, strict=True):
            assert locs.dtype == np.uint64
            assert locs.shape == (len(digest),)
            _assert_ancestor_or_equal(locs, _contributors(dfs_cache, grid, int(children[cell_i])))

    def test_counts_and_weights_exact_vs_pooled(self, monkeypatch):
        _force_tiny_blocks(monkeypatch)
        key = _shard_key()
        pooled_cfg = _config(_variables(located=True))
        spill_cfg = _config(_variables(located=True), streaming=_SPILL)
        grid = _grid(pooled_cfg)
        dfs = _granule_dfs(grid, key, _CELL_LISTS, obs_per_cell=100, seed=4)
        df_p, ragged_p, _ = _run(monkeypatch, pooled_cfg, grid, key, list(dfs))
        df_s, ragged_s, _ = _run(monkeypatch, spill_cfg, _grid(spill_cfg), key, list(dfs))
        pd.testing.assert_series_equal(df_p["count"], df_s["count"])
        vals_p, idx_p, _ = ragged_p["h_tdigest"]
        vals_s, idx_s, _ = ragged_s["h_tdigest"]
        assert idx_p == idx_s
        for dp, ds in zip(vals_p, vals_s, strict=True):
            # Total weight is the exact observation count either way.
            assert float(dp[:, 1].sum()) == float(ds[:, 1].sum())
            for q in (0.1, 0.5, 0.9):
                assert abs(quantile_from_tdigest(ds, q) - quantile_from_tdigest(dp, q)) < 1.0

    def test_below_knee_locations_are_exact_point_words(self, monkeypatch):
        # n <= delta across every block: the fold stays loss-free, every
        # centroid is weight 1, and its location is the exact order-29 point
        # word — the multiset of locations equals the contributor words.
        _force_tiny_blocks(monkeypatch)
        key = _shard_key()
        cfg = _config(_variables(located=True, delta=512), streaming=_SPILL)
        grid = _grid(cfg)
        dfs = _granule_dfs(grid, key, _CELL_LISTS, obs_per_cell=20, seed=8)
        _, ragged, _ = _run(monkeypatch, cfg, grid, key, list(dfs))
        vals, idx, locs = ragged["h_tdigest"]
        children = np.asarray(grid.children(key), dtype=np.uint64)
        assert len(vals) > 0
        for cell_i, digest, cell_locs in zip(idx, vals, locs, strict=True):
            assert (digest[:, 1] == 1.0).all()
            contributors = _contributors(dfs, grid, int(children[cell_i]))
            np.testing.assert_array_equal(np.sort(cell_locs), np.sort(contributors))

    def test_multi_block_actually_engaged(self, monkeypatch):
        _force_tiny_blocks(monkeypatch)
        closes = {"n": 0}
        orig = SpillAggregator._close_block

        def counting(self):
            closes["n"] += 1
            orig(self)

        monkeypatch.setattr(SpillAggregator, "_close_block", counting)
        key = _shard_key()
        cfg = _config(_variables(located=True), streaming=_SPILL)
        grid = _grid(cfg)
        dfs = _granule_dfs(grid, key, _CELL_LISTS, seed=2)
        _, ragged, meta = _run(monkeypatch, cfg, grid, key, dfs)
        assert closes["n"] >= len(_CELL_LISTS)
        assert meta["total_obs"] > 0
        assert len(ragged["h_tdigest"]) == 3  # located 3-tuple delivered


class TestStrataMultiBlock:
    """build_tdigest_where strata across forced block closes."""

    def _pooled_and_spill(self, monkeypatch, located=False, seed=4):
        key = _shard_key()
        pooled_cfg = _config(_variables(strata=True, located=located))
        spill_cfg = _config(_variables(strata=True, located=located), streaming=_SPILL)
        grid = _grid(pooled_cfg)
        dfs = _granule_dfs(grid, key, _CELL_LISTS, obs_per_cell=100, seed=seed)
        pooled = _run(monkeypatch, pooled_cfg, grid, key, list(dfs))
        _force_tiny_blocks(monkeypatch)
        spilled = _run(monkeypatch, spill_cfg, _grid(spill_cfg), key, list(dfs))
        return key, grid, dfs, pooled, spilled

    def test_stratum_weights_exact_and_quantiles_close_vs_pooled(self, monkeypatch):
        _, _, _, (df_p, ragged_p, _), (df_s, ragged_s, _) = self._pooled_and_spill(monkeypatch)
        pd.testing.assert_series_equal(df_p["count"], df_s["count"])
        for name in ("h_sig", "h_noise"):
            vals_p, idx_p = ragged_p[name]
            vals_s, idx_s = ragged_s[name]
            assert idx_p == idx_s
            for dp, ds in zip(vals_p, vals_s, strict=True):
                # Stratum membership is decided per row before the build, so
                # the stratum count (total weight) is exact across blocks.
                assert float(dp[:, 1].sum()) == float(ds[:, 1].sum())
                for q in (0.1, 0.5, 0.9):
                    assert abs(quantile_from_tdigest(ds, q) - quantile_from_tdigest(dp, q)) < 1.5

    def test_located_strata_locations_bounded_per_stratum(self, monkeypatch):
        key, grid, dfs, _, (_, ragged_s, _) = self._pooled_and_spill(
            monkeypatch, located=True, seed=6
        )
        children = np.asarray(grid.children(key), dtype=np.uint64)
        masks = {
            "h_sig": lambda df: df["h_ph"].values > 0,
            "h_noise": lambda df: ~(df["h_ph"].values > 0),
        }
        for name, mask_fn in masks.items():
            vals, idx, locs = ragged_s[name]
            assert len(vals) > 0
            for cell_i, digest, cell_locs in zip(idx, vals, locs, strict=True):
                assert cell_locs.shape == (len(digest),)
                contributors = _contributors(dfs, grid, int(children[cell_i]), mask_fn)
                # Only the stratum's own rows contribute to its locations.
                _assert_ancestor_or_equal(cell_locs, contributors)

    def test_strata_and_composition_config_is_mergeable(self):
        # The full stratified-product shape (issue #321 exemplar): strata +
        # composition + count, accepted by the spill fold probe (issue #370
        # option (a)). A non-fold scalar stays rejected alongside.
        variables = _variables(located=True, strata=True)
        variables["composition"] = _composition_field()
        cfg = _config(variables, streaming=_SPILL)
        validate_spill_fold(cfg)
        agg = SpillAggregator(cfg, _grid(cfg), "pandas", 1)
        assert agg._mergeable
        assert "composition" in agg._composition_fields
        agg.close()
        variables["h_mean"] = {"function": "mean", "source": "h_ph"}
        with pytest.raises(ValueError, match="h_mean.*no.*cross-block fold"):
            validate_spill_fold(_config(variables))

    def test_strata_invariant_to_block_placement(self, monkeypatch):
        # The same rows split into different granules (hence different block
        # boundaries under buffer_granules=1 + tiny blocks) must land the
        # exact same per-cell stratum weights, and close quantiles.
        _force_tiny_blocks(monkeypatch)
        key = _shard_key()
        cfg_a = _config(_variables(strata=True), streaming=_SPILL)
        grid = _grid(cfg_a)
        dfs = _granule_dfs(grid, key, _CELL_LISTS, obs_per_cell=80, seed=3)
        # Placement B: the same rows, re-partitioned into two granules.
        pooled_rows = pd.concat(dfs, ignore_index=True)
        half = len(pooled_rows) // 2
        dfs_b = [pooled_rows.iloc[:half].copy(), pooled_rows.iloc[half:].copy()]
        cfg_b = _config(_variables(strata=True), streaming=_SPILL)
        _, ragged_a, _ = _run(monkeypatch, cfg_a, grid, key, dfs)
        _, ragged_b, _ = _run(monkeypatch, cfg_b, _grid(cfg_b), key, dfs_b)
        for name in ("h_sig", "h_noise"):
            vals_a, idx_a = ragged_a[name]
            vals_b, idx_b = ragged_b[name]
            assert idx_a == idx_b
            for da, db in zip(vals_a, vals_b, strict=True):
                assert float(da[:, 1].sum()) == float(db[:, 1].sum())
                for q in (0.1, 0.5, 0.9):
                    assert abs(quantile_from_tdigest(da, q) - quantile_from_tdigest(db, q)) < 1.5


def _true_lane_counts(dfs, grid, cell, threshold=2):
    """Ground-truth composition lane counts + n_signal for one cell's rows."""
    conf_rows, finite_rows = [], []
    for df in dfs:
        in_cell = np.asarray(grid.cells_of(df["leaf_id"].values)) == cell
        conf_rows.append(df.loc[in_cell, list(_CONF_COLS)].to_numpy(np.int64))
        finite_rows.append(np.isfinite(df.loc[in_cell, "h_ph"].to_numpy(np.float64)))
    conf = np.concatenate(conf_rows)[np.concatenate(finite_rows)]
    signal = (conf >= threshold).any(axis=1)
    n = int(signal.sum())
    counts = np.zeros(8, dtype=np.int64)
    if n:
        csig = conf[signal]
        counts[:5] = (csig >= threshold).sum(axis=0)
        strongest = csig.max(axis=1)
        for i, level in enumerate((2, 3, 4)):
            counts[5 + i] = int((strongest == level).sum())
    return counts, n


class TestCompositionMultiBlock:
    """merge_composition across forced block closes (issue #370 option (a))."""

    def test_presence_exact_counts_within_fold_bound(self, monkeypatch):
        key = _shard_key()
        variables = {
            "count": {"function": "len", "source": "h_ph", "dtype": "int32", "fill_value": 0},
            "composition": _composition_field(),
        }
        pooled_cfg = _config(dict(variables))
        spill_cfg = _config(dict(variables), streaming=_SPILL)
        grid = _grid(pooled_cfg)
        dfs = _granule_dfs(grid, key, _CELL_LISTS, obs_per_cell=60, seed=21, conf=True)
        df_p, _, _ = _run(monkeypatch, pooled_cfg, grid, key, list(dfs))
        _force_tiny_blocks(monkeypatch)
        df_s, _, _ = _run(monkeypatch, spill_cfg, _grid(spill_cfg), key, list(dfs))
        np.testing.assert_array_equal(df_p["count"].values, df_s["count"].values)
        children = np.asarray(grid.children(key), dtype=np.uint64)
        checked = 0
        for i, cell in enumerate(children):
            word_p, word_s = int(df_p["composition"].values[i]), int(df_s["composition"].values[i])
            truth, n = _true_lane_counts(dfs, grid, int(cell))
            if n == 0:
                assert word_p == 0 and word_s == 0
                continue
            # Presence (lane > 0) survives every fold exactly — that is the
            # spec's floor guarantee — and matches the pooled word's presence.
            np.testing.assert_array_equal(unpack_composition(word_p) > 0, truth > 0)
            np.testing.assert_array_equal(unpack_composition(word_s) > 0, truth > 0)
            # Below n=254 the pooled word recovers counts exactly; the folded
            # word stays within the documented O(n/510)-per-fold error over
            # at most len(dfs) folds (one block per granule here).
            assert n <= 254
            np.testing.assert_array_equal(counts_from_composition(word_p, n), truth)
            tol = 1 + len(dfs) * n / 510.0
            assert np.abs(counts_from_composition(word_s, n) - truth).max() <= tol
            checked += 1
        assert checked >= 3

    def test_full_stratified_product_survives_block_closes(self, monkeypatch):
        # The issue #370 acceptance shape: located strata + composition +
        # count through a forced multi-block run lands every channel.
        key = _shard_key()
        variables = _variables(located=True, strata=True)
        variables["composition"] = _composition_field()
        pooled_cfg = _config(dict(variables))
        spill_cfg = _config(dict(variables), streaming=_SPILL)
        grid = _grid(pooled_cfg)
        dfs = _granule_dfs(grid, key, _CELL_LISTS, obs_per_cell=80, seed=9, conf=True)
        df_p, ragged_p, _ = _run(monkeypatch, pooled_cfg, grid, key, list(dfs))
        _force_tiny_blocks(monkeypatch)
        df_s, ragged_s, meta_s = _run(monkeypatch, spill_cfg, _grid(spill_cfg), key, list(dfs))
        assert meta_s["total_obs"] > 0
        np.testing.assert_array_equal(df_p["count"].values, df_s["count"].values)
        # Composition present and presence-consistent with pooled.
        for wp, ws in zip(df_p["composition"].values, df_s["composition"].values, strict=True):
            np.testing.assert_array_equal(
                unpack_composition(int(wp)) > 0, unpack_composition(int(ws)) > 0
            )
        # Both strata deliver the located 3-tuple with exact stratum weights.
        for name in ("h_sig", "h_noise"):
            vals_p, idx_p, _ = ragged_p[name]
            vals_s, idx_s, locs_s = ragged_s[name]
            assert idx_p == idx_s
            assert len(locs_s) == len(vals_s)
            for dp, ds, ls in zip(vals_p, vals_s, locs_s, strict=True):
                assert float(dp[:, 1].sum()) == float(ds[:, 1].sum())
                assert ls.shape == (len(ds),)
