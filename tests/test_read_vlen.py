"""Tests for the generic vlen reader primitives (issue #425).

A synthetic GEDI-shaped mini-granule (canned arrays behind the ``_FakeH5``
stub, the ``test_processing`` convention) exercises the three grammar
primitives — origin-aware ragged gather, expand-by-count, linspace coordinate
synthesis — plus the record-level filters and the sibling-asset (L2A) join,
all through the public ``_read_group`` dispatch.
"""

import numpy as np
import pytest

from zagg.config import (
    PipelineConfig,
    validate_config,
)
from zagg.processing.read import (
    _expand_mask_at_rows,
    _expand_mask_to_base,
    _link_parent_at_rows,
    _read_group,
    _validate_link_disjoint,
    link_base_extent,
)
from zagg.processing.read_vlen import (
    expand_link_indices,
    synthesize_linspace,
)

# ── synthetic mini-granule (both products) ───────────────────────────────────
#
# Four shots; rx_sample_start_index is ORIGIN-1 (the GEDI convention), so a
# missing index_base would shift every waveform by one sample — the tests pin
# exact values to catch that bug class.
#
#   shot   count  start(1-based)  lat    elev_bin0  elev_lastbin  degrade
#   101    5      1               10.0   100.0      96.0          0
#   102    3      6               11.0   50.0       48.0          0
#   103    1      9               12.0   30.0       30.0          1
#   104    4      10              13.0   80.0       77.0          0

SHOTS = np.array([101, 102, 103, 104], dtype=np.uint64)
COUNTS = np.array([5, 3, 1, 4], dtype=np.uint16)
STARTS = np.array([1, 6, 9, 10], dtype=np.uint64)  # origin-1
LATS = np.array([10.0, 11.0, 12.0, 13.0])
LONS = np.array([-100.0, -100.1, -100.2, -100.3])
E0 = np.array([100.0, 50.0, 30.0, 80.0])
E1 = np.array([96.0, 48.0, 30.0, 77.0])
DEGRADE = np.array([0, 0, 1, 0], dtype=np.uint8)
NOISE_MEAN = np.array([10.0, 12.0, 11.0, 9.0], dtype=np.float32)
WAVE = np.arange(13, dtype=np.float32) + 100.0  # samples 100..112


class _FakeH5:
    """Stub h5coro object (the ``test_processing`` convention)."""

    def __init__(self, arrays):
        self._arrays = arrays

    def readDatasets(self, datasets):  # noqa: N802 (mirror real h5coro API)
        out = {}
        for d in datasets:
            if isinstance(d, str):
                out[d] = self._arrays[d]
                continue
            path = d["dataset"]
            arr = self._arrays[path]
            hs = d["hyperslice"]
            if hs:
                lo, hi = hs[0]
                arr = arr[lo:hi]
            out[path] = arr
        return out


class _LatGrid:
    """Grid stub: shard key 1 for lat >= 11.5, else 0 (leaf id == shard id)."""

    @staticmethod
    def assign(lats, lons):
        return (np.asarray(lats) >= 11.5).astype(np.uint64)

    @staticmethod
    def shards_of(leaf_ids):
        return np.asarray(leaf_ids).astype(int)


class _OneShardGrid:
    """Grid stub: every row in shard 0, leaf id == row index."""

    @staticmethod
    def assign(lats, lons):
        return np.arange(len(lats), dtype=np.uint64)

    @staticmethod
    def shards_of(leaf_ids):
        return np.zeros(len(leaf_ids), dtype=int)


NOISE_STDDEV = np.array([2.0, 1.0, 1.5, 1.0], dtype=np.float32)
RX_ENERGY = np.array([50.0, 30.0, 5.0, 40.0], dtype=np.float32)


def _l1b_arrays(group="BEAM0000"):
    return {
        f"/{group}/rxwaveform": WAVE,
        f"/{group}/rx_sample_start_index": STARTS,
        f"/{group}/rx_sample_count": COUNTS,
        f"/{group}/shot_number": SHOTS,
        f"/{group}/noise_mean_corrected": NOISE_MEAN,
        f"/{group}/noise_stddev_corrected": NOISE_STDDEV,
        f"/{group}/rx_energy": RX_ENERGY,
        f"/{group}/geolocation/latitude_bin0": LATS,
        f"/{group}/geolocation/longitude_bin0": LONS,
        f"/{group}/geolocation/elevation_bin0": E0,
        f"/{group}/geolocation/elevation_lastbin": E1,
        f"/{group}/geolocation/degrade": DEGRADE,
    }


def _l2a_arrays(group="BEAM0000"):
    # Shot 103 has NO sibling row: the join must drop it, never borrow a row.
    return {
        f"/{group}/shot_number": np.array([101, 102, 104], dtype=np.uint64),
        f"/{group}/quality_flag": np.array([1, 0, 1], dtype=np.uint8),
        f"/{group}/sensitivity": np.array([0.95, 0.99, 0.50], dtype=np.float32),
        f"/{group}/rx_assess/rx_clipbin_count": np.array([0, 0, 0], dtype=np.uint16),
    }


def _vlen_ds(**extra):
    ds = {
        "coordinates": {
            "latitude": "/{group}/geolocation/latitude_bin0",
            "longitude": "/{group}/geolocation/longitude_bin0",
            "level": "shots",
        },
        "variables": {
            "rxwaveform": "/{group}/rxwaveform",
            "elevation": {
                "synthesize": "linspace",
                "level": "shots",
                "start": "/{group}/geolocation/elevation_bin0",
                "stop": "/{group}/geolocation/elevation_lastbin",
            },
        },
        "base_level": "samples",
        "levels": {
            "samples": {"path": "/{group}", "link": None},
            "shots": {
                "path": "/{group}/geolocation",
                "coordinates": {"latitude": "latitude_bin0", "longitude": "longitude_bin0"},
                "link": {
                    "to": "samples",
                    "index_beg": "/{group}/rx_sample_start_index",
                    "count": "/{group}/rx_sample_count",
                    "index_base": 1,
                },
                "variables": {
                    "shot_number": "/{group}/shot_number",
                    "noise_mean": "/{group}/noise_mean_corrected",
                },
            },
        },
    }
    ds.update(extra)
    return ds


def _expected_elevation(shot_idx):
    """Hand-computed linspace(elev_bin0, elev_lastbin, count) for one shot."""
    n = int(COUNTS[shot_idx])
    return np.linspace(E0[shot_idx], E1[shot_idx], n)


# ── primitive units ──────────────────────────────────────────────────────────


class TestExpandLinkIndices:
    def test_origin_1_placement(self):
        parent, within = expand_link_indices(STARTS, COUNTS, 1, 13)
        assert parent.tolist() == [0] * 5 + [1] * 3 + [2] + [3] * 4
        assert within.tolist() == [0, 1, 2, 3, 4, 0, 1, 2, 0, 0, 1, 2, 3]

    def test_gap_rows_unassigned(self):
        # Record 1 starts at 8 (1-based), leaving base rows 5..6 untiled.
        parent, within = expand_link_indices(np.array([1, 8]), np.array([5, 3]), 1, 10)
        assert parent.tolist() == [0, 0, 0, 0, 0, -1, -1, 1, 1, 1]

    def test_empty_record_skipped(self):
        # count == 0 with the origin-1 sentinel start 0 must not raise.
        parent, _ = expand_link_indices(np.array([1, 0, 6]), np.array([5, 0, 2]), 1, 7)
        assert parent.tolist() == [0] * 5 + [2] * 2

    def test_negative_start_raises(self):
        with pytest.raises(ValueError, match="less than index_base"):
            expand_link_indices(np.array([0]), np.array([3]), 1, 3)

    def test_overrun_raises(self):
        with pytest.raises(ValueError, match="exceeds base size"):
            expand_link_indices(np.array([1]), np.array([5]), 1, 3)


class TestLinkDisjointValidation:
    """Overlapping records are refused where a level's link is assembled (#452).

    The precondition was assumed everywhere and checked nowhere. The reviewer's
    construction ``index_beg=[0, 5]``, ``count=[10, 2]`` is the whole problem in
    three records' worth of data: the paint maps resolve rows 7-9 to record 0
    (later record shadows earlier, then the paint ends) while the searchsorted
    derivation resolves them to nobody, so coordinates and the synthesized
    column would disagree with every companion column and every record-level
    filter verdict, row by row, with nothing raised."""

    OVERLAP_BEG = np.array([0, 5], dtype=np.int64)
    OVERLAP_CNT = np.array([10, 2], dtype=np.int64)

    def test_the_two_owner_derivations_disagree_on_the_overlap(self):
        # Why the check has to exist: unvalidated, these are the columns that
        # would ship side by side out of one read.
        painted, _ = expand_link_indices(self.OVERLAP_BEG, self.OVERLAP_CNT, 0, 10)
        searched = _link_parent_at_rows(self.OVERLAP_BEG, self.OVERLAP_CNT, 0, np.arange(10))
        assert painted.tolist() == [0, 0, 0, 0, 0, 1, 1, 0, 0, 0]
        assert searched.tolist() == [0, 0, 0, 0, 0, 1, 1, -1, -1, -1]

    def test_overlapping_records_raise(self):
        with pytest.raises(ValueError, match="requires disjoint record ranges"):
            _validate_link_disjoint(self.OVERLAP_BEG, self.OVERLAP_CNT, 0)

    def test_the_message_names_the_level_and_both_records(self):
        with pytest.raises(ValueError) as e:
            _validate_link_disjoint(self.OVERLAP_BEG, self.OVERLAP_CNT, 0, level="shots")
        msg = str(e.value)
        assert "level 'shots'" in msg
        assert "record 0 covers base rows [0:10]" in msg
        assert "record 1 starts at 5" in msg

    def test_disjoint_links_pass(self):
        # The shipped fixtures, both packed and strided, plus an unsorted link
        # with gaps and an issue #116 empty (start sentinel 0 under origin-1).
        _validate_link_disjoint(STARTS, COUNTS, 1)
        _validate_link_disjoint(STARTS_STRIDED, COUNTS, 1)
        _validate_link_disjoint(np.array([10, 1, 0, 5]), np.array([2, 3, 0, 4]), 1)

    def test_touching_records_are_disjoint(self):
        # A record starting exactly where the previous ends is contiguous, not
        # overlapping — the ATL03 shape, which must keep passing.
        _validate_link_disjoint(np.array([0, 5, 8]), np.array([5, 3, 2]), 0)

    def test_the_vlen_route_refuses_an_overlapping_coordinates_link(self):
        arrays = _l1b_arrays()
        arrays["/BEAM0000/rx_sample_count"] = np.array([6, 3, 1, 4], dtype=np.uint16)
        with pytest.raises(ValueError, match="level 'shots'.*disjoint record ranges"):
            _read_group(_FakeH5(arrays), "BEAM0000", _vlen_ds(), 0, _OneShardGrid())


class TestSynthesizeLinspace:
    def test_endpoints_exact_and_single_sample(self):
        parent, within = expand_link_indices(STARTS, COUNTS, 1, 13)
        vals = synthesize_linspace(E0, E1, COUNTS, parent, within)
        # Endpoints are exact per shot; a 1-sample shot gets start.
        assert vals[0] == E0[0] and vals[4] == E1[0]
        assert vals[8] == E0[2]  # count == 1
        np.testing.assert_allclose(vals[:5], _expected_elevation(0))
        np.testing.assert_allclose(vals[9:13], _expected_elevation(3))

    def test_gap_rows_nan(self):
        parent, within = expand_link_indices(np.array([1, 8]), np.array([5, 3]), 1, 10)
        vals = synthesize_linspace(
            np.array([0.0, 10.0]), np.array([4.0, 12.0]), np.array([5, 3]), parent, within
        )
        assert np.isnan(vals[5]) and np.isnan(vals[6])
        np.testing.assert_allclose(vals[:5], [0, 1, 2, 3, 4])


class TestExpandMaskAtRowsEquivalence:
    """``_expand_mask_at_rows`` is ``_expand_mask_to_base(...)[base_rows]``,
    validation scope included (PR #454 review).

    The base-rate form's ``beg < 0`` check sits after the predicate guard, so a
    record the predicate rejects is never validated. The reviewer's
    construction is a record with a malformed origin-0 start that the mask
    filters *out*: the base-rate form reads it fine, and the at-rows twin must
    not fail-hard on the production route where the original does not."""

    def test_a_filtered_out_record_is_not_validated(self):
        ibeg = np.array([0, 1])
        cnt = np.array([2, 3])
        mask = np.array([False, True])
        base = _expand_mask_to_base(mask, ibeg, cnt, 1, 4)
        at_rows = _expand_mask_at_rows(mask, ibeg, cnt, 1, np.arange(4))
        assert base.tolist() == [True, True, True, False]
        assert at_rows.tolist() == base.tolist()

    def test_a_kept_record_with_a_malformed_start_still_raises(self):
        # The refusal is narrowed, not removed: flip the same record's verdict
        # to keep and both forms raise the same way.
        ibeg = np.array([0, 1])
        cnt = np.array([2, 3])
        mask = np.array([True, True])
        with pytest.raises(ValueError, match="less than index_base"):
            _expand_mask_to_base(mask, ibeg, cnt, 1, 4)
        with pytest.raises(ValueError, match="less than index_base"):
            _expand_mask_at_rows(mask, ibeg, cnt, 1, np.arange(4))

    def test_equivalence_on_the_shipped_strided_fixture(self):
        for mask in ([True] * 4, [False] * 4, [True, False, True, False]):
            m = np.array(mask)
            base = _expand_mask_to_base(m, STARTS_STRIDED, COUNTS, 1, STRIDED_EXTENT)
            rows = np.arange(STRIDED_EXTENT)
            np.testing.assert_array_equal(
                _expand_mask_at_rows(m, STARTS_STRIDED, COUNTS, 1, rows), base
            )


class TestPlannedGatherMap:
    """The planned arm's gather map: plan-rate, and identical to the full-rate
    map restricted to the planned rows (review finding, PR #432)."""

    def test_matches_the_full_map_on_planned_rows(self):
        from zagg.processing.read_vlen import _planned_gather_map

        parent, within = expand_link_indices(STARTS, COUNTS, 1, 13)
        gidx, p_plan, w_plan = _planned_gather_map(STARTS, COUNTS, 1, 13, [(0, 5), (9, 13)])
        assert gidx.tolist() == [0, 1, 2, 3, 4, 9, 10, 11, 12]
        assert p_plan.tolist() == parent[gidx].tolist()
        assert w_plan.tolist() == within[gidx].tolist()
        # Sized to the plan, not to the base extent.
        assert len(p_plan) == 9 < 13

    def test_gap_rows_stay_unassigned(self):
        from zagg.processing.read_vlen import _planned_gather_map

        # Record 1 starts at 8 (1-based), leaving base rows 5..6 untiled.
        _, parent, _ = _planned_gather_map(np.array([1, 8]), np.array([5, 3]), 1, 10, [(3, 8)])
        assert parent.tolist() == [0, 0, -1, -1, 1]

    def test_validation_matches_the_full_map(self):
        from zagg.processing.read_vlen import _planned_gather_map

        with pytest.raises(ValueError, match="less than index_base"):
            _planned_gather_map(np.array([0]), np.array([3]), 1, 3, [(0, 3)])
        with pytest.raises(ValueError, match="exceeds base size"):
            _planned_gather_map(np.array([1]), np.array([5]), 1, 3, [(0, 3)])

    def test_planned_route_never_builds_the_full_rate_map(self, monkeypatch):
        import zagg.processing.read_vlen as rv

        def _boom(*a, **k):
            raise AssertionError("the planned arm must not allocate at base rate")

        monkeypatch.setattr(rv, "expand_link_indices", _boom)
        ds = _vlen_ds(read_plan={"spatial_index": "shots", "pad": 0})
        df = _read_group(_FakeH5(_l1b_arrays()), "BEAM0000", ds, 1, _LatGrid())
        assert df["shot_number"].tolist() == [103] + [104] * 4


# ── the vlen read route (gather + expand + synthesis) ────────────────────────


class TestVlenReadGroup:
    def test_full_read_gather_expand_synthesis(self):
        df = _read_group(_FakeH5(_l1b_arrays()), "BEAM0000", _vlen_ds(), 0, _OneShardGrid())
        # Every sample of every shot; waveform values exact (origin-1 gather).
        assert len(df) == 13
        np.testing.assert_array_equal(df["rxwaveform"].to_numpy(), WAVE)
        # Coordinates expanded by count: 5x shot0, 3x shot1, 1x shot2, 4x shot3.
        assert df["shot_number"].tolist() == [101] * 5 + [102] * 3 + [103] + [104] * 4
        np.testing.assert_array_equal(
            df["noise_mean"].to_numpy(), np.repeat(NOISE_MEAN, COUNTS.astype(int))
        )
        np.testing.assert_allclose(
            df["elevation"].to_numpy(),
            np.concatenate([_expected_elevation(i) for i in range(4)]),
        )

    def test_planned_read_selects_shard_records(self):
        # Shard 1 holds shots 103 + 104 (lat >= 11.5); pad=0 keeps it exact.
        ds = _vlen_ds(read_plan={"spatial_index": "shots", "pad": 0})
        df = _read_group(_FakeH5(_l1b_arrays()), "BEAM0000", ds, 1, _LatGrid())
        assert df["shot_number"].tolist() == [103] + [104] * 4
        np.testing.assert_array_equal(df["rxwaveform"].to_numpy(), WAVE[8:13])

    def test_planned_matches_full(self):
        ds_plan = _vlen_ds(read_plan={"spatial_index": "shots", "pad": 1})
        ds_full = _vlen_ds()
        for shard in (0, 1):
            a = _read_group(_FakeH5(_l1b_arrays()), "BEAM0000", ds_plan, shard, _LatGrid())
            b = _read_group(_FakeH5(_l1b_arrays()), "BEAM0000", ds_full, shard, _LatGrid())
            np.testing.assert_array_equal(a["rxwaveform"].to_numpy(), b["rxwaveform"].to_numpy())
            np.testing.assert_array_equal(a["elevation"].to_numpy(), b["elevation"].to_numpy())

    def test_empty_shard_returns_none(self):
        ds = _vlen_ds(read_plan={"spatial_index": "shots", "pad": 0})
        assert _read_group(_FakeH5(_l1b_arrays()), "BEAM0000", ds, 7, _LatGrid()) is None

    def test_record_level_filter_expands(self):
        # degrade == 0 at the shots level drops shot 103's single sample.
        ds = _vlen_ds(
            filters=[
                {
                    "level": "shots",
                    "dataset": "/{group}/geolocation/degrade",
                    "op": "eq",
                    "value": 0,
                }
            ]
        )
        df = _read_group(_FakeH5(_l1b_arrays()), "BEAM0000", ds, 0, _OneShardGrid())
        assert 103 not in df["shot_number"].tolist()
        assert len(df) == 12

    def test_expression_filter_over_synthesized_column(self):
        ds = _vlen_ds(filters=[{"expression": "elevation >= 79.0"}])
        df = _read_group(_FakeH5(_l1b_arrays()), "BEAM0000", ds, 0, _OneShardGrid())
        # Shot 0: all five samples (96..100); shot 3: 80, 79 survive.
        assert (df["elevation"].to_numpy() >= 79.0).all()
        assert len(df) == 7

    def test_obs_read_counter(self):
        io_stats = {}
        _read_group(
            _FakeH5(_l1b_arrays()), "BEAM0000", _vlen_ds(), 0, _OneShardGrid(), io_stats=io_stats
        )
        assert io_stats["obs_read"] == 13

    def test_spatial_index_must_name_coordinates_level(self):
        ds = _vlen_ds(read_plan={"spatial_index": "samples"})
        with pytest.raises(ValueError, match="must name the coordinates level"):
            _read_group(_FakeH5(_l1b_arrays()), "BEAM0000", ds, 0, _OneShardGrid())


# ── the sibling-asset join (L2A) ─────────────────────────────────────────────


def _l2a_ds(**extra):
    return _vlen_ds(
        assets={
            "l2a": {
                "join": {
                    "left": "/{group}/shot_number",
                    "right": "/{group}/shot_number",
                }
            }
        },
        **extra,
    )


class TestSiblingAssetJoin:
    def test_join_filters_by_shot_number(self):
        # quality_flag == 1 keeps shots 101 + 104; shot 102 fails the flag and
        # shot 103 has no L2A row at all (unmatched -> dropped).
        ds = _l2a_ds(
            filters=[{"asset": "l2a", "dataset": "/{group}/quality_flag", "op": "eq", "value": 1}]
        )
        df = _read_group(
            _FakeH5(_l1b_arrays()),
            "BEAM0000",
            ds,
            0,
            _OneShardGrid(),
            siblings={"l2a": _FakeH5(_l2a_arrays())},
        )
        assert sorted(set(df["shot_number"].tolist())) == [101, 104]
        assert len(df) == 9

    def test_join_multiple_predicates_anded(self):
        ds = _l2a_ds(
            filters=[
                {"asset": "l2a", "dataset": "/{group}/quality_flag", "op": "eq", "value": 1},
                {"asset": "l2a", "dataset": "/{group}/sensitivity", "op": "ge", "value": 0.9},
            ]
        )
        df = _read_group(
            _FakeH5(_l1b_arrays()),
            "BEAM0000",
            ds,
            0,
            _OneShardGrid(),
            siblings={"l2a": _FakeH5(_l2a_arrays())},
        )
        # 104 fails sensitivity (0.50), 102 fails quality, 103 unmatched.
        assert sorted(set(df["shot_number"].tolist())) == [101]

    def test_unmatched_record_never_borrows(self):
        # With no predicates beyond the join itself there are no asset filters,
        # so declare a tautology: every matched shot passes, unmatched drop.
        ds = _l2a_ds(
            filters=[{"asset": "l2a", "dataset": "/{group}/quality_flag", "op": "ge", "value": 0}]
        )
        df = _read_group(
            _FakeH5(_l1b_arrays()),
            "BEAM0000",
            ds,
            0,
            _OneShardGrid(),
            siblings={"l2a": _FakeH5(_l2a_arrays())},
        )
        assert 103 not in df["shot_number"].tolist()

    def test_missing_sibling_handle_raises(self):
        ds = _l2a_ds(
            filters=[{"asset": "l2a", "dataset": "/{group}/quality_flag", "op": "eq", "value": 1}]
        )
        with pytest.raises(ValueError, match="no open sibling handle"):
            _read_group(_FakeH5(_l1b_arrays()), "BEAM0000", ds, 0, _OneShardGrid())


# ── config validation of the vlen grammar ────────────────────────────────────


def _cfg(ds):
    return PipelineConfig(
        data_source={"reader": "h5coro", "groups": ["BEAM0000"], **ds},
        aggregation={
            "coordinates": {"morton": {"dtype": "uint64", "fill_value": 0}},
            "variables": {"count": {"function": "len", "source": "rxwaveform"}},
        },
        output={"grid": {"type": "healpix", "parent_order": 9, "child_order": 18}},
    )


class TestVlenConfigValidation:
    def test_valid_vlen_config_passes(self):
        validate_config(_cfg(_l2a_ds()))

    def test_coordinates_level_must_exist(self):
        ds = _vlen_ds()
        ds["coordinates"]["level"] = "nope"
        with pytest.raises(ValueError, match="not a key in levels"):
            validate_config(_cfg(ds))

    def test_coordinates_level_must_link_to_base(self):
        ds = _vlen_ds()
        ds["coordinates"]["level"] = "samples"
        with pytest.raises(ValueError, match="must link directly to base level"):
            validate_config(_cfg(ds))

    def test_synthesize_requires_vlen_route(self):
        ds = _vlen_ds()
        del ds["coordinates"]["level"]
        with pytest.raises(ValueError, match="synthesize requires the vlen route"):
            validate_config(_cfg(ds))

    def test_synthesize_level_must_match(self):
        ds = _vlen_ds()
        ds["variables"]["elevation"]["level"] = "samples"
        with pytest.raises(ValueError, match="must equal"):
            validate_config(_cfg(ds))

    def test_synthesize_unknown_kind_rejected(self):
        ds = _vlen_ds()
        ds["variables"]["elevation"]["synthesize"] = "logspace"
        with pytest.raises(ValueError, match="must be 'linspace'"):
            validate_config(_cfg(ds))

    def test_asset_filter_requires_declared_asset(self):
        ds = _vlen_ds(
            filters=[{"asset": "l2a", "dataset": "/{group}/quality_flag", "op": "eq", "value": 1}]
        )
        with pytest.raises(ValueError, match="not declared in data_source.assets"):
            validate_config(_cfg(ds))

    def test_asset_filter_rejects_level(self):
        ds = _l2a_ds(
            filters=[
                {
                    "asset": "l2a",
                    "level": "shots",
                    "dataset": "/{group}/quality_flag",
                    "op": "eq",
                    "value": 1,
                }
            ]
        )
        with pytest.raises(ValueError, match="record-level by construction"):
            validate_config(_cfg(ds))

    def test_assets_join_requires_paths(self):
        ds = _vlen_ds(assets={"l2a": {"join": {"left": "/{group}/shot_number"}}})
        with pytest.raises(ValueError, match="join.right must be a non-empty string"):
            validate_config(_cfg(ds))


# ── paired-asset plumbing: runner entry resolution + worker sibling handles ──


class TestResolveGranuleEntries:
    def _records(self):
        return [
            {"id": "a", "s3": "s3://b/a.h5", "https": "https://h/a.h5"},
            {
                "id": "b",
                "s3": "s3://b/b.h5",
                "https": "https://h/b.h5",
                "assets": {"l2a": {"id": "sib", "s3": "s3://b/b2.h5", "https": "https://h/b2.h5"}},
            },
            {"id": "c", "s3": None, "https": "https://h/c.h5"},
        ]

    def test_single_asset_records_stay_plain_strings(self):
        from zagg.runner import _resolve_granule_entries, _resolve_urls

        entries = _resolve_granule_entries(self._records()[:1], "s3")
        assert entries == ["s3://b/a.h5"]
        assert entries == _resolve_urls(self._records()[:1], "s3")

    def test_paired_record_resolves_to_entry_mapping(self):
        from zagg.runner import _resolve_granule_entries

        entries = _resolve_granule_entries(self._records(), "s3")
        # href-less primary (c) dropped, same rule as _resolve_urls.
        assert entries == [
            "s3://b/a.h5",
            {"url": "s3://b/b.h5", "assets": {"l2a": "s3://b/b2.h5"}},
        ]

    def test_driver_selects_sibling_endpoint(self):
        from zagg.runner import _resolve_granule_entries

        entries = _resolve_granule_entries(self._records(), "https")
        assert entries[1] == {"url": "https://h/b.h5", "assets": {"l2a": "https://h/b2.h5"}}

    def test_sibling_without_driver_endpoint_omitted(self):
        from zagg.runner import _resolve_granule_entries

        rec = {
            "id": "b",
            "s3": "s3://b/b.h5",
            "https": None,
            "assets": {"l2a": {"id": "sib", "s3": None, "https": "https://h/b2.h5"}},
        }
        # The sibling has no s3 href: the asset is omitted (the read then
        # raises on the missing handle rather than reading unfiltered data).
        assert _resolve_granule_entries([rec], "s3") == ["s3://b/b.h5"]

    def test_raster_style_string_assets_stay_plain(self):
        from zagg.runner import _resolve_granule_entries

        rec = {
            "id": "r",
            "s3": "s3://b/r.h5",
            "https": None,
            "assets": {"red": "s3://b/red.tif"},
        }
        assert _resolve_granule_entries([rec], "s3") == ["s3://b/r.h5"]

    def test_count_matches_resolve_urls(self):
        from zagg.runner import _resolve_granule_entries, _resolve_urls

        for driver in ("s3", "https"):
            assert len(_resolve_granule_entries(self._records(), driver)) == len(
                _resolve_urls(self._records(), driver)
            )


class TestWorkerPairedEntries:
    """process_shard opens sibling handles for dict-form granule entries and
    threads them to the read seam presence-gated (issue #425)."""

    def _run(self, monkeypatch, entries):
        from zagg.config import default_config
        from zagg.grids import HealpixGrid
        from zagg.index.hierarchical import HierarchicalIndex
        from zagg.processing import process_shard

        opened, closed, captured = [], [], []

        class _RecH5:
            def __init__(self, resource, driver, **kwargs):
                self.resource = resource
                opened.append(resource)

            def close(self):
                closed.append(self.resource)

        def fake_read_group(h5obj, group, ds, shard_key, grid, **kwargs):
            captured.append(dict(kwargs))
            return None

        monkeypatch.setattr("zagg.processing._read_group", fake_read_group)
        monkeypatch.setattr("zagg.processing.h5coro.H5Coro", _RecH5)
        monkeypatch.setattr(
            "zagg.processing.worker.index_from_config", lambda cfg: HierarchicalIndex()
        )
        from mortie import geo2mort

        cfg = default_config()
        grid = HealpixGrid(6, 8, layout="fullsphere", config=cfg)
        shard_key = int(geo2mort(-78.5, -132.0, order=6)[0])
        process_shard(grid, shard_key, entries, s3_credentials={}, config=cfg)
        return opened, closed, captured

    def test_dict_entry_opens_and_closes_siblings(self, monkeypatch):
        opened, closed, captured = self._run(
            monkeypatch, [{"url": "s3://b/l1b.h5", "assets": {"l2a": "s3://b/l2a.h5"}}]
        )
        assert opened == ["b/l1b.h5", "b/l2a.h5"]  # s3:// scheme stripped for S3Driver
        assert sorted(closed) == sorted(opened)
        assert captured, "read seam should have been called"
        for kwargs in captured:
            assert set(kwargs["siblings"]) == {"l2a"}
            assert kwargs["siblings"]["l2a"].resource == "b/l2a.h5"

    def test_string_entries_pass_no_siblings(self, monkeypatch):
        opened, closed, captured = self._run(monkeypatch, ["s3://b/plain.h5"])
        assert opened == ["b/plain.h5"]
        for kwargs in captured:
            assert "siblings" not in kwargs

    def test_entry_url_helper(self):
        from zagg.processing.worker import _entry_url

        assert _entry_url("s3://b/x.h5") == "s3://b/x.h5"
        assert _entry_url({"url": "s3://b/x.h5", "assets": {}}) == "s3://b/x.h5"


class TestInlineWriteBackPrebuild:
    """``index: inline`` with ``write_back`` prebuilds the group's chunk maps
    before the read; both new grammar forms broke that helper, so the
    combination the template advertises raised before a byte was read (review
    finding, PR #432)."""

    def _built_paths(self, monkeypatch, data_source, group="BEAM0000"):
        import zagg.index.inline as inline_mod
        from zagg.index.inline import InlineIndex

        arrays = _l1b_arrays(group)
        built = []

        def _spy(h5obj, path):
            # Missing datasets raise KeyError, as a real chunk-map walk does:
            # anything this granule does not hold must not be in the set.
            arrays[path]
            built.append(path)
            return object()

        monkeypatch.setattr(inline_mod, "build_chunk_map", _spy)
        h5obj = _FakeH5(arrays)
        h5obj.resource = "s3://bucket/GEDI01_B_granule.h5"  # the write-back key
        idx = InlineIndex(write_back=True, store="s3://bucket/manifests")
        idx._prebuild_group_maps(h5obj, group, data_source)
        return built

    def test_template_prebuild_covers_the_read_set(self, monkeypatch):
        from zagg.config import default_config

        cfg = default_config("gedi01b_waveform_healpix_hive")
        built = self._built_paths(monkeypatch, cfg.data_source)
        assert "/BEAM0000/rxwaveform" in built  # the base-rate variable
        assert "/BEAM0000/rx_sample_start_index" in built  # the record link
        assert "/BEAM0000/rx_sample_count" in built
        assert "/BEAM0000/geolocation/latitude_bin0" in built  # record coordinates
        assert "/BEAM0000/geolocation/longitude_bin0" in built
        # The synthesized column is computed, not read — its record-rate
        # endpoints are what the read actually touches.
        assert "/BEAM0000/geolocation/elevation_bin0" in built
        assert "/BEAM0000/geolocation/elevation_lastbin" in built
        assert len(built) == len(set(built))  # one map per dataset

    def test_coordinates_level_is_not_a_dataset(self, monkeypatch):
        from zagg.config import default_config

        cfg = default_config("gedi01b_waveform_healpix_hive")
        assert "shots" not in self._built_paths(monkeypatch, cfg.data_source)

    def test_sibling_asset_filters_are_not_mapped(self, monkeypatch):
        from zagg.config import default_config

        cfg = default_config("gedi01b_waveform_healpix_hive")
        built = self._built_paths(monkeypatch, cfg.data_source)
        # The L2A predicates read the paired granule; this backend never opens
        # it, so mapping them here would KeyError on the primary.
        assert not [p for p in built if "quality_flag" in p or "rx_assess" in p]

    def test_vlen_prebuild_runs_without_a_read_plan(self, monkeypatch):
        # No read_plan: the record level still has to contribute its link, or
        # the full-read arm's link datasets go unmapped.
        built = self._built_paths(monkeypatch, _vlen_ds())
        assert "/BEAM0000/rx_sample_start_index" in built
        assert "/BEAM0000/rx_sample_count" in built


# ── the packaged gedi01b template ────────────────────────────────────────────


class TestGediTemplate:
    """The shipped gedi01b_waveform_healpix_hive template: validated shape,
    and the whole declared surface driven over the synthetic mini-granule —
    read (gather/expand/synthesis + degrade + the L2A trio) into cell stats
    (flux digest + companions)."""

    @pytest.fixture
    def cfg(self):
        from zagg.config import default_config

        return default_config("gedi01b_waveform_healpix_hive")

    def test_ratified_shape(self, cfg):
        grid = cfg.output["grid"]
        assert (grid["parent_order"], grid["child_order"], grid["chunk_inner"]) == (9, 18, 12)
        assert grid["sharded"] is True
        assert cfg.output["store_layout"] == "hive"
        assert cfg.output["pyramid"] is False
        assert "streaming" not in cfg.aggregation  # pooled; composability none
        flux = cfg.aggregation["variables"]["rx_flux"]
        assert flux["params"]["delta"] == 8192
        assert "operating_point" in flux["attrs"]  # clip provenance
        assert flux["attrs"]["gain_name"]
        assert len(cfg.data_source["groups"]) == 8

    def test_template_reads_and_aggregates_the_fixture(self, cfg):
        from zagg.processing import calculate_cell_statistics

        df = _read_group(
            _FakeH5(_l1b_arrays()),
            "BEAM0000",
            cfg.data_source,
            0,
            _OneShardGrid(),
            siblings={"l2a": _FakeH5(_l2a_arrays())},
        )
        # Survivors: shot 101 only — 103 fails degrade, 102 fails quality_flag,
        # 104 fails sensitivity (0.50 < 0.9), 103 also has no L2A row.
        assert df["shot_number"].tolist() == [101] * 5
        np.testing.assert_array_equal(df["rxwaveform"].to_numpy(), WAVE[:5])
        np.testing.assert_allclose(df["elevation"].to_numpy(), _expected_elevation(0))

        cell = {c: df[c].to_numpy() for c in df.columns if c != "leaf_id"}
        stats = calculate_cell_statistics(cell, config=cfg)
        assert stats["count"] == 5
        assert stats["shot_count"] == 1
        assert stats["shot_number"] == 101
        assert stats["noise_mean"] == pytest.approx(NOISE_MEAN[0])
        assert stats["rx_energy"] == pytest.approx(RX_ENERGY[0])
        assert stats["elevation_bin0"] == pytest.approx(E0[0])
        assert stats["elevation_lastbin"] == pytest.approx(E1[0])
        # Flux digest: WAVE values 100-104 sit far above the noise floor, so
        # all five samples survive the clip loss-free (weight = count - mean).
        flux = stats["rx_flux"]
        assert flux.shape == (5, 2)
        np.testing.assert_allclose(
            np.sort(flux[:, 1]), np.sort((WAVE[:5] - NOISE_MEAN[0]).astype(np.float32))
        )


# ── strided records (the GEDI L1B shape — issue #452) ────────────────────────
#
# The packed fixture above is the pathological case that hid the bug: its
# records tile ``rxwaveform`` end to end, so ``Σcount`` happens to equal the
# array length. Real GEDI L1B is STRIDED — ``rx_sample_start_index`` steps by a
# fixed 1,420-sample window per shot while ``rx_sample_count`` is the valid
# sample count — and the slack between a record's valid samples and the next
# window is ZERO-FILLED in the product. Same four shots, same counts, starts
# strided by 6, and 2 samples of tail padding past the last record:
#
#   row   0 1 2 3 4 | 5 | 6 7 8 | 9 10 11 | 12 | 13..17 | 18 19 20 21 | 22 23
#   shot  101 ..... | . | 102 . | ....... | 103| ...... | 104 ....... | .....
#
# ``Σcount`` is 13 but the link's extent is 22: under the old derivation
# ``plan_read`` clamped every run to ``base_end <= base_start`` and the group
# returned zero rows with no error (the 0.45.0 fleet smoke).

STARTS_STRIDED = np.array([1, 7, 13, 19], dtype=np.uint64)  # origin-1, stride 6
WAVE_STRIDED = np.zeros(24, dtype=np.float32)  # slack is zero-filled, as in GEDI
WAVE_STRIDED[0:5] = WAVE[0:5]
WAVE_STRIDED[6:9] = WAVE[5:8]
WAVE_STRIDED[12] = WAVE[8]
WAVE_STRIDED[18:22] = WAVE[9:13]
STRIDED_EXTENT = 22
SLACK_ROWS = [5, 9, 10, 11, 13, 14, 15, 16, 17, 22, 23]


def _l1b_arrays_strided(group="BEAM0000"):
    arrays = _l1b_arrays(group)
    arrays[f"/{group}/rxwaveform"] = WAVE_STRIDED
    arrays[f"/{group}/rx_sample_start_index"] = STARTS_STRIDED
    return arrays


class _LastShotGrid:
    """Grid stub: shard 1 for lat >= 12.5 (the last shot only), else 0."""

    @staticmethod
    def assign(lats, lons):
        return (np.asarray(lats) >= 12.5).astype(np.uint64)

    @staticmethod
    def shards_of(leaf_ids):
        return np.asarray(leaf_ids).astype(int)


class _FirstShotGrid:
    """Grid stub: shard 1 for lat <= 10.5 (the first shot only), else 0."""

    @staticmethod
    def assign(lats, lons):
        return (np.asarray(lats) <= 10.5).astype(np.uint64)

    @staticmethod
    def shards_of(leaf_ids):
        return np.asarray(leaf_ids).astype(int)


class TestStridedRecords:
    """A strided twin of the packed fixture: same rows out, none from the slack."""

    def test_extent_exceeds_sum_of_counts(self):
        assert link_base_extent(STARTS_STRIDED, COUNTS, 1) == STRIDED_EXTENT
        assert int(COUNTS.sum()) == 13 < STRIDED_EXTENT

    def test_full_read_row_parity_with_the_packed_fixture(self):
        strided = _read_group(
            _FakeH5(_l1b_arrays_strided()), "BEAM0000", _vlen_ds(), 0, _OneShardGrid()
        )
        packed = _read_group(_FakeH5(_l1b_arrays()), "BEAM0000", _vlen_ds(), 0, _OneShardGrid())
        assert len(strided) == len(packed) == 13
        for col in ("rxwaveform", "shot_number", "noise_mean", "elevation"):
            np.testing.assert_allclose(strided[col].to_numpy(), packed[col].to_numpy())

    def test_no_slack_row_leaks_into_the_output(self):
        df = _read_group(_FakeH5(_l1b_arrays_strided()), "BEAM0000", _vlen_ds(), 0, _OneShardGrid())
        # Every returned sample comes from a record's valid range; the slack is
        # zero-filled in the product, so a leaked row would show up as a real
        # zero-amplitude sample (and inflate the row count).
        assert (df["rxwaveform"].to_numpy() != 0.0).all()
        np.testing.assert_array_equal(
            df["rxwaveform"].to_numpy(), WAVE_STRIDED[[r for r in range(24) if r not in SLACK_ROWS]]
        )

    def test_planned_read_selects_shard_records(self):
        # Shard 1 holds shots 103 + 104 (lat >= 11.5) -> one run -> base rows
        # 12..21: 5 real samples and 5 slack, the ~1.9x read amplification
        # striding costs. Only the real ones come back.
        ds = _vlen_ds(read_plan={"spatial_index": "shots", "pad": 0})
        io_stats: dict = {}
        df = _read_group(
            _FakeH5(_l1b_arrays_strided()), "BEAM0000", ds, 1, _LatGrid(), io_stats=io_stats
        )
        assert io_stats["obs_read"] == 10  # rows DECODED, slack included
        assert len(df) == 5
        assert df["shot_number"].tolist() == [103] + [104] * 4
        np.testing.assert_array_equal(df["rxwaveform"].to_numpy(), WAVE[8:13])

    def test_planned_matches_full(self):
        ds_plan = _vlen_ds(read_plan={"spatial_index": "shots", "pad": 1})
        for shard in (0, 1):
            a = _read_group(_FakeH5(_l1b_arrays_strided()), "BEAM0000", ds_plan, shard, _LatGrid())
            b = _read_group(
                _FakeH5(_l1b_arrays_strided()), "BEAM0000", _vlen_ds(), shard, _LatGrid()
            )
            for col in ("rxwaveform", "shot_number", "noise_mean", "elevation"):
                np.testing.assert_allclose(a[col].to_numpy(), b[col].to_numpy())

    def test_sum_of_counts_extent_returns_the_silent_zero(self, monkeypatch):
        # The bug, pinned: with the extent derived as Σcount (13), the last
        # shot's run starts past the phantom end (row 18), ``plan_read`` clamps
        # it to ``base_end <= base_start``, and the group returns no rows and no
        # error -- the 0.45.0 fleet smoke, one shard at a time.
        import zagg.processing.read_vlen as rv

        ds = _vlen_ds(read_plan={"spatial_index": "shots", "pad": 0})
        h5, grid = _FakeH5(_l1b_arrays_strided()), _LastShotGrid()
        io_stats: dict = {}
        monkeypatch.setattr(rv, "link_base_extent", lambda ibeg, cnt, base: int(np.sum(cnt)))
        assert _read_group(h5, "BEAM0000", ds, 1, grid, io_stats=io_stats) is None
        assert io_stats["obs_read"] == 0
        monkeypatch.undo()
        df = _read_group(_FakeH5(_l1b_arrays_strided()), "BEAM0000", ds, 1, grid)
        assert df["shot_number"].tolist() == [104] * 4

    def test_record_level_filter_expands_at_planned_rate(self):
        # degrade == 0 at the shots level drops shot 103's single sample; the
        # cross-level expansion now gathers per planned row (searchsorted over
        # the level's link) instead of painting the base extent.
        ds = _vlen_ds(
            read_plan={"spatial_index": "shots", "pad": 0},
            filters=[
                {
                    "level": "shots",
                    "dataset": "/{group}/geolocation/degrade",
                    "op": "eq",
                    "value": 0,
                }
            ],
        )
        df = _read_group(_FakeH5(_l1b_arrays_strided()), "BEAM0000", ds, 1, _LatGrid())
        assert df["shot_number"].tolist() == [104] * 4
        np.testing.assert_array_equal(df["rxwaveform"].to_numpy(), WAVE[9:13])

    def test_sibling_join_expands_at_planned_rate(self):
        ds = _vlen_ds(
            read_plan={"spatial_index": "shots", "pad": 0},
            assets={
                "l2a": {"join": {"left": "/{group}/shot_number", "right": "/{group}/shot_number"}}
            },
            filters=[{"asset": "l2a", "dataset": "/{group}/quality_flag", "op": "eq", "value": 1}],
        )
        df = _read_group(
            _FakeH5(_l1b_arrays_strided()),
            "BEAM0000",
            ds,
            1,
            _LatGrid(),
            siblings={"l2a": _FakeH5(_l2a_arrays())},
        )
        # Shot 103 has no L2A row (dropped); 104 passes the flag.
        assert df["shot_number"].tolist() == [104] * 4

    def test_slack_rows_nan_under_linspace_synthesis(self):
        from zagg.processing.read_vlen import _planned_gather_map

        gidx, parent, within = _planned_gather_map(
            STARTS_STRIDED, COUNTS, 1, STRIDED_EXTENT, [(0, STRIDED_EXTENT)]
        )
        vals = synthesize_linspace(E0, E1, COUNTS, parent, within)
        slack = [r for r in SLACK_ROWS if r < STRIDED_EXTENT]
        assert np.isnan(vals[slack]).all()
        assert not np.isnan(vals[[0, 4, 6, 8, 12, 18, 21]]).any()
        np.testing.assert_allclose(vals[0:5], _expected_elevation(0))
        np.testing.assert_allclose(vals[18:22], _expected_elevation(3))
        assert gidx.tolist() == list(range(STRIDED_EXTENT))

    def test_gather_map_marks_every_slack_row_unassigned(self):
        parent, _ = expand_link_indices(STARTS_STRIDED, COUNTS, 1, STRIDED_EXTENT)
        assert [int(r) for r in np.flatnonzero(parent < 0)] == [
            r for r in SLACK_ROWS if r < STRIDED_EXTENT
        ]

    def test_dataset_shorter_than_the_link_extent_raises(self):
        arrays = _l1b_arrays_strided()
        arrays["/BEAM0000/rxwaveform"] = WAVE_STRIDED[:20]  # link reaches row 21
        with pytest.raises(ValueError, match="does not tile the declared base extent"):
            _read_group(_FakeH5(arrays), "BEAM0000", _vlen_ds(), 0, _OneShardGrid())


# ── memory posture: nothing at base rate on a planned vlen read (#452) ───────


class _RampBase:
    """Stand-in for a huge flat base dataset: value == row index, never
    materialized. Only the plan's hyperslices are ever sliced out of it, which
    is the property under test — a route that touched it whole would have to
    allocate ``n_base`` and would trip the spy below."""

    def __init__(self, n):
        self._n = n

    def __len__(self):
        return self._n

    def __getitem__(self, key):
        start, stop, step = key.indices(self._n)
        return np.arange(start, stop, step, dtype=np.float32) + 1.0


class _AllocSpy:
    """``numpy`` proxy recording the length of every array a module allocates.

    Installed over the module-global ``np`` of ``read`` / ``read_vlen`` so only
    those two modules' own allocations are seen (pandas, the grid stub and the
    fixture keep the real numpy)."""

    _TRACKED = ("zeros", "ones", "empty", "full", "arange", "repeat", "concatenate")

    def __init__(self, sizes: list[int]):
        self._sizes = sizes

    def __getattr__(self, name):
        attr = getattr(np, name)
        if name not in self._TRACKED:
            return attr

        def _wrapped(*args, **kwargs):
            out = attr(*args, **kwargs)
            self._sizes.append(int(np.size(out)))
            return out

        return _wrapped


class TestVlenMemoryPosture:
    """A planned vlen read allocates at PLANNED rate, never at base rate.

    The companion half of the fix: with a correct extent but base-rate
    expansion, one GEDI beam group builds ~7 GB of per-shot broadcasts against
    a 2 GB worker. The fixture strides the same four shots across 3,000,004
    base rows while the shard touches five of them, so any surviving base-rate
    allocation is six orders of magnitude larger than the plan and impossible
    to miss."""

    STRIDE = 1_000_000
    N_BASE = 3 * STRIDE + 4

    def _arrays(self, group="BEAM0000"):
        arrays = _l1b_arrays(group)
        arrays[f"/{group}/rx_sample_start_index"] = np.array(
            [1, self.STRIDE + 1, 2 * self.STRIDE + 1, 3 * self.STRIDE + 1], dtype=np.uint64
        )
        arrays[f"/{group}/rxwaveform"] = _RampBase(self.N_BASE)
        return arrays

    def _data_source(self):
        # All three expansion paths at once: a cross-level (record) filter, a
        # sibling-asset join, and the segment-level variable broadcasts the
        # shared fixture already declares (shot_number, noise_mean).
        return _vlen_ds(
            read_plan={"spatial_index": "shots", "pad": 0},
            assets={
                "l2a": {"join": {"left": "/{group}/shot_number", "right": "/{group}/shot_number"}}
            },
            filters=[
                {
                    "level": "shots",
                    "dataset": "/{group}/geolocation/degrade",
                    "op": "eq",
                    "value": 0,
                },
                {"asset": "l2a", "dataset": "/{group}/quality_flag", "op": "eq", "value": 1},
            ],
        )

    def test_planned_read_allocates_nothing_at_base_rate(self, monkeypatch):
        import zagg.processing.read as rd
        import zagg.processing.read_vlen as rv

        arrays = self._arrays()
        assert link_base_extent(arrays["/BEAM0000/rx_sample_start_index"], COUNTS, 1) == self.N_BASE

        sizes: list[int] = []
        spy = _AllocSpy(sizes)
        monkeypatch.setattr(rd, "np", spy)
        monkeypatch.setattr(rv, "np", spy)
        df = _read_group(
            _FakeH5(arrays),
            "BEAM0000",
            self._data_source(),
            1,
            _FirstShotGrid(),
            siblings={"l2a": _FakeH5(_l2a_arrays())},
        )
        monkeypatch.undo()

        # The shard holds shot 101 only: 5 planned rows, all kept.
        assert df["shot_number"].tolist() == [101] * 5
        np.testing.assert_array_equal(df["rxwaveform"].to_numpy(), np.arange(1.0, 6.0))
        assert sizes, "the allocation spy saw nothing — is it still wired to the read modules?"
        assert max(sizes) <= 64, f"a base-rate allocation survived: {sorted(sizes)[-3:]}"

    def test_full_read_keeps_the_base_rate_forms(self, monkeypatch):
        """The FULL arm must not borrow the planned arm's at-rows machinery.

        The at-rows helpers exist to avoid a length-``n_base`` intermediate. On
        the full arm there is nothing to avoid — every base row is decoded — so
        routing it through them buys nothing and costs a lot: it first has to
        materialize the identity ``arange(n_base)`` (8 B/row) and then pays
        33 B/row of searchsorted temporaries per companion where the paint pays
        1-4 B/row. Measured on 2,000 records x count 700 over a 1,420 stride
        (``n_base`` 19,998,560): ``_broadcast_segment_to_base`` 80.0 MB vs
        ``_gather_segment_at_rows`` 660.7 MB, ``_expand_mask_to_base`` 20.0 MB
        vs ``_expand_mask_at_rows`` 660.6 MB. So this pins the dispatch, not a
        byte count: on the full arm no row vector is built and no owner map is
        re-derived by searchsorted."""
        import zagg.processing.read as rd
        import zagg.processing.read_vlen as rv

        arrays = self._arrays()
        # A real flat array this time: the full arm decodes the whole dataset.
        arrays["/BEAM0000/rxwaveform"] = np.arange(1.0, self.N_BASE + 1.0, dtype=np.float32)

        at_rows_calls: list[str] = []
        for name in ("_link_parent_at_rows", "_expand_mask_at_rows", "_gather_segment_at_rows"):
            real = getattr(rd, name)

            def _spy(*a, _n=name, _f=real, **k):
                at_rows_calls.append(_n)
                return _f(*a, **k)

            monkeypatch.setattr(rd, name, _spy)
            if hasattr(rv, name):
                monkeypatch.setattr(rv, name, _spy)

        sizes: list[int] = []
        monkeypatch.setattr(rd, "np", _AllocSpy(sizes))
        monkeypatch.setattr(rv, "np", _AllocSpy(sizes))
        ds = self._data_source()
        del ds["read_plan"]  # the full arm
        df = _read_group(
            _FakeH5(arrays),
            "BEAM0000",
            ds,
            1,
            _FirstShotGrid(),
            siblings={"l2a": _FakeH5(_l2a_arrays())},
        )
        monkeypatch.undo()

        assert df["shot_number"].tolist() == [101] * 5
        np.testing.assert_array_equal(df["rxwaveform"].to_numpy(), np.arange(1.0, 6.0))
        assert not at_rows_calls, f"the full arm used the at-rows helpers: {at_rows_calls}"
        # No row vector: the identity map is never materialized. The full arm's
        # legitimate base-rate allocations (the gather map's two arrays, the
        # per-companion broadcasts) are all <= n_base; nothing may exceed it.
        assert sizes, "the allocation spy saw nothing — is it still wired to the read modules?"
        assert max(sizes) <= self.N_BASE, f"an over-base-rate allocation: {sorted(sizes)[-3:]}"
        assert sum(1 for s in sizes if s >= self.N_BASE) <= 6, (
            f"more base-rate allocations than the base-rate forms need: {sorted(sizes)[-8:]}"
        )
