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
from zagg.processing.read import _read_group
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
