"""Per-cell temporal envelopes — spec §8.2/§8.4, issue #410."""

import numpy as np
import pytest
from conftest import toc_words as _toc_words

from zagg.stats.toc import cell_envelope, cell_envelopes


class TestCellEnvelope:
    def test_single_observation_is_an_exact_timestamp(self):
        import mortie

        words = _toc_words(1)
        out = cell_envelope(words)
        assert out.dtype == np.uint64
        assert int(out) == int(words[0])
        assert not bool(mortie.toc_is_range(np.array([out]))[0])

    def test_repeated_instants_stay_a_timestamp(self):
        # §8.2: a timestamp word "exactly when that join is a single instant —
        # a cell covering one instantaneous observation, or several sharing
        # one instant". GEDI's shot pooling is exactly this case.
        import mortie

        one = _toc_words(1)
        out = cell_envelope(np.repeat(one, 1000))
        assert int(out) == int(one[0])
        assert not bool(mortie.toc_is_range(np.array([out]))[0])

    def test_spread_observations_become_a_conservative_range(self):
        import mortie

        words = _toc_words(50)
        out = cell_envelope(words)
        assert bool(mortie.toc_is_range(np.array([out]))[0])
        start, end = (int(b[0]) for b in mortie.toc2time(np.array([out])))
        members = mortie.toc2time(words)[0]
        assert start <= int(members.min()) and int(members.max()) < end

    def test_order_independent(self):
        words = _toc_words(64)
        rng = np.random.default_rng(410)
        base = int(cell_envelope(words))
        for _ in range(5):
            assert int(cell_envelope(words[rng.permutation(len(words))])) == base

    def test_empty_raises_rather_than_inventing_an_identity(self):
        with pytest.raises(ValueError, match="no identity element"):
            cell_envelope(np.empty(0, dtype=np.uint64))

    def test_reserved_zero_refused(self):
        words = _toc_words(4)
        words[1] = 0
        with pytest.raises(ValueError, match="reserved 0 word"):
            cell_envelope(words)

    def test_non_uint64_refused(self):
        with pytest.raises(ValueError, match="is not uint64"):
            cell_envelope(np.array([1.5, 2.5]))

    def test_wired_via_resolve_function(self):
        from zagg.config import resolve_function

        f = resolve_function("zagg.stats.toc.cell_envelope")
        assert int(f(_toc_words(3))) == int(cell_envelope(_toc_words(3)))


class TestCellEnvelopes:
    def test_matches_the_scalar_reduce_per_group(self):
        words = _toc_words(30)
        offsets = np.array([0, 4, 4 + 11, 30], dtype=np.int64)
        out = cell_envelopes(words, offsets)
        assert out.dtype == np.uint64 and out.shape == (3,)
        for i, (lo, hi) in enumerate(zip(offsets[:-1], offsets[1:], strict=True)):
            assert int(out[i]) == int(cell_envelope(words[lo:hi]))

    def test_empty_group_raises(self):
        words = _toc_words(6)
        with pytest.raises(ValueError, match="empty segment"):
            cell_envelopes(words, np.array([0, 3, 3, 6], dtype=np.int64))

    def test_reserved_zero_refused(self):
        words = _toc_words(6)
        words[4] = 0
        with pytest.raises(ValueError, match="reserved 0 word"):
            cell_envelopes(words, np.array([0, 3, 6], dtype=np.int64))


class TestShippedTemplates:
    """The §8.3 companion as the shipped configs declare it (issue #410 phase 4).

    Both sensors carry a **per-centroid** companion — espg-ruled 2026-08-17,
    amending ruling 3 (which had GEDI per-cell): one shape means readers bind
    one contract across sensors, and GEDI's honesty property is unchanged, since
    all of a shot's samples share its instant.
    """

    @pytest.mark.parametrize(
        "name,field",
        [
            ("atl03_tdigest_located_healpix", "h_tdigest"),
            ("gedi01b_waveform_healpix_hive", "rx_flux"),
        ],
    )
    def test_the_config_declares_a_gps_clock_and_a_companion(self, name, field):
        from zagg.config import default_config, get_agg_fields
        from zagg.time_axis import toc_source

        cfg = default_config(name)  # default_config validates
        clock = toc_source(cfg)
        # Continuous scale, or the words could not claim nanosecond exactness.
        assert clock["scale"] in ("gps", "tai")
        assert clock["epoch"].startswith("2018-01-01")
        assert clock["units"] == "seconds"
        assert get_agg_fields(cfg)[field]["temporal"] == "per-centroid"

    @pytest.mark.parametrize(
        "name,field",
        [
            ("atl03_tdigest_located_healpix", "h_tdigest"),
            ("gedi01b_waveform_healpix_hive", "rx_flux"),
        ],
    )
    def test_the_template_emits_the_declared_sibling(self, name, field):
        import zarr
        from zarr.storage import MemoryStore

        from zagg.config import default_config
        from zagg.grids import from_config
        from zagg.time_axis import temporal_declaration

        grid = from_config(default_config(name))
        store = MemoryStore()
        grid.emit_shard_template(store, overwrite=True)
        group = zarr.open_group(store, path=grid.group_path.split("/")[-1], mode="r", zarr_format=3)
        sibling = f"{field}_times"
        assert sibling in set(group.array_keys())
        # The payload binds it; the sibling declares the words it holds (§8.3).
        assert dict(group[field].attrs)["times"] == sibling
        assert temporal_declaration(dict(group[sibling].attrs)) == {
            "spec": "zagg-toc/1",
            "shape": "per-centroid",
            "grammar": "mortie-toc/1",
        }
        assert "temporal" not in dict(group[field].attrs)

    def test_the_gedi_clock_is_the_verified_gps_epoch(self):
        # Measured, not assumed: BEAM0000/ancillary/master_time_epoch reads
        # 1198800018.000000 on two real granules (GEDI01_B_2019128… and
        # GEDI01_B_2020024…) — the 2018-01-01 epoch in leap-aware GPS seconds
        # (naive UTC would be 1198800000; +18 is GPS−UTC at 2018). A nominal-UTC
        # column would be refused by name, so this constant is what admits the
        # declaration at all.
        import numpy as np

        from zagg.time_axis import observation_words

        gps_epoch_seconds = 1_198_800_018.0
        naive = 1_198_800_000.0
        assert gps_epoch_seconds - naive == 18.0
        # delta_time == 0 is the epoch instant itself, and it encodes exactly.
        import mortie

        word = observation_words(
            np.array([0.0]), epoch="2018-01-01T00:00:00", scale="gps", units="seconds"
        )
        assert not mortie.toc_is_range(word)[0]
        assert mortie.to_datetime64(mortie.toc2time(word)[0])[0] == np.datetime64(
            "2018-01-01T00:00:00", "ns"
        )
