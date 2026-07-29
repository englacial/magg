"""Tests for the packed signal-composition field and stratified reducer (issue #321)."""

import numpy as np
import pytest

from zagg.stats.composition import (
    LANES,
    counts_from_composition,
    merge_composition,
    pack_composition,
    unpack_composition,
)
from zagg.stats.tdigest import build_tdigest, build_tdigest_where


def _conf_kwargs(conf):
    """Split an (n, 5) conf matrix into the reducer's five column kwargs."""
    cols = ("conf_land", "conf_ocean", "conf_sea_ice", "conf_land_ice", "conf_inland_water")
    return {name: conf[:, i] for i, name in enumerate(cols)}


class TestPackComposition:
    def test_single_photon_lanes_are_exact_flags(self):
        # count == 1: the lanes ARE the photon's flags (fraction in {0, 255}).
        conf = np.array([[4, -1, 0, 3, 1]])
        word = pack_composition(np.array([10.0]), **_conf_kwargs(conf), threshold=2)
        lanes = unpack_composition(word)
        assert lanes.tolist() == [255, 0, 0, 255, 0, 0, 0, 255]  # land, land_ice; strongest=4

    def test_golden_word_pins_lsb_first_byte_order(self):
        # Every other assertion goes through ``unpack_composition``, so a
        # layout flip to MSB-first would keep the whole suite green while
        # breaking the ratified byte order for any independent reader
        # (moczarr). Pin the literal: lane i occupies bits 8i..8i+8, so lanes
        # [255, 0, 0, 255, 0, 0, 0, 255] -> land (b0) | land_ice (b3) | high
        # (b7). MSB-first would give 0xFF0000FF000000FF instead.
        conf = np.array([[4, -1, 0, 3, 1]])
        word = pack_composition(np.array([10.0]), **_conf_kwargs(conf), threshold=2)
        assert word == 0xFF000000FF0000FF

    def test_exact_count_recovery_below_knee(self):
        rng = np.random.default_rng(321)
        n = 200
        conf = rng.integers(-2, 5, size=(n, 5))
        conf[:, 0] = 4  # make every photon signal so N == n
        word = pack_composition(np.zeros(n), **_conf_kwargs(conf), threshold=2)
        counts = counts_from_composition(word, n)
        expected = (conf >= 2).sum(axis=0)
        assert counts[:5].tolist() == expected.tolist()
        strongest = conf.max(axis=1)
        assert counts[5:].tolist() == [(strongest == lv).sum() for lv in (2, 3, 4)]

    def test_presence_floor_keeps_rare_flags_nonzero(self):
        n = 300  # above the exact regime; 1/300 would round to 0 without the floor
        conf = np.full((n, 5), -1)
        conf[:, 0] = 4  # all signal via land
        conf[0, 4] = 2  # one photon also flags inland_water
        word = pack_composition(np.zeros(n), **_conf_kwargs(conf), threshold=2)
        lanes = unpack_composition(word)
        assert lanes[4] == 1  # floored, not rounded away
        assert lanes[0] == 255

    def test_no_signal_packs_zero(self):
        conf = np.full((10, 5), 1)  # buffer everywhere: below the >1 cut
        assert pack_composition(np.zeros(10), **_conf_kwargs(conf), threshold=2) == 0

    def test_nan_heights_align_with_signal_digest_weight(self):
        # N_signal must equal the signal digest's total weight: both drop
        # non-finite heights before counting.
        values = np.array([1.0, np.nan, 2.0, 3.0, np.nan])
        conf = np.full((5, 5), -1)
        conf[:, 3] = 4  # all rows signal (land_ice)
        word = pack_composition(values, **_conf_kwargs(conf), threshold=2)
        signal = (conf >= 2).any(axis=1)
        digest = build_tdigest_where(values, where=signal)
        n_digest = int(digest[:, 1].sum())
        assert n_digest == 3
        assert counts_from_composition(word, n_digest)[3] == 3

    def test_threshold_knob(self):
        conf = np.array([[4, 0, 0, 0, 0], [3, 0, 0, 0, 0], [2, 0, 0, 0, 0]])
        w_high_only = pack_composition(np.zeros(3), **_conf_kwargs(conf), threshold=4)
        lanes = unpack_composition(w_high_only)
        assert lanes[0] == 255 and lanes[7] == 255  # 1 signal photon, high
        # Documented behavior, not a surprise: level lanes are ABSOLUTE
        # (conf == 2/3/4 at every threshold), so raising the threshold empties
        # the lower lanes instead of renumbering them — one lane layout for
        # every product (see pack_composition's docstring).
        assert lanes[5] == 0 and lanes[6] == 0

    def test_row_mismatch_raises(self):
        conf = np.zeros((3, 5))
        with pytest.raises(ValueError, match="rows"):
            pack_composition(np.zeros(4), **_conf_kwargs(conf))


class TestMergeComposition:
    def test_identity_and_passthrough(self):
        conf = np.array([[4, -1, -1, 2, -1]])
        w = pack_composition(np.array([1.0]), **_conf_kwargs(conf))
        assert merge_composition(w, 1, 0, 0) == w
        assert merge_composition(0, 0, w, 1) == w

    def test_merge_matches_pooled_recompute(self):
        rng = np.random.default_rng(87)
        n_a, n_b = 40, 60
        conf = rng.integers(-2, 5, size=(n_a + n_b, 5))
        conf[:, 1] = 4  # everything signal
        ka = _conf_kwargs(conf[:n_a])
        kb = _conf_kwargs(conf[n_a:])
        wa = pack_composition(np.zeros(n_a), **ka)
        wb = pack_composition(np.zeros(n_b), **kb)
        merged = merge_composition(wa, n_a, wb, n_b)
        pooled = pack_composition(np.zeros(n_a + n_b), **_conf_kwargs(conf))
        # Merging re-quantizes already-quantized lanes, so the merged word may
        # sit 1 LSB off the pooled word; the contract is bounded count error
        # (±1 here) and exact presence, not word equality.
        m_counts = counts_from_composition(merged, n_a + n_b)
        p_counts = counts_from_composition(pooled, n_a + n_b)
        assert np.max(np.abs(m_counts - p_counts)) <= 1
        assert np.array_equal(unpack_composition(merged) > 0, unpack_composition(pooled) > 0)

    def test_presence_survives_deep_merges(self):
        # One rare flag in the first block must stay nonzero through a long
        # chain of merges with blocks that never carry it.
        conf0 = np.full((10, 5), -1)
        conf0[:, 0] = 4
        conf0[0, 4] = 3  # rare inland_water flag
        w = pack_composition(np.zeros(10), **_conf_kwargs(conf0))
        n = 10
        confk = np.full((100, 5), -1)
        confk[:, 0] = 4
        wk = pack_composition(np.zeros(100), **_conf_kwargs(confk))
        for _ in range(20):
            w = merge_composition(w, n, wk, 100)
            n += 100
        assert unpack_composition(w)[4] >= 1

    def test_n_is_the_signal_digest_weight(self):
        # The ratified law takes ``n`` from the signal DIGEST's total weight,
        # not from the cell's observation count. Every other merge test here
        # passes an ``n`` it computed itself on a fixture rigged so N == n, so
        # that half of the law was unpinned (review finding). Build two blocks
        # with non-signal rows AND non-finite heights, so N != len(values) on
        # both axes, and source ``n`` the way a reader would.
        rng = np.random.default_rng(321)
        conf = rng.integers(-2, 5, size=(100, 5))
        values = rng.standard_normal(100)
        values[::13] = np.nan  # dropped by BOTH the digest and the packer
        signal = (conf >= 2).any(axis=1)
        blocks = ((slice(0, 40), 40), (slice(40, 100), 60))
        words, weights = [], []
        for sl, size in blocks:
            digest = build_tdigest_where(values[sl], where=signal[sl])
            n = int(digest[:, 1].sum())
            assert 0 < n < size  # both drops actually bite
            words.append(pack_composition(values[sl], **_conf_kwargs(conf[sl]), threshold=2))
            weights.append(n)
        merged = merge_composition(words[0], weights[0], words[1], weights[1])

        n_total = weights[0] + weights[1]
        pooled_digest = build_tdigest_where(values, where=signal)
        assert int(pooled_digest[:, 1].sum()) == n_total  # weights are additive
        pooled = pack_composition(values, **_conf_kwargs(conf), threshold=2)
        m_counts = counts_from_composition(merged, n_total)
        p_counts = counts_from_composition(pooled, n_total)
        assert np.max(np.abs(m_counts - p_counts)) <= 1
        assert np.array_equal(unpack_composition(merged) > 0, unpack_composition(pooled) > 0)
        # Ground the recovered counts in the raw rows, so a wrong ``n`` cannot
        # pass: lanes are fractions of the SIGNAL stratum's finite rows.
        keep = signal & np.isfinite(values)
        assert p_counts[:5].tolist() == (conf[keep] >= 2).sum(axis=0).tolist()
        # The observation count is the wrong divisor and shows it.
        assert not np.array_equal(counts_from_composition(pooled, len(values)), p_counts)

    def test_symmetric(self):
        rng = np.random.default_rng(5)
        conf_a = rng.integers(-2, 5, size=(30, 5))
        conf_b = rng.integers(-2, 5, size=(50, 5))
        conf_a[:, 2] = 4
        conf_b[:, 2] = 4
        wa = pack_composition(np.zeros(30), **_conf_kwargs(conf_a))
        wb = pack_composition(np.zeros(50), **_conf_kwargs(conf_b))
        assert merge_composition(wa, 30, wb, 50) == merge_composition(wb, 50, wa, 30)


class TestBuildTdigestWhere:
    def test_equals_masked_build(self):
        rng = np.random.default_rng(11)
        values = rng.standard_normal(500)
        mask = rng.random(500) > 0.4
        assert np.array_equal(
            build_tdigest_where(values, delta=64, where=mask),
            build_tdigest(values[mask], delta=64),
        )

    def test_disjoint_strata_weights_sum_to_finite_count(self):
        rng = np.random.default_rng(12)
        values = rng.standard_normal(300)
        values[::17] = np.nan
        mask = rng.random(300) > 0.5
        d_sig = build_tdigest_where(values, where=mask)
        d_noise = build_tdigest_where(values, where=~mask)
        total = int(d_sig[:, 1].sum() + d_noise[:, 1].sum())
        assert total == int(np.isfinite(values).sum())

    def test_locations_subset_alongside(self):
        from conftest import point_words

        values = np.arange(20, dtype=np.float64)
        locs = point_words(20, seed=9)
        mask = np.zeros(20, dtype=bool)
        mask[3:9] = True
        digest, out_locs = build_tdigest_where(values, where=mask, locations=locs)
        assert np.array_equal(out_locs, locs[3:9])  # unit weights, sorted input
        assert int(digest[:, 1].sum()) == 6

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="where shape"):
            build_tdigest_where(np.zeros(3), where=np.zeros(4, dtype=bool))
        with pytest.raises(ValueError, match="locations shape"):
            build_tdigest_where(
                np.zeros(3), where=np.ones(3, dtype=bool), locations=np.zeros(4, dtype=np.uint64)
            )

    def test_lanes_constant_documented(self):
        assert LANES == (
            "land",
            "ocean",
            "sea_ice",
            "land_ice",
            "inland_water",
            "low",
            "med",
            "high",
        )
