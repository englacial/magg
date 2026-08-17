"""Tests for the sensor-generic waveform flux transform (issue #425).

Round-trip (stored rows + companions reconstruct the clipped waveform within
float32), rx_energy closure (stored mass + the noise-model remainder equals
the record's integrated energy), pooled multi-shot merge over the shared
t-digest fold, and the shot_count/shot_number sentinel companions.
"""

import numpy as np
import pytest

from zagg.stats.tdigest import merge_tdigests_kway
from zagg.stats.waveform import (
    build_waveform_digest,
    shot_count,
    shot_number,
    single_shot_value,
    threshold_sigma,
    waveform_weights,
)

# ── synthetic single shot ────────────────────────────────────────────────────
#
# 40 samples on a flat noise floor (mean 10, σ 1) with four signal bins well
# above threshold. gain 0.5 photoelectrons/count. Elevation descends bin0 →
# lastbin (the GEDI convention).

N = 40
NOISE_MEAN = 10.0
NOISE_STD = 1.0
GAIN = 0.5
E0, E1 = 100.0, 80.5  # elevation_bin0 / elevation_lastbin
SIGNAL_BINS = {12: 30.0, 13: 42.0, 14: 26.0, 25: 19.0}  # bin -> ADC count
FP = 1e-3


def _shot(seed=7):
    rng = np.random.default_rng(seed)
    counts = NOISE_MEAN + rng.normal(0.0, NOISE_STD, N)
    counts = np.clip(counts, None, NOISE_MEAN + 2.5 * NOISE_STD)  # keep noise sub-threshold
    for b, v in SIGNAL_BINS.items():
        counts[b] = v
    elev = np.linspace(E0, E1, N)
    return counts, elev


def _digest(counts, elev, **over):
    kw = dict(
        counts=counts,
        noise_mean=np.full_like(elev, NOISE_MEAN),
        noise_stddev=np.full_like(elev, NOISE_STD),
        gain=GAIN,
        false_positive_rate=FP,
        samples_per_record=N,
        correlation_length=1.0,
    )
    kw.update(over)
    return build_waveform_digest(elev, **kw)


class TestThresholdSigma:
    def test_single_trial_known_value(self):
        # fp = 1 - Φ(1) = 0.158655…: one trial puts the threshold at exactly 1σ.
        assert threshold_sigma(0.15865525393145707, 1) == pytest.approx(1.0, abs=1e-9)

    def test_more_trials_raise_the_bar(self):
        assert threshold_sigma(FP, 1420) > threshold_sigma(FP, 40) > threshold_sigma(FP, 1)

    def test_correlation_lowers_effective_trials(self):
        assert threshold_sigma(FP, 1420, correlation_length=5.0) == pytest.approx(
            threshold_sigma(FP, 284), abs=1e-12
        )

    def test_domain_errors(self):
        with pytest.raises(ValueError, match="false_positive_rate"):
            threshold_sigma(0.0)
        with pytest.raises(ValueError, match="samples_per_record"):
            threshold_sigma(FP, 0)
        with pytest.raises(ValueError, match="correlation_length"):
            threshold_sigma(FP, 10, correlation_length=0.0)


class TestWaveformWeights:
    def test_threshold_to_zero_and_gain(self):
        counts = np.array([10.0, 13.0, 14.5, 30.0])
        w = waveform_weights(counts, np.full(4, 10.0), np.ones(4), gain=GAIN, n_sigma=4.0)
        # 10, 13 and 14.0-threshold-edge zeroed; survivors are (count-mean)*g.
        assert w[0] == 0.0 and w[1] == 0.0
        assert w[2] == pytest.approx((14.5 - 10.0) * GAIN)
        assert w[3] == pytest.approx((30.0 - 10.0) * GAIN)

    def test_at_threshold_is_zero(self):
        counts = np.array([14.0])  # exactly mean + 4σ
        w = waveform_weights(counts, np.array([10.0]), np.array([1.0]), n_sigma=4.0)
        assert w[0] == 0.0


class TestRoundTrip:
    def test_loss_free_rows_are_the_clipped_samples(self):
        counts, elev = _shot()
        d = _digest(counts, elev)
        n_sigma = threshold_sigma(FP, N)
        w = waveform_weights(
            counts, np.full(N, NOISE_MEAN), np.full(N, NOISE_STD), gain=GAIN, n_sigma=n_sigma
        )
        kept = w > 0
        assert set(SIGNAL_BINS) == set(np.flatnonzero(kept)), "clip must keep exactly the signal"
        # Loss-free regime (n ≪ δ): one row per surviving sample, value-sorted.
        order = np.argsort(elev[kept], kind="stable")
        np.testing.assert_array_equal(d[:, 0], elev[kept][order].astype(np.float32))
        np.testing.assert_array_equal(d[:, 1], w[kept][order].astype(np.float32))

    def test_reconstruct_clipped_waveform_within_float32(self):
        # The issue #425 acceptance: stored rows + companions (elevation
        # endpoints, sample count, noise mean, gain) reconstruct the clipped
        # L1B waveform within float32.
        counts, elev = _shot()
        d = _digest(counts, elev)
        dz = (E1 - E0) / (N - 1)
        recon = np.zeros(N)  # threshold-to-zero: unstored bins are clipped
        for mean, weight in d:
            j = int(round((float(mean) - E0) / dz))
            recon[j] = weight / GAIN + NOISE_MEAN
        n_sigma = threshold_sigma(FP, N)
        thresh = NOISE_MEAN + n_sigma * NOISE_STD
        expected = np.where(counts > thresh, counts, 0.0)
        np.testing.assert_allclose(
            recon[recon > 0], expected[expected > 0], rtol=np.finfo(np.float32).eps * 8
        )
        assert (recon > 0).sum() == len(SIGNAL_BINS)

    def test_rx_energy_closure(self):
        # Stored mass + the noise-model remainder == the record's integrated
        # energy: sum(weights)/g accounts for every above-threshold sample,
        # the sub-threshold residual is exactly the rest.
        counts, elev = _shot()
        d = _digest(counts, elev)
        rx_energy = float(np.sum(counts - NOISE_MEAN))
        n_sigma = threshold_sigma(FP, N)
        thresh = NOISE_MEAN + n_sigma * NOISE_STD
        residual = float(np.sum((counts - NOISE_MEAN)[counts <= thresh]))
        stored_mass = float(np.sum(d[:, 1].astype(np.float64))) / GAIN
        assert stored_mass + residual == pytest.approx(rx_energy, rel=1e-6)

    def test_empty_when_nothing_survives(self):
        counts = np.full(N, NOISE_MEAN)  # pure floor
        elev = np.linspace(E0, E1, N)
        d = _digest(counts, elev)
        assert d.shape == (0, 2)

    def test_shape_mismatch_raises(self):
        counts, elev = _shot()
        with pytest.raises(ValueError, match="noise_mean shape"):
            build_waveform_digest(
                elev,
                counts=counts,
                noise_mean=np.zeros(3),
                noise_stddev=np.full(N, NOISE_STD),
            )


class TestPooledMultiShot:
    def _two_shots(self):
        counts_a, elev_a = _shot(seed=7)
        counts_b, elev_b = _shot(seed=11)
        elev_b = elev_b - 0.31  # distinct elevations, same cell
        return (counts_a, elev_a), (counts_b, elev_b)

    def test_pooled_build_equals_kway_merge(self):
        # The ratified pooled-merge design: multi-shot cells fold with the
        # EXISTING t-digest merge — below δ both routes are loss-free and
        # byte-identical.
        (ca, ea), (cb, eb) = self._two_shots()
        pooled = _digest(np.concatenate([ca, cb]), np.concatenate([ea, eb]))
        merged = merge_tdigests_kway([_digest(ca, ea), _digest(cb, eb)], delta=8192)
        np.testing.assert_array_equal(pooled, merged)

    def test_pooled_mass_is_additive(self):
        (ca, ea), (cb, eb) = self._two_shots()
        pooled = _digest(np.concatenate([ca, cb]), np.concatenate([ea, eb]))
        parts = _digest(ca, ea)[:, 1].sum() + _digest(cb, eb)[:, 1].sum()
        assert pooled[:, 1].sum() == pytest.approx(parts, rel=1e-6)


class TestCompanions:
    def test_shot_count_and_number_single(self):
        shots = np.full(9, 12345678901, dtype=np.uint64)
        assert shot_count(shots) == 1
        assert shot_number(shots) == 12345678901

    def test_shot_number_sentinel_when_pooled(self):
        shots = np.array([101, 101, 104], dtype=np.uint64)
        assert shot_count(shots) == 2
        assert shot_number(shots) == 0

    def test_empty_cell(self):
        empty = np.array([], dtype=np.uint64)
        assert shot_count(empty) == 0
        assert shot_number(empty) == 0
        assert np.isnan(single_shot_value(np.array([])))

    def test_single_shot_value_passthrough_and_sentinel(self):
        assert single_shot_value(np.full(5, 3.25)) == 3.25
        assert np.isnan(single_shot_value(np.array([3.25, 3.5])))


class TestAggregateWiring:
    def test_params_as_columns_through_cell_statistics(self):
        # The reducer receives counts/noise columns via the params-as-column
        # path and the scalars verbatim — the exact template wiring.
        from zagg.config import PipelineConfig, validate_config
        from zagg.processing import calculate_cell_statistics

        cfg = PipelineConfig(
            data_source={
                "reader": "h5coro",
                "groups": ["BEAM0000"],
                "coordinates": {"latitude": "/lat", "longitude": "/lon"},
                "variables": {
                    "rxwaveform": "/w",
                    "elevation": "/e",
                    "noise_mean": "/nm",
                    "noise_stddev": "/ns",
                },
            },
            aggregation={
                "coordinates": {"morton": {"dtype": "uint64", "fill_value": 0}},
                "variables": {
                    "count": {"function": "len", "source": "rxwaveform"},
                    "rx_flux": {
                        "kind": "ragged",
                        "function": "zagg.stats.waveform.build_waveform_digest",
                        "source": "elevation",
                        "inner_shape": [2],
                        "dtype": "float32",
                        "fill_value": 0,
                        "params": {
                            "delta": 8192,
                            "counts": "rxwaveform",
                            "noise_mean": "noise_mean",
                            "noise_stddev": "noise_stddev",
                            "gain": GAIN,
                            "false_positive_rate": FP,
                            "samples_per_record": N,
                        },
                    },
                },
            },
            output={"grid": {"type": "healpix", "parent_order": 9, "child_order": 18}},
        )
        validate_config(cfg)
        counts, elev = _shot()
        cell = {
            "rxwaveform": counts,
            "elevation": elev,
            "noise_mean": np.full(N, NOISE_MEAN),
            "noise_stddev": np.full(N, NOISE_STD),
        }
        result = calculate_cell_statistics(cell, config=cfg)
        assert result["count"] == N
        np.testing.assert_array_equal(result["rx_flux"], _digest(counts, elev))
