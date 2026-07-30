"""Overview sweep family (issue #201): D24 kernels, writer, pyramid block.

Phase A covers the composability-class helpers (the D24 derivation from the
aggregator merge-law set) and the per-field up-aggregation kernels: exact
folds byte-equal by construction, approximate (t-digest) folds via the
order-independent k-way merge.
"""

import numpy as np
import pytest

from zagg.semantics import (
    EXACT_MERGE_LAWS,
    composability_classes,
    field_composability,
)
from zagg.sweep_overview import (
    combine_dense,
    decode_digest,
    encode_digest,
    fold_dense,
    fold_digests,
)


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
