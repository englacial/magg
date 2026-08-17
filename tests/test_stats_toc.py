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
