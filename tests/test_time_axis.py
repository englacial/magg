"""Tests for the spec §8 temporal declaration (issue #443).

The encode/decode contract on its own: the declaration grammar and its
strict-checks, envelope -> word -> conservative range round-trips inside the
grammar's quantization, the ordering property the stored axis relies on, the
window predicate, and the legacy (absent-declaration) path that every
pre-§8 store still reads through.
"""

import numpy as np
import pytest
from mortie import Q_END_NS, Q_START_NS

from zagg.time_axis import (
    LEGACY_TIME_ATTRS,
    TOC_EPOCH,
    TOC_GRAMMAR,
    TOC_SPEC,
    decode_time_axis,
    encode_time_axis,
    temporal_declaration,
    time_axis_attrs,
    time_axis_dtype,
    time_axis_overlaps,
    time_encoding,
)

TOC_ATTRS = time_axis_attrs("toc")
#: 2025-06-15T15:06:40Z, a comfortable distance from any grid boundary.
BASE_US = 1_750_000_000_000_000


def _cfg(value):
    class _C:
        output = {} if value is None else {"time_encoding": value}

    return _C()


class TestDeclaration:
    def test_default_encoding_is_legacy(self):
        assert time_encoding(_cfg(None)) == "microseconds"
        assert time_axis_dtype("microseconds") == "int64"
        assert time_axis_attrs("microseconds") == LEGACY_TIME_ATTRS

    def test_toc_stamps_the_declaration_and_no_cf_attrs(self):
        # units/calendar would describe the words wrongly, so a CF-decoding
        # client must find nothing to decode rather than plausible garbage.
        assert set(TOC_ATTRS) == {"temporal"}
        block = TOC_ATTRS["temporal"]
        # Exactly the #410-ruled shape: {spec, shape, grammar revision}. NO
        # per-store epoch/timescale/quantum guards -- those are properties of
        # the cited grammar, and echoing them would only put a restated
        # constant into the fixture bytes and the §5 hash.
        assert block == {"spec": TOC_SPEC, "shape": "coordinate", "grammar": TOC_GRAMMAR}
        assert time_axis_dtype("toc") == "uint64"

    def test_the_cited_grammar_is_the_one_this_reader_decodes_with(self):
        # The citation is a fixed {name}/{major} revision token -- never a
        # documentation URL or a release stamp (store bytes must not move when
        # docs or a floor move) -- but it must not drift from the constants
        # the decode actually runs on.
        assert TOC_GRAMMAR == "mortie-toc/1"
        assert TOC_EPOCH == "1850-01-01T00:00:00"
        assert (int(Q_START_NS), int(Q_END_NS)) == (2**31, 2**32)

    def test_absent_key_is_legacy_never_a_refusal(self):
        for attrs in (None, {}, {"units": "microseconds since 1970-01-01T00:00:00"}):
            assert temporal_declaration(attrs) is None

    def test_unknown_spec_refused(self):
        block = {**TOC_ATTRS["temporal"], "spec": "zagg-toc/2"}
        with pytest.raises(ValueError, match="unknown temporal declaration spec"):
            temporal_declaration({"temporal": block})

    def test_unimplemented_shape_refused(self):
        # A shape outside the vocabulary is a future revision's: refuse it
        # rather than decode words under a layout this reader cannot know.
        block = {**TOC_ATTRS["temporal"], "shape": "per-granule"}
        with pytest.raises(ValueError, match="shape 'per-granule' is not implemented"):
            temporal_declaration({"temporal": block})

    def test_companion_shape_refused_where_a_coordinate_is_expected(self):
        # §8.2/§8.3 shapes are defined, but a caller that can only consume a
        # time AXIS must not read a companion's block as one.
        block = {**TOC_ATTRS["temporal"], "shape": "per-centroid"}
        assert temporal_declaration({"temporal": block}) == block
        with pytest.raises(ValueError, match="shape 'per-centroid' is not implemented"):
            temporal_declaration({"temporal": block}, shape="coordinate")

    def test_uncited_grammar_refused(self):
        block = {**TOC_ATTRS["temporal"], "grammar": "mortie-toc/2"}
        with pytest.raises(ValueError, match="cites word grammar"):
            temporal_declaration({"temporal": block})

    def test_informative_keys_are_ignored_not_refused(self):
        # §8: unrecognized keys are non-normative provenance, so a reader
        # passes them through rather than refusing the store.
        block = {**TOC_ATTRS["temporal"], "source_time_field": "start_datetime"}
        assert temporal_declaration({"temporal": block}) == block

    def test_non_mapping_declaration_refused(self):
        with pytest.raises(ValueError, match="must be a mapping"):
            temporal_declaration({"temporal": "zagg-toc/1"})

    def test_bad_config_value_refused(self):
        with pytest.raises(ValueError, match="output.time_encoding must be one of"):
            time_encoding(_cfg("datetime64"))


class TestRoundTrip:
    def test_span_decodes_to_a_containing_range_within_quantization(self):
        starts = np.array([BASE_US, BASE_US + 86_400_000_000])
        ends = np.array([BASE_US + 7_500_000, BASE_US + 86_400_000_000 + 12_000_000])
        words = encode_time_axis(starts, ends, encoding="toc")
        assert words.dtype == np.uint64
        lo, hi = decode_time_axis(words, TOC_ATTRS)
        real_lo = starts.astype("datetime64[us]").astype("datetime64[ns]")
        real_hi = ends.astype("datetime64[us]").astype("datetime64[ns]")
        # Conservative: the decoded envelope contains the real interval...
        assert (lo <= real_lo).all() and (hi > real_hi).all()
        # ...and is loose by at most one quantum at each end.
        assert ((real_lo - lo) < np.timedelta64(int(Q_START_NS), "ns")).all()
        assert ((hi - real_hi) <= np.timedelta64(int(Q_END_NS), "ns")).all()

    def test_degenerate_envelope_is_an_exact_timestamp(self):
        # A single-item acquisition stays exact to the nanosecond: the writer
        # never widens an instant into a range (§8.1).
        words = encode_time_axis([BASE_US], [BASE_US], encoding="toc")
        lo, hi = decode_time_axis(words, TOC_ATTRS)
        exact = np.datetime64(BASE_US, "us").astype("datetime64[ns]")
        assert lo[0] == exact and hi[0] == exact

    def test_mixed_variants_sort_by_envelope_start(self):
        # Unsigned word order IS order by encoded envelope start (§8.1), so
        # envelopes that themselves ascend encode to ascending words, mixing
        # timestamp and range variants.
        day = 86_400_000_000
        starts = np.array([BASE_US, BASE_US + day, BASE_US + 2 * day])
        ends = np.array([BASE_US + 9_000_000, BASE_US + day, BASE_US + 2 * day + 3_000_000])
        words = encode_time_axis(starts, ends, encoding="toc")
        np.testing.assert_array_equal(np.sort(words), words)

    def test_a_leading_envelope_start_breaks_stored_word_order(self):
        # The §8.1 divergence, at the encode layer: row order is the group's
        # earliest MEMBER time, but the word encodes the ENVELOPE start. A
        # later row whose envelope begins before an earlier row's encodes a
        # smaller word, so the stored axis is materially unsorted -- which is
        # why §8.1 forbids bisecting it.
        starts = np.array([BASE_US, BASE_US - 63_000_000])
        ends = np.array([BASE_US, BASE_US + 10_000_000])
        words = encode_time_axis(starts, ends, encoding="toc")
        assert words[1] < words[0]
        assert not np.array_equal(np.sort(words), words)

    def test_empty_axis(self):
        words = encode_time_axis([], [], encoding="toc")
        assert words.dtype == np.uint64 and words.size == 0
        lo, hi = decode_time_axis(words, TOC_ATTRS)
        assert lo.size == 0 and hi.size == 0

    def test_inverted_envelope_refused(self):
        with pytest.raises(ValueError, match="ends before it starts"):
            encode_time_axis([BASE_US], [BASE_US - 1], encoding="toc")

    def test_pre_epoch_time_refused(self):
        with pytest.raises(ValueError, match="precedes the toc epoch"):
            encode_time_axis([-4_000_000_000_000_000], [0], encoding="toc")

    def test_legacy_encoding_stores_the_start_only(self):
        values = encode_time_axis([BASE_US], [BASE_US + 7_500_000], encoding="microseconds")
        assert values.dtype == np.int64 and values[0] == BASE_US
        lo, hi = decode_time_axis(values, {})
        exact = np.datetime64(BASE_US, "us").astype("datetime64[ns]")
        assert lo[0] == exact and hi[0] == exact


class TestWindowSelection:
    def _axis(self):
        day = 86_400_000_000
        starts = np.array([BASE_US, BASE_US + day, BASE_US + 2 * day])
        ends = np.array([BASE_US + 7_500_000, BASE_US + day, BASE_US + 2 * day + 3_000_000])
        return starts, ends

    def test_toc_window_selects_by_overlap(self):
        starts, ends = self._axis()
        words = encode_time_axis(starts, ends, encoding="toc")
        mask = time_axis_overlaps(words, TOC_ATTRS, "2025-06-16T00:00:00", "2025-06-17T00:00:00")
        np.testing.assert_array_equal(mask, [False, True, False])

    def test_toc_window_never_under_reports(self):
        # A window that clips the middle of a range still selects it.
        starts, ends = self._axis()
        words = encode_time_axis(starts, ends, encoding="toc")
        mask = time_axis_overlaps(words, TOC_ATTRS, "2025-06-15T15:06:44", "2025-06-15T15:06:45")
        assert bool(mask[0]) and not mask[1:].any()

    def test_legacy_window_matches_the_toc_selection(self):
        starts, ends = self._axis()
        legacy = encode_time_axis(starts, ends, encoding="microseconds")
        words = encode_time_axis(starts, ends, encoding="toc")
        window = ("2025-06-16T00:00:00", "2025-06-17T00:00:00")
        np.testing.assert_array_equal(
            time_axis_overlaps(legacy, {}, *window),
            time_axis_overlaps(words, TOC_ATTRS, *window),
        )

    def test_inverted_window_refused(self):
        with pytest.raises(ValueError, match="window is inverted"):
            time_axis_overlaps(np.array([0], dtype="int64"), {}, "2025-01-02", "2025-01-01")


class TestObservationWords:
    """``observation_words`` — the per-observation toc encode (§8.3, #410)."""

    EPOCH = "2018-01-01T00:00:00"

    def _words(self, values, **kw):
        from zagg.time_axis import observation_words

        kw.setdefault("epoch", self.EPOCH)
        kw.setdefault("scale", "gps")
        kw.setdefault("units", "seconds")
        return observation_words(np.asarray(values, dtype=np.float64), **kw)

    def test_every_word_is_a_timestamp_never_a_range(self):
        # §8.3's MUST: an instant is never widened into a range, and zagg's
        # readers deliver per-observation instants.
        import mortie

        words = self._words([0.0, 1.5, 86400.25, 3.2e7])
        assert words.dtype == np.uint64
        assert not mortie.toc_is_range(words).any()

    def test_round_trips_to_the_declared_instants(self):
        import mortie

        offsets = [0.0, 1.5, 86400.25]
        start, end = mortie.toc2time(self._words(offsets))
        assert np.array_equal(start, end)  # a timestamp's two bounds coincide
        # Literal instants, NOT a re-derivation of the encode's own arithmetic:
        # re-deriving pins the round trip only where the formula is already
        # right, which is how the leap-shift bug stayed green (issue #410 review).
        expected = np.array(
            [
                "2018-01-01T00:00:00.000000000",
                "2018-01-01T00:00:01.500000000",
                "2018-01-02T00:00:00.250000000",
            ],
            dtype="datetime64[ns]",
        )
        assert np.array_equal(mortie.to_datetime64(start), expected)

    def test_days_units_scale(self):
        assert int(self._words([1.0], units="days")[0]) == int(self._words([86400.0])[0])

    def test_utc_scale_refused(self):
        with pytest.raises(ValueError, match="continuous timescale"):
            self._words([0.0], scale="utc")

    def test_unknown_units_refused(self):
        with pytest.raises(ValueError, match="units 'ticks' is not one of"):
            self._words([0.0], units="ticks")

    def test_pre_epoch_time_refused(self):
        # The grammar's 1850 origin is a hard floor; refusing names the gap
        # rather than wrapping into a plausible-looking word.
        with pytest.raises(ValueError, match="precedes the toc epoch"):
            self._words([-1e18], units="seconds")

    def test_non_finite_refused(self):
        # Dropping a NaN instant would misalign the companion from its payload,
        # which is a §8.3 row-alignment break rather than a lost observation.
        with pytest.raises(ValueError, match="non-finite observation time"):
            self._words([0.0, np.nan])

    def test_empty_input_returns_empty(self):
        out = self._words([])
        assert out.shape == (0,) and out.dtype == np.uint64

    def test_agrees_with_the_window_router_at_a_boundary(self):
        # The hazard this single-sourcing exists to prevent: an observation
        # routed into window W whose stored word reads as outside it. Both sides
        # derive from the same fixed-offset model, so a boundary instant's word
        # decodes back to the very instant the router converted.
        import mortie

        from zagg.windows import utc_to_offset

        boundary = np.datetime64("2019-01-01T00:00:00", "ns")
        offset = utc_to_offset(
            boundary.astype("datetime64[us]").astype(object), epoch=self.EPOCH, scale="gps"
        )
        decoded = mortie.to_datetime64(mortie.toc2time(self._words([offset]))[0])[0]
        assert decoded == boundary

    def test_is_exact_to_the_nanosecond_at_icesat2_magnitudes(self):
        # §8.3 MUSTs a timestamp word exact to the nanosecond. By 2026
        # ``delta_time`` is ~2.5e8 s, so ``offsets * 1e9`` lands near 2.5e17 where
        # a float64 ulp is 32 ns: one multiply quantizes before it rounds and
        # misses the nearest ns by up to 16 ns (issue #410 review). Compared
        # against exact rational arithmetic on the delivered float64 values.
        from fractions import Fraction

        import mortie

        from zagg.time_axis import _internal_ns

        values = np.random.default_rng(410).uniform(2.4e8, 2.5e8, 2000)
        base = int(_internal_ns(np.array([np.datetime64(self.EPOCH, "us")]))[0])
        start, _ = mortie.toc2time(self._words(values))
        exact = np.array([base + round(Fraction(float(v)) * 10**9) for v in values], dtype="uint64")
        assert np.array_equal(np.asarray(start, dtype="uint64"), exact)

    def test_pre_2017_epoch_is_not_leap_shifted(self):
        # Every other case here sits on the post-2017 ATLAS epoch, where a
        # scale-vs-UTC correction is 0 and therefore invisible: deleting or
        # sign-flipping one leaves the class green (issue #410 review). This case
        # runs the branch — a GPS-native epoch, a 2024 observation — against the
        # instant itself AND against the window router, so any correction
        # reintroduced here fails by 18 s rather than passing silently.
        import mortie

        from zagg.windows import offset_to_utc

        epoch, value = "1980-01-06T00:00:00", 1.4e9
        decoded = mortie.to_datetime64(mortie.toc2time(self._words([value], epoch=epoch))[0])[0]
        assert decoded == np.datetime64("2024-05-17T16:53:02", "ns")
        assert decoded == np.datetime64(
            offset_to_utc(value, epoch=epoch, scale="gps").replace(tzinfo=None), "ns"
        )

    def test_post_ceiling_time_refused(self):
        # ~2142 is the grammar's ceiling (1850 + 2^63 - 2^32 ns); past it the cast
        # would wrap into a plausible-looking word.
        with pytest.raises(ValueError, match="exceeds the toc grammar's range"):
            self._words([1e18])

    def test_the_ceiling_is_the_grammars_own_span_not_the_naive_63rd_bit(self):
        # mortie refuses at TOC_MAX_NS = 2^63 - 2^32 (its quantum-aligned span
        # ceiling), so a 2^63 - 1 bound here lets the last 4.29 s through and the
        # caller gets mortie's message instead of this one (issue #410 review).
        from mortie import TOC_MAX_NS

        from zagg.time_axis import _internal_ns

        base = int(_internal_ns(np.array([np.datetime64(self.EPOCH, "us")]))[0])
        # An offset landing inside [TOC_MAX_NS, 2^63) — refused by the grammar,
        # accepted by the naive bound.
        with pytest.raises(ValueError, match="exceeds the toc grammar's range"):
            self._words([(TOC_MAX_NS + 2**31 - base) / 1e9])
        # ...and the band just below it still encodes, so the bound is not simply
        # tightened past the domain.
        assert self._words([(TOC_MAX_NS - 2**31 - base) / 1e9]).dtype == np.uint64
