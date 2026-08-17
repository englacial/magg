"""Tests for the spec §8 temporal declaration (issue #443).

The encode/decode contract on its own: the declaration grammar and its
strict-checks, envelope -> word -> conservative range round-trips inside the
grammar's quantization, the ordering property the stored axis relies on, the
window predicate, and the legacy (absent-declaration) path that every
pre-§8 store still reads through.
"""

import numpy as np
import pytest
from mortie.toc import Q_END_NS, Q_START_NS

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
        # Exactly the #410-ruled shape: {spec, shape, versioned grammar
        # citation}. NO per-store epoch/timescale/quantum guards -- those are
        # properties of the cited grammar, and echoing them would only put a
        # restated constant into the fixture bytes and the §5 hash.
        assert block == {"spec": TOC_SPEC, "shape": "axis", "grammar": TOC_GRAMMAR}
        assert time_axis_dtype("toc") == "uint64"

    def test_the_cited_grammar_is_the_one_this_reader_decodes_with(self):
        # The citation is a fixed token, but it must not drift from the
        # constants the decode actually runs on.
        assert TOC_GRAMMAR.startswith("mortie/toc@")
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
        block = {**TOC_ATTRS["temporal"], "shape": "per-centroid"}
        with pytest.raises(ValueError, match="shape 'per-centroid' is not implemented"):
            temporal_declaration({"temporal": block})

    def test_uncited_grammar_refused(self):
        block = {**TOC_ATTRS["temporal"], "grammar": "mortie/toc@2.0.0"}
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
