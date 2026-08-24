"""Tests for the closest-observation ingest builder (issue #509).

Phase 1 — cover-driven epoch extraction: per-shard epochs from one or more
store roots' ``coverage.toc`` word-set covers (spec §10.5), union across
stores, AOI intersect, loud refusal when a store carries no readable cover.
Synthetic covers ride the same :func:`zagg.coverage_toc.build_cover_section`
producer the sweep uses; the committed golden ``coverage.toc`` fixture pins
the extraction against the frozen §10.5 grammar bytes.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from mortie import from_datetime64, time2toc, to_datetime64

from zagg.catalog.closest_obs import (
    ReferenceEpochs,
    _word_midpoints,
    nearest_acquisitions,
    reference_epochs,
)
from zagg.coverage_toc import (
    COVER_CAP,
    COVER_NAME,
    TEMPORAL_COVER_ORDER,
    build_cover_section,
    cover_words,
    quantize_words,
    read_cover,
    write_cover,
)
from zagg.grids.morton import morton_word

SPEC_DATA = Path(__file__).parent / "data" / "spec"
DAY_NS = 86_400 * 10**9
#: An arbitrary but realistic base instant on the §8 internal-ns scale
#: (mirrors ``test_coverage_toc``'s convention).
BASE_NS = 5_344_000_000_000_000_000
#: One order-18 cover bucket (2^45 ns) and half of it — the epoch-midpoint
#: error bound is half a BUCKET, not half a word (a word is a run of buckets).
BUCKET_NS = 2 ** (63 - TEMPORAL_COVER_ORDER)
HALF_BUCKET = np.timedelta64(BUCKET_NS // 2, "ns")

#: The golden fixture's one shard, and the cell centre / far-away rings the
#: AOI tests use (order 4; centre from ``mortie.mort2geo``).
SHARD = "11213"
SHARD_KEY = morton_word(SHARD)


def _ring(lat, lon, half=2.0):
    lats = np.array([lat - half, lat - half, lat + half, lat + half, lat - half])
    lons = np.array([lon - half, lon + half, lon + half, lon - half, lon - half])
    return [(lats, lons)]


AOI_AT_SHARD = _ring(14.54, 53.44)
AOI_ELSEWHERE = _ring(-40.0, -120.0)


def _write_store(root, shards: dict[str, np.ndarray], order: int = 4) -> str:
    """A store root carrying a §10.5 cover claiming ``shards``' instants."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    contributions = {
        decimal: [(None, None, None, quantize_words(time2toc(np.asarray(inst, dtype=np.uint64))))]
        for decimal, inst in shards.items()
    }
    write_cover(str(root), build_cover_section(contributions, ["h"], order))
    return str(root)


def _instants(*day_offsets) -> np.ndarray:
    return np.array([BASE_NS + d * DAY_NS for d in day_offsets], dtype=np.uint64)


def _utc(instants) -> np.ndarray:
    return np.sort(np.asarray(to_datetime64(np.asarray(instants, np.uint64)), "datetime64[ns]"))


def _nearest_gap(mids: np.ndarray, true: np.ndarray) -> np.timedelta64:
    """Largest distance from a true instant to its nearest epoch."""
    return np.abs(true[:, None] - mids[None, :]).min(axis=1).max()


class TestWordMidpoints:
    def test_empty_words_decode_to_no_epochs(self):
        out = _word_midpoints(np.empty(0, dtype=np.uint64))
        assert out.dtype == np.dtype("datetime64[ns]") and out.size == 0

    def test_a_quantized_word_midpoint_stays_within_half_a_bucket(self):
        inst = _instants(0, 5, 11)
        words = quantize_words(time2toc(inst))
        mids = np.sort(_word_midpoints(words))
        true = np.sort(np.asarray(to_datetime64(inst), dtype="datetime64[ns]"))
        assert mids.size == true.size
        assert (np.abs(mids - true) <= HALF_BUCKET).all()

    def test_an_exact_timestamp_word_decodes_to_its_bucket_midpoint(self):
        """An unquantized word still resolves to the bucket it falls in."""
        inst = _instants(3)
        mids = _word_midpoints(np.asarray(time2toc(inst), dtype=np.uint64))
        assert mids.size == 1
        assert abs(mids[0] - _utc(inst)[0]) <= HALF_BUCKET

    def test_two_passes_one_bucket_apart_yield_two_epochs(self):
        """``toc_normalize`` coalesces the abutting buckets into ONE word."""
        inst = np.array([BASE_NS, BASE_NS + BUCKET_NS], dtype=np.uint64)
        words = quantize_words(time2toc(inst))
        assert len(words) == 1  # the coalescing that motivates the expansion
        mids = np.sort(_word_midpoints(words))
        assert mids.size == 2
        assert _nearest_gap(mids, _utc(inst)) <= HALF_BUCKET

    def test_a_contiguous_campaign_yields_one_epoch_per_covered_bucket(self):
        """A 10-day, 6-hourly campaign is one word — and 25 covered buckets."""
        inst = np.array([BASE_NS + i * 6 * 3600 * 10**9 for i in range(40)], dtype=np.uint64)
        words = quantize_words(time2toc(inst))
        assert len(words) == 1
        covered = {int(t) >> (63 - TEMPORAL_COVER_ORDER) for t in inst}
        mids = np.sort(_word_midpoints(words))
        assert mids.size == len(covered) == 25
        assert _nearest_gap(mids, _utc(inst)) <= HALF_BUCKET

    def test_a_pass_straddling_a_bucket_edge_yields_two_epochs(self):
        """Benign over-selection: both epochs pick the same nearest granule."""
        edge = ((BASE_NS >> (63 - TEMPORAL_COVER_ORDER)) + 1) << (63 - TEMPORAL_COVER_ORDER)
        inst = np.array([edge - 60 * 10**9, edge + 60 * 10**9], dtype=np.uint64)
        words = quantize_words(time2toc(inst))
        assert len(words) == 1
        mids = np.sort(_word_midpoints(words))
        assert mids.size == 2
        assert _nearest_gap(mids, _utc(inst)) <= HALF_BUCKET

    def test_a_coarser_order_widens_the_buckets(self):
        """The block's effective order drives the expansion, not the pin."""
        inst = np.array([BASE_NS, BASE_NS + BUCKET_NS], dtype=np.uint64)
        coarse = quantize_words(time2toc(inst), TEMPORAL_COVER_ORDER - 2)
        mids = _word_midpoints(coarse, TEMPORAL_COVER_ORDER - 2)
        assert mids.size == 1  # both passes now share one 39 h bucket
        assert _nearest_gap(mids, _utc(inst)) <= 4 * HALF_BUCKET


class TestReferenceEpochs:
    def test_one_store_one_shard(self, tmp_path):
        root = _write_store(tmp_path, {SHARD: _instants(0, 5, 11)})
        out = reference_epochs(root)
        assert isinstance(out, ReferenceEpochs)
        assert out.order == 4
        assert list(out.epochs) == [SHARD_KEY]
        assert out.epochs[SHARD_KEY].size == 3 == out.total
        assert out.stores == [root]
        # Sorted unique datetime64[ns], the contract phase 2 searchsorted rides.
        e = out.epochs[SHARD_KEY]
        assert e.dtype == np.dtype("datetime64[ns]")
        assert (np.diff(e.astype("int64")) > 0).all()

    def test_union_across_stores_is_deduplicated(self, tmp_path):
        """Two stores quantized on the same grid: shared passes count once."""
        a = _write_store(tmp_path / "a", {SHARD: _instants(0, 5)})
        b = _write_store(tmp_path / "b", {SHARD: _instants(5, 11)})
        out = reference_epochs([a, b])
        assert out.epochs[SHARD_KEY].size == 3
        # Parity: the union equals each store's own epochs united.
        ea = reference_epochs(a).epochs[SHARD_KEY]
        eb = reference_epochs(b).epochs[SHARD_KEY]
        assert np.array_equal(out.epochs[SHARD_KEY], np.union1d(ea, eb))

    def test_partially_overlapping_covers_union_at_the_bucket_grid(self, tmp_path):
        """Unequal words for a shared pass must not double it (bucket union).

        Three passes one bucket apart; store A saw passes 1+2 and store B
        saw 2+3, so each store's cover carries a *different* two-bucket range
        word for the shared pass. Word-level ``np.unique`` keeps both and
        yields two displaced epochs for three passes; bucket-level union
        yields exactly one epoch per covered bucket.
        """
        passes = np.array([BASE_NS, BASE_NS + BUCKET_NS, BASE_NS + 2 * BUCKET_NS], np.uint64)
        a = _write_store(tmp_path / "a", {SHARD: passes[:2]})
        b = _write_store(tmp_path / "b", {SHARD: passes[1:]})
        # Each store really does emit one coalesced (and unequal) word.
        wa = cover_words(read_cover(a))[SHARD]
        wb = cover_words(read_cover(b))[SHARD]
        assert len(wa) == len(wb) == 1 and wa[0] != wb[0]
        out = reference_epochs([a, b])
        e = out.epochs[SHARD_KEY]
        assert e.size == 3
        assert _nearest_gap(e, _utc(passes)) <= HALF_BUCKET

    def test_a_shard_only_one_store_covers_still_contributes(self, tmp_path):
        a = _write_store(tmp_path / "a", {SHARD: _instants(0)})
        b = _write_store(tmp_path / "b", {SHARD: _instants(40), "11212": _instants(7)})
        out = reference_epochs([a, b])
        assert set(out.epochs) == {SHARD_KEY, morton_word("11212")}
        assert out.epochs[SHARD_KEY].size == 2

    def test_a_store_without_a_cover_refuses_loudly(self, tmp_path):
        (tmp_path / "empty").mkdir()
        with pytest.raises(ValueError, match="cover-driven"):
            reference_epochs(str(tmp_path / "empty"))

    def test_an_unknown_revision_cover_refuses_loudly(self, tmp_path):
        root = tmp_path / "future"
        root.mkdir()
        (root / COVER_NAME).write_text(
            json.dumps({"spec": "zagg-coverage-toc-cover/9", "shards": {}})
        )
        with pytest.raises(ValueError, match="unreadable"):
            reference_epochs(str(root))

    def test_a_shard_order_mismatch_between_stores_refuses(self, tmp_path):
        a = _write_store(tmp_path / "a", {SHARD: _instants(0)}, order=4)
        b = _write_store(tmp_path / "b", {SHARD: _instants(5)}, order=5)
        with pytest.raises(ValueError, match="not.*comparable|shard order"):
            reference_epochs([a, b])

    def test_a_cover_without_a_shard_order_refuses_by_name(self, tmp_path):
        """A missing ``order`` used to surface as an opaque ``TypeError``."""
        root = tmp_path / "orderless"
        root.mkdir()
        (root / COVER_NAME).write_text(
            json.dumps({"spec": "zagg-coverage-toc-cover/1", "shards": {}})
        )
        with pytest.raises(ValueError, match="non-integer cover shard order"):
            reference_epochs(str(root))

    def test_no_stores_refuses(self):
        with pytest.raises(ValueError, match="at least one"):
            reference_epochs([])

    def test_aoi_parts_restrict_the_shard_set(self, tmp_path):
        root = _write_store(tmp_path, {SHARD: _instants(0, 5)})
        assert set(reference_epochs(root, aoi=AOI_AT_SHARD).epochs) == {SHARD_KEY}
        assert reference_epochs(root, aoi=AOI_ELSEWHERE).epochs == {}

    def test_aoi_moc_restricts_the_shard_set(self, tmp_path):
        from mortie import Moc

        root = _write_store(tmp_path, {SHARD: _instants(0, 5)})
        keep = Moc.from_polygon(*AOI_AT_SHARD[0])
        drop = Moc.from_polygon(*AOI_ELSEWHERE[0])
        assert set(reference_epochs(root, aoi=keep).epochs) == {SHARD_KEY}
        assert reference_epochs(root, aoi=drop).epochs == {}

    def test_aoi_geojson_path_restricts_the_shard_set(self, tmp_path):
        root = _write_store(tmp_path, {SHARD: _instants(0)})
        lats, lons = AOI_AT_SHARD[0]
        geojson = {
            "type": "Polygon",
            "coordinates": [[[float(x), float(y)] for x, y in zip(lons, lats)]],
        }
        path = tmp_path / "aoi.geojson"
        path.write_text(json.dumps(geojson))
        assert set(reference_epochs(root, aoi=str(path)).epochs) == {SHARD_KEY}


class TestCoarsenedBlock:
    """§10.5 lets a block coarsen below the pin — the read half must be loud."""

    #: Enough single-bucket claims (2 buckets apart, so no gap coalesces) to
    #: blow the 512-word cap and force ``_cap_cover`` down an order.
    COARSE_INSTANTS = np.array(
        [BASE_NS + i * 2 * BUCKET_NS for i in range(COVER_CAP + 88)], dtype=np.uint64
    )

    def _root(self, tmp_path):
        return _write_store(tmp_path, {SHARD: self.COARSE_INSTANTS})

    def test_the_block_really_coarsened(self, tmp_path):
        root = self._root(tmp_path)
        block = read_cover(root)["shards"][SHARD]
        assert block["temporal_order"] < TEMPORAL_COVER_ORDER

    def test_a_coarsened_block_warns_and_reports_its_order(self, tmp_path, caplog):
        root = self._root(tmp_path)
        landed = read_cover(root)["shards"][SHARD]["temporal_order"]
        with caplog.at_level("WARNING", logger="zagg.catalog.closest_obs"):
            out = reference_epochs(root)
        assert f"temporal order {landed}" in caplog.text
        assert "below the pinned" in caplog.text
        assert out.orders == {SHARD_KEY: landed}
        assert out.tolerance(SHARD_KEY) == np.timedelta64(2 ** (62 - landed), "ns")

    def test_the_epochs_are_the_coarse_buckets_midpoints(self, tmp_path):
        root = self._root(tmp_path)
        landed = read_cover(root)["shards"][SHARD]["temporal_order"]
        k = 63 - landed
        internal = np.asarray(from_datetime64(reference_epochs(root).epochs[SHARD_KEY]), np.uint64)
        # Every epoch sits at an aligned coarse-bucket midpoint, and at the
        # coarse grid's buckets — not the pinned order's.
        assert set(int(t) % 2**k for t in internal) == {2 ** (k - 1) - 1}
        covered = {int(t) >> k for t in self.COARSE_INSTANTS}
        assert set(int(t) >> k for t in internal) == covered
        assert _nearest_gap(
            reference_epochs(root).epochs[SHARD_KEY], _utc(self.COARSE_INSTANTS)
        ) <= np.timedelta64(2 ** (k - 1), "ns")

    def test_an_uncoarsened_block_neither_warns_nor_hides_its_order(self, tmp_path, caplog):
        root = _write_store(tmp_path, {SHARD: _instants(0, 5)})
        with caplog.at_level("WARNING", logger="zagg.catalog.closest_obs"):
            out = reference_epochs(root)
        assert "below the pinned" not in caplog.text
        assert out.orders == {SHARD_KEY: TEMPORAL_COVER_ORDER}
        assert out.tolerance(SHARD_KEY) == HALF_BUCKET


class TestGoldenFixture:
    """The committed §7 ``temporal/`` fixture pins the frozen grammar bytes."""

    #: The two epochs the committed fixture decodes to, pinned as literals
    #: rather than recomputed with the function under test. Verified by hand:
    #: each is the midpoint of an order-18 bucket (internal ns congruent to
    #: 2^44 - 1 mod 2^45, buckets 151903 and 151915 — twelve apart, the
    #: fixture's five-day gap), and each sits within half a bucket of the
    #: generator's two campaign clusters (``TEMPORAL_BASE`` and +5 days).
    GOLDEN = np.array(
        ["2019-05-14T03:14:07.595891711", "2019-05-19T00:31:00.060957695"],
        dtype="datetime64[ns]",
    )

    def test_the_golden_cover_decodes_to_the_pinned_epochs(self):
        out = reference_epochs(str(SPEC_DATA / "temporal"))
        assert out.order == 4
        assert list(out.epochs) == [SHARD_KEY]
        assert np.array_equal(out.epochs[SHARD_KEY], self.GOLDEN)
        assert out.orders == {SHARD_KEY: TEMPORAL_COVER_ORDER}

    def test_the_golden_epochs_are_order_18_bucket_midpoints(self):
        """The arithmetic the ±4.9 h claim rests on, checked independently."""
        k = 63 - TEMPORAL_COVER_ORDER
        internal = np.asarray(from_datetime64(self.GOLDEN), dtype=np.uint64)
        assert [int(t) % 2**k for t in internal] == [2 ** (k - 1) - 1] * 2
        assert [int(t) >> k for t in internal] == [151903, 151915]

    def test_every_fixture_observation_has_an_epoch_within_half_a_bucket(self):
        """The property the ruling actually claims, against the generator's
        own recorded instants (``temporal.expected.json``), not against
        anything :mod:`zagg.catalog.closest_obs` computed."""
        expected = json.loads((SPEC_DATA / "temporal.expected.json").read_text())
        true = np.array(
            sorted({int(ns) for c in expected["cells"] for ns in c["obs_span_ns"]}),
            dtype="datetime64[ns]",
        )
        epochs = reference_epochs(str(SPEC_DATA / "temporal")).epochs[SHARD_KEY]
        assert _nearest_gap(epochs, true) <= HALF_BUCKET

    def test_the_golden_epochs_sit_inside_the_fixture_campaign(self):
        """Midpoints land inside the leaf's synthetic campaign window."""
        out = reference_epochs(str(SPEC_DATA / "temporal"))
        lo = np.datetime64("2019-01-01")
        hi = np.datetime64("2020-01-01")
        e = out.epochs[SHARD_KEY]
        assert ((e > lo) & (e < hi)).all()


class TestNearestAcquisitions:
    """Phase 2 — the vectorized closest-1 selection core."""

    def _dt(self, *hours):
        return np.array(
            [np.datetime64("2025-06-01T00:00") + np.timedelta64(h, "h") for h in hours]
        ).astype("datetime64[ns]")

    def test_each_epoch_selects_its_nearest_acquisition(self):
        times = self._dt(0, 10, 24)
        epochs = self._dt(1, 9, 23)
        sel, off = nearest_acquisitions(epochs, times)
        assert sel.tolist() == [0, 1, 2]
        assert off.astype("timedelta64[h]").astype(int).tolist() == [-1, 1, 1]

    def test_offsets_are_signed_acquisition_minus_epoch(self):
        times = self._dt(12)
        epochs = self._dt(10, 14)
        sel, off = nearest_acquisitions(epochs, times)
        assert sel.tolist() == [0, 0]
        assert off[0] == np.timedelta64(2, "h") and off[1] == -np.timedelta64(2, "h")

    def test_an_exact_match_has_zero_offset(self):
        times = self._dt(5, 7)
        sel, off = nearest_acquisitions(self._dt(7), times)
        assert sel.tolist() == [1] and off[0] == np.timedelta64(0, "ns")

    def test_a_tie_selects_the_earlier_acquisition(self):
        times = self._dt(0, 10)
        sel, off = nearest_acquisitions(self._dt(5), times)
        assert sel.tolist() == [0]
        assert off[0] == -np.timedelta64(5, "h")

    def test_selection_indices_refer_to_the_input_order(self):
        times = self._dt(24, 0, 10)  # unsorted catalog order
        sel, _ = nearest_acquisitions(self._dt(23, 1), times)
        assert sel.tolist() == [0, 1]

    def test_max_time_offset_boundary_exactly_at_selects(self):
        times = self._dt(0)
        sel, off = nearest_acquisitions(self._dt(6), times, max_time_offset=np.timedelta64(6, "h"))
        assert sel.tolist() == [0]

    def test_max_time_offset_one_ns_past_drops_but_still_reports(self):
        times = self._dt(0)
        cap = np.timedelta64(6, "h") - np.timedelta64(1, "ns")
        sel, off = nearest_acquisitions(self._dt(6), times, max_time_offset=cap)
        assert sel.tolist() == [-1]
        # The nearest offset is still reported — the loud record the drop rides.
        assert off[0] == -np.timedelta64(6, "h")

    def test_no_acquisitions_selects_nothing_and_reports_nat(self):
        sel, off = nearest_acquisitions(self._dt(1, 2), np.array([], dtype="datetime64[ns]"))
        assert sel.tolist() == [-1, -1]
        assert np.isnat(off).all()

    def test_no_epochs_is_empty(self):
        sel, off = nearest_acquisitions(np.array([], dtype="datetime64[ns]"), self._dt(0))
        assert sel.size == 0 and off.size == 0

    def test_a_negative_max_time_offset_refuses(self):
        with pytest.raises(ValueError, match="non-negative"):
            nearest_acquisitions(self._dt(1), self._dt(0), max_time_offset=-np.timedelta64(1, "h"))

    def test_duplicate_acquisition_times_select_the_first_record(self):
        # One datatake, three granules at the same instant, plus a far one.
        times = self._dt(0, 0, 0, 96)
        for epoch in (self._dt(-1), self._dt(0), self._dt(1)):
            sel, _ = nearest_acquisitions(epoch, times)
            assert sel.tolist() == [0]

    def _check_oracle(self, epochs, times, cap):
        """Brute force: nearest, ties to the earlier, then the lowest index."""
        sel, off = nearest_acquisitions(epochs, times, max_time_offset=cap)
        t = times.astype("int64")
        for k, e in enumerate(epochs.astype("int64")):
            d = np.abs(t - e)
            best = np.flatnonzero(d == d.min())
            # tie -> earlier acquisition; equal times -> lowest catalog index
            want = best[np.argmin(t[best])] if best.size > 1 else best[0]
            assert off[k] == np.timedelta64(int(t[want] - e), "ns")
            if d.min() <= cap.astype("timedelta64[ns]").astype("int64"):
                assert sel[k] == want
            else:
                assert sel[k] == -1

    def test_a_nat_acquisition_time_refuses(self):
        times = np.array(["2025-06-01T00", "NaT", "2025-06-01T10"], dtype="datetime64[ns]")
        with pytest.raises(ValueError, match="times carries NaT"):
            nearest_acquisitions(self._dt(1, 9), times)

    def test_a_nat_epoch_refuses(self):
        epochs = np.array(["2025-06-01T01", "NaT"], dtype="datetime64[ns]")
        with pytest.raises(ValueError, match="epochs carries NaT"):
            nearest_acquisitions(epochs, self._dt(0, 10))

    def test_matches_a_brute_force_oracle(self):
        rng = np.random.default_rng(7)
        base = np.datetime64("2025-01-01").astype("datetime64[ns]").astype("int64")
        times = (base + rng.integers(0, 400 * 86_400, 60) * 10**9).astype("datetime64[ns]")
        epochs = (base + rng.integers(0, 400 * 86_400, 45) * 10**9).astype("datetime64[ns]")
        self._check_oracle(epochs, times, np.timedelta64(2, "D"))

    def test_matches_the_oracle_with_every_acquisition_time_duplicated(self):
        rng = np.random.default_rng(7)
        base = np.datetime64("2025-01-01").astype("datetime64[ns]").astype("int64")
        raw = base + rng.integers(0, 400 * 86_400, 20) * 10**9
        # Every instant carried by two catalog records, interleaved out of order.
        times = np.concatenate([raw, raw]).astype("datetime64[ns]")
        epochs = (base + rng.integers(0, 400 * 86_400, 30) * 10**9).astype("datetime64[ns]")
        self._check_oracle(epochs, times, np.timedelta64(2, "D"))
