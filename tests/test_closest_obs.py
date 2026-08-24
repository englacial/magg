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
from mortie import time2toc, to_datetime64

from zagg.catalog.closest_obs import ReferenceEpochs, _word_midpoints, reference_epochs
from zagg.coverage_toc import (
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


class TestGoldenFixture:
    """The committed §7 ``temporal/`` fixture pins the frozen grammar bytes."""

    def test_the_golden_cover_decodes_to_epochs(self):
        out = reference_epochs(str(SPEC_DATA / "temporal"))
        assert out.order == 4
        assert list(out.epochs) == [SHARD_KEY]
        epochs = out.epochs[SHARD_KEY]
        assert epochs.size > 0
        # Every epoch is one committed cover word's midpoint, exactly.
        words = cover_words(read_cover(str(SPEC_DATA / "temporal")))[SHARD]
        assert np.array_equal(epochs, np.unique(_word_midpoints(words)))

    def test_the_golden_epochs_sit_inside_the_fixture_campaign(self):
        """Midpoints land inside the leaf's synthetic campaign window."""
        out = reference_epochs(str(SPEC_DATA / "temporal"))
        lo = np.datetime64("2019-01-01")
        hi = np.datetime64("2020-01-01")
        e = out.epochs[SHARD_KEY]
        assert ((e > lo) & (e < hi)).all()
