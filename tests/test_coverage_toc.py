"""The root coverage sidecar's temporal section — spec §10, issue #480.

Three things are asserted here that the §7 conformance suite cannot: the
order-independence of the root fold, the composition rules the GET-union-PUT
seam applies, and the byte-identity of a NON-temporal store's root object —
the promise that a store with no temporal channel is untouched by this
revision. The committed ``temporal/`` fixture is the real-store end of it;
``minimal/`` is the absence end.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from mortie import span2toc, time2toc, toc_merge, toc_overlaps, toc_reduce

from zagg.coverage import refresh_root_coverage
from zagg.coverage_toc import (
    COVER_CAP,
    COVER_KEY,
    COVER_NAME,
    COVER_SPEC,
    ROOT_TOC_DELTA,
    TEMPORAL_COVERAGE_SPEC,
    TEMPORAL_DAY_ORDER,
    build_cover_section,
    build_temporal_section,
    cover_unchanged,
    cover_words,
    coverage_toc,
    coverage_toc_digest,
    load_cover,
    load_temporal_coverage,
    merge_cover_sections,
    merge_temporal_sections,
    quantize_words,
    read_cover,
    section_unchanged,
    shards_overlapping,
    temporal_fields,
    write_cover,
)
from zagg.hive import build_root_coverage, read_root_coverage, write_root_coverage

SPEC_DATA = Path(__file__).parent / "data" / "spec"
#: A day on the toc scale, in internal ns — enough to keep the synthetic
#: leaves below in visibly distinct campaign clusters.
DAY_NS = 86_400 * 10**9
#: An arbitrary but realistic base instant on the §8 internal-ns scale.
BASE_NS = 5_344_000_000_000_000_000


def _leaf(seed: int, n: int = 12):
    """A synthetic per-leaf contribution: ``(word, digest, times, cover)``.

    Shaped exactly like :func:`zagg.coverage_toc.read_leaf_temporal`'s return
    — a valid ``(k, 2)`` digest sorted by mean, its row-aligned toc words,
    the join over them, and their §10.5 quantized cover.
    """
    rng = np.random.default_rng(seed)
    starts = np.sort(BASE_NS + seed * 40 * DAY_NS + rng.integers(0, 30 * DAY_NS, n)).astype(
        np.uint64
    )
    # Both variants, deliberately: a single-instant centroid keeps its exact
    # timestamp word, a spanning one gets a conservative range.
    words = np.array(
        [
            int(time2toc(int(t))) if i % 3 == 0 else int(span2toc(int(t), int(t) + 3600 * 10**9))
            for i, t in enumerate(starts)
        ],
        dtype=np.uint64,
    )
    digest = np.empty((n, 2), dtype=np.float32)
    digest[:, 0] = starts.astype(np.float64)
    digest[:, 1] = rng.integers(1, 20, n).astype(np.float64)
    order = np.lexsort((words, digest[:, 0]))
    return int(toc_reduce(words)), digest[order], words[order], quantize_words(words)


def _contributions(seeds):
    return {f"1121{i}": [_leaf(s)] for i, s in enumerate(seeds)}


class TestSectionGrammar:
    """§10.1 — what a built section carries, and what it refuses to carry."""

    def test_required_keys_and_string_words(self):
        section = build_temporal_section(_contributions([1, 2, 3]), ["h_tdigest"])
        assert section["spec"] == TEMPORAL_COVERAGE_SPEC
        assert set(section) == {"spec", "source", "generated_at", "fields", "shards", "digest"}
        assert section["fields"] == ["h_tdigest"]
        assert all(isinstance(w, str) and w.isdigit() for w in section["shards"].values())
        assert section["digest"]["delta"] == ROOT_TOC_DELTA
        assert section["digest"]["element"] == {"dtype": "float32", "shape": [-1, 2]}

    def test_an_empty_walk_builds_no_section(self):
        assert build_temporal_section({}, []) is None
        assert build_temporal_section({}, ["h_tdigest"]) is None

    def test_several_window_leaves_reduce_to_one_shard_word(self):
        a, b = _leaf(1), _leaf(2)
        section = build_temporal_section({"11213": [a, b]}, ["h_tdigest"])
        assert set(section["shards"]) == {"11213"}
        assert int(section["shards"]["11213"]) == int(toc_merge(a[0], b[0]))

    def test_weight_conservation(self):
        contributions = _contributions([1, 2, 3])
        section = build_temporal_section(contributions, ["h_tdigest"])
        total = sum(
            float(part[1][:, 1].sum()) for parts in contributions.values() for part in parts
        )
        payload, _words = coverage_toc_digest({"temporal": section})
        assert float(payload[:, 1].sum()) == pytest.approx(total)
        assert section["digest"]["weight_total"] == pytest.approx(total)

    def test_the_root_words_reduce_to_the_join_of_every_shard_word(self):
        section = build_temporal_section(_contributions([1, 2, 3]), ["h_tdigest"])
        _payload, words = coverage_toc_digest({"temporal": section})
        assert int(toc_reduce(words)) == int(
            toc_reduce(np.array([int(w) for w in section["shards"].values()], dtype=np.uint64))
        )


class TestOrderIndependence:
    """§10.3 — the fold is ONE k-way merge, so leaf order cannot matter."""

    def test_permuting_the_leaves_reproduces_the_section(self):
        contributions = _contributions([4, 7, 11, 13, 17])
        forward = build_temporal_section(contributions, ["h_tdigest"])
        keys = list(contributions)
        orders = [list(reversed(keys)), [keys[i] for i in (2, 0, 4, 1, 3)]]
        for order in orders:
            other = build_temporal_section({k: contributions[k] for k in order}, ["h_tdigest"])
            assert other["shards"] == forward["shards"]
            # Byte-for-byte: both the digest and its companion words, which is
            # what "permutation-independent in every channel" means.
            assert other["digest"]["payload"] == forward["digest"]["payload"]
            assert other["digest"]["times"] == forward["digest"]["times"]

    def test_the_fold_compresses(self):
        # δ is provenance, not a promise about k (§10.3) — the k1 budget is
        # scale-free, not a hard cap — but the fold must actually compress:
        # a root digest the size of its inputs would be no summary at all.
        contributions = _contributions(range(1, 12))
        rows = sum(len(part[1]) for parts in contributions.values() for part in parts)
        section = build_temporal_section(contributions, ["h_tdigest"])
        assert 0 < section["digest"]["centroids"] < rows
        assert section["digest"]["delta"] == ROOT_TOC_DELTA


class TestComposition:
    """§10.4 — how two sections meet at the GET-union-PUT seam."""

    def test_tier_one_unions_elementwise(self):
        a = build_temporal_section({"11211": [_leaf(1)], "11212": [_leaf(2)]}, ["h_tdigest"])
        b = build_temporal_section({"11212": [_leaf(3)], "11213": [_leaf(4)]}, ["h_tdigest"])
        merged = merge_temporal_sections(a, b)
        assert set(merged["shards"]) == {"11211", "11212", "11213"}
        assert int(merged["shards"]["11212"]) == int(
            toc_merge(int(a["shards"]["11212"]), int(b["shards"]["11212"]))
        )
        # The join is idempotent: re-merging changes nothing.
        assert merge_temporal_sections(merged, merged)["shards"] == merged["shards"]

    def test_a_partial_producer_drops_the_digest(self):
        whole = build_temporal_section({"11211": [_leaf(1)], "11212": [_leaf(2)]}, ["h_tdigest"])
        partial = build_temporal_section({"11213": [_leaf(3)]}, ["h_tdigest"])
        merged = merge_temporal_sections(whole, partial)
        # Neither side's map covers the union, so neither digest can vouch for
        # the store — tier 1 stands, tier 2 goes.
        assert set(merged["shards"]) == {"11211", "11212", "11213"}
        assert "digest" not in merged

    def test_a_whole_covering_producer_replaces_the_digest(self):
        old = build_temporal_section({"11211": [_leaf(1)]}, ["h_tdigest"])
        new = build_temporal_section({"11211": [_leaf(9)]}, ["h_tdigest"])
        merged = merge_temporal_sections(old, new)
        assert merged["digest"]["payload"] == new["digest"]["payload"]

    def test_a_producer_with_no_section_leaves_the_standing_one_alone(self):
        standing = build_temporal_section(_contributions([1, 2]), ["h_tdigest"])
        assert merge_temporal_sections(standing, None) == standing
        assert merge_temporal_sections(None, standing) == standing
        assert merge_temporal_sections(None, None) is None

    def test_an_unknown_revision_on_the_standing_side_is_preserved(self):
        """§10.4: readers add revisions, they never drop them.

        The merge is the WRITE composer — a ``None`` return deletes the key —
        so an unreadable standing section must survive both a producer with
        nothing to say and one carrying this revision's section. Otherwise the
        older zagg in a mixed fleet is the one that wins.
        """
        good = build_temporal_section(_contributions([1]), ["h_tdigest"])
        future = {**good, "spec": "zagg-coverage-toc/2", "shards": {"99999": "1"}}
        assert merge_temporal_sections(future, None) == future
        assert merge_temporal_sections(future, good) == future
        # Incoming side: this revision cannot read it, so it contributes
        # nothing and the standing section stands.
        assert merge_temporal_sections(good, future) == good
        # Unmarked debris claims no revision and does not wedge the key shut.
        assert merge_temporal_sections({}, good) == good
        assert merge_temporal_sections({"shards": {"1": "2"}}, good) == good

    def test_section_unchanged(self):
        a = build_temporal_section({"11211": [_leaf(1)], "11212": [_leaf(2)]}, ["h_tdigest"])
        assert section_unchanged(a, None)
        assert section_unchanged(a, a)
        assert not section_unchanged(None, a)
        assert not section_unchanged(
            build_temporal_section({"11211": [_leaf(1)]}, ["h_tdigest"]), a
        )
        # A standing section this revision cannot read is preserved verbatim
        # by the merge, so composing over it changes nothing either — the
        # skip test must not churn the object on a mixed-version store.
        assert section_unchanged({"spec": "zagg-coverage-toc/2"}, a)

    def test_a_partial_producer_converges_instead_of_re_putting_forever(self):
        """The composed digest, not the built one, is what the skip test sees.

        A producer that walked one shard of a two-shard store always builds a
        digest, and §10.4 always drops it at the seam. Comparing the built
        section against the standing one therefore never converges; comparing
        the MERGE against it does, on the very next pass.
        """
        first = build_temporal_section({"11211": [_leaf(1)]}, ["h_tdigest"])
        second = build_temporal_section({"11212": [_leaf(2)]}, ["h_tdigest"])
        standing = merge_temporal_sections(first, second)
        assert "digest" not in standing  # neither producer covered the store
        assert second.get("digest") is not None  # ... yet the producer built one
        assert section_unchanged(standing, second)
        assert section_unchanged(standing, first)


class TestAbsence:
    """§10's standing posture: absence composes, and is never a refusal."""

    @pytest.mark.parametrize(
        "envelope",
        [
            None,
            {},
            {"spec": "morton-moc/1", "encoding": "ranges"},
            {"temporal": None},
            {"temporal": {"spec": "zagg-coverage-toc/2"}},
            "not a dict",
        ],
    )
    def test_readers_return_none_cleanly(self, envelope):
        assert load_temporal_coverage(envelope) is None
        assert coverage_toc(envelope) is None
        assert coverage_toc_digest(envelope) is None
        assert shards_overlapping(envelope, 0, 10**18) is None

    def test_a_block_whose_buffers_disagree_with_k_is_refused(self):
        """§10.3's MUST-check, on all three of the block's shape claims."""
        section = build_temporal_section(_contributions([1, 2]), ["h_tdigest"])
        block = section["digest"]
        for bad in (
            {"centroids": block["centroids"] + 1},
            {"centroids": None},
            {"times": build_temporal_section(_contributions([3]), ["h"])["digest"]["times"]},
        ):
            envelope = {"temporal": {**section, "digest": {**block, **bad}}}
            with pytest.raises(ValueError, match="row-aligned"):
                coverage_toc_digest(envelope)

    def test_a_section_without_a_digest_still_prunes(self):
        section = build_temporal_section(_contributions([1, 2]), ["h_tdigest"])
        section.pop("digest")
        envelope = {"temporal": section}
        assert coverage_toc_digest(envelope) is None
        assert set(coverage_toc(envelope)) == set(section["shards"])

    def test_a_store_declaring_no_temporal_field_has_no_fields(self):
        manifest = json.loads((SPEC_DATA / "minimal" / "morton_hive.json").read_text())
        assert temporal_fields(manifest) == {}
        assert temporal_fields(None) == {}
        assert temporal_fields({}) == {}

    def test_a_temporal_store_declares_its_sibling(self):
        manifest = json.loads((SPEC_DATA / "temporal" / "morton_hive.json").read_text())
        fields = temporal_fields(manifest)
        assert set(fields) == {"h_tdigest"}
        assert fields["h_tdigest"]["sibling"] == "h_tdigest_times"


class TestPartialReadsDropTheShard:
    """§10.2 — a LISTED shard's word contains every instant in that shard.

    A word joined over whichever window leaves happened to read does not, so
    a failed read costs the shard its map entry. Absent reads as *unknown*
    (still a candidate); listed-but-partial reads as a promise the section
    cannot keep.
    """

    def test_a_failed_window_leaf_drops_its_whole_shard(self, monkeypatch):
        import zagg.coverage_toc as toc_module
        from zagg.sweep import MocFamily

        def reader(leaf, *args, **kwargs):
            if leaf.endswith("_2020.zarr"):
                raise OSError("truncated companion")
            return _leaf(1)

        monkeypatch.setattr(toc_module, "read_leaf_temporal", reader)
        family = MocFamily()
        family._temporal_fields = {"h_tdigest": {"sibling": "h_tdigest_times"}}
        family._accumulate_temporal("root", "11213", "root/11213_2019.zarr", {})
        assert "11213" in family._temporal
        family._accumulate_temporal("root", "11213", "root/11213_2020.zarr", {})
        assert "11213" not in family._temporal
        # A later window that DOES read cannot resurrect a half-read shard.
        family._accumulate_temporal("root", "11213", "root/11213_2021.zarr", {})
        assert "11213" not in family._temporal
        # ... and the failure is scoped to its own shard.
        family._accumulate_temporal("root", "11214", "root/11214_2019.zarr", {})
        assert "11214" in family._temporal


class TestPruning:
    """§10.2 — the tier-1 predicate, conservative by the grammar's own law."""

    def test_windows_select_the_right_shards(self):
        contributions = _contributions([1, 5])
        section = build_temporal_section(contributions, ["h_tdigest"])
        envelope = {"temporal": section}
        for shard, parts in contributions.items():
            word = np.array([parts[0][0]], dtype=np.uint64)
            lo = int(np.asarray(parts[0][1][:, 0]).min()) - DAY_NS
            hi = int(np.asarray(parts[0][1][:, 0]).max()) + DAY_NS
            assert bool(np.asarray(toc_overlaps(word, lo, hi))[0])
            assert shard in shards_overlapping(envelope, lo, hi)

    def test_a_window_past_every_shard_selects_none(self):
        section = build_temporal_section(_contributions([1, 5]), ["h_tdigest"])
        far = BASE_NS + 10_000 * DAY_NS
        assert shards_overlapping({"temporal": section}, far, far + DAY_NS) == []


class TestOnCommittedStores:
    """End to end, on the §7 fixtures: the writer, and the absence pin."""

    def _copy(self, tmp_path, name):
        dst = tmp_path / name
        shutil.copytree(SPEC_DATA / name, dst)
        return str(dst)

    def _clone_shard(self, root, src="11213", dst="11214"):
        """Give ``root`` a second shard, cloned from its committed leaf.

        The §7 ``temporal/`` fixture ships ONE shard, which is exactly the
        shape that hides the composition seam: a single-shard producer is
        always whole-covering, so its digest survives the merge and the skip
        test converges by accident.
        """
        from zagg.grids.morton import morton_word
        from zagg.hive import shard_leaf_path

        src_leaf = Path(shard_leaf_path(root, int(morton_word(src))))
        dst_leaf = Path(shard_leaf_path(root, int(morton_word(dst))))
        dst_leaf.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src_leaf, dst_leaf)
        return [(int(morton_word(src)), None)], [(int(morton_word(dst)), None)]

    def test_refresh_rebuilds_the_section_from_its_own_walk(self, tmp_path):
        root = self._copy(tmp_path, "temporal")
        committed = json.loads((SPEC_DATA / "temporal" / "coverage.moc").read_text())
        envelope = refresh_root_coverage(root)
        assert envelope["temporal"]["source"] == "refresh"
        # Same walk, same words: only the carrier's provenance differs.
        assert envelope["temporal"]["shards"] == committed["temporal"]["shards"]
        assert (
            envelope["temporal"]["digest"]["payload"] == committed["temporal"]["digest"]["payload"]
        )

    def test_a_sweep_writes_the_section_the_fixture_committed(self, tmp_path):
        from zagg.grids.morton import morton_word
        from zagg.sweep import run_sweep

        root = self._copy(tmp_path, "temporal")
        (Path(root) / "coverage.moc").unlink()
        leaves = [(int(morton_word("11213")), None)]
        summary = run_sweep(root, leaves, families=["moc"], record=False)
        assert summary["families"]["moc"]["temporal_shards"] == 1
        committed = json.loads((SPEC_DATA / "temporal" / "coverage.moc").read_text())
        written = read_root_coverage(root)
        assert written["temporal"]["shards"] == committed["temporal"]["shards"]
        assert (
            written["temporal"]["digest"]["payload"] == committed["temporal"]["digest"]["payload"]
        )
        # Idempotence: a second pass over an unchanged tree writes nothing.
        again = run_sweep(root, leaves, families=["moc"], record=False)
        assert again["families"]["moc"]["root_moc_written"] is False

    def test_a_truncated_companion_is_refused(self, tmp_path):
        """§1.1 row alignment, at ARRAY level (issue #452's failure shape).

        A companion with fewer rows than its payload aligns row for row over
        its own length, so the per-cell check never fires — the leaf would
        join a prefix of its cells and be published as whole.
        """
        import zarr

        from zagg.coverage_toc import read_leaf_temporal
        from zagg.grids.morton import morton_word
        from zagg.hive import shard_leaf_path

        root = self._copy(tmp_path, "temporal")
        manifest = json.loads((Path(root) / "morton_hive.json").read_text())
        fields = temporal_fields(manifest)
        leaf = shard_leaf_path(root, int(morton_word("11213")))
        group = zarr.open_group(leaf, path=str(manifest["cell_order"]), mode="a", zarr_format=3)
        rows = group["h_tdigest_times"].shape[0]
        group["h_tdigest_times"].resize((rows - 1,))
        with pytest.raises(ValueError, match="row-aligned"):
            read_leaf_temporal(leaf, int(manifest["cell_order"]), fields)

    def test_refresh_drops_only_the_shard_whose_leaf_failed(self, tmp_path, monkeypatch):
        import zagg.coverage_toc as toc_module

        root = self._copy(tmp_path, "temporal")
        self._clone_shard(root)
        real = toc_module.read_leaf_temporal

        def reader(leaf, *args, **kwargs):
            if "11214" in leaf:
                raise OSError("truncated companion")
            return real(leaf, *args, **kwargs)

        monkeypatch.setattr(toc_module, "read_leaf_temporal", reader)
        envelope = refresh_root_coverage(root)
        # The spatial walk still lists both shards; the temporal map lists
        # only the one it could read whole (§10.2's unknown-not-empty rule).
        assert set(envelope["temporal"]["shards"]) == {"11213"}

    def test_the_shard_word_unions_across_every_temporal_field(self, tmp_path):
        """§10.2's headline rule: coverage is "any data", not "data in field X".

        The committed fixture declares ONE temporal field, so the union is
        invisible on it. A second field is grafted onto a copy of the leaf —
        the same payload rows under a companion whose words sit in a different
        campaign — and the shard word must be the join across both.
        """
        import zarr
        from mortie import time2toc

        from zagg.coverage_toc import read_leaf_temporal
        from zagg.grids.morton import morton_word
        from zagg.hive import shard_leaf_path

        root = self._copy(tmp_path, "temporal")
        manifest = json.loads((Path(root) / "morton_hive.json").read_text())
        order = int(manifest["cell_order"])
        leaf = shard_leaf_path(root, int(morton_word("11213")))
        group = zarr.open_group(leaf, path=str(order), mode="a", zarr_format=3)
        payload, sibling = group["h_tdigest"], group["h_tdigest_times"]
        # A companion of the same per-row width, so §1.1 alignment holds, but
        # carrying instants a whole campaign away from the committed ones.
        far = np.empty(sibling.shape[0], dtype=object)
        for i, row in enumerate(sibling[:]):
            width = 0 if row is None else len(row) // 8
            far[i] = np.array(
                [int(time2toc(BASE_NS + 20_000 * DAY_NS + j * DAY_NS)) for j in range(width)],
                dtype="<u8",
            ).tobytes()
        for name, values in (("g_tdigest", payload[:]), ("g_tdigest_times", far)):
            group.create_array(
                name,
                shape=payload.shape,
                chunks=payload.chunks,
                dtype=payload.metadata.data_type,
                overwrite=True,
            )[:] = values

        fields = temporal_fields(manifest)
        second = {"g_tdigest": {**fields["h_tdigest"], "sibling": "g_tdigest_times"}}
        one = read_leaf_temporal(leaf, order, fields)
        other = read_leaf_temporal(leaf, order, second)
        both = read_leaf_temporal(leaf, order, {**fields, **second})
        assert both[0] == int(toc_reduce(np.array([one[0], other[0]], dtype=np.uint64)))
        assert both[0] not in (one[0], other[0])  # neither field alone covers it
        section = build_temporal_section({"11213": [both]}, ["g_tdigest", "h_tdigest"])
        assert int(section["shards"]["11213"]) == both[0]
        # §10.3's once-per-field counting rule, seen from the weight side.
        assert section["digest"]["weight_total"] == pytest.approx(
            float(one[1][:, 1].sum()) + float(other[1][:, 1].sum())
        )

    def test_a_manifest_without_a_cell_order_publishes_no_section(self, tmp_path, caplog):
        """A required key missing is a broken manifest, not group ``"0"``.

        Defaulting either asks for a group that does not exist — logged as an
        ordinary missing contribution, which hides the real problem — or, on a
        store whose cell order really is 0, reads the wrong group.
        """
        from zagg.grids.morton import morton_word
        from zagg.sweep import run_sweep

        root = self._copy(tmp_path, "temporal")
        path = Path(root) / "morton_hive.json"
        manifest = json.loads(path.read_text())
        del manifest["cell_order"]
        path.write_text(json.dumps(manifest, indent=1))
        (Path(root) / "coverage.moc").unlink()
        with caplog.at_level("WARNING"):
            summary = run_sweep(
                root, [(int(morton_word("11213")), None)], families=["moc"], record=False
            )
            assert "temporal_shards" not in summary["families"]["moc"]
            assert "temporal" not in read_root_coverage(root)
            assert refresh_root_coverage(root).get("temporal") is None
        assert caplog.text.count("cell_order") >= 2

    def test_refresh_never_deletes_the_section_when_every_leaf_fails(self, tmp_path, monkeypatch):
        """The escape hatch must not be the thing that destroys the section.

        Fail-open per leaf is fail-DESTRUCTIVE in aggregate: refresh PUTs its
        envelope outright, so an all-failed walk would publish a root object
        with the ``temporal`` key gone — during exactly the incident an
        operator reached for refresh to repair.
        """
        import zagg.coverage_toc as toc_module

        root = self._copy(tmp_path, "temporal")
        standing = json.loads((Path(root) / "coverage.moc").read_text())["temporal"]

        def reader(*args, **kwargs):
            raise OSError("credentials expired mid-walk")

        monkeypatch.setattr(toc_module, "read_leaf_temporal", reader)
        envelope = refresh_root_coverage(root)
        assert envelope["ranges"]  # the spatial refresh still succeeded
        assert envelope["temporal"] == standing
        assert coverage_toc(envelope) == coverage_toc({"temporal": standing})

    def test_refresh_composes_a_partial_rebuild_with_the_standing_section(
        self, tmp_path, monkeypatch
    ):
        import zagg.coverage_toc as toc_module

        root = self._copy(tmp_path, "temporal")
        self._clone_shard(root)
        refresh_root_coverage(root)  # both shards land in the standing section
        real = toc_module.read_leaf_temporal

        def reader(leaf, *args, **kwargs):
            if "11214" in leaf:
                raise OSError("truncated companion")
            return real(leaf, *args, **kwargs)

        monkeypatch.setattr(toc_module, "read_leaf_temporal", reader)
        envelope = refresh_root_coverage(root)
        # The shard the walk could not read keeps the word the last whole walk
        # published: a partial rebuild composes, it does not overwrite.
        assert set(envelope["temporal"]["shards"]) == {"11213", "11214"}

    def test_a_second_pass_over_a_multi_shard_store_writes_nothing(self, tmp_path):
        """Sweep idempotence where the seam actually bites (§10.4).

        Two shards, two incremental sweeps: neither producer covers the store,
        so the composed section carries no digest while every pass keeps
        building one. The skip test has to converge on what was WRITTEN, or
        the fleet re-PUTs a byte-identical root object forever.
        """
        from zagg.sweep import run_sweep

        root = self._copy(tmp_path, "temporal")
        (Path(root) / "coverage.moc").unlink()
        a, b = self._clone_shard(root)
        for leaves in (a, b):
            summary = run_sweep(root, leaves, families=["moc"], record=False)
            assert summary["families"]["moc"]["root_moc_written"] is True
        written = read_root_coverage(root)
        assert set(written["temporal"]["shards"]) == {"11213", "11214"}
        assert "digest" not in written["temporal"]
        for leaves in (b, a):
            again = run_sweep(root, leaves, families=["moc"], record=False)
            assert again["families"]["moc"]["root_moc_written"] is False
        assert read_root_coverage(root)["temporal"] == written["temporal"]

    def test_a_non_temporal_store_writes_byte_identical_bytes(self, tmp_path):
        """The §10 absence promise, as bytes.

        A store declaring no temporal field must produce EXACTLY the root
        object a pre-#480 zagg produced: no key added, no key reordered.
        """
        from zagg.grids.morton import morton_word
        from zagg.sweep import run_sweep

        root = self._copy(tmp_path, "minimal")
        summary = run_sweep(
            root, [(int(morton_word("11213")), None)], families=["moc"], record=False
        )
        assert "temporal_shards" not in summary["families"]["moc"]
        raw = (Path(root) / "coverage.moc").read_bytes()
        envelope = json.loads(raw)
        assert "temporal" not in envelope
        assert coverage_toc(envelope) is None
        # The reference: the same carrier built with the §10 parameter omitted
        # entirely — the pre-#480 call — serialized the pre-#480 way. Equal
        # bytes is the whole promise.
        reference = build_root_coverage([morton_word("11213")], 4, source="sweep")
        reference["generated_at"] = envelope["generated_at"]
        assert json.dumps(reference, indent=1).encode() == raw
        # And a re-write through the GET-union-PUT seam stays temporal-free.
        assert "temporal" not in write_root_coverage(root, reference)
        assert "temporal" not in json.loads((Path(root) / "coverage.moc").read_bytes())


#: One §10.5 day-order bucket, in internal ns (order 16 -> span 2^47).
BUCKET_NS = 1 << (63 - TEMPORAL_DAY_ORDER)


class TestQuantization:
    """§10.5's quantization law: widening only, gap-preserving, commuting."""

    def _instants(self, n_days=49, per_day=200, seed=489):
        rng = np.random.default_rng(seed)
        days = np.sort(rng.choice(2_700, n_days, replace=False))
        ts = np.concatenate(
            [BASE_NS + int(d) * DAY_NS + rng.integers(0, 20 * 60 * 10**9, per_day) for d in days]
        ).astype(np.uint64)
        return days, ts

    def test_the_cover_contains_every_instant(self):
        from mortie import toc2time

        _days, ts = self._instants()
        cover = quantize_words(time2toc(ts))
        start, end = toc2time(cover)
        assert all(np.any((start <= t) & (t < end)) for t in ts[::97])

    def test_buckets_are_aligned_at_the_pinned_order(self):
        from mortie import toc2time

        _days, ts = self._instants()
        start, end = toc2time(quantize_words(time2toc(ts)))
        assert np.all(np.asarray(start, np.uint64) % BUCKET_NS == 0)
        assert np.all(np.asarray(end, np.uint64) % BUCKET_NS == 0)

    def test_a_range_word_widens_to_its_buckets(self):
        word = span2toc(BASE_NS + 10, BASE_NS + BUCKET_NS)
        from mortie import toc2time

        start, end = toc2time(quantize_words([word]))
        assert int(start[0]) == (BASE_NS // BUCKET_NS) * BUCKET_NS
        assert int(end[0]) - int(start[0]) >= 2 * BUCKET_NS  # widened, never shrunk

    def test_the_pass_day_shape_compresses_to_the_day_clusters(self):
        # The CA store's shard 3231242244 shape, scaled: millions of exact
        # timestamps clustering into ~49 distinct pass-days must land at
        # ~one word per cluster, not one per instant.
        days, ts = self._instants()
        cover = quantize_words(time2toc(ts))
        assert len(cover) <= len(days)

    def test_days_far_from_any_pass_stay_uncovered(self):
        # Named for what it checks: days at least 3 clear of any pass, i.e.
        # well past the two-span floor the law above pins exactly.
        days, ts = self._instants()
        cover = quantize_words(time2toc(ts))
        far = [int(d) for d in range(2_700) if np.abs(days - d).min() >= 3][:60]
        assert far
        for d in far:
            q0, q1 = BASE_NS + d * DAY_NS, BASE_NS + (d + 1) * DAY_NS
            assert not np.any(toc_overlaps(cover, q0, q1))

    @pytest.mark.parametrize(
        ("frac", "spans", "survives"),
        [
            (0.9, 1.0, False),  # abutting buckets coalesce: one span never survives
            (0.0, 1.0, False),
            (0.0, 1.99, False),  # the [1, 2) band is alignment-decided, not length-decided
            (0.5, 1.5, True),
            (0.0, 2.0, True),  # two whole spans always leave a bucket free
            (0.9, 2.0, True),
        ],
    )
    def test_a_gap_survives_iff_it_holds_a_whole_aligned_bucket(self, frac, spans, survives):
        # §10.5's only resolution promise, pinned as bytes in both directions:
        # the guaranteed floor is TWO bucket spans (2 * 2^47 ns ~ 78 h), not one.
        t0 = (BASE_NS // BUCKET_NS) * BUCKET_NS + int(frac * BUCKET_NS)
        t1 = t0 + int(spans * BUCKET_NS)
        cover = quantize_words(time2toc(np.array([t0, t1], dtype=np.uint64)))
        assert (len(cover) == 2) is survives
        # Either way the law never false-negatives on the instants themselves.
        assert np.any(toc_overlaps(cover, t0, t0 + 1))
        assert np.any(toc_overlaps(cover, t1, t1 + 1))

    def test_quantization_commutes_with_union(self):
        _days, ts = self._instants()
        words = time2toc(ts)
        a, b = words[: len(words) // 2], words[len(words) // 2 :]
        joint = quantize_words(words)
        parts = quantize_words(np.concatenate([quantize_words(a), quantize_words(b)]))
        assert np.array_equal(joint, parts)

    def test_quantization_commutes_with_the_envelope_join(self):
        _days, ts = self._instants()
        words = time2toc(ts)
        lhs = int(toc_reduce(quantize_words(words)))
        rhs = int(toc_reduce(quantize_words([toc_reduce(words)])))
        assert lhs == rhs

    def test_idempotent_at_the_same_order(self):
        _days, ts = self._instants()
        once = quantize_words(time2toc(ts))
        assert np.array_equal(once, quantize_words(once))

    def test_the_scale_ceiling_clamps_without_losing_containment(self):
        from mortie import TOC_MAX_NS, toc2time

        word = span2toc(TOC_MAX_NS - 2 * BUCKET_NS, TOC_MAX_NS - 10**9)
        start, end = toc2time(quantize_words([word]))
        w_start, w_end = toc2time(np.asarray([word], np.uint64))
        assert int(start[0]) <= int(w_start[0]) and int(w_end[0]) <= int(end[0])

    def test_orders_outside_the_grammar_are_refused(self):
        with pytest.raises(ValueError, match="temporal order"):
            quantize_words([time2toc(BASE_NS)], 32)
        with pytest.raises(ValueError, match="temporal order"):
            quantize_words([time2toc(BASE_NS)], -1)

    def test_empty_in_empty_out(self):
        assert len(quantize_words(np.array([], dtype=np.uint64))) == 0


class TestCoverSection:
    """§10.5's object grammar, from the same contributions the section folds."""

    def test_grammar_and_decode_round_trip(self):
        contributions = _contributions([1, 2, 3])
        section = build_cover_section(contributions, ["h_tdigest"], 4)
        assert section["spec"] == COVER_SPEC
        assert section["order"] == 4
        assert section["temporal_order"] == TEMPORAL_DAY_ORDER
        assert section["cap"] == COVER_CAP
        assert section["element"] == {"dtype": "uint64", "shape": [-1]}
        decoded = cover_words(section)
        assert set(decoded) == set(contributions)
        for decimal, parts in contributions.items():
            expect = quantize_words(np.concatenate([p[3] for p in parts]))
            assert np.array_equal(decoded[decimal], expect)

    def test_an_empty_walk_builds_no_object(self):
        assert build_cover_section({}, [], 4) is None

    def test_window_leaves_union_into_one_shard_block(self):
        parts = [_leaf(3), _leaf(9)]
        section = build_cover_section({"11213": parts}, ["h"], 4)
        expect = quantize_words(np.concatenate([p[3] for p in parts]))
        assert np.array_equal(cover_words(section)["11213"], expect)

    def test_the_cap_coarsens_by_order_and_records_it(self, caplog):
        # 600 instants two buckets apart: 600 words at the pinned order,
        # over the 512 cap; one coarsening step (span doubles) lands at 300.
        ts = (BASE_NS + np.arange(600, dtype=np.uint64) * np.uint64(2 * BUCKET_NS)).astype(
            np.uint64
        )
        cover = quantize_words(time2toc(ts))
        assert len(cover) == 600
        contributions = {"11213": [(int(toc_reduce(cover)), np.empty((0, 2)), [], cover)]}
        with caplog.at_level("WARNING"):
            section = build_cover_section(contributions, ["h"], 4)
        block = section["shards"]["11213"]
        assert block["temporal_order"] == TEMPORAL_DAY_ORDER - 1
        assert block["count"] <= COVER_CAP
        assert "coarsened" in caplog.text
        # Widening only: every original instant is still covered.
        assert np.all(toc_overlaps(cover_words(section)["11213"], int(ts[0]), int(ts[-1]) + 1))

    def test_parity_with_the_tier_one_map(self):
        contributions = _contributions([1, 5, 11])
        section = build_temporal_section(contributions, ["h"], source="sweep")
        cover = build_cover_section(contributions, ["h"], 4)
        for decimal, block in cover["shards"].items():
            order = block.get("temporal_order", TEMPORAL_DAY_ORDER)
            words = cover_words(cover)[decimal]
            tier1 = int(section["shards"][decimal])
            assert int(toc_reduce(words)) == int(toc_reduce(quantize_words([tier1], order)))

    def test_a_block_whose_count_disagrees_is_refused(self):
        section = build_cover_section(_contributions([1]), ["h"], 4)
        (decimal,) = section["shards"]
        section["shards"][decimal]["count"] += 1
        with pytest.raises(ValueError, match="declares"):
            cover_words(section)


class TestCoverComposition:
    """§10.5's seam: per-shard union, requantize at the coarser order, re-cap."""

    def test_merge_unions_per_shard_and_carries_singletons(self):
        a = build_cover_section(_contributions([1, 2]), ["h"], 4)
        b = build_cover_section({"11211": [_leaf(7)], "11219": [_leaf(4)]}, ["g"], 4)
        merged = merge_cover_sections(a, b)
        assert set(merged["shards"]) == {"11210", "11211", "11219"}
        assert merged["fields"] == ["g", "h"]
        union = quantize_words(np.concatenate([_leaf(2)[3], _leaf(7)[3]]))
        assert np.array_equal(cover_words(merged)["11211"], union)
        assert np.array_equal(cover_words(merged)["11210"], cover_words(a)["11210"])

    def test_mixed_orders_requantize_at_the_coarser(self):
        fine = build_cover_section({"11213": [_leaf(1)]}, ["h"], 4)
        coarse = build_cover_section({"11213": [_leaf(2)]}, ["h"], 4)
        coarse["shards"]["11213"] = dict(coarse["shards"]["11213"])
        # Simulate a capped producer: re-encode the block at order 14.
        words14 = quantize_words(cover_words(coarse)["11213"], 14)
        from zagg.coverage_toc import _encode_cover_block

        coarse["shards"]["11213"] = _encode_cover_block(words14, 14)
        merged = merge_cover_sections(fine, coarse)
        block = merged["shards"]["11213"]
        assert block["temporal_order"] == 14
        expect = quantize_words(np.concatenate([cover_words(fine)["11213"], words14]), 14)
        assert np.array_equal(cover_words(merged)["11213"], expect)

    def test_a_standing_cover_at_another_shard_order_is_replaced(self, caplog):
        # D1 ids at two orders are not comparable, so the seam behaves like
        # the carrier's incompatible-envelope arm: the incoming side wins.
        a = build_cover_section(_contributions([1, 2]), ["h"], 4)
        b = build_cover_section({"11219": [_leaf(4)]}, ["g"], 5)
        with caplog.at_level("WARNING"):
            merged = merge_cover_sections(a, b)
        assert merged == b
        assert set(merged["shards"]) == {"11219"}
        assert merged["fields"] == ["g"]
        assert "not comparable" in caplog.text
        # And the same shard order still unions, so the gate is the only change.
        same = build_cover_section({"11219": [_leaf(4)]}, ["g"], 4)
        assert set(merge_cover_sections(a, same)["shards"]) == {"11210", "11211", "11219"}

    def test_an_unknown_incoming_revision_contributes_nothing(self):
        a = build_cover_section(_contributions([1]), ["h"], 4)
        assert merge_cover_sections(a, {"spec": "zagg-coverage-toc-cover/9"}) == a

    def test_an_unknown_standing_revision_is_preserved(self):
        b = build_cover_section(_contributions([1]), ["h"], 4)
        future = {"spec": "zagg-coverage-toc-cover/9", "shards": {}}
        assert merge_cover_sections(future, b) == future
        assert merge_cover_sections(future, None) == future

    def test_unmarked_debris_is_replaced(self):
        b = build_cover_section(_contributions([1]), ["h"], 4)
        assert merge_cover_sections({"shards": "junk"}, b) == b

    def test_cover_unchanged_converges(self):
        a = build_cover_section(_contributions([1, 2]), ["h"], 4)
        assert not cover_unchanged(None, a)
        merged = merge_cover_sections(None, a)
        assert cover_unchanged(merged, a)
        b = build_cover_section({"11219": [_leaf(4)]}, ["h"], 4)
        assert not cover_unchanged(merged, b)
        assert cover_unchanged(merge_cover_sections(merged, b), b)


class TestCoverObject:
    """The sibling object's transport: GET-union-PUT, replace, preservation."""

    def test_write_read_round_trip_accumulates(self, tmp_path):
        root = str(tmp_path)
        a = build_cover_section(_contributions([1]), ["h"], 4)
        b = build_cover_section({"11219": [_leaf(4)]}, ["h"], 4)
        write_cover(root, a)
        write_cover(root, b)
        standing = read_cover(root)
        assert set(standing["shards"]) == {"11210", "11219"}
        assert (tmp_path / COVER_NAME).exists()

    def test_replace_discards_the_standing_object(self, tmp_path):
        root = str(tmp_path)
        write_cover(root, build_cover_section(_contributions([1]), ["h"], 4))
        b = build_cover_section({"11219": [_leaf(4)]}, ["h"], 4)
        write_cover(root, b, replace=True)
        assert set(read_cover(root)["shards"]) == {"11219"}

    def test_replace_never_downgrades_a_future_revision(self, tmp_path):
        root = str(tmp_path)
        future = {"spec": "zagg-coverage-toc-cover/9", "shards": {}}
        (tmp_path / COVER_NAME).write_text(json.dumps(future))
        b = build_cover_section(_contributions([1]), ["h"], 4)
        write_cover(root, b, replace=True)
        assert read_cover(root) == future
        write_cover(root, b)
        assert read_cover(root) == future

    def test_no_contribution_leaves_the_standing_object_untouched(self, tmp_path):
        # The refresh escape hatch over a store with NO temporal channel:
        # `build_cover_section` answers None and `replace=True` must be a
        # no-op on that arm too, not a crash and not an overwrite (§10.5).
        root = str(tmp_path)
        a = build_cover_section(_contributions([1]), ["h"], 4)
        write_cover(root, a)
        raw = (tmp_path / COVER_NAME).read_bytes()
        assert write_cover(root, None, replace=True) == read_cover(root)
        assert (tmp_path / COVER_NAME).read_bytes() == raw
        assert write_cover(root, None) == read_cover(root)
        assert (tmp_path / COVER_NAME).read_bytes() == raw

    def test_no_contribution_on_an_empty_root_writes_nothing(self, tmp_path):
        assert write_cover(str(tmp_path), None, replace=True) is None
        assert write_cover(str(tmp_path), None) is None
        assert not (tmp_path / COVER_NAME).exists()

    def test_garbage_is_overwritten(self, tmp_path):
        root = str(tmp_path)
        (tmp_path / COVER_NAME).write_text("not json {")
        b = build_cover_section(_contributions([1]), ["h"], 4)
        write_cover(root, b)
        assert set(read_cover(root)["shards"]) == {"11210"}

    @pytest.mark.parametrize("damage", ["count", "no-words", "bad-words", "not-an-object"])
    def test_a_corrupt_standing_cover_is_replaced(self, tmp_path, caplog, damage):
        # JSON-valid but not a decodable cover. The regenerable-cache posture
        # is the same as for garbage bytes: log and overwrite, never take the
        # sweep's spatial rollup down with an accelerator (§10.5).
        root = str(tmp_path)
        standing = build_cover_section(_contributions([1]), ["h"], 4)
        (decimal,) = standing["shards"]
        if damage == "count":
            standing["shards"][decimal]["count"] += 1
        elif damage == "no-words":
            del standing["shards"][decimal]["words"]
        elif damage == "bad-words":
            standing["shards"][decimal]["words"] = "junk"
        else:
            standing["shards"][decimal] = "junk"
        (tmp_path / COVER_NAME).write_text(json.dumps(standing))
        incoming = build_cover_section({"11219": [_leaf(4)]}, ["h"], 4)
        # The skip-if-current test runs BEFORE the writer, so it must fail
        # open too rather than raise on the way in.
        assert not cover_unchanged(read_cover(root), incoming)
        with caplog.at_level("WARNING"):
            write_cover(root, incoming)
        assert "failed to parse" in caplog.text
        assert set(read_cover(root)["shards"]) == {"11219"}

    def test_the_read_accessor_stays_loud_on_a_corrupt_block(self):
        # Fail-open is the WRITE seam's posture only: a reader handed a block
        # that disagrees with its own buffer still gets the §10.5 MUST-check.
        section = build_cover_section(_contributions([1]), ["h"], 4)
        (decimal,) = section["shards"]
        del section["shards"][decimal]["words"]
        with pytest.raises(KeyError):
            cover_words(section)

    def test_absent_reads_none(self, tmp_path):
        assert read_cover(str(tmp_path)) is None
        assert load_cover(None) is None
        assert load_cover({"spec": "zagg-coverage-toc-cover/9"}) is None
        assert cover_words(None) is None


class TestCoverMarker:
    """§10.1's `cover` key: carried through the section merge, never invented."""

    def test_the_marker_survives_the_seam(self):
        a = build_temporal_section(_contributions([1]), ["h"], source="sweep")
        a[COVER_KEY] = COVER_SPEC
        b = build_temporal_section(_contributions([2]), ["h"], source="sweep")
        assert merge_temporal_sections(a, b)[COVER_KEY] == COVER_SPEC
        assert merge_temporal_sections(b, a)[COVER_KEY] == COVER_SPEC
        assert merge_temporal_sections(None, a)[COVER_KEY] == COVER_SPEC
        assert merge_temporal_sections(a, None)[COVER_KEY] == COVER_SPEC

    def test_no_marker_no_key(self):
        a = build_temporal_section(_contributions([1]), ["h"], source="sweep")
        b = build_temporal_section(_contributions([2]), ["h"], source="sweep")
        assert COVER_KEY not in merge_temporal_sections(a, b)

    def test_a_marker_change_is_a_content_change(self):
        a = build_temporal_section(_contributions([1]), ["h"], source="sweep")
        merged = merge_temporal_sections(None, a)
        assert section_unchanged(merged, a)
        marked = dict(a)
        marked[COVER_KEY] = COVER_SPEC
        assert not section_unchanged(merged, marked)
