"""``2^n`` morton-subtree sweep partitions (issue #377): the decomposition.

The standing claims this file pins. On the decomposition itself: the split
rounds to morton DIGIT boundaries (a power of four) and rejects an odd ``2^n``
by name; ownership is a pure prefix test over the D1 decimal id; and the
decomposition is a partition in the set-theoretic sense — disjoint and
covering. On a partitioned pass: no two partitions write the same object, the
union of every partition equals an unpartitioned pass minus the coarse nodes
above the split, the ``finish`` hook and the coarse overview orders are
deferred, and each partition stays independently idempotent and resumable.
"""

import itertools
import json

import obstore
import pytest

from zagg import sweep as sweep_mod
from zagg.grids.morton import morton_decimal, morton_word
from zagg.hive import MANIFEST_NAME, _decimal_order, shard_leaf_path
from zagg.store import open_object_store
from zagg.sweep import run_sweep
from zagg.sweep_partition import (
    normalize_partition,
    partition_index,
    partition_leaves,
    partition_split_order,
    select_partition,
    sweep_partitions,
)
from zagg.telemetry import build_record, write_sidecar

#: Every order-2 node of one base cell, plus a second base cell's, so the
#: "one order-k subtree per base cell" property is exercised, not assumed.
NODES = [f"{b}{i}{j}" for b in ("-3", "1") for i in "1234" for j in "1234"]

SHARD_ORDER = 2

#: The unpatched rollup writer, captured once so a re-installed spy never nests.
_PUT_ROLLUP = sweep_mod._put_rollup

#: Leaves spread over three of the four order-1 subtrees AND two base cells,
#: so a partitions=4 (split order 1) pass exercises a partition with several
#: shards, one with a single shard, one spanning two base cells, and an empty
#: one — the four cases a fan-out actually meets.
LEAVES = ("-311", "-312", "-321", "-341", "141", "142")


def _write_manifest(root, *, pyramid=None):
    manifest = {
        "spec": "morton-hive/1",
        "dataset": {"short_name": "TEST", "version": "1"},
        "cell_order": SHARD_ORDER + 2,
        "shard_order": SHARD_ORDER,
        "split_schedule": [1] * SHARD_ORDER,
        "pyramid": pyramid if pyramid is not None else {"orders": [], "aggregation": {}},
        "generated_at": "2026-01-01T00:00:00+00:00",
    }
    obstore.put(open_object_store(str(root)), MANIFEST_NAME, json.dumps(manifest).encode())


def _put_leaf(root, decimal, *, n_obs=10):
    record = build_record(
        shard_key=morton_word(decimal),
        metadata={"total_obs": n_obs, "cells_with_data": 2, "duration_s": 0.5},
        granule_ids=[f"g-{decimal}"],
    )
    write_sidecar(shard_leaf_path(str(root), morton_word(decimal)), record)
    return record


def _store(tmp_path, decimals=LEAVES, **kwargs):
    _write_manifest(tmp_path, **kwargs)
    for decimal in decimals:
        _put_leaf(tmp_path, decimal)
    return [(morton_word(d), None) for d in decimals]


def _written(monkeypatch):
    """Capture every rollup object key a pass PUTs (the write-conflict probe).

    Always delegates to the PRISTINE writer, so calling this twice in one test
    replaces the spy rather than nesting it — each returned list holds exactly
    the keys of the pass it was installed for.
    """
    keys: list[str] = []

    def spy(store, fam, decimal, envelope):
        keys.append(sweep_mod._rollup_key(fam, decimal))
        return _PUT_ROLLUP(store, fam, decimal, envelope)

    monkeypatch.setattr(sweep_mod, "_put_rollup", spy)
    return keys


class TestSplitOrder:
    def test_powers_of_four_split_on_a_digit(self):
        assert [partition_split_order(4**k) for k in range(5)] == [0, 1, 2, 3, 4]

    def test_odd_power_of_two_is_rejected_by_name(self):
        # The design fork: 2^n with odd n halves a 2-bit digit. Conservative
        # option ruled here — reject, naming both valid neighbours (issue #377).
        with pytest.raises(ValueError, match=r"splits a morton digit in half") as excinfo:
            partition_split_order(8)
        assert "use 4 or 16" in str(excinfo.value)

    @pytest.mark.parametrize("bad", [0, -4, 3, 6, 12])
    def test_non_power_of_two_is_rejected(self, bad):
        with pytest.raises(ValueError, match=r"power of two"):
            partition_split_order(bad)

    def test_one_partition_is_the_identity(self):
        # partitions=1 -> split order 0 -> a single unit owning the whole tree,
        # which is exactly today's unpartitioned sweep.
        assert partition_split_order(1) == 0
        assert {partition_index(d, 1) for d in NODES} == {0}


class TestOwnership:
    def test_index_is_the_prefix_rank(self):
        # partitions=4 splits at order 1: the first digit IS the index.
        assert [partition_index(f"-3{d}", 4) for d in "1234"] == [0, 1, 2, 3]
        assert [partition_index(f"1{d}77", 4) for d in "1234"] == [0, 1, 2, 3]

    def test_a_node_and_every_descendant_share_a_partition(self):
        for node in NODES:
            deep = [node + tail for tail in ("", "1", "4", "1234", "4321")]
            assert len({partition_index(d, 16) for d in deep}) == 1

    def test_one_subtree_per_base_cell(self):
        # 12 base cells x 4^k order-k subtrees / 4^k partitions == 12 nodes each.
        bases = [f"{s}{b}" for s in ("", "-") for b in "123456"]
        owned: dict[int, list[str]] = {}
        for base, i, j in itertools.product(bases, "1234", "1234"):
            owned.setdefault(partition_index(f"{base}{i}{j}", 16), []).append(f"{base}{i}{j}")
        assert sorted(owned) == list(range(16))
        assert all(len(v) == 12 for v in owned.values())

    def test_coarser_than_the_split_is_refused(self):
        # An order-1 node under a 16-way (order-2) split spans four partitions;
        # it is the finisher's, and asking which partition owns it is an error.
        with pytest.raises(ValueError, match=r"coarser than the partitions=16 split order 2"):
            partition_index("-31", 16)
        # Exactly at the split order is fine — that node is wholly owned.
        assert _decimal_order("-311") == 2 and partition_index("-311", 16) == 0


class TestNormalizePartition:
    def test_none_passes_through(self):
        assert normalize_partition(None) is None

    def test_valid_block_round_trips(self):
        assert normalize_partition({"index": 3, "of": 16}) == (3, 16)
        assert normalize_partition({"index": "3", "of": "16"}) == (3, 16)

    @pytest.mark.parametrize("block", [{"of": 4}, {"index": 0}, {"index": None, "of": 4}, 4, "0/4"])
    def test_malformed_block_is_refused(self, block):
        with pytest.raises(ValueError, match=r"partition must be"):
            normalize_partition(block)

    @pytest.mark.parametrize("index", [-1, 4, 99])
    def test_index_out_of_range_is_refused(self, index):
        with pytest.raises(ValueError, match=r"out of range"):
            normalize_partition({"index": index, "of": 4})

    def test_digit_boundary_rule_is_enforced_worker_side_too(self):
        # A hand-rolled event carrying an odd 2^n must not sweep the wrong
        # subtree: the same validator runs where the fold happens.
        with pytest.raises(ValueError, match=r"splits a morton digit in half"):
            normalize_partition({"index": 0, "of": 8})


class TestPartitionLeaves:
    def _refs(self, decimals, window=None):
        return [(morton_word(d), window) for d in decimals]

    def test_split_is_disjoint_and_covering(self):
        refs = self._refs(NODES)
        buckets = partition_leaves(refs, 16)
        assert len(buckets) == 16
        flat = [r for b in buckets for r in b]
        assert sorted(flat) == sorted(refs)  # covering, and no duplication
        for a, b in itertools.combinations(buckets, 2):
            assert not set(a) & set(b)

    def test_every_ref_lands_in_its_own_index(self):
        for index, bucket in enumerate(partition_leaves(self._refs(NODES), 16)):
            for key, _window in bucket:
                assert partition_index(morton_decimal(key), 16) == index

    def test_windows_ride_along_and_bare_keys_normalize(self):
        buckets = partition_leaves([morton_word("-311"), (morton_word("-321"), "2019")], 4)
        assert buckets[0] == [(morton_word("-311"), None)]
        assert buckets[1] == [(morton_word("-321"), "2019")]

    def test_empty_partitions_keep_their_slot(self):
        buckets = partition_leaves(self._refs(["-311"]), 16)
        assert len(buckets) == 16 and buckets[0] and not any(buckets[1:])

    def test_one_partition_returns_the_whole_work_set(self):
        refs = self._refs(NODES)
        assert partition_leaves(refs, 1) == [refs]


class TestSelectPartition:
    def test_filters_and_counts_foreign_shards(self):
        by_shard = {d: {None} for d in NODES}
        kept, foreign = select_partition(by_shard, 16, 0)
        assert set(kept) == {d for d in NODES if partition_index(d, 16) == 0}
        assert foreign == len(NODES) - len(kept)

    def test_union_over_indices_is_the_whole_work_set(self):
        by_shard = {d: {None} for d in NODES}
        seen: set[str] = set()
        for index in range(16):
            kept, _foreign = select_partition(by_shard, 16, index)
            assert not seen & set(kept)
            seen |= set(kept)
        assert seen == set(by_shard)


class TestPartitionsValidatedAsAParameter:
    """A bad ``partitions`` is a PARAMETER error, not a per-ref one."""

    def test_empty_work_set_still_refuses_an_odd_power_of_two(self):
        with pytest.raises(ValueError, match=r"splits a morton digit in half"):
            partition_leaves([], 8)

    def test_refs_are_indexed_before_the_buckets_exist(self):
        # A too-deep split is caught by the ownership test, not after 4^k
        # empty lists have been materialized.
        with pytest.raises(ValueError, match=r"coarser than the partitions="):
            partition_leaves([morton_word("-311")], 4**15)

    def test_select_partition_range_checks_its_index(self):
        with pytest.raises(ValueError, match=r"out of range"):
            select_partition({d: {None} for d in NODES}, 4, 9)


class TestPartitionedPass:
    """``run_sweep(partition=...)``: exactly what one partition worker writes."""

    def test_writes_only_its_own_subtrees_and_stops_at_the_split(self, tmp_path, monkeypatch):
        refs = _store(tmp_path)
        keys = _written(monkeypatch)
        summary = run_sweep(
            str(tmp_path), refs, families=["stats"], partition={"index": 0, "of": 4}
        )
        # Partition 0 owns the "-31" subtree: its two shards plus their order-1
        # parent. The base node -3 sits at order 0, above the split -> nobody's.
        assert summary["partition"] == {"index": 0, "of": 4, "split_order": 1}
        assert summary["n_leaves"] == 2 and summary["foreign_leaves"] == 4
        assert sorted(keys) == [
            "-3/1/1/stats.rollup.json",
            "-3/1/2/stats.rollup.json",
            "-3/1/stats.rollup.json",
        ]

    def test_no_two_partitions_write_the_same_object(self, tmp_path, monkeypatch):
        # The disjointness obligation, against the real writer rather than the
        # index arithmetic: the four partitions' key sets are pairwise disjoint.
        refs = _store(tmp_path)
        per_partition = []
        for index in range(4):
            keys = _written(monkeypatch)
            run_sweep(str(tmp_path), refs, families=["stats"], partition={"index": index, "of": 4})
            per_partition.append(set(keys))
        for a, b in itertools.combinations(per_partition, 2):
            assert not a & b
        assert per_partition[2] == set()  # no leaf under -33/13: an empty partition
        assert "1/4/stats.rollup.json" in per_partition[3]  # and one spanning two base cells

    def test_union_equals_an_unpartitioned_pass_minus_the_coarse_nodes(self, tmp_path, monkeypatch):
        whole, split = tmp_path / "whole", tmp_path / "split"
        refs_whole, refs_split = _store(whole), _store(split)
        full = _written(monkeypatch)
        run_sweep(str(whole), refs_whole, families=["stats"], record=False)
        full_keys = set(full)
        parted = _written(monkeypatch)
        for index in range(4):
            run_sweep(
                str(split),
                refs_split,
                families=["stats"],
                record=False,
                partition={"index": index, "of": 4},
            )
        # Every node the whole-tree pass wrote — except the order-0 base nodes,
        # which span partitions — was written by exactly one partition.
        coarse = {"-3/stats.rollup.json", "1/stats.rollup.json"}
        assert coarse <= full_keys
        assert set(parted) == full_keys - coarse

    def test_finish_is_deferred_so_partitions_never_race_on_the_root_moc(self, tmp_path):
        refs = _store(tmp_path)
        result = run_sweep(str(tmp_path), refs, families=["moc"], partition={"index": 0, "of": 4})
        moc = result["families"]["moc"]
        assert moc["finish_deferred"] is True
        assert "root_moc_written" not in moc
        assert not (tmp_path / "coverage.moc").exists()

    def test_identity_partition_matches_an_unpartitioned_pass(self, tmp_path, monkeypatch):
        whole, one = tmp_path / "whole", tmp_path / "one"
        refs_whole, refs_one = _store(whole), _store(one)
        keys_whole = _written(monkeypatch)
        run_sweep(str(whole), refs_whole, families=["stats", "moc"], record=False)
        keys_one = _written(monkeypatch)
        summary = run_sweep(
            str(one),
            refs_one,
            families=["stats", "moc"],
            record=False,
            partition={"index": 0, "of": 1},
        )
        assert sorted(keys_one) == sorted(keys_whole)
        # of=1 splits at order 0: the base nodes ARE owned, so the finish hook
        # runs (it reports its own key) rather than being deferred.
        assert "root_moc_written" in summary["families"]["moc"]
        assert "finish_deferred" not in summary["families"]["moc"]


class TestWorkerSideFiltering:
    """The partition filter lives where the fold happens, not in the dispatcher."""

    def test_a_discover_style_whole_store_work_set_is_filtered(self, tmp_path):
        # ``discover: true`` re-derives the WHOLE store's leaves worker-side; a
        # partition invoke must still sweep only its own.
        refs = _store(tmp_path)
        summary = run_sweep(
            str(tmp_path), refs, families=["stats"], partition={"index": 3, "of": 4}
        )
        assert summary["n_leaves"] == 3  # -341, 141, 142
        assert summary["foreign_leaves"] == 3

    def test_split_finer_than_the_shard_order_is_refused(self, tmp_path):
        refs = _store(tmp_path)
        with pytest.raises(ValueError, match=r"splits at order 3, finer than the store's"):
            run_sweep(str(tmp_path), refs, families=["stats"], partition={"index": 0, "of": 64})


class TestPartitionIdempotence:
    """The D22 generation stamps make each partition resumable on its own."""

    def test_second_pass_of_one_partition_writes_nothing(self, tmp_path):
        refs = _store(tmp_path)
        block = {"index": 0, "of": 4}
        first = run_sweep(str(tmp_path), refs, families=["stats"], partition=block)
        second = run_sweep(str(tmp_path), refs, families=["stats"], partition=block)
        assert first["families"]["stats"]["written"] == 3
        assert second["families"]["stats"]["written"] == 0
        assert second["families"]["stats"]["current"] == 3

    def test_a_rerun_leaf_rewrites_only_its_own_partition_chain(self, tmp_path, monkeypatch):
        refs = _store(tmp_path)
        for index in range(4):
            run_sweep(str(tmp_path), refs, families=["stats"], partition={"index": index, "of": 4})
        _put_leaf(tmp_path, "-341", n_obs=99)
        keys = _written(monkeypatch)
        for index in range(4):
            run_sweep(str(tmp_path), refs, families=["stats"], partition={"index": index, "of": 4})
        assert sorted(keys) == ["-3/4/1/stats.rollup.json", "-3/4/stats.rollup.json"]


class TestPartitionRecord:
    """Per-partition timing rows — the issue #354 record, extended."""

    def test_record_key_carries_the_partition_so_the_fan_out_cannot_clobber(self, tmp_path):
        refs = _store(tmp_path)
        records = [
            run_sweep(str(tmp_path), refs, families=["stats"], partition={"index": i, "of": 4})[
                "record"
            ]
            for i in (0, 1)
        ]
        # The 2^n invokes of one fan-out land in the same wall-clock second.
        assert records[0].endswith("_p0of4.json") and records[1].endswith("_p1of4.json")
        body = json.loads((tmp_path / records[1]).read_text())
        assert body["partition"] == {"index": 1, "of": 4, "split_order": 1}
        assert body["families"]["stats"]["duration_s"] >= 0.0

    def test_an_unpartitioned_record_keeps_its_name_and_carries_no_partition(self, tmp_path):
        refs = _store(tmp_path)
        summary = run_sweep(str(tmp_path), refs, families=["stats"])
        assert summary["record"].endswith("Z.json") and "_p" not in summary["record"]
        assert "partition" not in summary and "foreign_leaves" not in summary


class TestOverviewOrdersAreClamped:
    def test_orders_coarser_than_the_split_are_deferred_to_the_finisher(self, tmp_path):
        decl = {
            "overview": {
                "orders": [1, 0],
                "fields": {"count": {"class": "exact", "method": "sum", "dtype": "int32"}},
            }
        }
        refs = _store(tmp_path, pyramid=decl)
        parted = run_sweep(
            str(tmp_path), refs, families=["overview"], partition={"index": 0, "of": 4}
        )
        assert parted["families"]["overview"]["deferred_orders"] == [0]
        whole = run_sweep(str(tmp_path), refs, families=["overview"])
        assert "deferred_orders" not in whole["families"]["overview"]

    def test_the_manifest_pyramid_rmw_is_left_to_the_finisher(self, tmp_path, monkeypatch):
        # `_update_manifest_pyramid` is a GET-modify-PUT of the ONE store-root
        # manifest: 2^n partitions racing it would lose updates, so a
        # partitioned pass must not touch it at all.
        from zagg import sweep_overview as ov

        decl = {
            "overview": {
                "orders": [1],
                "fields": {"count": {"class": "exact", "method": "sum", "dtype": "int32"}},
            }
        }
        refs = _store(tmp_path, pyramid=decl)
        monkeypatch.setattr(
            ov,
            "_update_manifest_pyramid",
            lambda *a, **k: pytest.fail("manifest RMW in a partition"),
        )
        before = (tmp_path / MANIFEST_NAME).read_bytes()
        result = run_sweep(
            str(tmp_path), refs, families=["overview"], partition={"index": 0, "of": 4}
        )["families"]["overview"]
        assert result["manifest_deferred"] is True
        assert "manifest_updated" not in result
        assert (tmp_path / MANIFEST_NAME).read_bytes() == before


class TestSweepPartitionsDriver:
    def test_runs_every_non_empty_partition_in_index_order(self, tmp_path):
        refs = _store(tmp_path)
        summaries = sweep_partitions(str(tmp_path), refs, partitions=4, families=["stats"])
        assert [s["partition"]["index"] for s in summaries] == [0, 1, 3]  # 2 is empty
        assert sum(s["n_leaves"] for s in summaries) == len(LEAVES)
