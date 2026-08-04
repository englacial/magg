"""``2^n`` morton-subtree sweep partitions (issue #377): the decomposition.

The standing claims this file pins: the split rounds to morton DIGIT
boundaries (a power of four) and rejects an odd ``2^n`` by name; partition
ownership is a pure prefix test over the D1 decimal id; and the decomposition
is a partition in the set-theoretic sense — disjoint and covering — so no two
partitions can ever address the same node.
"""

import itertools

import pytest

from zagg.grids.morton import morton_decimal, morton_word
from zagg.hive import _decimal_order
from zagg.sweep_partition import (
    normalize_partition,
    partition_index,
    partition_leaves,
    partition_split_order,
    select_partition,
)

#: Every order-2 node of one base cell, plus a second base cell's, so the
#: "one order-k subtree per base cell" property is exercised, not assumed.
NODES = [f"{b}{i}{j}" for b in ("-3", "1") for i in "1234" for j in "1234"]


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
