"""Staged dense sweep (issue #384): planner, worker, finisher, orchestration.

Standing claims:

- tuple grouping is orchestration-only — dispatch nodes at ``0 mod width``,
  ragged finest tuple, child columns one tuple down;
- source classification is derived: gather at/below the shard's resolution
  window, merge above it; every merge consumes the relayed gen-1 partials
  (the espg merge-source ruling), so ``tuple_width=1`` and ``tuple_width=3``
  builds of the same store are byte-identical;
- scope is a MOC over node prefixes (shardmap keys as sugar), composing with
  partitions by intersection;
- soft barriers: partial coverage is recorded (``source_children``) and
  self-heals; skip-if-current keys on summed child generations;
- the finisher owns the root singletons; the lease serializes sweeps.
"""

from __future__ import annotations

import pytest

from zagg.pyramid import PYRAMID_SPEC_V2, expand_overviews
from zagg.sweep_stage import (
    STAGE_GATHER,
    STAGE_MERGE,
    classify_level,
    column_members,
    compose_scope,
    ladder_entries,
    normalize_scope,
    partition_words,
    scope_admits,
    stage_tuples,
)


def _pyramid_block(resolutions, shard_order):
    return {
        "spec": PYRAMID_SPEC_V2,
        "overviews": expand_overviews(resolutions, parent_order=shard_order),
        "overview": {"all_time": False, "fold_source": "cascade", "fields": {}},
    }


class TestLadderEntries:
    def test_excludes_leaf_entry_and_sorts_finest_first(self):
        entries = ladder_entries(_pyramid_block([13], 9), 9)
        assert [e["node"] for e in entries] == list(range(8, -1, -1))
        assert all(e["cells"] == [e["node"] + 4] for e in entries)

    def test_refuses_v1_block(self):
        with pytest.raises(ValueError, match="zagg-pyramid/2"):
            ladder_entries({"spec": "zagg-pyramid/1", "overview": {"orders": [3]}}, 9)

    def test_refuses_empty_overviews(self):
        block = {"spec": PYRAMID_SPEC_V2, "overviews": []}
        with pytest.raises(ValueError, match="absent or empty"):
            ladder_entries(block, 9)

    def test_refuses_multi_member_ladder_entry(self):
        block = _pyramid_block([13], 9)
        block["overviews"][1]["cells"] = [12, 11]
        with pytest.raises(ValueError, match="exactly one"):
            ladder_entries(block, 9)


class TestStageTuples:
    def test_reference_geometry_o9_width_3(self):
        tuples = stage_tuples(9, tuple_width=3)
        assert [t["dispatch"] for t in tuples] == [6, 3, 0]
        assert [t["orders"] for t in tuples] == [[8, 7, 6], [5, 4, 3], [2, 1, 0]]
        assert [t["child_order"] for t in tuples] == [9, 6, 3]

    def test_ragged_finest_tuple(self):
        tuples = stage_tuples(8, tuple_width=3)
        assert [t["orders"] for t in tuples] == [[7, 6], [5, 4, 3], [2, 1, 0]]
        assert [t["child_order"] for t in tuples] == [8, 6, 3]

    def test_width_1_is_one_tuple_per_order(self):
        tuples = stage_tuples(3, tuple_width=1)
        assert [t["orders"] for t in tuples] == [[2], [1], [0]]
        assert [t["child_order"] for t in tuples] == [3, 2, 1]

    def test_width_wider_than_ladder_is_one_root_tuple(self):
        (t,) = stage_tuples(3, tuple_width=3)
        assert t == {"dispatch": 0, "orders": [2, 1, 0], "child_order": 3}

    def test_orders_cover_ladder_exactly_once(self):
        for width in (1, 2, 3, 4):
            covered = [k for t in stage_tuples(9, tuple_width=width) for k in t["orders"]]
            assert sorted(covered) == list(range(9))

    def test_refusals(self):
        with pytest.raises(ValueError, match="tuple_width"):
            stage_tuples(9, tuple_width=0)
        with pytest.raises(ValueError, match="no above-shard ladder"):
            stage_tuples(0)


class TestClassifyLevel:
    def test_reference_geometry_first_merge_at_node_4(self):
        # o9/d=4: gather at nodes >= 5 (cells 9..12), merge at nodes <= 4
        # (#381 point (6): "the first true k-way merge appears at node 4").
        for e in ladder_entries(_pyramid_block([13], 9), 9):
            expected = STAGE_GATHER if e["node"] >= 5 else STAGE_MERGE
            assert classify_level(e["cells"][0], shard_order=9) == expected

    def test_boundary_is_the_shard_order(self):
        assert classify_level(9, shard_order=9) == STAGE_GATHER
        assert classify_level(8, shard_order=9) == STAGE_MERGE


class TestColumnMembers:
    def test_reference_geometry(self):
        levels = expand_overviews([13], parent_order=9)
        # o6 column: relay (9) plus the members nodes 5..0 gather — cells
        # {9..12} from nodes {5,4,3,2,1,0} intersect >= 9 -> {9} only at
        # node 5; nodes 4.. are merges. Members: {9} == relay alone... plus
        # gatherable cells strictly: node 5 gathers 9 (the relay itself).
        assert column_members(levels, 6, shard_order=9) == [9]
        # o3 column: coarser levels' gatherable cells (7..2 + 4) are all < 9,
        # so the relay is the only member.
        assert column_members(levels, 3, shard_order=9) == [9]

    def test_finer_declaration_widens_the_gather_tier(self):
        # d=1 (overviews [10] on o9): node-8 level gathers cells 9 == relay;
        # a width-3 column at 6 carries relay only; nothing else >= shard.
        levels = expand_overviews([10], parent_order=9)
        assert column_members(levels, 6, shard_order=9) == [9]

    def test_multi_resolution_leaf_declaration(self):
        # overviews [16, 13] on o9: d = 13 - 9 = 4; ladder unchanged, and the
        # finer 16 member is leaf-only (no coarser level gathers it).
        levels = expand_overviews([16, 13], parent_order=9)
        assert column_members(levels, 6, shard_order=9) == [9]

    def test_wide_window_carries_gather_members(self):
        # d=5 on o6 (overviews [11]): coarser levels than the o3 column with
        # gatherable cells are node 2 (cells 7) and node 1 (cells 6); its own
        # level (cells 8) is an artifact, never one of its members.
        levels = expand_overviews([11], parent_order=6)
        assert column_members(levels, 3, shard_order=6) == [7, 6]
        assert column_members(levels, 1, shard_order=6) == [6]


class TestScope:
    def test_none_passes_through(self):
        assert normalize_scope(None) is None
        assert scope_admits("-4211", None)

    def test_decimals_ints_and_shardmap_keys(self):
        from zagg.grids.morton import morton_word

        w = morton_word("-42113")
        assert list(normalize_scope(["-42113"])) == [w]
        assert list(normalize_scope([w])) == [w]
        assert list(normalize_scope({"-42113": {"granules": []}})) == [w]

    def test_empty_scope_refuses(self):
        with pytest.raises(ValueError, match="empty"):
            normalize_scope([])

    def test_ancestor_prefixes_admit(self):
        scope = normalize_scope(["-42113221", "-42113222"])
        for node in ("-4", "-42", "-421", "-4211", "-42113221"):
            assert scope_admits(node, scope)
        assert not scope_admits("-43", scope)
        assert not scope_admits("3", scope)

    def test_partition_compose(self):
        scope = normalize_scope(["-42113", "31222"])
        # order-1 split: partition of index rank('2')==1 owns every subtree
        # whose first digit is 2.
        part = partition_words(4, 1)
        composed = compose_scope(scope, part)
        assert scope_admits("-42113", composed)
        assert not scope_admits("31222", composed)
        assert compose_scope(scope, None) is scope
        assert compose_scope(None, part) is part

    def test_partition_words_identity(self):
        assert partition_words(1, 0).size == 12
