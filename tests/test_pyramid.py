"""The ``(node, cells)`` pyramid level grammar (issue #382).

Grammar-only coverage for :mod:`zagg.pyramid`: normalization (scalar sugar,
shape refusals), validation (the issue #382 ordering/range rules, the gather
exception, ``cells == node``), the derived default schedule bound to the grid
block (#381 point (5)), and the ``validate_config`` wiring (whole-list
replacement of the ``/1`` ``orders``/``spacing`` knobs). Declaration/manifest
behavior lives in ``tests/test_sweep_overview.py``; the spec fixture in
``tests/test_spec_conformance.py``.
"""

import pytest

from zagg.pyramid import default_levels, normalize_levels, validate_levels

#: The reference 19/13/9 geometry from the issue #381 design record.
REF = {"parent_order": 9, "child_order": 19}


def norm(*entries):
    return normalize_levels(list(entries))


class TestNormalizeLevels:
    def test_scalar_cells_is_sugar_for_one_element_list(self):
        assert norm({"node": 7, "cells": 11}) == [{"node": 7, "cells": [11]}]

    def test_list_cells_kept_grouped_and_intified(self):
        assert norm({"node": 9, "cells": [16, 13]}) == [{"node": 9, "cells": [16, 13]}]

    def test_order_preserved_never_sorted(self):
        raw = [{"node": 9, "cells": 13}, {"node": 1, "cells": 5}, {"node": 5, "cells": 9}]
        assert [e["node"] for e in normalize_levels(raw)] == [9, 1, 5]

    @pytest.mark.parametrize("raw", [None, {}, [], "levels", 7])
    def test_non_list_or_empty_refused(self, raw):
        with pytest.raises(ValueError, match="non-empty list"):
            normalize_levels(raw)

    @pytest.mark.parametrize(
        "entry",
        [
            1,
            "node",
            {"node": 9},
            {"cells": [13]},
            {"node": 9, "cells": [13], "depth": 4},
            {"node": 9, "resolutions": [13]},
        ],
    )
    def test_entry_shape_refused(self, entry):
        with pytest.raises(ValueError, match="exactly 'node' and 'cells'"):
            norm(entry)

    # True/False are ints in Python: a typo'd YAML `node: true` must refuse,
    # never launder into order 1.
    @pytest.mark.parametrize("node", [None, -1, "9", 9.0, True, False])
    def test_bad_node_refused(self, node):
        with pytest.raises(ValueError, match="node must be an int >= 0"):
            norm({"node": node, "cells": [13]})

    @pytest.mark.parametrize(
        "cells", [None, [], "13", [13, "11"], [13.0], {"r": 13}, True, False, [True]]
    )
    def test_bad_cells_refused(self, cells):
        with pytest.raises(ValueError, match="cells must be an int or a non-empty"):
            norm({"node": 9, "cells": cells})


class TestValidateLevels:
    def test_reference_default_schedule_is_valid(self):
        validate_levels(
            norm(
                {"node": 9, "cells": 13},
                {"node": 7, "cells": 11},
                {"node": 5, "cells": 9},
                {"node": 3, "cells": 7},
                {"node": 1, "cells": 5},
            ),
            **REF,
        )

    def test_lone_entry_declares_a_whole_pyramid(self):
        # Whole-list replacement: this is a complete two-level declaration.
        validate_levels(norm({"node": 9, "cells": [16, 13]}), **REF)

    def test_cells_equal_node_is_the_one_cell_group(self):
        validate_levels(norm({"node": 9, "cells": [13, 9]}), **REF)

    def test_cells_equal_node_coexists_with_a_coarser_entry(self):
        # The whole-footprint member (#381 point (2)) is the writer's universal
        # partial — it serves the coarser folds below it, so it never sets the
        # descent floor. The floor here is 13, so the (7,[11]) tail is legal.
        validate_levels(norm({"node": 9, "cells": [13, 9]}, {"node": 7, "cells": [11]}), **REF)

    def test_bare_node_order_entry_constrains_nothing(self):
        # An entry declaring ONLY its node-order member has no reader-facing
        # member at all, so it imposes no cross-entry constraint.
        validate_levels(norm({"node": 9, "cells": [9]}, {"node": 7, "cells": [8]}), **REF)

    def test_gather_same_cells_at_coarser_node(self):
        validate_levels(norm({"node": 9, "cells": [13, 11]}, {"node": 5, "cells": [13, 11]}), **REF)

    def test_gather_does_not_reset_the_descent_floor(self):
        # After the gather the next comparison is against the GATHER entry:
        # its reader-facing floor is still 11, so (3,[7]) descends cleanly.
        validate_levels(
            norm(
                {"node": 9, "cells": [13, 11]},
                {"node": 5, "cells": [13, 11]},
                {"node": 3, "cells": [7]},
            ),
            **REF,
        )

    def test_node_zero_and_leaf_node_are_both_legal(self):
        validate_levels(norm({"node": 9, "cells": 13}, {"node": 0, "cells": 4}), **REF)

    def test_node_above_parent_order_refused(self):
        with pytest.raises(ValueError, match=r"outside \[0, parent_order = 9\]"):
            validate_levels(norm({"node": 10, "cells": 13}), **REF)

    @pytest.mark.parametrize("cells", [[11, 13], [13, 13]])
    def test_cells_not_strictly_descending_refused(self, cells):
        with pytest.raises(ValueError, match="must strictly descend"):
            validate_levels(norm({"node": 9, "cells": cells}), **REF)

    def test_cells_coarser_than_node_refused(self):
        with pytest.raises(ValueError, match="coarser than their own node"):
            validate_levels(norm({"node": 9, "cells": [13, 8]}), **REF)

    @pytest.mark.parametrize("cell", [19, 20])
    def test_cells_at_or_past_base_order_refused(self, cell):
        # Binding at the leaf node per the issue; enforced on every entry —
        # the base level is the store itself, never a declarable member.
        with pytest.raises(ValueError, match="base data's own cell order"):
            validate_levels(norm({"node": 9, "cells": [cell, 13]}), **REF)

    @pytest.mark.parametrize("nodes", [(9, 9), (7, 9)])
    def test_nodes_not_strictly_descending_refused(self, nodes):
        first, second = nodes
        with pytest.raises(ValueError, match="nodes must strictly descend"):
            validate_levels(
                norm({"node": first, "cells": 13}, {"node": second, "cells": 13}), **REF
            )

    def test_cross_entry_overlap_refused(self):
        # Neither descending nor a gather (the cells lists differ): refuse by
        # name, never widen.
        with pytest.raises(ValueError, match="do not descend from"):
            validate_levels(
                norm({"node": 9, "cells": [16, 13]}, {"node": 7, "cells": [13, 11]}), **REF
            )

    def test_cross_entry_equal_resolution_without_gather_refused(self):
        # 11 is reader-facing on the leaf entry (it is not the node's own
        # order), so it IS the floor: an o7 entry restating it must spell the
        # whole cells list to read as the declared gather.
        with pytest.raises(ValueError, match="do not descend from"):
            validate_levels(norm({"node": 9, "cells": [13, 11]}, {"node": 7, "cells": [11]}), **REF)


class TestDefaultLevels:
    def test_reference_geometry_pins_the_ratified_schedule(self):
        assert default_levels(9, 13, child_order=19) == [
            {"node": 9, "cells": [13]},
            {"node": 7, "cells": [11]},
            {"node": 5, "cells": [9]},
            {"node": 3, "cells": [7]},
            {"node": 1, "cells": [5]},
        ]

    def test_spec_fixture_geometry(self):
        assert default_levels(4, 5, child_order=6) == [
            {"node": 4, "cells": [5]},
            {"node": 2, "cells": [3]},
            {"node": 0, "cells": [1]},
        ]

    def test_even_shard_order_walks_down_to_node_zero(self):
        # The espg ruling on the node walk's tail (PR #389 thread): step by 2
        # until you run out — node 0 on even shard orders, node 1 on odd,
        # matching the /1 default's range(shard - 2, -1, -2) reach.
        assert [e["node"] for e in default_levels(6, 8, child_order=12)] == [6, 4, 2, 0]
        assert [e["node"] for e in default_levels(9, 13, child_order=19)] == [9, 7, 5, 3, 1]

    def test_k_one_degenerates_to_whole_footprint_groups(self):
        # K == 1: the grid's RESOLVED chunk order equals parent_order, so
        # d = 0 and every level is the 1-cell whole-footprint group.
        assert default_levels(4, 4, child_order=6) == [
            {"node": 4, "cells": [4]},
            {"node": 2, "cells": [2]},
            {"node": 0, "cells": [0]},
        ]

    def test_chunk_order_at_child_order_refused(self):
        # HealpixGrid admits chunk_order == child_order; the base entry would
        # BE the base data, so the default refuses loudly at derive time.
        with pytest.raises(ValueError, match="base data's own cell order"):
            default_levels(9, 19, child_order=19)

    @pytest.mark.parametrize("chunk_order", [9, 11, 13, 18])
    def test_default_round_trips_through_validate(self, chunk_order):
        # Valid by construction for every geometry the grid admits between
        # chunk_order == parent_order (K == 1) and child_order - 1.
        validate_levels(default_levels(9, chunk_order, child_order=19), **REF)


class TestConfigWiring:
    """``output.pyramid.levels`` through ``validate_config`` (issue #382)."""

    def _cfg(self, **output):
        from zagg.config import default_config

        cfg = default_config("atl06")  # grid: parent_order 6, child_order 12
        cfg.output.update(output)
        return cfg

    def _validate(self, pyramid):
        from zagg.config import validate_config

        validate_config(self._cfg(store_layout="hive", pyramid=pyramid))

    def test_valid_levels_block(self):
        self._validate(
            {
                "levels": [
                    {"node": 6, "cells": 9},
                    {"node": 4, "cells": 7},
                    {"node": 2, "cells": [5, 2]},
                ]
            }
        )

    def test_levels_with_orders_or_spacing_refused(self):
        for extra in ({"orders": [4]}, {"spacing": 2}):
            with pytest.raises(ValueError, match="declare levels OR orders/spacing"):
                self._validate({"levels": [{"node": 6, "cells": 9}], **extra})

    def test_levels_grammar_errors_surface_through_validate_config(self):
        with pytest.raises(ValueError, match="non-empty list"):
            self._validate({"levels": []})
        with pytest.raises(ValueError, match=r"outside \[0, parent_order = 6\]"):
            self._validate({"levels": [{"node": 7, "cells": 9}]})
        with pytest.raises(ValueError, match="base data's own cell order"):
            self._validate({"levels": [{"node": 6, "cells": 12}]})

    def test_missing_child_order_refuses_by_name(self):
        # Pinned on _validate_pyramid directly: validate_config's own grid
        # check catches a missing child_order first, so this is the only place
        # the accessor's named refusal is observable — the levels branch must
        # not inherit the raw `.get(..., 0)` fallback, which would refuse with
        # "cells [9] reach the base data's own cell order (0)" instead.
        from zagg.config import _validate_pyramid

        cfg = self._cfg(store_layout="hive", pyramid={"levels": [{"node": 6, "cells": 9}]})
        cfg.output["grid"] = {k: v for k, v in cfg.output["grid"].items() if k != "child_order"}
        with pytest.raises(ValueError, match="output.grid.child_order is required"):
            _validate_pyramid(cfg)

    def test_levels_require_hive_layout(self):
        from zagg.config import validate_config

        with pytest.raises(ValueError, match="output.pyramid requires"):
            validate_config(
                self._cfg(
                    store_layout="flat",
                    coverage_moc=False,
                    sweep=False,
                    pyramid={"levels": [{"node": 4, "cells": 6}]},
                )
            )
