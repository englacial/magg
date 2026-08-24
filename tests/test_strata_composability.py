"""Issue #515: strata + composition through the D24 composability gate.

Phase 1 — characterization. ``build_tdigest_where`` (the strata builder) and
the packed composition word are both admitted by the SPILL fold gate
(``_TDIGEST_SPILL_FUNCTIONS`` / the ``validate_spill_fold`` scalar branch,
issues #370/#321) — the published CA store's 2,726 shards were built by k-way
merging per-block strata partials under exactly those laws — yet D24
classifies every one of their fields ``none``, so the live manifest promises
no overview fold for the very payloads whose bytes rest on one. These tests
pin that drift before the admission changes it:

- the LIVE CA-manifest declaration, vendored byte-for-byte as an in-tree
  fixture (``tests/data/ca_atl03_tdigest_o9_morton_hive.json``, fetched from
  ``atl03_tdigest_o9.zarr/morton_hive.json`` on 2026-08-24) — the permanent
  BEFORE record no later phase edits;
- today's classifier verdicts and the spill-gate admissions — the pins later
  phases flip deliberately, one per admission.
"""

import json
from pathlib import Path

from zagg.config import default_config
from zagg.pyramid import declared_fields
from zagg.semantics import composability_classes, field_composability

CA_MANIFEST = Path(__file__).parent / "data" / "ca_atl03_tdigest_o9_morton_hive.json"

#: The live store's per-field declaration, exactly as published (issue #515):
#: count folds, everything the store is FOR does not.
CA_FIELDS_BEFORE = {
    "count": {
        "class": "exact",
        "method": "sum",
        "nan_policy": "skip",
        "dtype": "int32",
        "fill_value": 0,
    },
    "h_tdigest_signal": {"class": "none"},
    "h_tdigest_noise": {"class": "none"},
    "composition": {"class": "none"},
}


class TestLiveCaManifestFixture:
    """The vendored CA manifest is the historical BEFORE — never edited."""

    def test_fixture_pins_the_live_declaration(self):
        # The known-answer half of the issue #515 before/after: the published
        # store declares its strata and composition non-composable while its
        # own bytes were built by their merge laws (spill fold, issue #370).
        manifest = json.loads(CA_MANIFEST.read_text())
        assert manifest["spec"] == "morton-hive/1"
        assert manifest["shard_order"] == 9
        assert manifest["cell_order"] == 19
        # Identity of the record: this is the store the issue names, not a
        # synthetic reconstruction.
        assert (
            manifest["semantic_hash"]
            == "b9b15fdde78f147c15c929da8ca93de21930ad5c552ae082c5d8998fb83ada21"
        )
        assert manifest["pyramid"]["overview"]["fields"] == CA_FIELDS_BEFORE

    def test_fixture_declares_a_full_cascade_ladder(self):
        # The retrofit context: the ladder is declared top to bottom (nodes
        # 9..0), fold_source cascade — so the ONLY thing standing between this
        # store and digest pyramids is the per-field class map above.
        manifest = json.loads(CA_MANIFEST.read_text())
        pyramid = manifest["pyramid"]
        assert pyramid["spec"] == "zagg-pyramid/2"
        assert [e["node"] for e in pyramid["overviews"]] == list(range(9, -1, -1))
        assert pyramid["overview"]["fold_source"] == "cascade"
        assert pyramid["overview"]["exact_levels"] == 1


class TestSpillGateAdmissions:
    """The spill fold already carries both laws the D24 gate declines to state."""

    def test_where_builder_is_spill_admitted(self):
        from zagg.processing.streaming import (
            _TDIGEST_SPILL_FUNCTIONS,
            _TDIGEST_WHERE_FUNCTION,
        )

        assert _TDIGEST_WHERE_FUNCTION == "zagg.stats.tdigest.build_tdigest_where"
        assert _TDIGEST_WHERE_FUNCTION in _TDIGEST_SPILL_FUNCTIONS

    def test_composition_is_spill_admitted(self):
        # The composition spill law is ``merge_composition_kway`` over packed
        # ``(word, n_signal)`` pairs (issues #321/#370, option (a)); the gate
        # admits the reducer by name in its scalar branch.
        from zagg.processing.streaming import _COMPOSITION_FUNCTION

        assert _COMPOSITION_FUNCTION == "zagg.stats.composition.pack_composition"

    def test_strata_template_passes_the_spill_gate(self):
        # The CA config shape survives a block close: both strata AND the
        # composition word have cross-block fold laws (issue #370) — this is
        # the admission the D24 gate drifted from.
        from zagg.processing.streaming import validate_spill_fold

        validate_spill_fold(default_config("atl03_tdigest_strata_healpix"))


class TestD24BeforeAdmission:
    """Today's classifier verdicts — the drift itself. Phases 2–3 flip these
    deliberately, one per admission (espg ruling 2026-08-24, issue #515)."""

    def test_composition_field_classifies_none_today(self):
        meta = {
            "function": "zagg.stats.composition.pack_composition",
            "source": "h_ph",
            "dtype": "uint64",
            "fill_value": 0,
            "params": {"threshold": 2},
            "attrs": {"composition": {"of": "h_tdigest_signal", "threshold": 2}},
        }
        assert field_composability(meta) == "none"

    def test_strata_template_classes_today(self):
        classes = composability_classes(default_config("atl03_tdigest_strata_healpix"))
        assert classes["count"] == "exact"
        assert classes["h_tdigest_signal"] == "none"
        assert classes["h_tdigest_noise"] == "none"
        assert classes["composition"] == "none"

    def test_declared_fields_reproduces_the_live_ca_declaration(self):
        # The other half of the known-answer: today's classifier over the
        # shipped strata template reproduces the live store's declaration
        # EXACTLY — the manifest fixture above is what this code writes, so
        # the before/after is pinned against reality, not a synthetic shape.
        fields, excluded = declared_fields(default_config("atl03_tdigest_strata_healpix"))
        assert fields == CA_FIELDS_BEFORE
        assert sorted(excluded) == ["composition", "h_tdigest_noise", "h_tdigest_signal"]
