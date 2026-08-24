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

Phase 2 — the ``build_tdigest_where`` admission (espg-ruled admit,
2026-08-24): the strata builder joins ``_DIGEST_FAMILY_FUNCTIONS``, so strata
fields classify ``approximate`` and fold through the pyramid by the k-way law
with both companion channels; parity is asserted through the sweep's own fold
seam (``fold_digests``).
"""

import json
from pathlib import Path

import numpy as np

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


class TestD24Classification:
    """The classifier verdicts, phase by phase: phase 1 pinned ``none`` for
    all three; phase 2 admitted the strata builder (espg ruling 2026-08-24);
    phase 3 gives composition its own class."""

    def test_composition_field_classifies_none_today(self):
        # Flipped by phase 3: ``pack_composition`` has a fold law
        # (``merge_composition_kway``, issues #321/#370) that neither D24 arm
        # states yet, so today the classifier says ``none``.
        meta = {
            "function": "zagg.stats.composition.pack_composition",
            "source": "h_ph",
            "dtype": "uint64",
            "fill_value": 0,
            "params": {"threshold": 2},
            "attrs": {"composition": {"of": "h_tdigest_signal", "threshold": 2}},
        }
        assert field_composability(meta) == "none"

    def test_strata_template_classes(self):
        classes = composability_classes(default_config("atl03_tdigest_strata_healpix"))
        assert classes["count"] == "exact"
        # Phase 2: the strata builder is in the digest family (issue #515).
        assert classes["h_tdigest_signal"] == "approximate"
        assert classes["h_tdigest_noise"] == "approximate"
        # Phase 3 flips this one.
        assert classes["composition"] == "none"

    def test_declared_fields_no_longer_reproduce_the_live_ca_declaration(self):
        # The before/after known-answer, after phase 2: phase 1 asserted
        # ``fields == CA_FIELDS_BEFORE`` (today's classifier reproduced the
        # live store's declaration exactly); the admission changes ONLY the
        # strata entries — count is untouched and composition still declares
        # its recorded absence until phase 3.
        fields, excluded = declared_fields(default_config("atl03_tdigest_strata_healpix"))
        assert fields != CA_FIELDS_BEFORE
        assert fields["count"] == CA_FIELDS_BEFORE["count"]
        assert fields["composition"] == {"class": "none"}
        assert excluded == ["composition"]
        for name in ("h_tdigest_signal", "h_tdigest_noise"):
            assert fields[name] == {
                "class": "approximate",
                "method": "tdigest_kway",
                "dtype": "float32",
                "inner_shape": [2],
                # The template's leaf budget and its split pyramid-fold budget
                # (issue #424), recorded RESOLVED.
                "delta": 8192,
                "overview_delta": 512,
                # Located strata IS the default (espg ruling on PR #334); the
                # manifest entry is the only description the overview writer
                # has, so the channel must ride it (issue #410).
                "location": "leaf_id",
            }


class TestWhereBuilderFoldParity:
    """Phase 2: the admission's substance — a stratum payload built by
    ``build_tdigest_where`` folds by the digest family's k-way law with BOTH
    companion channels, matching the pooled build within the documented
    bounds (weights exact, quantiles close, located words to the common
    ancestor, temporal words to the toc envelope)."""

    def _blocks(self, k=4, n=400, seed=515):
        import mortie
        from conftest import toc_words

        from zagg.stats.tdigest import build_tdigest_where

        rng = np.random.default_rng(seed)
        values = rng.normal(30.0, 5.0, k * n)
        signal = rng.random(k * n) < 0.6
        lats = np.clip(37.0 + rng.uniform(-1e-4, 1e-4, k * n), -89.9, 89.9)
        lons = -119.0 + rng.uniform(-1e-4, 1e-4, k * n)
        locs = np.asarray(
            mortie.MortonIndexArray.from_latlon(lats, lons, points=True), dtype=np.uint64
        )
        times = toc_words(k * n)
        digests, words, tocs = [], [], []
        for i in range(k):
            sl = slice(i * n, (i + 1) * n)
            d, w, t = build_tdigest_where(
                values[sl],
                delta=64,
                where=signal[sl],
                locations=locs[sl],
                temporal=times[sl],
            )
            digests.append(d)
            words.append(w)
            tocs.append(t)
        return values, signal, locs, times, digests, words, tocs

    def test_payload_and_companions_through_the_fold(self):
        import mortie
        from mortie import common_ancestor

        from zagg.stats.tdigest import build_tdigest_where, quantile_from_tdigest
        from zagg.sweep_overview import decode_digest, encode_digest, fold_digests

        values, signal, locs, times, digests, words, tocs = self._blocks()
        # Through the SWEEP's fold seam, exactly as an overview cell folds its
        # contributors (payload and channels in one call, spec §9.1/§8.3).
        payload, loc_bytes, toc_bytes = fold_digests(
            digests,
            delta=64,
            dtype="float32",
            channels={"locations": list(words), "temporal": list(tocs)},
        )
        folded = decode_digest(payload, "float32", (2,))
        out_locs = decode_digest(loc_bytes, "uint64", ())
        out_tocs = decode_digest(toc_bytes, "uint64", ())
        assert len(out_locs) == len(out_tocs) == len(folded)
        # Weights are exact stratum counts, additive through the fold.
        assert int(folded[:, 1].sum()) == int(signal.sum())
        # Quantiles within the k-way law's documented closeness vs pooled.
        pooled, _, _ = build_tdigest_where(
            values, delta=64, where=signal, locations=locs, temporal=times
        )
        for q in (0.1, 0.5, 0.9):
            assert abs(quantile_from_tdigest(folded, q) - quantile_from_tdigest(pooled, q)) < 1.0
        # Located companion: every merged word contains the whole stratum's
        # common ancestor's refinement — cheap strong form: each output word
        # is an ancestor-or-equal of at least the stratum hull, so the join
        # of output words equals the join of input words (spec §9.1).
        assert int(common_ancestor(out_locs)) == int(common_ancestor(locs[signal]))
        # Temporal companion: the outputs' envelope is the inputs' envelope
        # (spec §8.3) — selected rows only, the WHERE mask masks the channel.
        starts, ends = (np.asarray(b, dtype="int64") for b in mortie.toc2time(out_tocs))
        m_starts, m_ends = (np.asarray(b, dtype="int64") for b in mortie.toc2time(times[signal]))
        assert int(starts.min()) <= int(m_starts.min())
        assert int(ends.max()) >= int(m_ends.max())
        # Round-trip stability: encode/decode is the identity on the payload.
        assert np.array_equal(
            decode_digest(encode_digest(folded, "float32"), "float32", (2,)), folded
        )

    def test_fold_is_order_independent(self):
        # The k-way law's defining property, on where-built payloads: a
        # permutation of the contributors returns identical bytes.
        from zagg.sweep_overview import fold_digests

        _, _, _, _, digests, words, tocs = self._blocks(k=5)
        a = fold_digests(
            digests,
            delta=64,
            dtype="float32",
            channels={"locations": list(words), "temporal": list(tocs)},
        )
        perm = [3, 0, 4, 2, 1]
        b = fold_digests(
            [digests[i] for i in perm],
            delta=64,
            dtype="float32",
            channels={"locations": [words[i] for i in perm], "temporal": [tocs[i] for i in perm]},
        )
        assert a == b
