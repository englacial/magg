"""The root coverage sidecar's temporal section — ``zagg-coverage-toc/1`` (issue #480).

Two tiers, both derived from the §8.3 per-centroid toc siblings a store
already carries, both riding the ONE object a reader already GETs to
bootstrap discovery (``{store_root}/coverage.moc``):

1. **the per-shard envelope word map** — one toc word per populated shard
   (``mortie.toc_reduce`` over that shard's sibling words, unioned across the
   store's temporal-carrying fields). This is the spatiotemporal pruning
   tier: "which shards hold data DURING my window" resolves from metadata,
   before any leaf is opened;
2. **the optional root time-digest** — a t-digest over acquisition times
   whose weights are observation counts, with the §8.3 per-centroid toc
   envelopes as its companion, in the store's NATIVE ragged ``(k, 2)`` +
   word-sibling form (base64 of the §1.4 element bytes), so a reader needs
   no grammar it does not already implement for the leaves.

Absence composes: a store with no temporal channel writes no section, and
every accessor here returns ``None`` for it. That is never a refusal — the
standing absence posture of the sidecar grammars.

The writer rides :class:`zagg.sweep.MocFamily`, the walk that already visits
every leaf to roll the spatial coverage up to the root; this module owns the
per-leaf read, the fold, the section grammar, and the reader-side accessors
zagg's own tests need. The normative grammar is ``docs/specification.md``
§10.
"""

from __future__ import annotations

import base64
import logging

import numpy as np

logger = logging.getLogger(__name__)

#: The versioned key discipline: the section carries its own spec marker, so
#: a reader gates on it exactly as it gates the enclosing ``morton-moc/1``
#: envelope and refuses an unknown revision rather than guessing.
TEMPORAL_COVERAGE_SPEC = "zagg-coverage-toc/1"

#: The section's key on the root coverage envelope.
TEMPORAL_KEY = "temporal"

#: Compression budget for the root time-digest (the issue's δ≈64): coarse by
#: design — the digest answers "how MUCH data in this window", while the
#: per-centroid companion words carry the exact temporal claim.
ROOT_TOC_DELTA = 64

#: The §8.3 shape this section is derived from. A ``"per-cell"`` or
#: ``"coordinate"`` declaration is a different array grammar and contributes
#: nothing here (see the §10 open question).
PER_CENTROID = "per-centroid"


def temporal_fields(manifest: dict | None) -> dict[str, dict]:
    """The store's temporal-carrying digest fields, from the manifest declaration.

    Keyed by payload field name, valued with the fold metadata plus the
    resolved ``sibling`` name. Discovery is declaration-driven — the same
    ``pyramid.overview.fields`` block :func:`zagg.sweep_overview.field_companions`
    reads — never a member enumeration of the leaf (a foreign or orphaned
    array prefix must not be able to steer this), and never a naming
    convention reconstructed by the reader. A store that declares no
    ``temporal`` field returns ``{}``, which is what makes the whole section
    absent for it.
    """
    from zagg.grids.base import ragged_times_name

    if not isinstance(manifest, dict):
        return {}
    decl = (manifest.get("pyramid") or {}).get("overview")
    fields = (decl or {}).get("fields") if isinstance(decl, dict) else None
    out: dict[str, dict] = {}
    for name, meta in (fields or {}).items():
        if not isinstance(meta, dict) or meta.get(TEMPORAL_KEY) != PER_CENTROID:
            continue
        out[name] = {**meta, "sibling": ragged_times_name(name)}
    return out


def _centroid_times(words: np.ndarray) -> np.ndarray:
    """Representative instants (internal ns, float64) for toc words.

    ``mortie.toc2time`` decodes a timestamp to ``(t, t)`` and a range to its
    conservative ``[start, end)`` envelope, so the midpoint is the instant
    itself for a timestamp and the envelope's centre for a range. The
    digest's value axis is approximate by construction (§10): the word
    beside each centroid stays the exact claim.
    """
    from mortie import toc2time

    start, end = toc2time(np.asarray(words, dtype=np.uint64))
    return (np.asarray(start, dtype=np.float64) + np.asarray(end, dtype=np.float64)) / 2.0


def read_leaf_temporal(leaf_root: str, cell_order: int, fields: dict, **store_kwargs):
    """One leaf's contribution: ``(envelope_word, digest, times)`` or ``None``.

    Reads each declared field's payload and its §8.3 sibling by NAME (never a
    member enumeration), row-aligned per §1.1. The envelope word is
    ``toc_reduce`` over every sibling word the leaf holds, unioned across the
    declared fields — coverage as "any data", the issue's proposed default.
    The digest is one flat k-way merge over the leaf's per-cell time digests,
    each built from that cell's centroid instants (:func:`_centroid_times`)
    weighted by the payload's own centroid weights, so its total weight is
    the leaf's temporal observation count. ``None`` when the leaf holds no
    temporal row at all (an unpopulated or pre-companion leaf), which is
    absence, not failure.
    """
    import zarr
    from mortie import toc_reduce

    from zagg.stats.tdigest import merge_tdigests_kway
    from zagg.store import open_store
    from zagg.sweep_overview import decode_digest

    group = zarr.open_group(
        open_store(leaf_root, **store_kwargs), path=str(cell_order), mode="r", zarr_format=3
    )
    all_words: list[np.ndarray] = []
    digests: list[np.ndarray] = []
    times: list[np.ndarray] = []
    for name in sorted(fields):
        meta = fields[name]
        try:
            sibling, payload = group[meta["sibling"]], group[name]
        except KeyError:
            # Schema evolution: the field postdates this leaf. It contributes
            # nothing, exactly as the pyramid fold treats the same gap.
            logger.debug(f"coverage[toc]: leaf {leaf_root} lacks field {name!r}")
            continue
        dtype = meta.get("dtype") or "float32"
        raw_words, raw_payload = sibling[:], payload[:]
        for i, row in enumerate(raw_words):
            if row is None or not len(row):
                continue
            words = decode_digest(row, "uint64", ())
            cell = decode_digest(raw_payload[i], dtype, (2,))
            if len(cell) != len(words):
                raise ValueError(
                    f"{meta['sibling']} cell {i} has {len(words)} words for a "
                    f"{len(cell)}-centroid payload — the companion must be row-aligned "
                    f"with its digest (spec §1.1)"
                )
            all_words.append(words)
            t = _centroid_times(words)
            # Sort into value (= time) order: the stored rows are in the
            # payload's own value order (§8.3), which is not time order, and a
            # digest's rows MUST ascend by mean (§2.1). The word breaks ties so
            # the per-cell array is a function of its contents, not of row order.
            order = np.lexsort((words, t))
            arr = np.empty((len(t), 2), dtype=np.float32)
            arr[:, 0] = t[order]
            arr[:, 1] = cell[order, 1]
            digests.append(arr)
            times.append(np.asarray(words, dtype=np.uint64)[order])
    if not all_words:
        return None
    word = int(toc_reduce(np.concatenate(all_words)))
    digest, folded = merge_tdigests_kway(digests, delta=ROOT_TOC_DELTA, temporal=times)
    return word, digest, folded


def build_temporal_section(contributions: dict, fields, *, source: str = "sweep") -> dict | None:
    """The ``zagg-coverage-toc/1`` section from per-leaf contributions.

    ``contributions`` maps a shard's D1 decimal id to the LIST of
    ``(word, digest, times)`` triples :func:`read_leaf_temporal` returned for
    it — one per window leaf, so a windowed shard's several leaves reduce to
    the one envelope word the shard-keyed map holds. Returns ``None`` for an
    empty map: a store with no temporal channel gets no section, and its root
    object stays byte-identical to a pre-#480 one.

    The root digest is ONE flat k-way merge over the per-leaf digests
    (:func:`zagg.stats.tdigest.merge_tdigests_kway` with the ``temporal``
    channel), so it is permutation-independent in the leaf order and its
    companion words are the envelopes of the centroid partition that merge
    produced — the shipped law, not a second pass over it.
    """
    from mortie import toc_reduce

    from zagg.hive import _utcnow
    from zagg.stats.tdigest import merge_tdigests_kway

    if not contributions:
        return None
    shards: dict[str, int] = {}
    digests, times = [], []
    for decimal in sorted(contributions):
        parts = contributions[decimal]
        shards[decimal] = int(toc_reduce(np.asarray([p[0] for p in parts], dtype=np.uint64)))
        for _word, digest, folded in parts:
            if len(digest):
                digests.append(np.asarray(digest, dtype=np.float32))
                times.append(np.asarray(folded, dtype=np.uint64))
    section = {
        "spec": TEMPORAL_COVERAGE_SPEC,
        "source": source,
        "generated_at": _utcnow(),
        "fields": sorted(fields),
        "shards": {d: str(w) for d, w in sorted(shards.items())},
    }
    if digests:
        payload, words = merge_tdigests_kway(digests, delta=ROOT_TOC_DELTA, temporal=times)
        section["digest"] = _encode_digest_block(payload, words)
    return section


def _encode_digest_block(payload: np.ndarray, words: np.ndarray) -> dict:
    """The tier-2 block: the native ragged ``(k, 2)`` + word sibling, base64'd.

    The bytes are exactly what the same digest would occupy as a
    ``zagg-ragged/1`` element and its §8.3 companion row — little-endian
    C-order at the declared dtype (§1.4) — so a reader decodes them with the
    leaf decoder it already has, base64 being the only wrapper a JSON
    carrier forces.
    """
    from zagg.sweep_overview import encode_digest

    payload = np.asarray(payload, dtype=np.float32)
    words = np.asarray(words, dtype=np.uint64)
    return {
        "delta": ROOT_TOC_DELTA,
        "weights": "counts",
        "value": "toc-ns",
        "element": {"dtype": "float32", "shape": [-1, 2]},
        "encoding": "base64",
        "centroids": int(len(payload)),
        "weight_total": float(payload[:, 1].sum()) if len(payload) else 0.0,
        "payload": base64.b64encode(encode_digest(payload, "float32")).decode("ascii"),
        "times": base64.b64encode(encode_digest(words, "uint64")).decode("ascii"),
    }


def merge_temporal_sections(existing, incoming) -> dict | None:
    """Compose two temporal sections for the root object's GET-union-PUT.

    Tier 1 unions elementwise under the grammar's ``toc_merge`` join —
    idempotent and exact, so a re-sweep of unchanged leaves reproduces the
    same words. Tier 2 is deliberately NOT unioned: its weights are
    observation counts, and merging two digests over overlapping shard sets
    would double-count them. It is REPLACED instead, and only by a producer
    that covered every shard the merged map lists; a partial producer drops
    it and leaves tier 1 standing (§10). An unknown-spec section on either
    side is ignored rather than merged — the same strict gate the enclosing
    envelope uses.
    """
    from mortie import toc_merge

    a, b = _usable(existing), _usable(incoming)
    if a is None and b is None:
        return None
    if a is None:
        return dict(b)  # type: ignore[arg-type]
    if b is None:
        return dict(a)
    shards = {d: int(w) for d, w in (a.get("shards") or {}).items()}
    for d, w in (b.get("shards") or {}).items():
        prior = shards.get(d)
        shards[d] = int(w) if prior is None else int(toc_merge(prior, int(w)))
    merged = {
        "spec": TEMPORAL_COVERAGE_SPEC,
        "source": b.get("source", a.get("source")),
        "generated_at": b.get("generated_at", a.get("generated_at")),
        "fields": sorted(set(a.get("fields") or []) | set(b.get("fields") or [])),
        "shards": {d: str(w) for d, w in sorted(shards.items())},
    }
    # Whole-coverage test, newest producer first: only a section whose own map
    # listed every shard in the union can vouch for a store-wide digest.
    listed = set(merged["shards"])
    for side in (b, a):
        if side.get("digest") is not None and set(side.get("shards") or {}) >= listed:
            merged["digest"] = side["digest"]
            break
    return merged


def section_covers(existing, incoming) -> bool:
    """Whether ``existing`` already carries everything ``incoming`` would add.

    The temporal half of the MOC family's skip-if-current test: an unchanged
    re-sweep of a temporal store must write no root object either. Compared
    on CONTENT (the shard words and whether a digest is present), never on
    the whole section — ``generated_at`` churns per pass by construction.
    """
    if incoming is None:
        return True
    a = _usable(existing)
    if a is None:
        return False
    have = {d: int(w) for d, w in (a.get("shards") or {}).items()}
    if any(have.get(d) != int(w) for d, w in (incoming.get("shards") or {}).items()):
        return False
    return not (incoming.get("digest") is not None and a.get("digest") is None)


def _usable(section) -> dict | None:
    """A section dict at the spec revision this module implements, else ``None``."""
    if not isinstance(section, dict) or section.get("spec") != TEMPORAL_COVERAGE_SPEC:
        if section is not None:
            logger.debug("coverage[toc]: ignoring a section with an unknown spec")
        return None
    return section


# ---------------------------------------------------------------------------
# Reader side — the minimum zagg's own tests (and demos) need. The external
# reader's `coverage_toc` / `when=` surface is espg/moczarr#45, decoded from
# the spec text and the §7 fixtures alone.
# ---------------------------------------------------------------------------


def load_temporal_coverage(envelope) -> dict | None:
    """The temporal section of a root coverage envelope, or ``None``.

    Absence — no section, an unknown spec revision, a malformed carrier — all
    read as ``None``. A store with no temporal channel is the common case and
    is never an error.
    """
    if not isinstance(envelope, dict):
        return None
    return _usable(envelope.get(TEMPORAL_KEY))


def coverage_toc(envelope) -> dict[str, int] | None:
    """The per-shard envelope word map, ``{shard decimal: toc word}``.

    ``None`` when the store carries no temporal section. Words come back as
    Python ints (the JSON carries decimal STRINGS — a uint64 exceeds 2^53 and
    a float-based parser would mangle a raw number, the same rule the spatial
    ranges follow).
    """
    section = load_temporal_coverage(envelope)
    if section is None:
        return None
    return {d: int(w) for d, w in (section.get("shards") or {}).items()}


def coverage_toc_digest(envelope):
    """The root time-digest as ``(payload, words)``, or ``None``.

    ``payload`` is the §2.1 ``(k, 2)`` float32 centroid array whose value
    column is an instant on the §8 internal-ns scale and whose weight column
    is an observation count; ``words`` is its row-aligned §8.3 companion.
    """
    section = load_temporal_coverage(envelope)
    block = (section or {}).get("digest")
    if not isinstance(block, dict):
        return None
    from zagg.sweep_overview import decode_digest

    payload = decode_digest(base64.b64decode(block["payload"]), "float32", (2,))
    words = decode_digest(base64.b64decode(block["times"]), "uint64", ())
    if len(payload) != len(words):
        raise ValueError(
            f"root time-digest has {len(payload)} centroids and {len(words)} companion "
            f"words — the companion must be row-aligned with its digest (spec §1.1)"
        )
    return payload, words


def shards_overlapping(envelope, q_start_ns: int, q_end_ns: int) -> list[str] | None:
    """Shard ids whose envelope word intersects ``[q_start_ns, q_end_ns)``.

    The tier-1 pruning answer, on the grammar's own predicate
    (``mortie.toc_overlaps``): conservative — it never under-reports, and may
    over-report by up to one quantum at a window edge. ``None`` when the
    store carries no temporal section, which a caller MUST read as "no
    temporal information", never as "no shards".
    """
    from mortie import toc_overlaps

    words = coverage_toc(envelope)
    if words is None:
        return None
    keys = sorted(words)
    if not keys:
        return []
    hit = toc_overlaps(np.asarray([words[k] for k in keys], dtype=np.uint64), q_start_ns, q_end_ns)
    return [k for k, ok in zip(keys, np.atleast_1d(hit), strict=True) if bool(ok)]


__all__ = [
    "PER_CENTROID",
    "ROOT_TOC_DELTA",
    "TEMPORAL_COVERAGE_SPEC",
    "TEMPORAL_KEY",
    "build_temporal_section",
    "coverage_toc",
    "coverage_toc_digest",
    "load_temporal_coverage",
    "merge_temporal_sections",
    "read_leaf_temporal",
    "section_covers",
    "shards_overlapping",
    "temporal_fields",
]
