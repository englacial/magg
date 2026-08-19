"""The root coverage sidecar's temporal section — ``zagg-coverage-toc/1`` (issue #480).

Two tiers, both derived from the §8.3 per-centroid toc siblings a store
already carries, both riding the ONE object a reader already GETs to
bootstrap discovery (``{store_root}/coverage.moc``):

1. **the per-shard envelope word map** — one toc word per populated shard
   (``mortie.toc_reduce`` over that shard's sibling words, unioned across the
   store's temporal-carrying fields). This is the spatiotemporal pruning
   tier: "which shards hold data DURING my window" resolves from metadata,
   before any leaf is opened;
2. **the optional root time-digest** — a t-digest whose weights are
   observation counts and whose value axis is the §8.3 per-centroid toc
   envelopes' MIDPOINTS (:func:`_centroid_times`), with those same envelopes
   as its companion, in the store's NATIVE ragged ``(k, 2)`` + word-sibling
   form (base64 of the §1.4 element bytes), so a reader needs no grammar it
   does not already implement for the leaves. Its total weight is exact; the
   placement of that weight is only as time-resolved as the leaf digest's own
   centroid partition, which is a partition of VALUE (§10.3).

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


def temporal_cell_order(manifest: dict | None) -> int | None:
    """The manifest's ``cell_order`` — the leaf group the arrays live under.

    ``None`` when the key is absent or unparsable, and a caller MUST read that
    as "this store publishes no temporal coverage" rather than substitute a
    default. ``cell_order`` is a required manifest key, and a silent ``0``
    either asks for a group that does not exist — surfacing as an ordinary
    "no temporal contribution" log line, which hides the manifest problem —
    or, on a store whose cell order really IS 0, reads the wrong group and
    publishes words for it.
    """
    if not isinstance(manifest, dict):
        return None
    try:
        return int(manifest["cell_order"])
    except (KeyError, TypeError, ValueError):
        return None


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

    **Cost.** One leaf-sized read per declared field — ``payload[:]`` plus its
    sibling — and a per-cell decode before the merge, the same shape as the
    overview family's own leaf read (:func:`zagg.sweep_overview._fold_node`,
    which reads ``arr[:]`` per field and accumulates per-cell centroid lists
    the same way). What it returns is bounded regardless: the k-way merge here
    compresses the whole leaf to ~``ROOT_TOC_DELTA`` centroids, so the caller
    accumulating leaves (:func:`build_temporal_section`) holds
    ``n_leaves × ~δ`` rows, not ``n_leaves × n_cells × k``. For a
    2,726-shard store that is ~1.4 MB of centroids plus ~1.4 MB of companion
    words, which is why the root fold needs no chunk batching of its own.
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
        if len(raw_words) != len(raw_payload):
            # A SHORT companion aligns row for row over its own length, so the
            # per-cell check below never fires: the leaf would publish a word
            # joined over a PREFIX of its cells and be listed as complete. This
            # is the truncated-array shape the read path refuses everywhere
            # else (issue #452); refuse it here too, per shard (§10.2).
            raise ValueError(
                f"{meta['sibling']} has {len(raw_words)} rows for a "
                f"{len(raw_payload)}-row {name} payload — the companion must be "
                f"row-aligned with its digest (spec §1.1)"
            )
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
    it and leaves tier 1 standing (§10).

    ``fields`` unions with tier 1, because it is the provenance of the SHARD
    MAP and the map is the thing that unions. It is deliberately not narrowed
    to the surviving digest's own fields: doing so would describe the map with
    a list that no longer covers it. §10.1 says so, and makes the list an
    upper bound for the once-per-field weight rule rather than an exact
    description of the installed digest.

    An unknown-spec section on the INCOMING side contributes nothing — the
    same strict gate the enclosing envelope uses. On the EXISTING side it is
    kept **verbatim** instead: this function is the write composer, and a
    ``None`` return deletes the key. A section at a revision this zagg cannot
    read was written by a producer that knew more than this one, so it is
    neither dropped by a producer with no contribution nor downgraded by one
    carrying a ``zagg-coverage-toc/1`` section — the page's standing rule that
    readers add revisions and never drop them (§10.4).
    """
    from mortie import toc_merge

    a, b = _usable(existing), _usable(incoming)
    if b is None:
        return dict(a) if a is not None else _preserved(existing)
    if a is None:
        newer = _preserved(existing)
        if newer is not None:
            logger.warning(
                f"coverage[toc]: keeping the standing {newer.get('spec')!r} section — "
                f"{TEMPORAL_COVERAGE_SPEC} does not read it and MUST NOT downgrade it"
            )
            return newer
        return dict(b)
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


def warn_if_section_missing(store_root: str, envelope, manifest) -> bool:
    """Warn when a temporal-declaring store's root carries no §10 section.

    The belt for the one gap the §10.4 succession rule cannot close (issue
    #488): the rule binds producers that know it, and a pre-#481 producer —
    an old laptop, a stale worker — rebuilds the root envelope from the keys
    it knows and drops the section it never learned to copy. Severity is
    bounded by D9 (the section is a regenerable accelerator: a reader that
    loses it falls back to opening every candidate, slow and correct), so
    this is a logged nudge naming :func:`zagg.coverage.refresh_root_coverage`
    as the remedy — deliberately NOT a locking scheme, the #208 no-lock
    ruling stands. Returns whether it warned.

    An absent section is an OBSERVATION, and the message says only that.
    Two causes produce it and this seat cannot tell them apart: a producer
    dropped it, or no walk has ever built one — only the walk writes a
    section (:meth:`zagg.sweep.MocFamily.finish` and
    :func:`zagg.coverage.refresh_root_coverage` are its two writers; the
    runner's ingest and :func:`zagg.sweep_stages.run_finisher` both write
    section-less root envelopes). In the default pipeline the walk does
    precede the staged finisher — ``runner.py`` runs
    :func:`zagg.sweep.sweep_after_run` over
    :data:`zagg.sweep.DEFAULT_FAMILIES` (``moc`` among them) BEFORE
    :func:`zagg.sweep_stages.stage_sweep_after_run` in the same post-run
    hook — so a section missing at the finisher is genuinely anomalous
    there. It is NOT anomalous for a store swept with ``--stages``
    standalone, or one whose fail-open families sweep failed, and the same
    remedy fixes every one of them.

    Detection is DECLARATION-driven, like every other reach for the temporal
    channel here (:func:`temporal_fields`): a store whose manifest declares
    §8.3 ``"per-centroid"`` companions should have a section, and one that
    declares none never should. That makes the check free — no leaf is
    opened — at the cost of also firing on a temporal-declaring store whose
    leaves genuinely hold no temporal row yet (freshly templated, or leaves
    written before the declaration was added). The remedy named is right for
    that store too: the walk rebuilds what is there, and writes no section
    when the answer is honestly nothing.

    A standing section at a revision this zagg cannot read is NOT missing —
    §10.4 preserves it verbatim, and warning about it would invite an
    operator to run a walk that downgrades nothing but says the same thing
    again. Only an absent key, or an unmarked carrier (debris), warns.
    """
    from zagg.hive import ROOT_COVERAGE_NAME

    if not temporal_fields(manifest):
        return False
    section = envelope.get(TEMPORAL_KEY) if isinstance(envelope, dict) else None
    if _usable(section) is not None or _preserved(section) is not None:
        return False
    logger.warning(
        f"coverage[toc]: {store_root} declares temporal fields but its root "
        f"{ROOT_COVERAGE_NAME} carries no {TEMPORAL_COVERAGE_SPEC} section — no walk has "
        f"built one yet, or a producer dropped it; either way `when=` pruning degrades to "
        f"opening every candidate until it is built, so run "
        f"zagg.coverage.refresh_root_coverage({store_root!r})"
    )
    return True


def section_unchanged(existing, incoming) -> bool:
    """Whether composing ``incoming`` into ``existing`` would change nothing.

    The temporal half of the MOC family's skip-if-current test: an unchanged
    re-sweep of a temporal store must write no root object either. The test is
    on what would actually be **written** — :func:`merge_temporal_sections`'s
    own output — compared against the standing section on CONTENT (the shard
    words, the digest block, the field list), never on the whole section:
    ``source`` and ``generated_at`` churn per pass by construction.

    Testing the merge rather than the inputs is what makes it converge. A
    producer that walked only part of the store always builds a digest, and
    §10.4 always drops that digest at the seam; a test asking "does the
    standing section already carry everything this one holds" therefore
    answers *no* forever on any store with more than one shard, and every
    incremental sweep re-PUTs a byte-identical object.
    """
    merged = merge_temporal_sections(existing, incoming)
    return _content(merged) == _content(existing if isinstance(existing, dict) else None)


def _content(section) -> tuple | None:
    """The part of a section a no-op pass must reproduce exactly."""
    if section is None:
        return None
    return (
        {d: str(w) for d, w in (section.get("shards") or {}).items()},
        section.get("digest"),
        sorted(section.get("fields") or []),
    )


def _usable(section) -> dict | None:
    """A section dict at the spec revision this module implements, else ``None``."""
    if not isinstance(section, dict) or section.get("spec") != TEMPORAL_COVERAGE_SPEC:
        if section is not None:
            logger.debug("coverage[toc]: ignoring a section with an unknown spec")
        return None
    return section


def _preserved(section) -> dict | None:
    """A standing section carrying a spec MARKER this revision does not implement.

    Distinguished from plain malformation: a marked section is some future
    revision's, and §10.4 keeps it verbatim rather than clobber it. An
    unmarked carrier (no ``spec``, or a non-string one) claims no revision,
    so it is debris a producer may legitimately replace — otherwise one bad
    write would wedge the key shut forever.
    """
    if not isinstance(section, dict):
        return None
    spec = section.get("spec")
    if isinstance(spec, str) and spec and spec != TEMPORAL_COVERAGE_SPEC:
        return dict(section)
    return None


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

    §10.3's MUST-check is on all THREE: the two buffers against each other and
    both against the block's declared ``centroids``. A ``k`` that agrees with
    neither is a broken block, not a decorative field — and a reference
    accessor that skipped the check would leave the external reader
    (moczarr) implementing one zagg does not.
    """
    section = load_temporal_coverage(envelope)
    block = (section or {}).get("digest")
    if not isinstance(block, dict):
        return None
    from zagg.sweep_overview import decode_digest

    payload = decode_digest(base64.b64decode(block["payload"]), "float32", (2,))
    words = decode_digest(base64.b64decode(block["times"]), "uint64", ())
    if len(payload) != len(words) or block.get("centroids") != len(payload):
        raise ValueError(
            f"root time-digest declares {block.get('centroids')!r} centroids and decodes "
            f"{len(payload)} of them with {len(words)} companion words — the companion "
            f"must be row-aligned with its digest at the declared k (spec §1.1, §10.3)"
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
    "section_unchanged",
    "shards_overlapping",
    "temporal_cell_order",
    "temporal_fields",
    "warn_if_section_missing",
]
