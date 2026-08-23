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

A third surface lives BESIDE the bootstrap object (issue #489): the
**word-set cover sibling** ``{store_root}/coverage.toc`` — per shard, the
``mortie.toc_normalize`` canonical cover of that shard's companion words,
quantized to a spec-pinned temporal order (§10.5). Tier 1's one word makes
gaps invisible below shard granularity; the cover preserves them (the
grammar's never-bridge law) at bucket resolution, so "is there data in
``[t0, t1)``" resolves per shard without opening a leaf. It is a sibling
object, not a section: at CA scale it is ~300× the bootstrap object's size,
and spatial-only readers must not pay that. ``coverage.moc``'s temporal
section carries only a presence marker for it (:data:`COVER_KEY`).

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

#: The word-set cover sibling's own spec marker (§10.5, issue #489) — an
#: OBJECT-level marker: the cover is a root sibling, not a section, so it
#: gates itself the way the carrier envelope does.
COVER_SPEC = "zagg-coverage-toc-cover/1"

#: The sibling object's name under the store root, beside ``coverage.moc``.
COVER_NAME = "coverage.toc"

#: The presence-marker key on the §10.1 temporal section: its value is the
#: sibling object's spec string, so a temporal consumer knows one GET ahead
#: that (and at which revision) a cover stands beside the root. A hint under
#: the sidecar staleness posture, never a promise.
COVER_KEY = "cover"

#: The §10.5 day order: temporal order ``o`` partitions the toc scale into
#: ``2**o`` aligned buckets of ``2**(63 - o)`` ns. Order 16 (span 2^47 ns
#: ≈ 39.1 h) is the FINEST order whose bucket span is at least one day —
#: the spec-pinned quantization for the cover, chosen so bucket bounds are
#: exactly representable on the grammar's own encoding grids (§10.5).
TEMPORAL_DAY_ORDER = 16

#: The §10.5 overflow cap: a shard's cover holds at most this many words.
#: A cover that lands above it coarsens by order (each step halves the
#: bucket count) until it fits — widening only, loudly recorded.
COVER_CAP = 512


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
    """One leaf's contribution: ``(envelope_word, digest, times, cover)`` or ``None``.

    Reads each declared field's payload and its §8.3 sibling by NAME (never a
    member enumeration), row-aligned per §1.1. The envelope word is
    ``toc_reduce`` over every sibling word the leaf holds, unioned across the
    declared fields — coverage as "any data", the issue's proposed default.
    The digest is one flat k-way merge over the leaf's per-cell time digests,
    each built from that cell's centroid instants (:func:`_centroid_times`)
    weighted by the payload's own centroid weights, so its total weight is
    the leaf's temporal observation count. ``cover`` is the leaf's §10.5
    word-set cover — :func:`quantize_words` over every sibling word the leaf
    holds, at the pinned day order — reduced here (a few dozen words per
    leaf) so the accumulator never holds the raw word multiset: a CA-scale
    shard carries millions of words, and the cover of a union is the
    normalize of the union of covers, exactly. ``None`` when the leaf holds
    no temporal row at all (an unpopulated or pre-companion leaf), which is
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
    raw = np.concatenate(all_words)
    word = int(toc_reduce(raw))
    digest, folded = merge_tdigests_kway(digests, delta=ROOT_TOC_DELTA, temporal=times)
    return word, digest, folded, quantize_words(raw)


def build_temporal_section(contributions: dict, fields, *, source: str = "sweep") -> dict | None:
    """The ``zagg-coverage-toc/1`` section from per-leaf contributions.

    ``contributions`` maps a shard's D1 decimal id to the LIST of
    ``(word, digest, times, cover)`` tuples :func:`read_leaf_temporal`
    returned for it — one per window leaf, so a windowed shard's several
    leaves reduce to the one envelope word the shard-keyed map holds (the
    ``cover`` element is :func:`build_cover_section`'s input and is ignored
    here). Returns ``None`` for an
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
        for _word, digest, folded, _cover in parts:
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
    # The §10.5 presence marker carries through the seam (§10.4): the sibling
    # object composes under its own merge, so a producer that wrote no cover
    # must not erase the standing pointer to one.
    marker = b.get(COVER_KEY, a.get(COVER_KEY))
    if marker is not None:
        merged[COVER_KEY] = marker
    return merged


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
        section.get(COVER_KEY),
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
# The word-set cover sibling — `zagg-coverage-toc-cover/1` (§10.5, issue
# #489). A root object BESIDE the bootstrap sidecar, GET-on-demand by
# temporal consumers only: per shard, the canonical gap-preserving cover of
# its companion words, quantized to the pinned day order.
# ---------------------------------------------------------------------------


def quantize_words(words, order: int = TEMPORAL_DAY_ORDER) -> np.ndarray:
    """The §10.5 quantization: toc words, widened to aligned order-``o`` buckets.

    Temporal order ``o`` partitions the toc scale (2^63 ns from the grammar's
    1850 epoch) into ``2**o`` aligned buckets of ``2**(63 - o)`` ns. Each
    input word's conservative envelope (``toc2time``) is widened to the
    bucket grid — start floored, end ceiled — re-encoded as a range word,
    and the set canonicalized with ``toc_normalize``. Bucket bounds are
    exactly representable on the grammar's own grids for every ``o <= 31``
    (a bucket start is a multiple of 2^31 ns and its end of 2^32 ns), so the
    encoding round-trip adds no rounding of its own.

    **Widening only.** The output's coverage contains the input's — a cover
    may over-claim, never false-negative — and the never-bridge law holds at
    bucket resolution: a real gap of at least one bucket survives exactly,
    while a sub-bucket gap may close. The one clamp is the scale ceiling:
    the top bucket's end exceeds the grammar's maximum encodable end
    (``mortie.TOC_MAX_NS``), so it is clamped there — still containing every
    encodable input word, because no encoder-produced envelope reaches past
    the ceiling either.

    Quantization commutes with union and with the envelope join
    (``toc_reduce``), which is what makes the per-leaf fold exact
    (:func:`read_leaf_temporal`) and the §10.5 parity invariant testable.
    Junk words carry the grammar's own posture: garbage in, garbage out,
    deterministically.
    """
    from mortie import TOC_MAX_NS, span2toc, toc2time, toc_normalize

    if not 0 <= int(order) <= 31:
        raise ValueError(
            f"temporal order {order} is outside [0, 31] — coarser than the scale, or a "
            f"bucket finer than the grammar's 2^32 ns end grid can encode (spec §10.5)"
        )
    words = np.asarray(words, dtype=np.uint64)
    if words.size == 0:
        return words
    k = np.uint64(63 - int(order))
    start, end = toc2time(words)
    start = np.atleast_1d(np.asarray(start, dtype=np.uint64))
    end = np.atleast_1d(np.asarray(end, dtype=np.uint64))
    lo = (start >> k) << k
    # A timestamp decodes to (t, t); a range's decoded end is exclusive. The
    # last covered instant is therefore max(end, start + 1) - 1, uniformly.
    last = np.maximum(end, start + np.uint64(1)) - np.uint64(1)
    hi = np.minimum(((last >> k) + np.uint64(1)) << k, np.uint64(TOC_MAX_NS))
    return toc_normalize(span2toc(lo, hi - np.uint64(1)))


def _cap_cover(cover: np.ndarray, order: int, cap: int = COVER_CAP) -> tuple[np.ndarray, int]:
    """Coarsen a cover by order until it fits the §10.5 cap.

    Each step halves the bucket count (order − 1 doubles the span), so the
    loop terminates: order 0 is a single bucket. The same widening law as
    the quantization itself — never a truncation of the word list, which
    would silently drop coverage.
    """
    while len(cover) > cap and order > 0:
        order -= 1
        cover = quantize_words(cover, order)
    return cover, order


def build_cover_section(contributions: dict, fields, shard_order: int, *, source="sweep"):
    """The ``zagg-coverage-toc-cover/1`` object body from per-leaf contributions.

    Same ``contributions`` mapping :func:`build_temporal_section` folds —
    this consumes the tuples' ``cover`` element: per shard, the union of its
    window leaves' covers, requantized at the pinned day order (canonical,
    and exact: quantization commutes with union) and coarsened to the cap.
    ``None`` for an empty map — the standing absence rule, so a store with
    no temporal channel gets no sibling object at all.

    A shard that had to coarsen below :data:`TEMPORAL_DAY_ORDER` records the
    order it landed at in its own block (``temporal_order``), and the
    coarsening is logged — §10.5's "widening only, loudly recorded".
    """
    from zagg.hive import _utcnow

    if not contributions:
        return None
    shards: dict[str, dict] = {}
    for decimal in sorted(contributions):
        cover = np.concatenate([np.asarray(p[3], dtype=np.uint64) for p in contributions[decimal]])
        cover = quantize_words(cover, TEMPORAL_DAY_ORDER)
        cover, order = _cap_cover(cover, TEMPORAL_DAY_ORDER)
        if order != TEMPORAL_DAY_ORDER:
            logger.warning(
                f"coverage[toc]: shard {decimal} cover coarsened to temporal order {order} "
                f"(span 2^{63 - order} ns) to fit the {COVER_CAP}-word cap (spec §10.5)"
            )
        shards[decimal] = _encode_cover_block(cover, order)
    return {
        "spec": COVER_SPEC,
        "source": source,
        "generated_at": _utcnow(),
        "order": int(shard_order),
        "temporal_order": TEMPORAL_DAY_ORDER,
        "cap": COVER_CAP,
        "fields": sorted(fields),
        "element": {"dtype": "uint64", "shape": [-1]},
        "encoding": "base64",
        "shards": shards,
    }


def _encode_cover_block(cover: np.ndarray, order: int) -> dict:
    """One shard's block: base64 of the §1.4 element bytes, plus its k.

    ``count`` is the §10.5 MUST-check (the §10.3 ``centroids`` rule, one
    buffer instead of two); ``temporal_order`` appears only when the shard
    coarsened below the object's pinned order — absence means the pin.
    """
    from zagg.sweep_overview import encode_digest

    block = {
        "words": base64.b64encode(encode_digest(np.asarray(cover, np.uint64), "uint64")).decode(
            "ascii"
        ),
        "count": int(len(cover)),
    }
    if order != TEMPORAL_DAY_ORDER:
        block["temporal_order"] = int(order)
    return block


def _decode_cover_block(decimal: str, block) -> tuple[np.ndarray, int]:
    """One shard's ``(words, temporal_order)``, MUST-checked against ``count``."""
    from zagg.sweep_overview import decode_digest

    if not isinstance(block, dict):
        raise ValueError(f"cover shard {decimal} block is not an object (spec §10.5)")
    words = decode_digest(base64.b64decode(block["words"]), "uint64", ())
    if block.get("count") != len(words):
        raise ValueError(
            f"cover shard {decimal} declares {block.get('count')!r} words and decodes "
            f"{len(words)} — the block must agree with its own buffer (spec §10.5)"
        )
    return words, int(block.get("temporal_order", TEMPORAL_DAY_ORDER))


def merge_cover_sections(existing, incoming):
    """Compose two cover bodies for the sibling object's GET-union-PUT.

    Per shard: the union of the two word sets, requantized at the coarser of
    the two recorded orders (union alone could interleave two orders' bucket
    bounds, and the §10.5 parity invariant is a claim at ONE order), then
    re-capped — which may coarsen further, under the same widening law. A
    shard on one side carries over unchanged. The unknown-revision rules are
    the §10.4 ones verbatim: an incoming unknown revision contributes
    nothing; a standing one is preserved and never downgraded.
    """
    a, b = _cover_usable(existing), _cover_usable(incoming)
    if b is None:
        return dict(a) if a is not None else _cover_preserved(existing)
    if a is None:
        newer = _cover_preserved(existing)
        if newer is not None:
            logger.warning(
                f"coverage[toc]: keeping the standing {newer.get('spec')!r} cover — "
                f"{COVER_SPEC} does not read it and MUST NOT downgrade it"
            )
            return newer
        return dict(b)
    shards: dict[str, dict] = {}
    for decimal in sorted(set(a.get("shards") or {}) | set(b.get("shards") or {})):
        sides = [
            _decode_cover_block(decimal, side["shards"][decimal])
            for side in (a, b)
            if decimal in (side.get("shards") or {})
        ]
        if len(sides) == 1:
            ((words, order),) = sides
        else:
            order = min(o for _, o in sides)
            words = quantize_words(np.concatenate([w for w, _ in sides]), order)
        words, order = _cap_cover(words, order)
        shards[decimal] = _encode_cover_block(words, order)
    return {
        "spec": COVER_SPEC,
        "source": b.get("source", a.get("source")),
        "generated_at": b.get("generated_at", a.get("generated_at")),
        "order": b.get("order", a.get("order")),
        "temporal_order": TEMPORAL_DAY_ORDER,
        "cap": COVER_CAP,
        "fields": sorted(set(a.get("fields") or []) | set(b.get("fields") or [])),
        "element": {"dtype": "uint64", "shape": [-1]},
        "encoding": "base64",
        "shards": shards,
    }


def cover_unchanged(existing, incoming) -> bool:
    """Whether composing ``incoming`` into ``existing`` would change nothing.

    The sibling object's half of the MOC family's skip-if-current test,
    built exactly as :func:`section_unchanged` is: on the MERGE's content
    (the per-shard blocks and the field list), never on the carrier fields
    that churn per pass.
    """
    merged = merge_cover_sections(existing, incoming)
    return _cover_content(merged) == _cover_content(
        existing if isinstance(existing, dict) else None
    )


def _cover_content(section) -> tuple | None:
    if section is None:
        return None
    shards = {
        d: (b.get("words"), b.get("count"), b.get("temporal_order"))
        for d, b in (section.get("shards") or {}).items()
        if isinstance(b, dict)
    }
    return (shards, sorted(section.get("fields") or []), section.get("order"))


def _cover_usable(section) -> dict | None:
    if not isinstance(section, dict) or section.get("spec") != COVER_SPEC:
        if section is not None:
            logger.debug("coverage[toc]: ignoring a cover with an unknown spec")
        return None
    return section


def _cover_preserved(section) -> dict | None:
    """A standing cover at a spec marker this revision does not implement."""
    if not isinstance(section, dict):
        return None
    spec = section.get("spec")
    if isinstance(spec, str) and spec and spec != COVER_SPEC:
        return dict(section)
    return None


def write_cover(store_root: str, section: dict | None, *, replace: bool = False, **store_kwargs):
    """PUT the ``coverage.toc`` sibling, composing across the standing object.

    The default is the same GET-union-PUT seam ``coverage.moc`` rides
    (incremental sweeps accumulate; concurrent runs race benignly, last
    writer wins, D9). ``replace=True`` is the refresh escape hatch: the
    caller's walk is authoritative and the standing object is discarded —
    except a standing cover at an UNKNOWN revision, which §10.4's succession
    rule preserves on both paths (a producer never downgrades what it cannot
    read). An unparsable standing object is logged and overwritten, the
    regenerable-cache posture. Returns what was left standing.

    A ``None`` section — :func:`build_cover_section`'s empty-walk answer — is
    a no-op on BOTH arms: §10.5's "a producer with no temporal contribution
    leaves the standing object untouched", so the refresh escape hatch over a
    store with no temporal channel writes nothing rather than clobbering (or
    crashing on ``dict(None)``).
    """
    import json

    import obstore

    from zagg.hive import _read_json, open_object_store

    store = open_object_store(store_root, **store_kwargs)
    try:
        existing = _read_json(store, COVER_NAME)
    except ValueError:
        logger.warning(
            f"existing {COVER_NAME} at {store_root} is not JSON; overwriting "
            f"(regenerable cache — the sweep is the authoritative rebuilder)"
        )
        existing = None
    if section is None:
        # Nothing to say. The default arm reaches the same answer through
        # `merge_cover_sections(existing, None)`; short-circuiting keeps the
        # two arms in agreement and skips a PUT that would only rewrite the
        # standing bytes with themselves.
        return existing if isinstance(existing, dict) else None
    if replace:
        preserved = _cover_preserved(existing)
        if preserved is not None:
            logger.warning(
                f"coverage[toc]: keeping the standing {preserved.get('spec')!r} cover at "
                f"{store_root} — {COVER_SPEC} does not read it and MUST NOT downgrade it"
            )
        merged = preserved if preserved is not None else dict(section)
    else:
        merged = merge_cover_sections(existing, section)
    if merged is None:
        return None
    obstore.put(store, COVER_NAME, json.dumps(merged, indent=1).encode())
    return merged


def read_cover(store_root: str, **store_kwargs):
    """Read the store-root ``coverage.toc``; ``None`` when absent.

    Raises ``ValueError`` on garbage JSON, exactly as the carrier's reader
    does — a corrupt cache must be loud, never a plausible partial answer.
    """
    from zagg.hive import _read_json, open_object_store

    return _read_json(open_object_store(store_root, **store_kwargs), COVER_NAME)


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


def load_cover(obj) -> dict | None:
    """A ``coverage.toc`` body at the revision this module implements, else ``None``.

    The strict-gate-then-degrade rule at object level: an unknown revision, a
    non-dict carrier, or plain absence all read as ``None`` — the cover is an
    accelerator whose truth is in the leaves, so degraded means "fall back to
    tier 1 (or open leaves)", never a refusal.
    """
    return _cover_usable(obj)


def cover_words(obj) -> dict[str, np.ndarray] | None:
    """The decoded per-shard word sets, ``{shard decimal: uint64 words}``.

    ``None`` when ``obj`` is not a readable cover. Each block's ``count`` is
    MUST-checked against its own buffer (§10.5) — the same rule
    :func:`coverage_toc_digest` applies to the digest block's ``centroids``.
    """
    section = load_cover(obj)
    if section is None:
        return None
    return {
        d: _decode_cover_block(d, block)[0] for d, block in (section.get("shards") or {}).items()
    }


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
    "COVER_CAP",
    "COVER_KEY",
    "COVER_NAME",
    "COVER_SPEC",
    "PER_CENTROID",
    "ROOT_TOC_DELTA",
    "TEMPORAL_COVERAGE_SPEC",
    "TEMPORAL_DAY_ORDER",
    "TEMPORAL_KEY",
    "build_cover_section",
    "build_temporal_section",
    "cover_unchanged",
    "cover_words",
    "coverage_toc",
    "coverage_toc_digest",
    "load_cover",
    "load_temporal_coverage",
    "merge_cover_sections",
    "merge_temporal_sections",
    "quantize_words",
    "read_cover",
    "read_leaf_temporal",
    "section_unchanged",
    "shards_overlapping",
    "temporal_cell_order",
    "temporal_fields",
    "write_cover",
]
