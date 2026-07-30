"""Hive commit stamp and coverage payloads, leaf through root (issue #330).

Split out of the single-file ``zagg.hive`` (issue #330 phase 1); the public
surface is unchanged and re-exported from :mod:`zagg.hive`. Holds the D4 commit
stamp, the §4 tier-0 coverage envelope and its exact-occupancy bitmap sidecar
(issue #200 phase 2), and the store-root ranges MOC rollup (issue #200 phase 3).
"""

from __future__ import annotations

import json
import logging

import numpy as np
import zarr
from zarr.errors import GroupNotFoundError

from zagg.hive.layout import HIVE_SPEC, HIVE_SPEC_V2, _read_json, _utcnow
from zagg.store import open_object_store
from zagg.windows import split_leaf_name, union_time_range

logger = logging.getLogger(__name__)

#: Root-group attrs key carrying the commit stamp (D4).
COMMIT_ATTR = "morton_hive_commit"
#: Convention version of the stamp's coverage payload (§4 tier 0, issue #200).
COVERAGE_SPEC = "morton-moc/1"
#: Fixed slot count of the tier-0 morton box (2-4 members, null-padded).
COVERAGE_BOX_SLOTS = 4
#: In-leaf occupancy-bitmap sidecar object name (issue #200 phase 2, O8) —
#: the one recorded exception to the "vanilla zarr v3 leaf" claim: a foreign
#: key inside ``{full_id}.zarr/`` that zarr readers ignore (data reads are
#: unaffected; ``members()``/``tree()`` emit a ``ZarrUserWarning`` and skip
#: it — review finding, PR #208 round 2).
COVERAGE_SIDECAR = "coverage.moc"
#: zstd level for the sidecar bitmap — fixed so identical occupancy produces
#: byte-identical sidecars across workers and backends.
_ZSTD_LEVEL = 3
#: Store-ROOT coverage object name (issue #200 phase 3): the shard-order MOC
#: for the one-GET bootstrap — the second root-only exception to the node
#: invariant, next to the manifest. Same name as the in-leaf sidecar
#: (:data:`COVERAGE_SIDECAR`), different location and encoding.
ROOT_COVERAGE_NAME = "coverage.moc"


def build_coverage(
    shard_key, occupied, cell_order: int, *, bitmap: bytes | None = None, full: bool = False
) -> dict:
    """Coverage payload for one shard's commit stamp (§4, issue #200).

    ``occupied`` is the shard's occupied cell words (mixed order allowed —
    the cells ``cells_with_data`` counts); the box is their canonical
    <= 4-member cover (:func:`zagg.grids.morton.morton_box`). ``None``/empty
    falls back to the trivial 1-member cover, the shard id itself — always a
    valid ancestor of its own coverage. Members are serialized as decimal
    morton strings (D1), padded to exactly :data:`COVERAGE_BOX_SLOTS` slots
    with trailing ``None`` (JSON null) sentinels — the recorded pad lean.
    ``cell_order`` records the order occupancy was measured at; ``source``
    the producer (``"worker"`` at the leaf tier — phase-3 root and
    sweep-composed payloads record theirs). ``generated_at`` is DELIBERATELY
    omitted at the leaf (review finding, PR #208): the payload rides the
    commit stamp, whose ``written_at`` is the one clock and one writer;
    root/ancestor carriers add their own timestamp fields under this same
    spec (per-carrier-optional).

    ``bitmap`` (phase 2, the O8 resolution) is the encoded sidecar payload
    from :func:`encode_coverage_bitmap`; when given the envelope grows the
    ``encoding``/``sidecar`` pointer plus compressed/raw byte sizes. A
    box-only envelope (``None``, the phase-1 shape) omits those keys — a
    reader treats their absence as "box only". Raises ``ValueError`` if the
    box escapes the shard's subtree (occupied cells from another shard are
    an upstream bug, never stamped).

    ``full`` (issue #246, D14) marks whole-subtree coverage: the ``encoding``
    discriminator becomes ``"full"`` and NO sidecar is written or pointed to
    — the shard id itself is the exact MOC, so readers skip the sidecar GET
    entirely. Decided by one popcount at stamp time (the caller's job);
    mutually exclusive with ``bitmap``.
    """
    if full and bitmap is not None:
        raise ValueError(
            "full=True and a bitmap payload are mutually exclusive: a fully "
            "occupied subtree writes no sidecar (D14)"
        )
    from zagg.grids.morton import morton_box, morton_decimal

    shard = morton_decimal(shard_key)
    if occupied is None or len(occupied) == 0:
        labels = [shard]
    else:
        labels = [morton_decimal(w) for w in morton_box(occupied)]
    if len(labels) > COVERAGE_BOX_SLOTS or any(not s.startswith(shard) for s in labels):
        raise ValueError(
            f"coverage box {labels} escapes shard {shard}'s subtree — occupied "
            f"cells must be the shard's own (the shard id is always a valid "
            f"trivial cover, so this is an upstream cell-assignment bug)"
        )
    coverage = {
        "spec": COVERAGE_SPEC,
        "box": labels + [None] * (COVERAGE_BOX_SLOTS - len(labels)),
        "cell_order": int(cell_order),
        "source": "worker",
    }
    if full:
        coverage["encoding"] = "full"
    elif bitmap is not None:
        n_bits = 4 ** (int(cell_order) - _decimal_order(shard))
        coverage.update(
            encoding="bitmap",
            sidecar=COVERAGE_SIDECAR,
            nbytes=len(bitmap),
            raw_nbytes=-(-n_bits // 8),
        )
    return coverage


def _decimal_order(decimal: str) -> int:
    """HEALPix order of a D1 decimal id (one digit per level past the base)."""
    return len(decimal) - (2 if decimal.startswith("-") else 1)


def _cell_ranks(shard: str, cells, cell_order: int) -> np.ndarray:
    """Bit index of each cell in the shard-subtree bitmap (frozen convention).

    Bit ``i`` is the i-th cell of the shard subtree at ``cell_order`` in
    ascending packed-word (Z-)order — equivalently the base-4 value of the
    cell's D1 digit tail with digits ``1..4`` mapped to ``0..3``. Raises
    ``ValueError`` for a cell outside the subtree or not at ``cell_order``
    (the bitmap is exact-order by construction; there is nothing conservative
    to fall back to).
    """
    from zagg.grids.morton import to_morton_array

    depth = int(cell_order) - _decimal_order(shard)
    ranks = np.empty(len(cells), dtype=np.int64)
    for i, dec in enumerate(to_morton_array(cells).decimal_repr()):
        tail = dec[len(shard) :]
        if not dec.startswith(shard) or len(tail) != depth:
            raise ValueError(
                f"cell {dec} is not an order-{cell_order} cell of shard {shard}; "
                f"the coverage bitmap encodes exact cell-order occupancy only"
            )
        rank = 0
        for ch in tail:
            rank = rank * 4 + (int(ch) - 1)
        ranks[i] = rank
    return ranks


def encode_coverage_bitmap(shard_key, occupied, cell_order: int) -> bytes:
    """zstd-compressed exact occupancy bitmap for one shard (issue #200 phase 2).

    The O8-resolved leaf encoding: a bit field over the shard subtree at
    ``cell_order`` — ``4^(cell_order - shard_order)`` bits, bit ``i`` per the
    :func:`_cell_ranks` convention (ascending packed-word order; base-4 digit
    tail), packed MSB-first within each byte (``np.packbits``), zstd-
    compressed at a fixed level. Raw size is deterministic
    (``ceil(4^depth / 8)`` bytes) regardless of fragmentation — the property
    that beat coarsen-to-fit ranges in the #202 item (6) measurement; the
    bit-order convention freezes with the mortie-side spec. zstd rides
    numcodecs, already in the tree via zarr's codec stack — no new
    dependency.
    """
    from numcodecs import Zstd

    from zagg.grids.morton import morton_decimal

    shard = morton_decimal(shard_key)
    depth = int(cell_order) - _decimal_order(shard)
    if depth <= 0:
        raise ValueError(f"cell_order {cell_order} is not below shard {shard}'s order")
    # Staging is one uint8 per BIT — 8x the raw bitmap (1 MB at the design
    # point: order-9 shards, order-19 cells). It is bounded by the shard's
    # cell count, which the worker already materializes for the leaf
    # template, so no extra guard here; coarse-shard + deep-cell configs
    # beyond that envelope are out of scope (review note, PR #208 round 2).
    bits = np.zeros(4**depth, dtype=np.uint8)
    bits[_cell_ranks(shard, occupied, cell_order)] = 1
    return bytes(Zstd(level=_ZSTD_LEVEL).encode(np.packbits(bits).tobytes()))


def decode_coverage_bitmap(payload: bytes, shard_key, cell_order: int) -> np.ndarray:
    """Occupied cell words from a sidecar bitmap payload (issue #200 phase 2).

    The inverse of :func:`encode_coverage_bitmap`: returns the sorted packed
    ``uint64`` cell words at ``cell_order`` whose bits are set — exact
    occupancy, no over-coverage. Posture (review finding, PR #208 round 2):
    a CORRUPT payload — zstd garbage, or a decompressed size that is not the
    exact raw bitmap size for the depth — raises loudly rather than
    zero-padding/truncating to a plausible partial cell set (a false
    negative, the one thing D9 forbids; the exact truth is intact in the
    leaf, so surfacing beats under-reporting). A MISSING sidecar degrades to
    ``None`` in :func:`read_coverage_bitmap`.
    """
    from numcodecs import Zstd

    from zagg.grids.morton import morton_decimal, morton_words_from_decimals

    shard = morton_decimal(shard_key)
    depth = int(cell_order) - _decimal_order(shard)
    raw = np.frombuffer(bytes(Zstd().decode(payload)), dtype=np.uint8)
    expected = -(-(4**depth) // 8)
    if raw.size != expected:
        raise ValueError(
            f"coverage sidecar decompressed to {raw.size} B; an order-{cell_order} bitmap "
            f"for shard {shard} is exactly {expected} B — refusing to zero-pad or truncate "
            f"(a partial cell set would be a false negative)"
        )
    bits = np.unpackbits(raw, count=4**depth)
    decimals = [shard + _rank_tail(int(rank), depth) for rank in np.flatnonzero(bits)]
    return np.sort(morton_words_from_decimals(decimals))


def write_coverage_sidecar(leaf_root: str, payload: bytes, **store_kwargs) -> None:
    """PUT the occupancy bitmap sidecar into a leaf (issue #200 phase 2).

    One object at ``{leaf}/coverage.moc`` — the recorded exception to the
    vanilla-v3 leaf, ignored by zarr readers (member enumeration warns and
    skips it; data reads are unaffected). Written BEFORE the commit
    stamp so the stamp stays the leaf's FINAL write (D4): in an unstamped
    prefix the sidecar is debris like everything else, and the wholesale
    retry re-template clears it.
    """
    import obstore

    obstore.put(open_object_store(leaf_root, **store_kwargs), COVERAGE_SIDECAR, payload)


def read_coverage_bitmap(
    leaf_root: str, *, coverage: dict | None = None, **store_kwargs
) -> np.ndarray | None:
    """A leaf's exact occupied cell words from its sidecar, or ``None``.

    Gates on the committed stamp's envelope (:func:`read_coverage`): no
    stamp, a box-only phase-1 payload (no ``encoding``/``sidecar`` keys), an
    unknown encoding, or a missing sidecar object all read ``None`` — the
    box is then the only index and readers degrade per D9, never to wrong
    answers. An ``encoding: "full"`` envelope (issue #246, D14) also reads
    ``None`` here — there IS no sidecar; the shard id itself is the exact
    MOC and :func:`zagg.coverage.bitmap_and` short-circuits on it. Pass an
    already-read ``coverage`` envelope to skip the stamp GET. A PRESENT-but-corrupt sidecar raises instead (see
    :func:`decode_coverage_bitmap` — degrading a corrupt payload would be
    indistinguishable from healthy box-only coverage). The shard id comes
    from the leaf basename — ``{full_id}.zarr``, or the windowed
    ``{full_id}_{window}.zarr`` (issue #246) — via the frozen first-``_``
    split; ``cell_order`` from the envelope. One GET, paid only by readers
    that want cell-level filtering.
    """
    import obstore
    from obstore.exceptions import NotFoundError

    from zagg.grids.morton import morton_word
    from zagg.store import open_store

    if coverage is None:
        coverage = read_coverage(open_store(leaf_root, **store_kwargs))
    if not coverage or coverage.get("encoding") != "bitmap" or not coverage.get("sidecar"):
        return None
    # Windowed leaves (issue #246) carry `{full_id}_{window}.zarr` basenames;
    # the shard id is the part before the first `_` (the frozen parse rule).
    shard = morton_word(split_leaf_name(leaf_root.rstrip("/").rsplit("/", 1)[-1])[0])
    store = open_object_store(leaf_root, **store_kwargs)
    try:
        data = obstore.get(store, str(coverage["sidecar"])).bytes()
    except (FileNotFoundError, NotFoundError):
        return None
    return decode_coverage_bitmap(bytes(data), shard, int(coverage["cell_order"]))


def stamp_commit(
    leaf_store,
    *,
    cells_with_data: int,
    granule_count: int,
    coverage: dict | None = None,
    window: str | None = None,
    time_range: tuple | list | None = None,
) -> None:
    """Stamp a shard leaf complete — the shard's FINAL write (D4).

    One small PUT rewriting the leaf's root ``zarr.json`` (which the template
    already created), not consolidation. Until this lands, the leaf prefix is
    debris: a worker that dies mid-shard leaves no stamp, and a retry may
    overwrite the prefix wholesale. ``coverage`` (issue #200) attaches the
    tier-0 payload from :func:`build_coverage`; ``None`` writes the
    pre-coverage stamp unchanged.

    ``window``/``time_range`` (issue #246, D15): a windowed leaf's stamp is
    the TRUTH half of the temporal split — the window label plus the actual
    ``[t_min, t_max]`` written, as ISO-8601 UTC strings (ratified #246 Q2;
    the manifest keeps only the static schedule). A windowed stamp declares
    ``spec: "morton-hive/2"``; unwindowed stamps stay ``/1`` byte-identical.
    ``time_range`` without ``window`` is rejected (no unwindowed extent claim).
    """
    if window is None and time_range is not None:
        raise ValueError(
            "time_range rides windowed stamps only (D15: unwindowed leaves make "
            "no extent claim in the stamp); pass window= as well"
        )
    group = zarr.open_group(leaf_store, path="", mode="r+", zarr_format=3)
    stamp: dict = {
        "spec": HIVE_SPEC if window is None else HIVE_SPEC_V2,
        "complete": True,
        "cells_with_data": int(cells_with_data),
        "granule_count": int(granule_count),
        "written_at": _utcnow(),
    }
    if window is not None:
        stamp["window"] = str(window)
        if time_range is not None:
            # The stamp is the D15 TRUTH half — it fails CLOSED on a bad range
            # (unlike the fail-open cache union). Validate a 2-sequence of
            # parseable UTC instants with t_min <= t_max before it becomes
            # durable truth; the production path already builds this via
            # windows.iso_time_range (worker min/max, ordered), so this guards
            # direct callers. (review finding, PR #248)
            from zagg.windows import parse_utc

            try:
                lo, hi = time_range
                lo_dt, hi_dt = parse_utc(lo), parse_utc(hi)
            except (TypeError, ValueError) as e:
                raise ValueError(
                    f"time_range must be a 2-sequence of parseable UTC instants "
                    f"[t_min, t_max]; got {time_range!r} ({e})"
                ) from e
            if lo_dt > hi_dt:
                raise ValueError(
                    f"time_range is reversed: t_min {lo!r} follows t_max {hi!r} "
                    f"(the D15 stamp is truth and must fail closed on a bad range)"
                )
            stamp["time_range"] = [str(t) for t in time_range]
    if coverage is not None:
        stamp["coverage"] = coverage
    group.attrs[COMMIT_ATTR] = stamp


def read_commit(leaf_store) -> dict | None:
    """The leaf's commit stamp, or ``None`` for debris / absent leaves (D4).

    Absence (no root group at all) and an unstamped root are the same answer:
    the shard is not complete. Presence requires the stamp — never infer
    completeness from the ``.zarr/`` prefix existing.
    """
    try:
        group = zarr.open_group(leaf_store, path="", mode="r", zarr_format=3)
    except (FileNotFoundError, GroupNotFoundError):
        return None
    stamp = group.attrs.get(COMMIT_ATTR)
    # A malformed (non-mapping) stamp is debris too — never half-trusted.
    return dict(stamp) if isinstance(stamp, dict) else None


def read_coverage(leaf_store) -> dict | None:
    """The leaf's tier-0 coverage payload, or ``None`` when absent (issue #200).

    Rides :func:`read_commit`: debris and absent leaves read ``None``, and so
    does a committed pre-coverage stamp (issue #199 stores carry no
    ``coverage`` key) — older stores keep reading fine. STRICT on the spec
    (review finding, PR #208): only ``spec == "morton-moc/1"`` payloads are
    returned; a malformed dict or an unknown/future spec reads as absent
    rather than half-parsed, so a new envelope version must be adopted here
    deliberately instead of leaking through to box consumers. Box members are
    decimal morton strings; parse one back with
    :func:`zagg.grids.morton.morton_word`.
    """
    stamp = read_commit(leaf_store)
    if stamp is None:
        return None
    coverage = stamp.get("coverage")
    if not isinstance(coverage, dict) or coverage.get("spec") != COVERAGE_SPEC:
        return None
    return dict(coverage)


def _decimal_base(decimal: str) -> str:
    """The ``{sign+base}`` component of a D1 decimal id."""
    return decimal[:2] if decimal.startswith("-") else decimal[:1]


def _decimal_rank(decimal: str) -> int:
    """Base-4 value of a D1 digit tail (digits ``1..4`` -> ``0..3``)."""
    rank = 0
    for ch in decimal[len(_decimal_base(decimal)) :]:
        rank = rank * 4 + (int(ch) - 1)
    return rank


def _rank_tail(rank: int, depth: int) -> str:
    """Inverse of :func:`_decimal_rank`: the width-``depth`` digit tail."""
    digits = []
    for _ in range(depth):
        digits.append(str(rank % 4 + 1))
        rank //= 4
    return "".join(reversed(digits))


def build_root_coverage(
    shard_keys, order: int, *, source: str = "dispatcher", time_range: tuple | list | None = None
) -> dict:
    """Store-root coverage envelope from completed shard keys (issue #200 phase 3).

    The O1 serialization: JSON ranges under the ``morton-moc/1`` envelope,
    with ``encoding: "ranges"`` (vs the leaf sidecar's ``"bitmap"``), the
    shard ``order``, ``source`` and ``generated_at`` — the root carrier's
    staleness discriminators (per-carrier fields under the same spec; the
    leaf payload deliberately omits them, see :func:`build_coverage`). A
    range is an inclusive ``[first, last]`` run of same-order cells within
    ONE base cell, consecutive in base-4 digit-tail rank (ascending
    packed-word order — the bitmap's rank convention at the root). Endpoints
    are D1 decimal STRINGS: packed u64 words exceed 2^53, so raw JSON
    numbers would be silently mangled by any float-based parser (O1).

    ``time_range`` (issue #246, D15): the root summary optionally carries the
    ``[min, max]`` ISO-8601 UTC union of the run's leaf-stamp time ranges —
    CACHE, never truth (the per-leaf stamps are the truth; the walk and the
    sweep regenerate this). Omitted for unwindowed stores, keeping their root
    object byte-identical to pre-#246 runs.
    """
    from zagg.grids.morton import to_morton_array

    words = np.unique(np.asarray(shard_keys, dtype=np.uint64))
    if words.size == 0:
        raise ValueError("build_root_coverage requires at least one shard key")
    decs = list(to_morton_array(words).decimal_repr())
    bad = [d for d in decs if _decimal_order(d) != int(order)]
    if bad:
        raise ValueError(f"shard keys {bad[:3]} are not at shard order {order}")
    # np.unique sorts by packed word; at a fixed order the words of one base
    # cell are contiguous and rank-ascending, so one linear pass finds runs.
    ranges = []
    start = prev = decs[0]
    for dec in decs[1:]:
        same_run = (
            _decimal_base(dec) == _decimal_base(prev)
            and _decimal_rank(dec) == _decimal_rank(prev) + 1
        )
        if same_run:
            prev = dec
            continue
        ranges.append([start, prev])
        start = prev = dec
    ranges.append([start, prev])
    envelope = {
        "spec": COVERAGE_SPEC,
        "encoding": "ranges",
        "order": int(order),
        "source": source,
        "generated_at": _utcnow(),
        "ranges": ranges,
    }
    if time_range is not None:
        envelope["time_range"] = [str(t) for t in time_range]
    return envelope


def root_coverage_words(envelope: dict) -> np.ndarray:
    """Shard words from a root envelope's ranges (inverse of the builder).

    Raises ``ValueError`` on malformed ranges (base-crossing, wrong order,
    reversed endpoints) — same loud posture as the bitmap decoder: a corrupt
    cache must never yield a plausible partial answer.

    Scale note (review, PR #208 round 3): expansion is O(covered shards) in
    a Python loop — milliseconds at coherent-run scale (the design point,
    shard order <= 11 regional products), but a full-sphere accumulated root
    (~3M order-9 / ~50M order-11 shards) would take minutes worker-side. An
    interval-space union on ``[base, lo_rank, hi_rank]`` triples (O(ranges),
    no word materialization) is the upgrade path if root objects ever reach
    continental-accumulation scale; out of scope here.
    """
    from zagg.grids.morton import morton_words_from_decimals

    order = int(envelope["order"])
    decimals = []
    for lo, hi in envelope["ranges"]:
        base = _decimal_base(lo)
        lo_rank, hi_rank = _decimal_rank(lo), _decimal_rank(hi)
        ok = _decimal_base(hi) == base and lo_rank <= hi_rank
        ok = ok and _decimal_order(lo) == order and _decimal_order(hi) == order
        if not ok:
            raise ValueError(f"malformed coverage range [{lo}, {hi}] at order {order}")
        decimals.extend(base + _rank_tail(r, order) for r in range(lo_rank, hi_rank + 1))
    return np.unique(morton_words_from_decimals(decimals))


def write_root_coverage(store_root: str, envelope: dict, **store_kwargs) -> dict:
    """GET-union-PUT the store-root ``coverage.moc`` (issue #200 phase 3).

    Incremental runs accumulate: a parsable existing object with the same
    spec/encoding/order is UNIONED with ``envelope`` before the PUT. An
    unparsable or incompatible existing object is logged and OVERWRITTEN —
    the root MOC is a regenerable cache (D9): the leaf stamps are the
    durable truth and the §7 sweep is the authoritative rebuilder, so
    merging with garbage would be worse than replacing it. CONCURRENT runs
    race benignly (review finding, PR #208 round 3): GET-union-PUT is not
    atomic and S3 has no compare-and-swap, so the last writer wins and its
    union may miss the loser's shards until the sweep or the next run
    re-unions — accepted under D9/O7 (a missing listing degrades to "reader
    doesn't see the newest run", never a wrong answer; do NOT add a lock).
    Returns the payload actually written.
    """
    import obstore

    store = open_object_store(store_root, **store_kwargs)
    try:
        existing = _read_json(store, ROOT_COVERAGE_NAME)
    except ValueError:
        logger.warning(
            f"existing {ROOT_COVERAGE_NAME} at {store_root} is not JSON; overwriting "
            f"(regenerable cache — the sweep is the authoritative rebuilder)"
        )
        existing = None
    merged = envelope
    if isinstance(existing, dict):
        compatible = (
            existing.get("spec") == envelope.get("spec")
            and existing.get("encoding") == envelope.get("encoding")
            and existing.get("order") == envelope.get("order")
        )
        if compatible:
            try:
                union = np.union1d(root_coverage_words(existing), root_coverage_words(envelope))
                merged = build_root_coverage(
                    union,
                    int(envelope["order"]),
                    source=envelope.get("source", "dispatcher"),
                    # D15: incremental runs accumulate the time union too —
                    # cache semantics identical to the spatial ranges.
                    time_range=union_time_range(
                        existing.get("time_range"), envelope.get("time_range")
                    ),
                )
            except (KeyError, TypeError, ValueError) as e:
                logger.warning(
                    f"existing {ROOT_COVERAGE_NAME} at {store_root} failed to parse ({e}); "
                    f"overwriting (regenerable cache — the sweep rebuilds authoritatively)"
                )
        else:
            logger.warning(
                f"existing {ROOT_COVERAGE_NAME} at {store_root} has an incompatible "
                f"envelope; overwriting (regenerable cache)"
            )
    obstore.put(store, ROOT_COVERAGE_NAME, json.dumps(merged, indent=1).encode())
    return merged


def read_root_coverage(store_root: str, **store_kwargs) -> dict | None:
    """Read the store-root ``coverage.moc``; ``None`` when absent."""
    return _read_json(open_object_store(store_root, **store_kwargs), ROOT_COVERAGE_NAME)
