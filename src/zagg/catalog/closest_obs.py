"""Closest-observation ingest builder: cover-driven epochs -> paired shard map.

Issue #509 — the consumer of the #489/#507 ``coverage.toc`` surface. One
Sentinel-2 (or any raster) store serves several point-cloud reference stores
(ATL03 + GEDI): for every reference *epoch* a shard's covers claim, the
builder selects the single **nearest** acquisition from the raster catalog —
closest observation, not a two-sided bracket; multiple passes bracket
naturally (espg design ruling, 2026-08-23/24).

Epochs are **store-derived**: each reference store's ``coverage.toc`` sibling
(spec §10.5) carries per-shard word-set covers quantized at order 18
(2^45 ns ≈ 9.77 h buckets). A cover *word* is a maximal RUN of those buckets
(``toc_normalize`` coalesces ranges that merely abut), so the epochs are the
midpoints of a word's **constituent buckets**, one per bucket — each within
±4.9 h of every instant its bucket covers, against Sentinel-2's ~4.3-day
revisit. Granule catalogs are *not* an
epoch source — the leaf sub-maps record the dispatched assignment verbatim
and inherit the CMR-hull over-assignment (~70 assigned vs 49 contributing
pass-days on shard ``3231422244``-class cases); covers reflect only data
that landed.

The pairing is a property of the ingest *query*, not the store schema: the
raster store stays a plain raster store, which granules were ingested **is**
the pairing, and coincidence at read time is toc intersection.

Word semantics (bit layout, decode, midpoints) are mortie's; the §10.5 cover
accessors are :mod:`zagg.coverage_toc`'s. This module owns only the join:
covers -> epochs, epochs -> nearest acquisitions, and the resulting
:class:`~zagg.catalog.shardmap.ShardMap`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from zagg.coverage_toc import TEMPORAL_COVER_ORDER

logger = logging.getLogger(__name__)


def _word_midpoints(words: np.ndarray, order: int = TEMPORAL_COVER_ORDER) -> np.ndarray:
    """Cover words -> UTC ``datetime64[ns]`` midpoints of their *buckets*.

    A cover word is not one bucket. :func:`zagg.coverage_toc.quantize_words`
    widens each instant to an aligned order-``order`` bucket and then
    canonicalizes with ``toc_normalize``, which "coalesces ranges that merely
    abut" (§10.5) — so a word is a maximal RUN of contiguous buckets and its
    envelope midpoint would name one epoch per *campaign*, not per pass. This
    expands every word back into its constituent buckets and emits one
    midpoint each, which is what restores the ±4.9 h bound: a bucket spans
    ``2**(63 - order)`` ns, so its midpoint is within half a bucket of every
    instant the bucket covers.

    ``toc2time`` decodes a word's conservative envelope ``(start, end)`` on
    the internal-ns scale — ``end`` exclusive for a range, ``end == start``
    for an exact timestamp — so the last covered instant is
    ``max(end, start + 1) - 1``, the same uniform rule ``quantize_words``
    applies, and the covered buckets are ``start >> k`` through ``last >> k``
    inclusive with ``k = 63 - order``. Bucket midpoints use that same rule
    within the bucket: ``(b << k) + 2**(k - 1) - 1``.

    One pass that straddles a bucket edge (its word's envelope reaches into
    the neighbouring bucket) yields **two** epochs rather than one. That is
    benign over-selection, not error: both epochs sit within half a bucket of
    the pass, both pick the same nearest acquisition, and the builder dedupes
    granule ids per shard.

    ``order`` is the block's *effective* temporal order (§10.5 lets a block
    coarsen below the object's pin), defaulting to
    :data:`zagg.coverage_toc.TEMPORAL_COVER_ORDER`.
    """
    import mortie

    words = np.asarray(words, dtype=np.uint64)
    if words.size == 0:
        return np.empty(0, dtype="datetime64[ns]")
    k = int(63 - int(order))
    start, end = mortie.toc2time(words)
    start = np.atleast_1d(np.asarray(start, dtype=np.uint64))
    end = np.atleast_1d(np.asarray(end, dtype=np.uint64))
    last = np.maximum(end, start + np.uint64(1)) - np.uint64(1)
    # Bucket index run [b0, b1] per word; expanded with repeat/arange
    # arithmetic rather than a Python loop over words.
    b0 = (start >> np.uint64(k)).astype(np.int64)
    b1 = (last >> np.uint64(k)).astype(np.int64)
    counts = b1 - b0 + 1
    offsets = np.cumsum(counts) - counts
    within = np.arange(int(counts.sum()), dtype=np.int64) - np.repeat(offsets, counts)
    buckets = np.unique(np.repeat(b0, counts) + within).astype(np.uint64)
    half = np.uint64((1 << k) // 2 - 1)
    mid = np.minimum((buckets << np.uint64(k)) + half, np.uint64(mortie.TOC_MAX_NS))
    return np.asarray(mortie.to_datetime64(mid), dtype="datetime64[ns]")


def _aoi_shard_set(aoi, order: int) -> set[int] | None:
    """Resolve an ``aoi`` argument to the set of shard keys it covers.

    ``None`` passes through (no restriction). Accepted forms, matching the
    catalog layer's existing vocabulary:

    - ``mortie.Moc`` — cast to the flat cell list at ``order``;
    - ``str`` — a GeoJSON path (:func:`zagg.catalog.load_polygon`);
    - ``[(lats, lons), ...]`` ring parts — the ``coverage``/``region`` form.
    """
    if aoi is None:
        return None
    import mortie

    if isinstance(aoi, mortie.Moc):
        return {int(c) for c in aoi.to_order(order)}
    if isinstance(aoi, str):
        from zagg.catalog import load_polygon

        aoi = load_polygon(aoi)
    from zagg.grids.aoi import healpix_aoi_moc

    moc = healpix_aoi_moc(aoi, order)
    return {int(c) for c in mortie.moc_to_order(moc, order)}


def _shard_order(cover: dict, root: str) -> int:
    """A cover object's declared shard (dispatch) ``order``, or a loud refusal.

    §10.5 makes ``order`` mandatory; a body missing it (or carrying a
    non-integer) is debris this builder cannot key shards from, and the bare
    ``int(...)`` it used to get raised an opaque ``TypeError`` naming nothing.
    """
    value = cover.get("order")
    try:
        return int(value)
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"reference_epochs: store {root!r} declares a non-integer cover shard order "
            f"{value!r} — §10.5 requires it, and shard keys are parsed against it"
        ) from e


@dataclass
class ReferenceEpochs:
    """Per-shard reference epochs decoded from one or more store covers.

    Attributes
    ----------
    order : int
        The covers' shard (dispatch) order — every contributing store must
        agree on it, and the raster grid the epochs pair against must match.
    epochs : dict of int -> np.ndarray
        Shard key (packed morton word, the canonical in-memory form D1
        renders as a decimal string externally) -> sorted unique UTC
        ``datetime64[ns]`` epoch midpoints, the union across the contributing
        stores' covers. Shards whose cover block decodes to an empty word set
        are omitted (they claim nothing).
    stores : list of str
        The store roots that contributed, in the order given (provenance).
    orders : dict of int -> int
        Shard key -> the **coarsest** temporal order that contributed to that
        shard's epochs. Normally the pinned
        :data:`~zagg.coverage_toc.TEMPORAL_COVER_ORDER`, but §10.5 lets a
        block coarsen below the pin to fit the cover cap, and a coarser order
        means a wider midpoint bound (``2**(62 - order)`` ns). Phase 2's
        ``max_time_offset`` reasoning reads this rather than assuming the pin.
    """

    order: int
    epochs: dict[int, np.ndarray]
    stores: list[str] = field(default_factory=list)
    orders: dict[int, int] = field(default_factory=dict)

    @property
    def total(self) -> int:
        """Total epoch count across shards (post-union, post-dedupe)."""
        return sum(e.size for e in self.epochs.values())

    def tolerance(self, shard: int) -> np.timedelta64:
        """Half a bucket at ``shard``'s effective order — its epoch bound."""
        order = self.orders.get(shard, TEMPORAL_COVER_ORDER)
        return np.timedelta64(2 ** (62 - int(order)), "ns")


def reference_epochs(reference_stores, *, aoi=None, **store_kwargs) -> ReferenceEpochs:
    """Per-shard epochs from the reference stores' ``coverage.toc`` covers.

    For each store root: fetch the §10.5 sibling
    (:func:`zagg.coverage_toc.read_cover`), strict-decode its per-shard word
    sets (:func:`zagg.coverage_toc.cover_words`), and expand each word into
    its constituent buckets' midpoints (:func:`_word_midpoints`). Per shard
    the result is the **union** across stores, deduplicated (espg ruling: one
    raster store serves both sensors, epochs are the union across the
    reference stores).

    The union is canonical at the **bucket** level, not the word level. Words
    are post-``toc_normalize`` runs whose extent depends on that store's
    *other* data, so two stores sharing one pass routinely emit
    overlapping-but-**unequal** range words — a raw ``np.unique`` over words
    would keep both and represent the shared pass twice, at two displaced
    midpoints. Expanding to buckets first removes that degree of freedom:
    the bucket grid is fixed by the order alone, so a shared pass contributes
    the same bucket midpoint from every store and dedupes exactly.

    §10.5 lets a block coarsen **below** the object's pinned temporal order to
    fit the cover cap, and its landed order is recorded in the block itself.
    :func:`zagg.coverage_toc.cover_words` returns only the words, so this
    reads each block's ``temporal_order`` straight from the grammar, expands
    that block's words at **its** order, logs a warning whenever a
    block sits below the pin (the read half of §10.5's "widening only,
    *loudly recorded*"), and reports the coarsest contributing order per
    shard on :attr:`ReferenceEpochs.orders`.

    A store that carries **no readable cover refuses loudly** — this builder
    is cover-driven by design (store-derived epochs, never the granule
    catalogs), so "no cover yet" means "sweep the store first", not "fall
    back silently". Likewise a shard-order mismatch between stores: D1 ids
    at two orders are not comparable.

    Parameters
    ----------
    reference_stores : str or sequence of str
        Store roots whose covers drive the epochs (e.g. ATL03 + GEDI).
    aoi : optional
        Restrict the shard set: a ``mortie.Moc``, a GeoJSON path, or
        ``[(lats, lons), ...]`` ring parts (see :func:`_aoi_shard_set`).
    **store_kwargs
        Forwarded to the object-store open (region, credentials, ...).

    Returns
    -------
    ReferenceEpochs
    """
    from zagg.coverage_toc import COVER_NAME, cover_words, load_cover, read_cover
    from zagg.grids.morton import morton_word

    if isinstance(reference_stores, str):
        reference_stores = [reference_stores]
    reference_stores = list(reference_stores)
    if not reference_stores:
        raise ValueError("reference_epochs: at least one reference store root is required")

    order: int | None = None
    mids_by_shard: dict[int, list[np.ndarray]] = {}
    orders: dict[int, int] = {}
    for root in reference_stores:
        obj = read_cover(root, **store_kwargs)
        cover = load_cover(obj)
        if cover is None:
            detail = (
                "no coverage.toc object"
                if obj is None
                else f"unreadable {COVER_NAME} (spec {obj.get('spec') if isinstance(obj, dict) else obj!r})"
            )
            raise ValueError(
                f"reference_epochs: store {root!r} has {detail} — epochs are cover-driven "
                f"(spec §10.5, issue #509); run the rollup sweep that materializes the "
                f"cover before pairing against this store"
            )
        store_order = _shard_order(cover, root)
        if order is None:
            order = store_order
        elif store_order != order:
            raise ValueError(
                f"reference_epochs: store {root!r} covers shard order {store_order}, "
                f"previous stores cover order {order} — D1 ids at two orders are not "
                f"comparable (spec §10.5)"
            )
        # cover_words strict-decodes (and validates the object's pin); the
        # per-block order it drops is read back off the same grammar.
        decoded = cover_words(obj)
        pinned = int(cover.get("temporal_order", TEMPORAL_COVER_ORDER))
        blocks = cover.get("shards") or {}
        for decimal, words in decoded.items():
            if not len(words):
                continue
            block = blocks.get(decimal) or {}
            effective = int(block.get("temporal_order", pinned))
            if effective < TEMPORAL_COVER_ORDER:
                logger.warning(
                    f"reference_epochs: store {root!r} shard {decimal} cover sits at "
                    f"temporal order {effective}, below the pinned {TEMPORAL_COVER_ORDER} "
                    f"(§10.5 cap coarsening) — its buckets span 2^{63 - effective} ns, so "
                    f"these epochs are good to ±2^{62 - effective} ns, not ±4.9 h"
                )
            # Cover blocks are keyed by the D1 decimal id (the external
            # string form, sign included); shard maps key on the packed
            # morton word — parse at the boundary (issue #199).
            shard = morton_word(decimal)
            mids_by_shard.setdefault(shard, []).append(_word_midpoints(words, effective))
            orders[shard] = min(orders.get(shard, effective), effective)

    assert order is not None  # non-empty store list, every cover carried an order
    keep = _aoi_shard_set(aoi, order)
    epochs: dict[int, np.ndarray] = {}
    for shard in sorted(mids_by_shard):
        if keep is not None and shard not in keep:
            continue
        mids = np.unique(np.concatenate(mids_by_shard[shard]))
        if mids.size:
            epochs[shard] = mids
    return ReferenceEpochs(
        order, epochs, reference_stores, {k: v for k, v in orders.items() if k in epochs}
    )


def nearest_acquisitions(epochs, times, *, max_time_offset=None):
    """Nearest acquisition per epoch — the vectorized closest-1 core.

    The phase-2 selection of the closest-observation join (issue #509): for
    each reference epoch, the single nearest acquisition — closest-1, never a
    two-sided bracket (espg ruling); several epochs bracketing one
    acquisition each select it, and the builder dedupes granules downstream.

    Parameters
    ----------
    epochs : array-like of datetime64[ns]
        One shard's reference epochs (:func:`reference_epochs`). Any order.
        ``NaT`` refuses — see the note below.
    times : array-like of datetime64[ns]
        The shard's acquisition times, in catalog record order. Any order —
        the returned selection indexes THIS array's positions. ``NaT``
        refuses — see the note below.
    max_time_offset : np.timedelta64 or int, optional
        An epoch whose nearest acquisition lies further than this selects
        nothing. Exactly-at selects; one ns past does not. A bare int means
        nanoseconds. ``None`` (default) always selects the nearest, however
        far. Negative refuses, and so does a duration that will not convert
        to nanoseconds (``np.timedelta64(1000, "Y")`` overflows int64 ns). Callers gating against the epochs' own
        precision should widen by :meth:`ReferenceEpochs.tolerance` — a
        cover epoch is a bucket midpoint, good to half a bucket, not exact.

    Returns
    -------
    selection : np.ndarray of int64
        Per epoch, the index into ``times`` of the selected acquisition, or
        ``-1`` where the epoch selects nothing (no acquisitions at all, or
        nearest beyond ``max_time_offset``).
    offsets : np.ndarray of timedelta64[ns]
        Per epoch, the SIGNED offset ``times[nearest] - epoch`` of the
        nearest acquisition — positive when the acquisition follows the
        epoch. Reported for every epoch, dropped ones included (the loud
        record a drop rides — the builder's report needs the near-miss
        distance, not just the fact of the drop). ``NaT`` when ``times`` is
        empty, and for the offset that will not fit ``timedelta64[ns]`` —
        selection is exact over the whole ``datetime64[ns]`` span (~584
        years), but a *difference* past ~292 years is unrepresentable, so it
        saturates to ``NaT`` rather than reporting a wrapped duration. Such
        an epoch selects nothing under any cap (no finite tolerance reaches
        it) and still selects its nearest with ``max_time_offset=None``.

    Raises
    ------
    ValueError
        If either input carries ``NaT``. A missing instant has no nearest
        anything: ``NaT`` sorts last but casts to ``iinfo(int64).min``, so it
        would leave ``ts`` unsorted (silently mis-pairing its neighbours) and
        wrap the offset subtraction. :func:`reference_epochs` never emits
        ``NaT`` and the catalog's time parser refuses a missing acquisition
        time, so ``NaT`` here is caller debris — refused loudly, this
        module's posture.

    Notes
    -----
    A tie — an epoch exactly equidistant between two acquisitions — selects
    the EARLIER acquisition, deterministically. Equal acquisition times (one
    Sentinel-2 datatake stamps many granules with the same instant) are
    broken by catalog record order: the FIRST record of the equal-time run,
    whichever flank the epoch approaches it from.
    """
    epochs = np.asarray(epochs, dtype="datetime64[ns]")
    times = np.asarray(times, dtype="datetime64[ns]")
    for name, arr in (("epochs", epochs), ("times", times)):
        if np.isnat(arr).any():
            raise ValueError(f"{name} carries NaT ({int(np.isnat(arr).sum())} of {arr.size})")
    cap = None
    if max_time_offset is not None:
        offset = np.timedelta64(max_time_offset)
        if np.isnat(offset):
            raise ValueError(f"max_time_offset must be a real duration (got {max_time_offset!r})")
        as_ns = offset.astype("timedelta64[ns]")
        if as_ns.astype(offset.dtype) != offset:
            raise ValueError(
                "max_time_offset does not convert exactly to nanoseconds "
                f"(got {max_time_offset!r}; timedelta64[ns] spans ~292 years)"
            )
        cap = int(as_ns.astype("int64"))
        if cap < 0:
            raise ValueError(f"max_time_offset must be non-negative (got {max_time_offset!r})")
    selection = np.full(epochs.shape, -1, dtype=np.int64)
    offsets = np.full(epochs.shape, np.timedelta64("NaT"), dtype="timedelta64[ns]")
    if epochs.size == 0 or times.size == 0:
        return selection, offsets
    order = np.argsort(times, kind="stable")
    ts = times[order].astype("int64")
    e = epochs.astype("int64")
    pos = np.searchsorted(ts, e)  # left insertion point
    # Flank distances as UNSIGNED magnitudes. Both are non-negative by
    # construction (``e >= ts[pos - 1]`` and ``ts[pos] >= e``), and mod-2^64
    # subtraction of the int64 bit patterns is exact across the whole
    # datetime64[ns] span — an int64 ``e - ts`` wraps silently for a pair
    # more than ~292 years apart, which then passes the cap gate.
    tsu, eu = ts.view(np.uint64), e.view(np.uint64)
    far = np.uint64(2**64 - 1)  # a missing flank, wider than any real span
    left = np.where(pos > 0, eu - tsu[np.maximum(pos - 1, 0)], far)
    right = np.where(pos < ts.size, tsu[np.minimum(pos, ts.size - 1)] - eu, far)
    # Strict ``<`` keeps a tie on the LEFT (earlier) neighbor; an epoch equal
    # to an acquisition has ``right == 0`` and selects it exactly.
    take_right = right < left
    nearest = np.where(take_right, np.minimum(pos, ts.size - 1), np.maximum(pos - 1, 0))
    # Equal acquisition times form one run in ``ts``; the left flank lands on
    # its END and the right flank on its START, so snap to the run start —
    # with a stable ``argsort`` that is the run's first catalog record, from
    # either side.
    nearest = np.searchsorted(ts, ts[nearest], side="left")
    selection = order[nearest].astype(np.int64)
    magnitude = np.where(take_right, right, left)
    # A magnitude past int64 ns is a real distance the report cannot carry:
    # saturate it to NaT rather than wrap it into a plausible-looking day.
    fits = magnitude <= np.uint64(np.iinfo(np.int64).max)
    signed = np.minimum(magnitude, np.uint64(np.iinfo(np.int64).max)).astype(np.int64)
    offsets = np.where(take_right, signed, -signed).astype("timedelta64[ns]")
    offsets = np.where(fits, offsets, np.timedelta64("NaT"))
    if cap is not None:
        selection = np.where(magnitude <= np.uint64(cap), selection, np.int64(-1))
    return selection, offsets


__all__ = ["ReferenceEpochs", "nearest_acquisitions", "reference_epochs"]
