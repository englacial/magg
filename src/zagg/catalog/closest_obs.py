"""Closest-observation ingest builder: cover-driven epochs -> paired shard map.

Issue #509 — the consumer of the #489/#507 ``coverage.toc`` surface. One
Sentinel-2 (or any raster) store serves several point-cloud reference stores
(ATL03 + GEDI): for every reference *epoch* a shard's covers claim, the
builder selects the single **nearest** acquisition from the raster catalog —
closest observation, not a two-sided bracket; multiple passes bracket
naturally (espg design ruling, 2026-08-23/24).

Epochs are **store-derived**: each reference store's ``coverage.toc`` sibling
(spec §10.5) carries per-shard word-set covers quantized at order 18
(2^45 ns ≈ 9.77 h buckets), so a word midpoint names its pass epoch to
±4.9 h against Sentinel-2's ~4.3-day revisit. Granule catalogs are *not* an
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

logger = logging.getLogger(__name__)


def _word_midpoints(words: np.ndarray) -> np.ndarray:
    """Cover words -> UTC ``datetime64[ns]`` midpoints of their envelopes.

    ``toc2time`` decodes a word's conservative envelope ``(start, end)`` on
    the internal-ns scale — ``end`` exclusive for a range, ``end == start``
    for an exact timestamp — so the midpoint of the *covered* instants is
    ``start + (last - start) // 2`` with ``last = max(end, start + 1) - 1``,
    the same uniform last-covered-instant rule
    :func:`zagg.coverage_toc.quantize_words` applies. At the pinned cover
    order (:data:`zagg.coverage_toc.TEMPORAL_COVER_ORDER`) a bucket spans
    2^45 ns, so a midpoint is within ±4.9 h of every instant its word covers.
    """
    import mortie

    words = np.asarray(words, dtype=np.uint64)
    if words.size == 0:
        return np.empty(0, dtype="datetime64[ns]")
    start, end = mortie.toc2time(words)
    start = np.atleast_1d(np.asarray(start, dtype=np.uint64))
    end = np.atleast_1d(np.asarray(end, dtype=np.uint64))
    last = np.maximum(end, start + np.uint64(1)) - np.uint64(1)
    mid = start + (last - start) // np.uint64(2)
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
    """

    order: int
    epochs: dict[int, np.ndarray]
    stores: list[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        """Total epoch count across shards (post-union, post-dedupe)."""
        return sum(e.size for e in self.epochs.values())


def reference_epochs(reference_stores, *, aoi=None, **store_kwargs) -> ReferenceEpochs:
    """Per-shard epochs from the reference stores' ``coverage.toc`` covers.

    For each store root: fetch the §10.5 sibling
    (:func:`zagg.coverage_toc.read_cover`), strict-decode its per-shard word
    sets (:func:`zagg.coverage_toc.cover_words`), and decode each word's
    envelope midpoint (mortie). Per shard the result is the **union** across
    stores, deduplicated — two stores quantized on the same order-18 grid
    yield the same word for the same pass window, so the union is exact,
    never doubled (espg ruling: one raster store serves both sensors, epochs
    are the union across the reference stores).

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
    words_by_shard: dict[int, list[np.ndarray]] = {}
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
        store_order = int(cover.get("order"))
        if order is None:
            order = store_order
        elif store_order != order:
            raise ValueError(
                f"reference_epochs: store {root!r} covers shard order {store_order}, "
                f"previous stores cover order {order} — D1 ids at two orders are not "
                f"comparable (spec §10.5)"
            )
        for decimal, words in cover_words(obj).items():
            if len(words):
                # Cover blocks are keyed by the D1 decimal id (the external
                # string form, sign included); shard maps key on the packed
                # morton word — parse at the boundary (issue #199).
                words_by_shard.setdefault(morton_word(decimal), []).append(
                    np.asarray(words, dtype=np.uint64)
                )

    assert order is not None  # non-empty store list, every cover carried an order
    keep = _aoi_shard_set(aoi, order)
    epochs: dict[int, np.ndarray] = {}
    for shard in sorted(words_by_shard):
        if keep is not None and shard not in keep:
            continue
        words = np.unique(np.concatenate(words_by_shard[shard]))
        mids = np.unique(_word_midpoints(words))
        if mids.size:
            epochs[shard] = mids
    return ReferenceEpochs(order, epochs, reference_stores)


__all__ = ["ReferenceEpochs", "reference_epochs"]
