"""Packed per-cell signal composition — the ``zagg-composition/1`` field (issue #321).

One ``uint64`` per cell carries eight 8-bit lanes of quantized fractions of the
cell's **signal stratum** (``N_signal`` = the signal digest's total weight —
magnitude lives in the digest, composition here):

======  =========================================================
bytes   lanes
======  =========================================================
0–4     per-surface fractions, ``signal_conf_ph`` column order:
        land, ocean, sea_ice, land_ice, inland_water — the count of
        signal photons whose per-surface confidence clears the
        threshold, over ``N_signal``
5–7     low / med / high fractions: signal photons whose *strongest*
        per-surface confidence is exactly 2 / 3 / 4, over ``N_signal``
======  =========================================================

Quantization uses a **presence floor**: ``k = round(255 * c / n)``, except any
nonzero count quantizes to at least 1 — so ``lane > 0`` is exactly "this flag
occurred" at every ``n``, through arbitrary merge chains. Count recovery
``round(k * n / 255)`` is exact for ``n <= 254`` (quantization error
``<= n/510 < 1/2``) — the entire below-compression-knee regime.

The per-surface lanes are **overlapping marginals** (``surf_type`` is
multi-hot): they do not sum to 255 and cannot split the height distribution
per surface. Empty signal stratum packs to 0.

Merge law (order-independent monoid over ``(word, n)`` pairs)::

    lane_merged = quantize((n_a * lane_a + n_b * lane_b) / (n_a + n_b))

with the same presence floor, so presence survives re-quantization exactly and
count error stays O(n/510) per merge.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "COMPOSITION_SPEC",
    "LANES",
    "counts_from_composition",
    "merge_composition",
    "pack_composition",
    "unpack_composition",
]

COMPOSITION_SPEC = "zagg-composition/1"

#: Lane order, LSB byte first: five surfaces (``signal_conf_ph`` column
#: order), then the three confidence levels of the union-signal stratum.
LANES = ("land", "ocean", "sea_ice", "land_ice", "inland_water", "low", "med", "high")

_SURFACES = 5


def _quantize(counts: np.ndarray, n: int) -> np.ndarray:
    """Quantize lane counts to u8 fractions of ``n`` with the presence floor."""
    if n <= 0:
        return np.zeros(len(counts), dtype=np.uint64)
    k = np.rint(255.0 * np.asarray(counts, dtype=np.float64) / n)
    k = np.where((np.asarray(counts) > 0) & (k == 0), 1.0, k)
    return np.clip(k, 0, 255).astype(np.uint64)


def _pack(lanes: np.ndarray) -> int:
    word = np.uint64(0)
    for i, lane in enumerate(lanes):
        word |= np.uint64(lane) << np.uint64(8 * i)
    return int(word)


def unpack_composition(word: int) -> np.ndarray:
    """Unpack a composition word into its eight u8 lanes (``LANES`` order)."""
    w = np.uint64(word)
    return np.array([(w >> np.uint64(8 * i)) & np.uint64(0xFF) for i in range(8)], dtype=np.uint8)


def counts_from_composition(word: int, n_signal: int) -> np.ndarray:
    """Recover per-lane counts; exact whenever ``n_signal <= 254``."""
    lanes = unpack_composition(word).astype(np.float64)
    return np.rint(lanes * n_signal / 255.0).astype(np.int64)


def pack_composition(
    values: np.ndarray,
    *,
    conf_land: np.ndarray,
    conf_ocean: np.ndarray,
    conf_sea_ice: np.ndarray,
    conf_land_ice: np.ndarray,
    conf_inland_water: np.ndarray,
    threshold: int = 2,
) -> int:
    """Per-cell reducer: pack the cell's signal composition into one uint64.

    ``values`` is the height column (the digest fields' ``source``); rows with
    non-finite heights are dropped first so ``N_signal`` here equals the signal
    digest's total weight exactly. The five ``conf_*`` kwargs are the
    row-aligned per-surface ``signal_conf_ph`` columns (delivered by the
    ``params``-as-columns mechanism). A photon is signal when **any** surface
    clears ``threshold`` (the ATBD predicate at the default ``threshold=2``,
    i.e. ``> 1``); its level lane is its *strongest* per-surface confidence.
    """
    values = np.asarray(values, dtype=np.float64)
    conf = np.column_stack(
        [conf_land, conf_ocean, conf_sea_ice, conf_land_ice, conf_inland_water]
    ).astype(np.int64)
    if conf.shape[0] != values.shape[0]:
        raise ValueError(f"conf columns have {conf.shape[0]} rows, values has {values.shape[0]}")
    finite = np.isfinite(values)
    conf = conf[finite]

    signal = (conf >= threshold).any(axis=1)
    n = int(signal.sum())
    if n == 0:
        return 0
    csig = conf[signal]

    counts = np.empty(8, dtype=np.int64)
    counts[:_SURFACES] = (csig >= threshold).sum(axis=0)
    strongest = csig.max(axis=1)
    for i, level in enumerate((2, 3, 4)):
        counts[_SURFACES + i] = int((strongest == level).sum())
    return _pack(_quantize(counts, n))


def merge_composition(word_a: int, n_a: int, word_b: int, n_b: int) -> int:
    """Fold two ``(word, n_signal)`` pairs — the digest-weighted-mean monoid.

    Presence floor preserved: a lane nonzero on either side stays nonzero.
    Identity element is ``(0, 0)``; the operation is symmetric and, up to the
    bounded re-quantization error, associative — fold order does not affect
    presence at all and affects counts only within O(n/510).
    """
    if n_a <= 0:
        return int(word_b)
    if n_b <= 0:
        return int(word_a)
    la = unpack_composition(word_a).astype(np.float64)
    lb = unpack_composition(word_b).astype(np.float64)
    n = n_a + n_b
    merged = (n_a * la + n_b * lb) / n
    k = np.rint(merged)
    k = np.where(((la > 0) | (lb > 0)) & (k == 0), 1.0, k)
    return _pack(np.clip(k, 0, 255).astype(np.uint64))
