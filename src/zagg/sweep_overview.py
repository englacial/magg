"""Overview-zarr sweep family: pyramid generation at ancestor nodes (issue #201).

The fourth D22 derived-artifact family (``docs/design/sparse_coverage.md`` §7,
D11/D22/D24): for each manifest-declared overview order *k*, the sweep folds
committed source leaves into an **overview zarr at the ancestor digit node** —
the same leaf structure one order family up (dense per-field arrays + the
``morton`` coordinate + ragged t-digest vlen arrays), at cell order
``cell_order - (shard_order - k)`` (constant depth: cells coarsen 4x per
order of ascent, so the pyramid is the store's resolution axis, partially
materialized — D24).

Field inclusion is gated by the D24 composability class
(:func:`zagg.semantics.field_composability`): ``exact`` fields fold by their
merge law (count/sum by addition, min/max by extremum — byte-equal to direct
aggregation at the coarser order), ``approximate`` fields (t-digests) merge
via the order-independent k-way fold (``np.isclose`` equality class), and
``none`` fields are **excluded** — they exist only at native resolution, with
the absence declared in the manifest's pyramid block (the ruled D24 default;
declared derived summaries are the opt-in, deferred until a roster-kind
consumer exists — issue #265).

Naming and attrs are the ratified D23/D11 layout moczarr plans against:
overviews inherit window naming — ``{window}.zarr`` at the ancestor node,
``all.zarr`` for the unwindowed / all-time fold (the reserved token) — and
every overview carries ``role: overview`` plus source orders, per-field
aggregation methods, and the D22 generation stamp in its root attrs (never
inferred from tree position: a shallow zarr may equally be coarse source).

Like every sweep artifact, overviews are regenerable caches (D9): deleting
every one leaves all leaf reads intact. Discovery is from the run-record
work set plus the root coverage MOC — never a recursive LIST.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

#: Envelope version of the per-node overview attrs payload this module writes.
OVERVIEW_SPEC = "zagg-overview/1"
#: Root-group attrs key carrying the overview provenance payload (D11).
OVERVIEW_ATTR = "zagg_overview"
#: Root-group attrs key classifying the zarr (D11/D24: never inferred from
#: position; source leaves carry no role — absence means source).
ROLE_ATTR = "role"
#: Version string of the manifest ``pyramid`` block this module declares.
PYRAMID_SPEC = "zagg-pyramid/1"
#: Default overview-order spacing (espg-ratified on the issue #201 thread:
#: display schedule is every 2 orders — 16x per step, ~1/15 extra storage).
DEFAULT_SPACING = 2

#: Fold law name recorded for approximate (t-digest) fields: the
#: order-independent k-way merge (issue #279 — deterministic under
#: permutation, unlike a pairwise left-fold).
TDIGEST_LAW = "tdigest_kway"


# ---------------------------------------------------------------------------
# Phase A: per-field up-aggregation kernels (the D24 merge laws over arrays).
# ---------------------------------------------------------------------------


def _is_missing(values: np.ndarray, fill_value) -> np.ndarray:
    """Boolean missing-mask for a dense field slab (fill/NaN sentinel cells)."""
    if values.dtype.kind == "f":
        fill = _fill_scalar(fill_value, values.dtype)
        if np.isnan(fill):
            return np.isnan(values)
        return np.isnan(values) | (values == fill)
    return values == _fill_scalar(fill_value, values.dtype)


def _fill_scalar(fill_value, dtype):
    """The numeric fill sentinel (zarr's ``"NaN"`` JSON token included)."""
    if isinstance(fill_value, str):
        return np.array(fill_value, dtype=dtype)[()]  # "NaN"/"Infinity" tokens
    return np.array(fill_value, dtype=dtype)[()]


#: law -> (reduce ufunc, identity kind). The identity replaces missing cells
#: before the reduce; all-missing groups are restored to the fill afterwards.
_LAW_REDUCERS = {
    "sum": np.add,
    "min": np.minimum,
    "max": np.maximum,
}


def fold_dense(values: np.ndarray, factor: int, law: str, fill_value) -> np.ndarray:
    """Fold a dense per-cell slab ``factor``-to-one under an exact merge law.

    ``values`` is a leaf-ordered 1-D array (row i = the i-th subtree cell in
    ascending packed-word order — the leaf invariant), so ``factor``
    consecutive rows share one coarser parent: the fold is a pure
    ``reshape(-1, factor)`` reduce. Missing cells (the field's fill sentinel,
    NaN for float fields) are excluded from the reduce; an all-missing group
    folds back to the fill. Exact by construction for the D24 exact class:
    addition/extremum over the same values in the same ascending order a
    direct coarser aggregation would visit.
    """
    if law not in _LAW_REDUCERS:
        raise ValueError(f"unknown exact fold law {law!r}; known: {sorted(_LAW_REDUCERS)}")
    values = np.asarray(values)
    if factor <= 0 or values.shape[0] % factor:
        raise ValueError(f"cannot fold {values.shape[0]} cells {factor}-to-one")
    groups = values.reshape(-1, factor)
    missing = _is_missing(groups, fill_value)
    if law == "sum":
        identity = np.zeros((), dtype=groups.dtype)
    elif law == "min":
        ident = np.inf if groups.dtype.kind == "f" else np.iinfo(groups.dtype).max
        identity = np.array(ident, dtype=groups.dtype)
    else:  # max
        ident = -np.inf if groups.dtype.kind == "f" else np.iinfo(groups.dtype).min
        identity = np.array(ident, dtype=groups.dtype)
    out = _LAW_REDUCERS[law].reduce(np.where(missing, identity, groups), axis=1)
    fill = _fill_scalar(fill_value, groups.dtype)
    return np.where(missing.all(axis=1), fill, out).astype(groups.dtype, copy=False)


def combine_dense(a: np.ndarray, b: np.ndarray, law: str, fill_value) -> np.ndarray:
    """Element-wise combine of two partial folds under one exact law.

    The accumulation form of :func:`fold_dense` (interleave the operands and
    fold 2-to-one), for overview cells assembled from more than one source
    leaf: coarser-than-shard overview orders, and the all-time fold's
    cross-window accumulation.
    """
    a, b = np.asarray(a), np.asarray(b)
    stacked = np.stack([a, b], axis=1).reshape(-1)
    return fold_dense(stacked, 2, law, fill_value)


def decode_digest(raw, dtype, inner_shape=(2,)) -> np.ndarray:
    """One cell's ragged payload bytes -> its ``(k, *inner_shape)`` array.

    The D18 vlen framing (raw little-endian C-order values at the element
    dtype); an empty payload decodes to the zero-length array, matching the
    reader (``zagg.readers.tdigest_tensor._decode_cell``).
    """
    dt = np.dtype(dtype).newbyteorder("<")
    return np.frombuffer(bytes(raw), dtype=dt).reshape((-1, *inner_shape))


def encode_digest(digest: np.ndarray, dtype) -> bytes:
    """The inverse of :func:`decode_digest`: C-order little-endian bytes."""
    dt = np.dtype(dtype).newbyteorder("<")
    return np.ascontiguousarray(np.asarray(digest, dtype=dt)).tobytes()


def fold_digests(cell_digests: list, *, delta: int, dtype="float32") -> bytes:
    """Merge one overview cell's accumulated t-digests into its payload bytes.

    The approximate-class fold law (D24): the **order-independent k-way
    merge** (:func:`zagg.stats.tdigest.merge_tdigests_kway`, issue #279), so
    the overview digest is permutation-stable — a re-sweep over unchanged
    leaves reproduces identical bytes. An empty accumulation folds to the
    empty payload (the ragged fill).
    """
    from zagg.stats.tdigest import merge_tdigests_kway

    digests = [d for d in cell_digests if len(d)]
    if not digests:
        return b""
    if len(digests) == 1:
        return encode_digest(digests[0], dtype)
    return encode_digest(merge_tdigests_kway(digests, delta=int(delta)), dtype)
