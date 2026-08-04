"""``2^n`` morton-subtree partitions for the parallel sweep (issue #377).

The unified sweep (:mod:`zagg.sweep`) is ONE walk over a hive store's digit
tree — one ``mode="sweep"`` worker invoke, whose wall clock and resident
memory both scale with the whole store. This module decomposes that walk into
``2^n`` **isolated** units so a fan-out of workers runs it in parallel, each
unit sized ``ceil(leaves / 2^n)``.

**Digit boundaries only.** A D1 decimal morton id is ``{sign+base}{digit}*``
with one base-4 digit — 2 bits — per order, so a ``2^n`` split lands on a
digit boundary exactly when ``n`` is even: ``n = 2k`` splits at **order k**,
and the partition index is the base-4 value of the id's first ``k`` digits.
``partitions`` is therefore a power of FOUR. An odd ``n`` (2, 8, 32, …
partitions) would halve a digit — two partitions sharing one order-k node's
four children — and is **rejected**, loudly and by name, rather than silently
rounded to a neighbouring power of four (:func:`partition_split_order`); the
bit-level alternative is the open design fork on issue #377.

**Disjointness.** Partition *i* owns every id whose first ``k`` digits rank
*i* — one order-k subtree per HEALPix base cell, twelve in all
(``12 * 4^k / 2^n == 12``). Every node at order ``>= k``, and every leaf and
window beneath it, therefore lies wholly inside ONE partition: no two
partitions can write the same ``(node, window)`` artifact, so the workers need
no coordination and the existing D22 generation stamps keep each partition
independently idempotent and resumable.

**What a partition deliberately does not do.** Nodes at orders COARSER than
``k`` span partitions, as does the store-root ``coverage.moc`` refresh
(:meth:`zagg.sweep.MocFamily.finish`). Those belong to the **coarse-level
finisher** — one invoke after the partitions land, folding from the
partitions' already-materialized overview slabs (issue #377, sequenced behind
the issue #376 cascade). A partitioned pass stops its walk at order ``k`` and
defers the finish hook; a partition worker never writes above the split.

Windowed stores expose a second, free parallelism dimension — the window axis
is already independent per ``(node, window)`` artifact, so per-(partition,
window) invokes decompose further. Noted, not implemented (issue #377 v1).
"""

from __future__ import annotations


def partition_split_order(partitions: int) -> int:
    """Morton order a ``partitions``-way split lands on (``2^(2k)`` -> ``k``).

    ``partitions = 1`` returns 0 — the identity partition, one unit owning the
    whole tree, byte-identical to an unpartitioned sweep. Raises ``ValueError``
    on a non-power-of-two, and on an ODD power of two (which would split a
    2-bit morton digit in half); the message names the two valid neighbours so
    the caller fixes it in one pass.
    """
    n = int(partitions)
    if n < 1 or n & (n - 1):
        raise ValueError(f"sweep partitions must be a power of two >= 1 (got {partitions!r})")
    bits = n.bit_length() - 1
    if bits % 2:
        raise ValueError(
            f"sweep partitions={n} (2^{bits}) splits a morton digit in half; a digit is 2 "
            f"bits, so partitions round to digit boundaries — use {n // 2} or {n * 2} "
            f"(bit-level splits are the open fork on issue #377)"
        )
    return bits // 2


def partition_index(decimal: str, partitions: int) -> int:
    """Index of the partition owning the D1 morton id ``decimal``.

    The base-4 value of the id's first ``k`` digits (:func:`_decimal_rank`'s
    convention, digits ``1..4`` -> ``0..3``), where ``k`` is the split order —
    so ownership is a pure prefix test and is disjoint by construction. Raises
    ``ValueError`` for a node COARSER than the split order: such a node spans
    partitions and belongs to the finisher, never to a partition worker.
    """
    from zagg.hive import _decimal_base, _decimal_order, _decimal_rank

    k = partition_split_order(partitions)
    order = _decimal_order(decimal)
    if order < k:
        raise ValueError(
            f"node {decimal} is at order {order}, coarser than the partitions={partitions} "
            f"split order {k}; coarse nodes span partitions (they are the finisher's, "
            f"issue #377)"
        )
    return _decimal_rank(decimal[: len(_decimal_base(decimal)) + k])


def normalize_partition(partition) -> tuple[int, int] | None:
    """Validate a wire ``{"index", "of"}`` block into ``(index, of)``.

    ``None`` passes through as ``None`` (an unpartitioned pass). The ``of``
    count goes through :func:`partition_split_order`, so the digit-boundary
    rule is enforced worker-side too — a malformed or hand-rolled event never
    silently sweeps the wrong subtree.
    """
    if partition is None:
        return None
    try:
        index, of = int(partition["index"]), int(partition["of"])
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(
            f"sweep partition must be {{'index': int, 'of': int}} (got {partition!r})"
        ) from e
    partition_split_order(of)
    if not 0 <= index < of:
        raise ValueError(f"sweep partition index {index} is out of range for of={of}")
    return index, of


def partition_leaves(leaves, partitions: int) -> list[list]:
    """Split a ``(shard_key, window)`` work set into ``partitions`` disjoint lists.

    The dispatch-side half of the decomposition: each returned list is one
    partition's inline work set, in the same ``(int key, window)`` currency
    :func:`zagg.sweep.run_sweep` takes, index-aligned with the partition index.
    Empty lists are kept in place (a partition with no dirty leaf still has an
    index) — callers skip them rather than fire an empty invoke.

    ``partitions`` is validated FIRST and every ref's index is resolved BEFORE
    the buckets exist: this is the dispatch-side entry point, reached long
    before any manifest names a shard order, so a bad count must fail as a
    parameter error (even on an empty work set) rather than as a per-ref one.
    """
    from zagg.grids.morton import morton_decimal

    partition_split_order(partitions)
    indexed = [
        (partition_index(morton_decimal(int(key)), partitions), int(key), window)
        for key, window in (r if isinstance(r, (tuple, list)) else (r, None) for r in leaves)
    ]
    buckets: list[list] = [[] for _ in range(int(partitions))]
    for index, key, window in indexed:
        buckets[index].append((key, window))
    return buckets


def select_partition(by_shard: dict, partitions: int, index: int) -> tuple[dict, int]:
    """Restrict a normalized work set to one partition; ``(kept, n_foreign)``.

    The worker-side half: the ``discover: true`` transport re-derives the WHOLE
    store's work set from its run records, and a hand-rolled invoke may carry
    anything, so the partition filter is applied where the fold happens rather
    than trusted from the dispatcher. ``by_shard`` is
    :func:`zagg.sweep._normalize_leaves`' ``{shard_decimal: {window, ...}}``.
    ``(index, partitions)`` is re-validated here for the same reason: an
    out-of-range index would otherwise filter EVERYTHING out and read as a
    clean "nothing to do" — the silent wrong answer this filter exists to
    prevent.
    """
    normalize_partition({"index": index, "of": partitions})
    kept = {d: w for d, w in by_shard.items() if partition_index(d, partitions) == index}
    return kept, len(by_shard) - len(kept)
