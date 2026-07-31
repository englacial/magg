"""O11 content hashes: the writer-side port of the spec §5 recipe (issue #342).

The logical content hash of a hive leaf is **per-array sha256 over decoded
values** — never stored object bytes, so codec/packaging changes are invisible
by construction while any value change flips the hash (exact bytes, no float
tolerance). The recipe is NOT this module's to adjust: the normative text is
``docs/specification.md`` §5 and the reference implementation is
``moczarr.stats.hash_arrays`` / ``combined_hash`` (espg/moczarr PR #23),
adopted verbatim here and pinned byte-for-byte by the committed conformance
fixtures (``tests/data/spec/*.expected.json``, spec §7).

Recipe summary (§5.2/§5.3):

- fixed-width arrays hash as their full decoded contents, raw C-order
  **little-endian** bytes at the declared dtype;
- vlen (object-dtype) arrays hash per cell in flat C order as
  ``u64le(len(payload)) || payload`` — the length prefix makes the digest
  injective and covers the cell grid, not just the payloads;
- an element with no byte recipe RAISES (a silently wrong digest is worse
  than none);
- ``combined`` = sha256 over the SORTED per-array hex digests joined by
  ``"\\n"``, hashed as ASCII (array names deliberately excluded).

Hashes are recorded in the leaf's D20 stats sidecar under ``content_hashes``
in the §5.3 structured shape (:func:`content_hashes_record`); absence reads
as *unverifiable, not tampered*.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Mapping

import numpy as np

logger = logging.getLogger(__name__)

#: Width of the §5.2 vlen recipe's per-cell length prefix (uint64, LE).
VLEN_LENGTH_PREFIX = 8


def _element_bytes(element: object) -> bytes:
    """One vlen cell's decoded payload as raw little-endian bytes (§5.2).

    ``variable_length_bytes`` cells decode to :class:`bytes` (zagg's ragged
    payloads and their ``{field}_locations`` siblings). ``None`` — an
    unwritten cell may decode as ``None``, not ``b""`` — is zero-length;
    ``str`` covers a vlen-utf8 future; an ndarray (a typed ``zagg-ragged/2``
    cell, spec §6) normalizes to C-contiguous little-endian bytes at its
    element dtype. Anything else RAISES rather than hashing a ``repr`` — a
    digest that is silently wrong is worse than no digest at all.
    """
    if element is None:
        return b""
    if isinstance(element, (bytes, bytearray, memoryview)):
        return bytes(element)
    if isinstance(element, str):
        return element.encode()
    if isinstance(element, np.ndarray):
        values = np.ascontiguousarray(element)
        if values.dtype.byteorder == ">":
            values = values.astype(values.dtype.newbyteorder("<"))
        return values.tobytes()
    raise ValueError(
        f"vlen element of type {type(element).__name__} has no O11 byte recipe "
        f"(expected bytes, str, ndarray, or None)"
    )


def hash_array(values: np.ndarray) -> str:
    """The §5.2 hash of ONE decoded array (fixed-width or vlen/object dtype)."""
    values = np.ascontiguousarray(values)
    if values.dtype.kind == "O":  # vlen: length-prefixed payloads, flat C order
        digest = hashlib.sha256()
        for element in values.ravel(order="C"):
            payload = _element_bytes(element)
            digest.update(len(payload).to_bytes(VLEN_LENGTH_PREFIX, "little"))
            digest.update(payload)
        return digest.hexdigest()
    if values.dtype.byteorder == ">":  # canonical form is little-endian
        values = values.astype(values.dtype.newbyteorder("<"))
    return hashlib.sha256(values.tobytes()).hexdigest()


def hash_arrays(group: Any, *, staged: Mapping[str, np.ndarray] | None = None) -> dict[str, str]:
    """O11 per-array hashes of every named zarr array beneath ``group``.

    ``group`` is an open zarr v3 group at the leaf ROOT; the scope is
    discovery-based (§5.1): ``members(max_depth=None)``, so the key set is
    whatever named arrays exist beneath the root — data fields, ragged vlen
    payloads and their locations siblings, ``morton``, every coordinate —
    keyed by the array's path relative to the leaf root (e.g. ``"8/morton"``).

    ``staged`` maps array keys to the IN-MEMORY values the writer just wrote
    (the ratified issue #342 hash source: the leaf write path already holds
    each array's full slab, so hashing is a memory-bandwidth pass, no data
    GETs). A staged value is hashed in place of a store read-back; enumerated
    arrays with no staged value (e.g. ``resolution: chunk`` companions, which
    are written per chunk-block rather than as one slab) fall back to reading
    that array back. The two sources are pinned equal by the write-read
    parity test in ``tests/test_content_hash.py`` — the CI guard for the
    dtype-cast/fill drift class §5 exists to catch.

    A staged key matching NO enumerated array warns: the digests stay correct
    (that array was read back) but the staged seam is broken — silent
    key-composition drift between writer and enumeration would otherwise turn
    every hash into a read-back with no observable symptom.
    """
    import zarr

    staged = staged or {}
    hashes: dict[str, str] = {}
    for key, node in group.members(max_depth=None):
        if not isinstance(node, zarr.Array):
            continue
        hashes[key] = hash_array(staged[key] if key in staged else node[...])
    unused = set(staged) - set(hashes)
    if unused:
        logger.warning(
            "staged values for %s match no array beneath the leaf root; those arrays "
            "were hashed from a store read-back instead (staged keys must be array "
            "paths relative to the leaf root, e.g. '8/morton')",
            sorted(unused),
        )
    return hashes


def combined_hash(hashes: Mapping[str, str]) -> str:
    """The O11 combined hash: sha256 of the sorted per-array hex digests.

    Serialization pinned by §5.3 (and the committed fixtures): the digests
    sorted lexically and joined with ``"\\n"``, hashed as ASCII — array
    *names* deliberately excluded ("hash of the sorted per-array hashes"),
    so ``combined`` is immune to enumeration order.
    """
    return hashlib.sha256("\n".join(sorted(hashes.values())).encode()).hexdigest()


def content_hashes_record(hashes: Mapping[str, str]) -> dict:
    """The §5.3 structured ``content_hashes`` sidecar record.

    The ``arrays`` map is key-sorted so a regenerated record diffs cleanly
    (``combined`` is order-immune either way — it sorts the digests).
    """
    return {"arrays": dict(sorted(hashes.items())), "combined": combined_hash(hashes)}
