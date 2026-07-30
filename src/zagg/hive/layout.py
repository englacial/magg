"""Hive layout primitives: spec versions, product names, leaf paths (issue #330).

Split out of the single-file ``zagg.hive`` (issue #330 phase 1); the public
surface is unchanged and re-exported from :mod:`zagg.hive`. This is the
dependency-free bottom of the package — the D19 product-name grammar, the D2/D3
leaf-path arithmetic and the D5 node invariant, plus the two small shared
helpers (:func:`_read_json`, :func:`_utcnow`) that the manifest and coverage
halves both use.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone

import numpy as np

from zagg.windows import leaf_name, split_leaf_name

#: Convention version recorded in the manifest and the commit stamp (D6).
HIVE_SPEC = "morton-hive/1"
#: The temporal superset convention (issue #246, D13/D15): declared iff the
#: manifest carries a temporal block; a ``/1`` store *is* a ``/2`` store with
#: ``schedule: none``, so ``/1`` stays the spec written for unwindowed stores.
HIVE_SPEC_V2 = "morton-hive/2"

#: Product-name grammar (D19; normative on the mortie spec page §6.5):
#: URL-safe lowercase alphanumerics plus ``-``/``_``, at most
#: :data:`PRODUCT_NAME_MAX` chars. The base-component exclusion (``-?[1-6]``,
#: :func:`_is_base_component`) is checked separately — a name shaped like a
#: hive base component would make the walker's child classification ambiguous
#: at a multi-product store root.
_PRODUCT_NAME_RE = re.compile(r"^[a-z0-9_-]+$")

#: Product-name length cap (D19; espg ruling mirrored from the mortie spec page
#: §6.5). Derivation: a POSIX filename component is 255 bytes; the immutable-
#: provenance decoration reserves 13 (``+`` + a 12-hex digest) ⇒ a 242-byte
#: hard ceiling. 192 sits 50 under that, and leaves the S3 total-key and
#: PATH_MAX budgets comfortable (the name also nests under the store prefix and
#: the digit tree). The grammar is single-byte per char, so char count == byte
#: count here.
PRODUCT_NAME_MAX = 192


def validate_product_name(name: str) -> str:
    """Validate a D19 product name; returns it.

    Grammar (normative on the mortie spec page §6.5): one or more of
    ``[a-z0-9_-]``, at most :data:`PRODUCT_NAME_MAX` characters, and never
    matching the morton base-component grammar ``-?[1-6]`` — the root-form
    discrimination (:func:`classify_store_root`) depends on names and digit
    components being disjoint. Names are URL-safe by construction (they appear
    in gridlook deep-links and web paths).
    """
    if not isinstance(name, str) or not _PRODUCT_NAME_RE.match(name):
        raise ValueError(
            f"product name {name!r} does not match the D19 grammar ([a-z0-9_-]+, "
            f"lowercase; normative on the mortie spec page)"
        )
    if len(name) > PRODUCT_NAME_MAX:
        raise ValueError(
            f"product name {name!r} is {len(name)} chars; the D19 grammar caps names "
            f"at {PRODUCT_NAME_MAX} (normative on the mortie spec page §6.5 — a POSIX "
            f"255-byte filename component less the 13-byte immutable-provenance "
            f"decoration)"
        )
    if _is_base_component(name):
        raise ValueError(
            f"product name {name!r} matches the morton base-component grammar "
            f"(-?[1-6]); such names are excluded so a store root's children stay "
            f"unambiguous (D19)"
        )
    return name


def product_root(store_root: str, name: str) -> str:
    """Root prefix of product ``name`` under a multi-product store (D19).

    A product subtree is a COMPLETE, unmodified morton-hive store (bare-named
    manifest + MOC + digit tree), so everything that takes a ``store_root``
    takes this value unchanged.
    """
    return f"{store_root.rstrip('/')}/{validate_product_name(name)}"


def effective_store_root(store_path: str, config) -> str:
    """The store root a run writes into: the product root when one is named.

    ``output.product_name`` (issue #299) prefixes the configured store path
    with the D19 ``{name}/`` product root; absent, the bare single-product
    layout is unchanged (byte-identical stores — the D19 revision is
    additive).
    """
    from zagg.config import get_product_name

    name = get_product_name(config)
    return product_root(store_path, name) if name else store_path


def _is_valid_product_name(name: str) -> bool:
    """Whether ``name`` satisfies the D19 product-name grammar (no raise)."""
    try:
        validate_product_name(name)
    except ValueError:
        return False
    return True


def shard_leaf_path(store_root: str, shard_key, window: str | None = None) -> str:
    """Absolute path of a shard's leaf zarr under ``store_root`` (D2/D3).

    Computed by mortie's ``hive_path`` — the layout convention is owned by the
    mortie spec — and re-checked against the node invariant (D5) so a future
    drift in either side fails loudly instead of writing a stray prefix.
    ``window`` (issue #246, D13) selects the shard's time-windowed leaf,
    ``{full_id}_{window}.zarr``, at the same node; ``None`` is the bare
    schedule-``none`` leaf, byte-identical to pre-windowing paths. Raises
    ``ValueError`` on an invalid shard key or window label.
    """
    from mortie import MortonIndexArray

    word = int(shard_key)
    if word < 0:
        raise ValueError(
            f"shard key must be a packed morton word (got {word}); parse a decimal "
            f"id with zagg.grids.morton.morton_word first"
        )
    rel = MortonIndexArray.from_words(np.asarray([word], dtype=np.uint64)).hive_path()[0]
    if window is not None:
        node, _sep, bare = rel.rpartition("/")
        rel = f"{node}/{leaf_name(bare.removesuffix('.zarr'), window)}"
    check_node_invariant(rel)
    return f"{store_root.rstrip('/')}/{rel}"


def check_node_invariant(rel_path: str) -> None:
    """Raise unless ``rel_path`` is a legal hive leaf path (D5).

    Below the root only digit components are allowed — ``{sign+base}``
    (optional ``-``, one digit ``1..6``) at the first level, one ``1..4`` digit
    per level after — terminating in ``{full_id}.zarr`` (or the windowed
    ``{full_id}_{window}.zarr``, issue #246: split on the first ``_``, window
    label per the frozen grammar) whose id equals the concatenated components.
    This is the walker's contract: any other name under the root (bar the
    manifest and the root ``coverage.moc``) breaks child classification.
    """
    parts = rel_path.strip("/").split("/")
    leaf = parts[-1]
    ok = len(parts) >= 2 and leaf.endswith(".zarr")
    if ok:
        head, digits = parts[0], parts[1:-1]
        try:
            full_id, _window = split_leaf_name(leaf)
        except ValueError:
            full_id = None  # malformed window label -> not a legal leaf
        ok = _is_base_component(head)
        ok = ok and all(len(d) == 1 and d in "1234" for d in digits)
        ok = ok and full_id == head + "".join(digits)
    if not ok:
        raise ValueError(f"path {rel_path!r} violates the hive node invariant (D5)")


def _is_base_component(name: str) -> bool:
    """Whether ``name`` is a ``{sign+base}``-shaped hive root child (D5)."""
    base = name[1:] if name.startswith("-") else name
    return len(base) == 1 and base in "123456"


def _read_json(obj_store, key: str) -> dict | None:
    """GET+parse one small JSON object; ``None`` when it does not exist."""
    import obstore
    from obstore.exceptions import NotFoundError

    try:
        data = obstore.get(obj_store, key).bytes()
    except (FileNotFoundError, NotFoundError):
        return None
    return json.loads(bytes(data))


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
