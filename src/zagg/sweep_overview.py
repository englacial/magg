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
merge law (count/sum by addition, min/max by extremum — byte-equal to a direct
**nan-skipping** aggregation at the coarser order; see
:data:`EXACT_NAN_POLICY`), ``approximate`` fields (t-digests) merge
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

import hashlib
import json
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

#: NaN policy the exact fold actually implements, recorded per field in the
#: pyramid declaration and in every overview's attrs (review finding, issue
#: #201). A leaf's stored NaN is the same bytes whether it is the fill sentinel
#: or a NaN datum, so :func:`fold_dense` cannot distinguish them and skips
#: both: the exact-class fold is byte-equal to a direct ``nansum``/``nanmin``/
#: ``nanmax`` at the coarser order, NOT to a NaN-propagating ``sum``/``min``/
#: ``max`` (which the store encoding makes unrecoverable either way — the plain
#: and nan-aware aggregators share one law in
#: :data:`zagg.semantics.EXACT_MERGE_LAWS`).
EXACT_NAN_POLICY = "skip"

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
    direct coarser aggregation would visit — with the one premise
    :data:`EXACT_NAN_POLICY` names: NaN is ALWAYS missing (fill and datum are
    the same bytes), so the match is against the nan-skipping reduction.
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


# ---------------------------------------------------------------------------
# The manifest pyramid block: template-time declaration (D22, per-family
# schedules), sweep-populated actuals.
# ---------------------------------------------------------------------------


def build_pyramid_block(config, shard_order: int) -> dict:
    """The manifest ``pyramid`` block: the template-time declaration (D22).

    Declares the overview family's order schedule (``output.pyramid`` knob;
    default every :data:`DEFAULT_SPACING` orders below the shard order —
    espg-ratified) and each field's composability class + fold method (D24).
    ``none`` fields are declared with their class ONLY — the recorded absence
    that gives readers zero-open filtering (option A, the ruled default) —
    and warned about loudly at template time, per the D24 ruling. The sweep
    later adds ``materialized`` actuals; it never rewrites the declaration.
    """
    from zagg.config import get_agg_fields, get_pyramid
    from zagg.semantics import EXACT_MERGE_LAWS, _fold_function_name, composability_classes

    knob = get_pyramid(config)
    if knob is None:  # output.pyramid: false — declared off
        return {"spec": PYRAMID_SPEC, "overview": {"orders": []}}
    spacing = int(knob.get("spacing") or DEFAULT_SPACING)
    if knob.get("orders") is not None:
        orders = sorted({int(k) for k in knob["orders"]}, reverse=True)
    else:
        orders = list(range(int(shard_order) - spacing, -1, -spacing))
    agg = get_agg_fields(config)
    fields: dict = {}
    excluded = []
    for name, cls in composability_classes(config).items():
        meta = agg[name]
        if cls == "exact":
            fields[name] = {
                "class": "exact",
                "method": EXACT_MERGE_LAWS[_fold_function_name(meta.get("function")) or ""],
                "nan_policy": EXACT_NAN_POLICY,
                "dtype": meta.get("dtype", "float32"),
                "fill_value": _json_fill(meta.get("fill_value", "NaN")),
            }
        elif cls == "approximate":
            inner = meta.get("inner_shape") or (2,)
            fields[name] = {
                "class": "approximate",
                "method": TDIGEST_LAW,
                "dtype": meta.get("dtype", "float32"),
                "inner_shape": [int(inner)] if isinstance(inner, int) else [int(x) for x in inner],
                "delta": int((meta.get("params") or {}).get("delta", 512)),
            }
        else:
            fields[name] = {"class": "none"}
            excluded.append(name)
    if excluded and orders:
        logger.warning(
            f"pyramid: fields {excluded} are non-composable (D24 class 'none') and will "
            f"exist ONLY at native resolution — excluded from every overview order (the "
            f"per-field-exclusion default; declare a derived summary to opt in, issue #201)"
        )
    overview: dict = {
        "spacing": spacing,
        "orders": orders,
        "all_time": bool(knob.get("all_time", False)),
        "fields": fields,
    }
    if knob.get("summarize"):
        overview["summarize"] = {str(k): dict(v) for k, v in knob["summarize"].items()}
    return {"spec": PYRAMID_SPEC, "overview": overview}


def _json_fill(fill_value):
    """A JSON-safe fill token (zarr v3 uses the string ``"NaN"``)."""
    if isinstance(fill_value, float) and np.isnan(fill_value):
        return "NaN"
    return fill_value


def _update_manifest_pyramid(store_root, orders, store_kwargs) -> bool:
    """Record materialized overview orders in the manifest pyramid block.

    The one manifest key the sweep may touch (D11: the block is populated/
    updated by the §7 sweep; it is excluded from the frozen resume keys, so
    this RMW can never brick appends). Fail-open — the declaration readers
    key on is untouched; ``materialized`` is a convenience actual.
    """
    import obstore

    from zagg.hive import MANIFEST_NAME, _utcnow, read_manifest
    from zagg.store import open_object_store

    try:
        fresh = read_manifest(store_root, **store_kwargs)
        if fresh is None:
            return False
        block = fresh.setdefault("pyramid", {}).setdefault("overview", {})
        known = set((block.get("materialized") or {}).get("orders") or [])
        block["materialized"] = {
            "orders": sorted(known | {int(k) for k in orders}),
            "generated_at": _utcnow(),
        }
        obstore.put(
            open_object_store(store_root, **store_kwargs),
            MANIFEST_NAME,
            json.dumps(fresh, indent=1).encode(),
        )
        return True
    except Exception as e:
        logger.warning(f"sweep[overview]: manifest pyramid update failed (fail-open): {e}")
        return False


def declare_pyramid(store_root: str, config, *, store_kwargs=None) -> dict:
    """Install/refresh the manifest ``pyramid`` declaration on an EXISTING store.

    The issue #358 retrofit tool: declaration is no longer birth-only. The
    block is re-derived from the supplied pipeline config (the authoritative
    source — the same :func:`build_pyramid_block` path template time uses) at
    the manifest's own ``shard_order``, validated against store truth, and
    RMW'd into the manifest. Unlike the sweep's fail-open ``materialized``
    update (:func:`_update_manifest_pyramid`), this is the user's explicit
    operation: every failure raises — a missing or malformed manifest, a config
    whose semantics the store's frozen hash denies, a declaration the store
    contradicts — and nothing is half-written (the one PUT is the whole write).

    Two validation sources, covering the two halves of the recipe:

    * **Typing** — ONE committed leaf, found through the store's run records
      (:func:`zagg.sweep.discover_leaves`, the sweep's own tree-enumeration-free
      discovery: one shallow root LIST of the run records, never a recursive
      LIST), read at the manifest ``cell_order``. Leaves are the layer the
      overview fold actually reads, so they are what can falsify the declared
      **presence, dtype, and D18 ragged element shape** of each field. They
      cannot falsify more than that: no leaf records which reducer produced it.
      A store with no committed leaf yet is still declarable — field-level
      checks are then skipped and the summary says which skip it was.
    * **Semantics** — the manifest's own frozen ``semantic_hash``
      (:func:`_semantic_guard`), which is what covers the ``method``/
      ``nan_policy`` half the leaf is silent on. The D19 ``aggregation.yaml``
      core is NOT used (it is itself config-derived and written fail-open).

    Idempotent RMW: an identical existing declaration is not re-PUT; a
    changed one is rewritten PRESERVING any ``materialized`` actuals (they
    inventory overview artifacts already on disk — overviews at
    now-undeclared orders stay as regenerable-cache debris, D24 option A).
    An ``output.pyramid: false`` config installs the declared-off block:
    recording absence is a valid retrofit.

    Returns a summary dict: ``orders`` (the declared schedule), ``fields``
    (``{name: class}``), ``validated`` (what store truth was checked),
    ``previous`` (``absent``/``identical``/``replaced``), and ``updated``
    (whether a PUT happened).
    """
    import obstore

    from zagg.hive import MANIFEST_NAME, _frozen_matches, read_manifest
    from zagg.store import open_object_store

    store_kwargs = dict(store_kwargs or {})
    manifest = read_manifest(store_root, **store_kwargs)
    if manifest is None:
        raise ValueError(
            f"no {MANIFEST_NAME} at {store_root} — not a hive store root; "
            f"declare_pyramid retrofits existing hive stores only"
        )
    # Both order keys up front: ``cell_order`` is only consumed by the leaf probe,
    # and a manifest that parses without it should refuse HERE rather than
    # KeyError after the whole discovery cost has been paid.
    if not isinstance(manifest, dict) or any(
        manifest.get(k) is None for k in ("shard_order", "cell_order")
    ):
        raise ValueError(
            f"the {MANIFEST_NAME} at {store_root} declares no shard_order/cell_order — "
            f"not a hive store manifest; declare_pyramid retrofits existing hive stores only"
        )
    shard_order = int(manifest["shard_order"])
    block = build_pyramid_block(config, shard_order)
    # Compare (and write) canonical JSON: ``prior`` came back through
    # ``json.loads``, so any non-JSON-primitive surviving the derivation (a tuple
    # in a ``summarize`` declaration, say) would differ from its round-tripped
    # self and re-PUT on every call — against the headline idempotency property.
    # It also surfaces an unserializable block HERE rather than at the PUT,
    # after the whole store-truth probe has been paid for.
    block = json.loads(json.dumps(block))
    bad = [k for k in block["overview"].get("orders") or [] if not 0 <= int(k) < shard_order]
    if bad:
        raise ValueError(
            f"declared orders {bad} are not ancestor orders of the manifest "
            f"shard_order {shard_order} — the config does not match this store"
        )
    semantic = _semantic_guard(manifest, config)
    validated = _validate_block_against_store(store_root, manifest, block, store_kwargs)
    # Re-read immediately before the RMW, the same discipline
    # :func:`_update_manifest_pyramid` uses: validation above is slow (run-record
    # discovery + leaf GETs), this PUT rewrites the WHOLE manifest, and a
    # concurrent sweep's ``materialized`` update landing in that window would be
    # silently reverted by the pre-validation copy. Frozen keys are re-checked so
    # the block is never installed on a manifest validation never saw.
    fresh = read_manifest(store_root, **store_kwargs)
    if fresh is None or not _frozen_matches(fresh, manifest):
        raise ValueError(
            f"the {MANIFEST_NAME} at {store_root} changed under declare_pyramid's "
            f"validation window (frozen keys differ, or it vanished) — nothing was "
            f"written; re-run against the settled store"
        )
    prior = fresh.get("pyramid")
    # A non-dict prior (hand-edited ``"pyramid": "off"``) is not an error: it is
    # not this module's grammar, so it carries no actuals to preserve and is
    # simply replaced — but it must not AttributeError on the way there.
    prior_overview = prior.get("overview") if isinstance(prior, dict) else None
    materialized = prior_overview.get("materialized") if isinstance(prior_overview, dict) else None
    if materialized is not None:
        block["overview"]["materialized"] = materialized
    summary = {
        "orders": list(block["overview"].get("orders") or []),
        "fields": {n: m.get("class") for n, m in (block["overview"].get("fields") or {}).items()},
        "validated": f"{validated}; {semantic}",
        "previous": "absent" if prior is None else "identical" if prior == block else "replaced",
        "updated": prior != block,
    }
    if prior == block:
        logger.info("declare_pyramid: the manifest already carries this declaration; no write")
        return summary
    fresh["pyramid"] = block
    obstore.put(
        open_object_store(store_root, **store_kwargs),
        MANIFEST_NAME,
        json.dumps(fresh, indent=1).encode(),
    )
    return summary


def _semantic_guard(manifest: dict, config) -> str:
    """Refuse a config whose semantics the store's frozen ``semantic_hash`` denies.

    The leaf probe (:func:`_field_drift`) can falsify TYPING only — no leaf
    records which reducer produced a field — so nothing there contradicts a
    declaration of ``method: "max"`` over a store holding minima, and
    :func:`build_pyramid_block` puts exactly that fold law into the block. The
    store does record its reducers, in the one place the repo calls
    authoritative: the D19 frozen ``semantic_hash`` (issue #299), a digest over
    the whole ``aggregation`` block. Comparing it here is what makes the issue
    #358 contract ("a wrong config cannot install a fold recipe the store
    contradicts") true of the fold LAW and not just of dtypes.

    ``output.*`` is not in the semantic core, so the intended retrofit config —
    the original plus ``output.pyramid`` — hashes identically; the guard cannot
    false-refuse on the pyramid edit itself. It compares only when the manifest
    declares the key, the same both-sides-present exemption
    :func:`zagg.hive._frozen_matches` gives pre-#299 stores (a pre-#344 retrofit
    target may well be one). Returns the note recorded in the summary.
    """
    from zagg.semantics import semantic_fingerprint, semantic_hash

    stored = manifest.get("semantic_hash")
    if not stored:
        logger.warning(
            "declare_pyramid: the manifest carries no semantic_hash (pre-#299 store) — "
            "the config's aggregation semantics could NOT be verified against the store; "
            "the declared fold methods are taken on trust"
        )
        return "semantic_hash absent (pre-#299 store — fold methods unverified)"
    supplied = semantic_hash(config)
    if supplied != stored:
        raise ValueError(
            f"config semantics {semantic_fingerprint(supplied)} != the store's frozen "
            f"semantic_hash {semantic_fingerprint(stored)} — this config did not build this "
            f"store, so the fold methods it declares (which no leaf records, and which the "
            f"field checks cannot falsify) are not the store's; declare_pyramid refuses. "
            f"Retrofit with the ORIGINAL config: output.* is not in the semantic core, so "
            f"adding output.pyramid to it hashes identically"
        )
    return f"semantic_hash {semantic_fingerprint(stored)}"


def _validate_block_against_store(store_root, manifest, block, store_kwargs) -> str:
    """The pre-write store-truth check: refuse a recipe the store contradicts.

    Probes the run records for a committed leaf (the first hit wins — discovery
    has already parsed every record, so the scan costs one GET per ref until it
    lands) and checks every declared field against the
    leaf's stored array: presence, dense dtype for the exact class, the D18
    ragged element declaration (element dtype + ``inner_shape``) for the
    approximate class. Returns the "what was checked" summary string;
    raises ``ValueError`` naming every drifted field. No committed leaf is
    not an error — the store is declarable before its first commit — but
    the skip is loud and recorded in the summary.
    """
    import zarr

    from zagg.hive import read_commit, shard_leaf_path
    from zagg.store import open_store
    from zagg.sweep import discover_leaves

    fields = block["overview"].get("fields") or {}
    if not fields:
        # The declared-off block carries no ``fields`` key at all; an ON
        # declaration over an empty aggregation carries an empty one. Reporting
        # both as "declared off" would read as a lie in the summary.
        return (
            "nothing to check (declared off)"
            if "fields" not in block["overview"]
            else "nothing to check (no aggregation variables to declare)"
        )
    # Un-capped on purpose: ``discover_leaves`` has ALREADY paid the whole cost
    # (a root LIST plus a GET + parse of every run record), so truncating its
    # result bounds nothing — it only risks giving up while committed leaves
    # remain. Each extra ref costs one ``read_commit`` GET, and the loop stops at
    # the first hit; the read is read-only, which on S3 also picks the shorter
    # readonly retry policy (issue #186).
    refs = discover_leaves(store_root, store_kwargs=store_kwargs)
    leaf = None
    for key, window in refs:
        path = shard_leaf_path(store_root, key, window=window)
        if read_commit(open_store(path, read_only=True, **store_kwargs)) is not None:
            leaf = path
            break
    if leaf is None:
        # Two very different stores that must not report the same thing: no run
        # records at all is a genuinely pre-commit store, while refs that are all
        # torn/rolled-back debris is a store whose validation was skipped over
        # real work. Neither refuses — a store with no committed leaf stays
        # declarable, which is the point of the retrofit — but the skip is loud
        # and says which case it was.
        detail = (
            "no run records at the product root"
            if not refs
            else f"none of the {len(refs)} run-record refs is a committed leaf"
        )
        logger.warning(
            f"declare_pyramid: no committed leaf to probe ({detail}) — field-level "
            f"validation skipped; only the manifest root was checked"
        )
        return f"manifest only ({detail}; fields unvalidated)"
    group = zarr.open_group(
        open_store(leaf, **store_kwargs),
        path=str(int(manifest["cell_order"])),
        mode="r",
        zarr_format=3,
    )
    drift = [e for n, m in sorted(fields.items()) if (e := _field_drift(group, n, m))]
    if drift:
        raise ValueError(
            f"pyramid declaration contradicts the store (checked leaf {leaf}): " + "; ".join(drift)
        )
    return f"leaf {leaf}"


def _field_drift(group, name, meta) -> str | None:
    """One declared field's mismatch against its stored leaf array, or ``None``.

    The ragged branch also gates the D18 attrs' **spec revision** against
    :data:`zagg.grids.base.RAGGED_SPEC`, mirroring the readers' posture
    (``readers.tdigest_tensor._open_ragged`` raises on mismatch) rather than
    inventing a second policy. A store whose ragged layout this zagg cannot
    read is the purest instance of the tool's contract — a fold recipe the
    store contradicts: without the gate the retrofit installs a declaration
    promising overviews the sweep then fails to read at FOLD time, later and
    with a worse error. ``zagg-ragged/2`` is a live migration path (issue
    #210 moves the element declaration into the zarr data type), so this is
    not hypothetical (espg-ruled, issue #358).
    """
    from zagg.grids.base import RAGGED_ELEMENT_ATTR, RAGGED_SPEC

    try:
        arr = group[name]
    except KeyError:
        return f"field {name!r} is declared but absent from the leaf"
    if meta["class"] == "approximate":
        raw = arr.attrs.get(RAGGED_ELEMENT_ATTR)
        ragged = dict(raw) if isinstance(raw, dict) else {}
        element = ragged.get("element") or {}
        if not element:
            return (
                f"field {name!r}: declared approximate (ragged) but the stored "
                f"array carries no ragged element declaration"
            )
        if ragged.get("spec") != RAGGED_SPEC:
            return (
                f"field {name!r}: the stored array declares ragged spec "
                f"{ragged.get('spec')!r}; this zagg understands {RAGGED_SPEC!r} only — "
                f"a newer writer's layout must be adopted deliberately, not folded "
                f"blind (declaring overviews over it would promise a fold that fails "
                f"at sweep time)"
            )
        declared_dt = np.dtype(meta.get("dtype") or "float32")
        # Explicit, because ``np.dtype(None)`` is float64: an element block that
        # declares no dtype would otherwise silently VALIDATE a float64
        # declaration against a store that says nothing.
        stored_dt = element.get("dtype")
        if stored_dt is None:
            return f"field {name!r}: the stored ragged element declaration carries no dtype"
        if np.dtype(stored_dt) != declared_dt:
            return f"field {name!r}: ragged element dtype {stored_dt} != declared {declared_dt}"
        stored_inner = [int(s) for s in (element.get("shape") or [-1])[1:]]
        declared_inner = [int(s) for s in meta.get("inner_shape") or [2]]
        if stored_inner != declared_inner:
            return f"field {name!r}: ragged inner_shape {stored_inner} != declared {declared_inner}"
    elif meta["class"] == "exact":
        declared_dt = np.dtype(meta.get("dtype") or "float32")
        if arr.dtype != declared_dt:
            return f"field {name!r}: dtype {arr.dtype} != declared {declared_dt}"
    return None


# ---------------------------------------------------------------------------
# Phase B: the overview writer — the family's whole-tree sweep.
# ---------------------------------------------------------------------------

#: Per-node overview bookkeeping object (window inventory + skip-if-current
#: stamps), sibling to the other families' ``{family}.rollup.json`` — the same
#: closed name grammar, so the §5 walker's child classification is unaffected.
ENVELOPE_NAME = "overview.rollup.json"


def sweep_overviews(store_root: str, manifest: dict, by_shard: dict, *, store_kwargs=None) -> dict:
    """Generate/refresh overview zarrs at the manifest-declared orders (D22).

    ``by_shard`` is the engine's normalized dirty work set
    (``{shard_decimal: {window, ...}}``); the walk visits ONLY ancestor nodes
    of those shards at each declared order. The full descendant leaf set per
    node comes from the dirty set unioned with the root ``coverage.moc``
    (run record + MOC — never a LIST); with the default family order the MOC
    family has just refreshed that root in the same pass.

    Idempotent: a (node, window) whose stored generation stamp (merged-leaf
    count + max leaf stamp timestamp) AND content hash both match the freshly
    folded payload — and whose zarr is confirmed present and stamped, since the
    envelope and the artifact are two objects (D9) — is skipped; the hash is
    the same-second backstop the engine's payload compare provides for JSON
    families. Returns the standard
    ``written``/``current``/``empty``/``failed`` counts plus ``declared``.
    """
    from zagg.store import open_object_store

    store_kwargs = dict(store_kwargs or {})
    counts: dict = {"written": 0, "current": 0, "empty": 0, "failed": 0, "declared": True}
    decl = (manifest.get("pyramid") or {}).get("overview")
    if not isinstance(decl, dict) or not decl.get("orders"):
        counts["declared"] = False
        logger.info(
            "sweep[overview]: no pyramid overview declaration in the manifest "
            "(template-time, D22); nothing to generate"
        )
        return counts
    if decl.get("summarize"):
        logger.warning(
            f"sweep[overview]: declared derived summaries {sorted(decl['summarize'])} are not "
            f"generated yet — no roster-kind ragged field ships (D24 opt-in, issue #265); skipping"
        )
    fields = {
        n: dict(m)
        for n, m in (decl.get("fields") or {}).items()
        if isinstance(m, dict) and m.get("class") in ("exact", "approximate")
    }
    if not fields:
        logger.info("sweep[overview]: no composable fields declared; nothing to generate")
        return counts
    shard_order = int(manifest["shard_order"])
    cell_order = int(manifest["cell_order"])
    windowed = manifest.get("temporal") is not None
    orders = sorted({int(k) for k in decl["orders"]}, reverse=True)
    bad = [k for k in orders if not (0 <= k < shard_order)]
    if bad:
        logger.warning(
            f"sweep[overview]: declared orders {bad} are not ancestor orders of "
            f"shard_order {shard_order}; skipping them"
        )
        orders = [k for k in orders if k not in bad]
    candidates = _candidate_decimals(store_root, shard_order, by_shard, store_kwargs)
    store = open_object_store(store_root, **store_kwargs)
    materialized: set[int] = set()
    for k in orders:
        nodes = sorted({_node_at(d, k) for d in by_shard})
        for node in nodes:
            node_shards = sorted(d for d in candidates if d.startswith(node))
            dirty_windows = set()
            for d in by_shard:
                if d.startswith(node):
                    dirty_windows |= by_shard[d]
            envelope = _read_envelope(store, node)
            entries = dict((envelope or {}).get("windows") or {})
            for key, fold_windows in _window_work(decl, windowed, dirty_windows, entries):
                entry = _roll_node(
                    store_root,
                    node,
                    k,
                    key,
                    fold_windows,
                    node_shards,
                    fields,
                    cell_order,
                    shard_order,
                    windowed,
                    entries.get(key),
                    counts,
                    store_kwargs,
                )
                if entry is not None:
                    entries[key] = entry
            if entries:
                materialized.add(k)
            fresh = {
                "spec": _sweep().SWEEP_SPEC,
                "family": "overview",
                "node": node,
                "order": k,
                "windows": entries,
            }
            if entries and fresh != envelope:
                import obstore

                obstore.put(
                    store,
                    f"{_node_rel(node)}/{ENVELOPE_NAME}",
                    json.dumps(fresh, indent=1).encode(),
                )
    if counts["written"]:
        counts["manifest_updated"] = _update_manifest_pyramid(
            store_root, sorted(materialized), store_kwargs
        )
    return counts


def _sweep():
    """The engine module (lazy: sweep.py registers this module's family)."""
    import zagg.sweep

    return zagg.sweep


def _node_rel(decimal: str) -> str:
    return _sweep()._node_rel(decimal)


def _node_at(decimal: str, order: int) -> str:
    """The ancestor prefix of a shard decimal at ``order`` (0 = base)."""
    from zagg.hive import _decimal_base

    return decimal[: len(_decimal_base(decimal)) + order]


def _rel_rank(decimal: str, node: str) -> int:
    """Base-4 rank of ``decimal``'s digit tail beyond the ``node`` prefix."""
    rank = 0
    for ch in decimal[len(node) :]:
        rank = rank * 4 + (int(ch) - 1)
    return rank


def _candidate_decimals(store_root, shard_order, by_shard, store_kwargs) -> set:
    """Descendant-leaf candidates: the dirty set unioned with the root MOC.

    Discovery stays LIST-free (D22): untouched sibling shards contribute via
    the root ``coverage.moc`` (default-on for hive; refreshed by the MOC
    family earlier in the same default pass). A missing/unusable root MOC
    degrades to the dirty set with a loud warning — the overview then covers
    only the given leaves until a full sweep repairs it (D9: regenerable).
    """
    from zagg.grids.morton import morton_decimal
    from zagg.hive import read_root_coverage, root_coverage_words

    decimals = set(by_shard)
    try:
        env = read_root_coverage(store_root, **store_kwargs)
    except ValueError:
        env = None
    if isinstance(env, dict) and env.get("order") == shard_order:
        try:
            decimals |= {morton_decimal(int(w)) for w in root_coverage_words(env)}
            return decimals
        except (KeyError, TypeError, ValueError) as e:
            logger.warning(f"sweep[overview]: unusable root coverage.moc ({e})")
    if decimals:
        logger.warning(
            "sweep[overview]: no usable root coverage.moc — overviews will fold ONLY "
            "the run's own leaves; sweep the moc family (or the default set) to repair"
        )
    return decimals


def _window_work(decl, windowed, dirty_windows, entries) -> list:
    """``(envelope key, [windows to fold])`` work items for one node.

    Per-window overviews inherit window naming (D23); the reserved ``all``
    token is the unwindowed leaf AND the opt-in all-time fold on a windowed
    store (``all_time`` in the declaration; a preexisting all-time entry stays
    maintained even if the declaration later drops the flag). A window whose
    label IS that token can therefore never own a separate overview — same
    basename, same envelope key — so it yields the all-time item alone.
    """
    from zagg.windows import SCHEDULE_NONE_TOKEN

    if not windowed:
        return [(SCHEDULE_NONE_TOKEN, [None])]
    labels = sorted(
        {w for w in dirty_windows if w is not None}
        | {key for key in entries if key != SCHEDULE_NONE_TOKEN}
    )
    work = [(w, [w]) for w in labels]
    if decl.get("all_time") or SCHEDULE_NONE_TOKEN in entries:
        if SCHEDULE_NONE_TOKEN in labels:
            # A window literally labeled with the reserved token (config
            # validation now rejects it; a hand-edited or pre-guard manifest can
            # still carry one). Its per-window overview would resolve to the
            # SAME basename and envelope key as the all-time fold, be
            # overwritten, and never regenerate — so yield only the all-time
            # item, which folds this window in anyway (review finding, #201).
            logger.warning(
                f"sweep[overview]: window label {SCHEDULE_NONE_TOKEN!r} is the reserved "
                f"all-time token (D23) — no separate per-window overview is written for "
                f"it; its leaves fold into the all-time overview instead"
            )
            work = [item for item in work if item[0] != SCHEDULE_NONE_TOKEN]
        work.append((SCHEDULE_NONE_TOKEN, labels))
    return work


def _read_envelope(store, node: str) -> dict | None:
    """The node's stored overview envelope, or ``None`` (strict, D9 cache)."""
    import obstore
    from obstore.exceptions import NotFoundError

    try:
        data = obstore.get(store, f"{_node_rel(node)}/{ENVELOPE_NAME}").bytes()
    except (FileNotFoundError, NotFoundError):
        return None
    try:
        envelope = json.loads(bytes(data))
    except ValueError:
        envelope = None
    usable = (
        isinstance(envelope, dict)
        and envelope.get("spec") == _sweep().SWEEP_SPEC
        and envelope.get("family") == "overview"
        and isinstance(envelope.get("windows"), dict)
    )
    if not usable:
        logger.debug(f"sweep[overview]: unusable envelope at node {node}; ignoring")
        return None
    return envelope


def _roll_node(
    store_root,
    node,
    k,
    key,
    fold_windows,
    node_shards,
    fields,
    cell_order,
    shard_order,
    windowed,
    existing_entry,
    counts,
    store_kwargs,
):
    """Fold one (node, window) overview; write its zarr unless current.

    Returns the fresh envelope entry, the existing one when current (or when
    no leaf contributed — an emptied window keeps its prior overview, the
    same append-only posture as the engine's interior fallback), or ``None``.
    """
    try:
        fold = _fold_node(
            store_root,
            node,
            k,
            fold_windows,
            node_shards,
            fields,
            cell_order,
            shard_order,
            counts,
            store_kwargs,
        )
    except Exception as e:
        logger.warning(f"sweep[overview]: fold failed at node {node} window {key!r}; ({e})")
        counts["failed"] += 1
        return None
    if fold is None:
        counts["empty"] += 1
        return existing_entry
    if (
        isinstance(existing_entry, dict)
        and existing_entry.get("generation") == fold["generation"]
        and existing_entry.get("content_hash") == fold["content_hash"]
        and _overview_committed(store_root, node, existing_entry.get("object"), store_kwargs)
    ):
        counts["current"] += 1
        return existing_entry
    try:
        basename = _write_overview(
            store_root, node, k, key, fold, fields, cell_order, shard_order, windowed, store_kwargs
        )
    except Exception as e:
        logger.warning(f"sweep[overview]: write failed at node {node} window {key!r}; ({e})")
        counts["failed"] += 1
        return None
    counts["written"] += 1
    return {
        "object": basename,
        "generation": fold["generation"],
        "content_hash": fold["content_hash"],
    }


def _overview_committed(store_root, node, basename, store_kwargs) -> bool:
    """Whether the entry's overview zarr is really present AND stamped (D9).

    Unlike the JSON families — whose bookkeeping IS the artifact, so they
    self-heal for free — the overview's envelope entry and its zarr are two
    objects: a generation+hash match proves nothing about the zarr surviving.
    One commit-stamp GET makes skip-if-current self-healing (a deleted
    overview regenerates) and doubles as proof the D4 stamp landed, so a torn
    prior write no longer reads as ``current`` (review finding, issue #201).
    """
    from zagg.hive import read_commit
    from zagg.store import open_store

    if not basename:
        return False
    try:
        leaf = open_store(f"{store_root}/{_node_rel(node)}/{basename}", **store_kwargs)
        return read_commit(leaf) is not None
    except Exception as e:  # an unreadable object is not a current one
        logger.debug(f"sweep[overview]: cannot confirm {node}/{basename} ({e})")
        return False


def _fold_node(
    store_root,
    node,
    k,
    fold_windows,
    node_shards,
    fields,
    cell_order,
    shard_order,
    counts,
    store_kwargs,
):
    """Fold the node's committed descendant leaves into per-field slabs.

    Reads leaf DATA by declared array name only — never a member enumeration
    — so orphan array prefixes from schema evolution (issue #341 Bug A) and
    foreign objects (status prefixes, #327) cannot crash the fold. A leaf
    missing one declared field contributes fill for it (schema evolution:
    the field postdates the leaf); a leaf at an unexpected cell order or with
    an unreadable group is skipped loudly (``failed``), never fatal.
    """
    import zarr

    from zagg.grids.morton import morton_word
    from zagg.hive import read_commit, shard_leaf_path
    from zagg.store import open_store
    from zagg.windows import union_time_range

    target_order = cell_order - (shard_order - k)
    n_cells = 4 ** (target_order - k)
    leaf_cells = 4 ** (cell_order - shard_order)
    slabs: dict = {}
    digests: dict = {}
    for name, meta in fields.items():
        if meta["class"] == "exact":
            dtype = np.dtype(meta.get("dtype") or "float32")
            slabs[name] = np.full(
                n_cells, _fill_scalar(meta.get("fill_value", "NaN"), dtype), dtype
            )
        else:
            digests[name] = [[] for _ in range(n_cells)]
    if target_order >= shard_order:
        span = 4 ** (target_order - shard_order)
        fold_factor = 4 ** (shard_order - k)
    else:
        span = 1
        fold_factor = leaf_cells
    n_leaves, timestamps, granules, ranges = 0, [], 0, []
    for dec in node_shards:
        if target_order >= shard_order:
            start = _rel_rank(dec, node) * span
        else:
            start = _rel_rank(_node_at(dec, target_order), node)
        for window in fold_windows:
            leaf = shard_leaf_path(store_root, morton_word(dec), window=window)
            leaf_store = open_store(leaf, **store_kwargs)
            stamp = read_commit(leaf_store)
            if stamp is None:
                continue  # absent leaf or unstamped debris (D4)
            # Fold the whole leaf's contribution BEFORE touching the slabs, so
            # a corrupt leaf skips cleanly instead of half-applying.
            try:
                group = zarr.open_group(leaf_store, path=str(cell_order), mode="r", zarr_format=3)
                morton = group["morton"]
                if morton.shape != (leaf_cells,):
                    raise ValueError(
                        f"morton shape {morton.shape} is not the manifest cell_order "
                        f"{cell_order} subtree ({leaf_cells} cells); mixed-order source "
                        f"leaves are unsupported this round (issue #347: sweep fold "
                        f"semantics + the writer-side append guard)"
                    )
                partials: dict = {}
                cell_digests: dict = {}
                for name, meta in fields.items():
                    try:
                        values = group[name][:]
                    except KeyError:
                        # Schema evolution: the field postdates this leaf — it
                        # contributes fill, exactly what re-running would write.
                        logger.debug(f"sweep[overview]: leaf {leaf} lacks field {name!r}")
                        continue
                    if meta["class"] == "exact":
                        partials[name] = fold_dense(
                            values, fold_factor, meta.get("method"), meta.get("fill_value", "NaN")
                        )
                    else:
                        dtype = meta.get("dtype") or "float32"
                        inner = tuple(meta.get("inner_shape") or (2,))
                        cell_digests[name] = [
                            (start + i // fold_factor, decode_digest(payload, dtype, inner))
                            for i, payload in enumerate(values)
                            if payload is not None and len(payload)
                        ]
            except Exception as e:
                logger.warning(f"sweep[overview]: skipping unreadable leaf {leaf} ({e})")
                counts["failed"] += 1
                continue
            seg = slice(start, start + span)
            for name, partial in partials.items():
                meta = fields[name]
                slabs[name][seg] = combine_dense(
                    slabs[name][seg], partial, meta.get("method"), meta.get("fill_value", "NaN")
                )
            for name, decoded in cell_digests.items():
                for j, digest in decoded:
                    digests[name][j].append(digest)
            n_leaves += 1
            timestamps.append(stamp.get("written_at"))
            granules += int(stamp.get("granule_count") or 0)
            if stamp.get("time_range") is not None:
                ranges.append(stamp["time_range"])
    if n_leaves == 0:
        return None
    for name, acc in digests.items():
        meta = fields[name]
        slab = np.full(n_cells, b"", dtype=object)
        for j, cell in enumerate(acc):
            if cell:
                slab[j] = fold_digests(
                    cell, delta=int(meta.get("delta") or 512), dtype=meta.get("dtype") or "float32"
                )
        slabs[name] = slab
    stamps = [t for t in timestamps if t is not None]
    return {
        "slabs": slabs,
        "generation": {
            "n_leaves": int(n_leaves),
            "max_leaf_timestamp": max(stamps) if stamps else None,
        },
        "content_hash": _content_hash(node, k, target_order, fields, slabs),
        "granule_count": granules,
        "time_range": union_time_range(*ranges) if ranges else None,
    }


def _content_hash(node, k, target_order, fields, slabs) -> str:
    """sha256 over the folded per-field values (decoded bytes, O11-style).

    The skip-if-current backstop: leaf stamps resolve to whole seconds, so a
    same-second leaf re-run carries an unchanged generation stamp; the hash
    catches the content change without re-reading the written overview.
    """
    h = hashlib.sha256(f"{OVERVIEW_SPEC}:{node}:{k}:{target_order}".encode())
    for name in sorted(slabs):
        h.update(name.encode())
        slab = slabs[name]
        if slab.dtype == object:
            for payload in slab:
                h.update(len(payload).to_bytes(4, "little"))
                h.update(payload)
        else:
            h.update(np.ascontiguousarray(slab).tobytes())
    return h.hexdigest()


def _overview_config(fields):
    """A minimal PipelineConfig whose leaf template matches the overview.

    The overview zarr reuses the leaf template machinery
    (``HealpixGrid.emit_shard_template``) so structure — dtypes, fills, the
    D18 ragged attrs, the D16 dggs attrs — cannot drift from source leaves.
    """
    from zagg.config import PipelineConfig

    variables = {}
    for name, meta in fields.items():
        if meta["class"] == "exact":
            variables[name] = {
                "function": meta.get("method"),
                "dtype": meta.get("dtype", "float32"),
                "fill_value": meta.get("fill_value", "NaN"),
            }
        else:
            variables[name] = {
                "kind": "ragged",
                "function": "zagg.stats.tdigest.build_tdigest",
                "inner_shape": list(meta.get("inner_shape") or [2]),
                "dtype": meta.get("dtype", "float32"),
                "fill_value": 0,
            }
    return PipelineConfig(
        aggregation={
            "coordinates": {"morton": {"dtype": "uint64", "fill_value": 0}},
            "variables": variables,
        }
    )


def _write_overview(
    store_root, node, k, key, fold, fields, cell_order, shard_order, windowed, store_kwargs
) -> str:
    """Write one overview zarr at its ancestor node; returns the basename.

    D23 naming: overviews inherit window naming — ``{window}.zarr``, with the
    reserved ``all`` token for the unwindowed / all-time fold. Write order is
    pinned like a leaf's: template (wholesale overwrite — a prior overview or
    torn debris is replaced, D4 retry semantics) -> arrays -> role/provenance
    attrs -> commit stamp LAST, so an interrupted writer leaves ignorable
    debris and presence certifies the ``role`` attr landed (D11).
    """
    import zarr
    from mortie import generate_morton_children
    from zarr import open_array

    from zagg.grids.healpix import HealpixGrid
    from zagg.grids.morton import morton_word
    from zagg.hive import _utcnow, stamp_commit
    from zagg.store import open_store
    from zagg.windows import SCHEDULE_NONE_TOKEN, leaf_name_v3

    target_order = cell_order - (shard_order - k)
    basename = leaf_name_v3(None if key == SCHEDULE_NONE_TOKEN else key)
    path = f"{store_root}/{_node_rel(node)}/{basename}"
    grid = HealpixGrid(k, target_order, config=_overview_config(fields), sharded=True)
    store = open_store(path, **store_kwargs)
    grid.emit_shard_template(store, overwrite=True)
    words = np.asarray(generate_morton_children(morton_word(node), target_order), dtype=np.uint64)
    arr = open_array(store, path=f"{target_order}/morton", zarr_format=3, consolidated=False)
    arr[:] = words
    for name, slab in fold["slabs"].items():
        arr = open_array(store, path=f"{target_order}/{name}", zarr_format=3, consolidated=False)
        arr[:] = slab
    populated = _populated_mask(fold["slabs"], fields)
    root = zarr.open_group(store, path="", mode="r+", zarr_format=3)
    root.attrs.update(
        {
            ROLE_ATTR: "overview",
            OVERVIEW_ATTR: {
                "spec": OVERVIEW_SPEC,
                "node": node,
                "order": int(k),
                "cell_order": int(target_order),
                "source_shard_order": int(shard_order),
                "source_cell_order": int(cell_order),
                "window": key,
                "fields": {n: _field_provenance(m) for n, m in fields.items()},
                "generation": fold["generation"],
                "content_hash": fold["content_hash"],
                "generated_at": _utcnow(),
            },
        }
    )
    stamp_window = key if windowed else None
    stamp_commit(
        store,
        cells_with_data=int(populated.sum()),
        granule_count=int(fold["granule_count"]),
        window=stamp_window,
        time_range=fold["time_range"] if stamp_window is not None else None,
    )
    return basename


def _field_provenance(meta: dict) -> dict:
    """One field's per-field entry in the overview attrs: class + fold law.

    Exact fields also carry :data:`EXACT_NAN_POLICY`, so a reader of the
    overview knows the reduction it actually got — nan-skipping, never
    NaN-propagating (review finding, issue #201).
    """
    entry = {"class": meta["class"], "method": meta.get("method", TDIGEST_LAW)}
    if meta["class"] == "exact":
        entry["nan_policy"] = EXACT_NAN_POLICY
    return entry


def _populated_mask(slabs: dict, fields: dict) -> np.ndarray:
    """Cells carrying any non-fill value in any included field."""
    populated = None
    for name, slab in slabs.items():
        if slab.dtype == object:
            mask = np.fromiter((len(p) > 0 for p in slab), dtype=bool, count=len(slab))
        else:
            mask = ~_is_missing(slab, fields[name].get("fill_value", "NaN"))
        populated = mask if populated is None else (populated | mask)
    return populated if populated is not None else np.zeros(0, dtype=bool)
