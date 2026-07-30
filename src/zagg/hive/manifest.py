"""Hive manifest: the static ``morton_hive.json`` and product discovery (issue #330).

Split out of the single-file ``zagg.hive`` (issue #330 phase 1); the public
surface is unchanged and re-exported from :mod:`zagg.hive`. Holds the §3/D6
manifest builder, the frozen-key resume guard (:func:`validate_manifest` /
:func:`ensure_manifest`), the D19 semantic-core convenience object, and the
root-form discrimination that reads them.
"""

from __future__ import annotations

import json
import logging

from zagg.hive.layout import (
    HIVE_SPEC,
    HIVE_SPEC_V2,
    _is_base_component,
    _is_valid_product_name,
    _read_json,
    _utcnow,
    product_root,
)
from zagg.store import open_object_store

logger = logging.getLogger(__name__)

#: Root manifest object name (the root-only exception to the node invariant).
MANIFEST_NAME = "morton_hive.json"

#: Canonical semantic core object name at the product root (D19): a DERIVED
#: convenience (D9 cache class) — deterministic YAML of the output-defining
#: subset. The manifest's ``semantic_hash`` is the truth; divergence resolves
#: to the hash.
AGGREGATION_CORE_NAME = "aggregation.yaml"


def classify_store_root(store_root: str, **store_kwargs) -> str:
    """Root-form discrimination by CONTENT (D19): what sits at ``store_root``.

    Returns ``"bare"`` (a manifest at the root ⇒ a single-product store — the
    digit tree lives right here), ``"products"`` (no manifest, and at least
    one name-shaped child prefix ⇒ a directory of product roots), or
    ``"empty"`` (neither). Base-component-shaped children without a manifest
    still classify as ``"bare"`` — the manifest write is async (issue #252),
    so a store mid-first-run must not be misread as a product directory.
    """
    import obstore

    store = open_object_store(store_root, **store_kwargs)
    if _read_json(store, MANIFEST_NAME) is not None:
        return "bare"
    listing = obstore.list_with_delimiter(store)
    children = [p.rstrip("/").split("/")[-1] for p in listing["common_prefixes"]]
    if any(_is_base_component(c) for c in children):
        return "bare"
    names = [c for c in children if _is_valid_product_name(c)]
    return "products" if names else "empty"


def list_products(store_root: str, **store_kwargs) -> dict:
    """``{name: manifest}`` for every product under a multi-product root (D19).

    Discovery is the root listing itself — no name↔hash translation layer:
    each name-shaped child prefix is read for its bare-named
    ``morton_hive.json``. Children without a readable manifest are skipped
    (debris or mid-write; the D4 posture — absence of a manifest means the
    product is not yet discoverable).
    """
    import obstore

    store = open_object_store(store_root, **store_kwargs)
    listing = obstore.list_with_delimiter(store)
    children = [p.rstrip("/").split("/")[-1] for p in listing["common_prefixes"]]
    products = {}
    for name in sorted(c for c in children if _is_valid_product_name(c)):
        manifest = read_manifest(product_root(store_root, name), **store_kwargs)
        if manifest is not None:
            products[name] = manifest
    return products


def build_manifest(grid, dataset: dict | None = None, windowing: dict | None = None) -> dict:
    """Build the static ``morton_hive.json`` payload for one store (§3, D6).

    ``grid`` supplies the orders; ``dataset`` (typically the ShardMap's
    ``metadata``) supplies identity — only ``short_name`` and ``version`` are
    recorded. The split schedule is implicit under D2 (one digit per level down
    to the shard order) but recorded explicitly for forward compatibility. The
    ``pyramid`` block carries the template-time overview declaration —
    per-family order schedule + per-field D24 composability classes
    (:func:`zagg.sweep_overview.build_pyramid_block`, issue #201); the §7
    sweep populates ``materialized`` actuals but never rewrites the
    declaration (overviews are a second-pass sweep, never written at
    fan-out time — D11).

    ``windowing`` (issue #246) is the normalized declaration from
    :func:`zagg.config.get_windowing`; when given, the manifest declares
    ``spec: "morton-hive/2"`` and carries the D15 **temporal block** — the
    STATIC schema half of the temporal split (schedule, time encoding, the
    membership ``time_field``, the explicit windows list, and the append
    policy). Temporal EXTENT deliberately never lives here: actual ranges are
    leaf-stamp truth and root-summary cache. ``None`` writes the ``/1``
    manifest byte-identical to pre-windowing runs.
    """
    from zagg.semantics import semantic_hash
    from zagg.sweep_overview import build_pyramid_block

    dataset = dataset or {}
    manifest = {
        "spec": HIVE_SPEC_V2 if windowing else HIVE_SPEC,
        "dataset": {
            "short_name": dataset.get("short_name"),
            "version": dataset.get("version"),
        },
        # D19 (issue #299): the FULL sha256 of the canonicalized semantic
        # core — a frozen key, so reusing a product name/root with different
        # aggregation semantics refuses up front like an orders mismatch.
        "semantic_hash": semantic_hash(grid.config),
        "cell_order": int(grid.child_order),
        "shard_order": int(grid.parent_order),
        "split_schedule": [1] * int(grid.parent_order),
        # D21: how many morton digits each path component chunks. Existing
        # stores are retroactively 1 (the _frozen normalization); new stores
        # declare it explicitly.
        "path_grouping": 1,
        "pyramid": build_pyramid_block(grid.config, int(grid.parent_order)),
        "generated_at": _utcnow(),
    }
    if windowing:
        temporal = {
            "schedule": windowing["schedule"],
            "time_field": windowing["time_field"],
            "epoch": windowing["epoch"],
            "scale": windowing["scale"],
            "units": windowing["units"],
            # D15 records calendar alongside encoding/units/epoch. Only
            # proleptic_gregorian is supported this round: the three scales
            # (utc/gps/tai) all derive from stdlib datetime, which is proleptic
            # Gregorian.
            "calendar": "proleptic_gregorian",
            # Generative schedules append by adding leaves the schedule already
            # describes (manifest untouched); the explicit list is the noted
            # D15 exception — appending outside it re-templates the manifest.
            "append_policy": (
                "re-template" if windowing["schedule"] == "explicit" else "new-window"
            ),
        }
        if windowing.get("windows"):
            temporal["windows"] = windowing["windows"]
        manifest["temporal"] = temporal
    return manifest


def validate_manifest(
    store_root: str, manifest: dict, *, overwrite: bool = False, **store_kwargs
) -> dict | None:
    """Read-only frozen-key precheck — the fail-fast half of the manifest guard.

    Split out of :func:`ensure_manifest` when the manifest WRITE came off the
    synchronous pre-dispatch path (issue #252). The write moving off the
    critical path is fine for readers, but it dragged the writer-side guard
    along with it — so an incompatible rerun could write mixed-order leaves
    before the check ever fired (D2). This function is the guard on its own:
    the lambda ping runs it BEFORE fan-out to refuse an incompatible existing
    store up front, while the write itself rides the async init-time setup
    invoke with a finalize backstop (issue #252 hybrid; the local dispatcher
    writes directly at init via :func:`ensure_manifest`, which calls this
    first).

    Performs exactly the checks :func:`ensure_manifest` does and writes nothing:
    reads the existing manifest; on an existing store whose FROZEN keys mismatch
    (:data:`_FROZEN_MANIFEST_KEYS`) it raises — the same ``does not match this
    run`` refusal without ``overwrite``, and the same ``list_with_delimiter``
    shard-data refusal with ``overwrite`` (the D2 old-order-masquerade footgun).
    Returns the existing manifest (``None`` on a fresh root).
    """
    import obstore

    store = open_object_store(store_root, **store_kwargs)
    existing = _read_json(store, MANIFEST_NAME)
    frozen_matches = _frozen_matches(existing, manifest)
    if existing is not None and not overwrite:
        if not frozen_matches:
            raise ValueError(
                f"{MANIFEST_NAME} at {store_root} does not match this run "
                f"(existing {existing!r} vs {manifest!r}); this store was templated "
                f"for different orders/identity — clear the store root (or pick a "
                f"new one) before writing with this configuration"
            )
        return existing
    if overwrite and existing is not None and not frozen_matches:
        # One delimiter-LIST: a {sign+base}-shaped child means shards were
        # already written under the OLD configuration. Their leaves are
        # stamped and walker-discoverable, so replacing just the manifest
        # would leave them masquerading as legal mixed-order data (D2).
        # ``semantic_hash`` is a frozen key (D19), so overwrite does NOT
        # bypass the semantic-hash refusal when both manifests carry one —
        # a changed aggregation block over existing data refuses right here.
        listing = obstore.list_with_delimiter(store)
        children = [p.rstrip("/").split("/")[-1] for p in listing["common_prefixes"]]
        if any(_is_base_component(c) for c in children):
            raise ValueError(
                f"refusing to overwrite {MANIFEST_NAME} at {store_root} with "
                f"different orders/identity: the digit tree already has shard "
                f"data (e.g. {children[0]!r}/), and overwrite replaces the "
                f"manifest only — clear the store root first"
            )
    # Hash-guard coherence (issue #341): ``_frozen_matches`` EXEMPTS
    # ``semantic_hash`` when EITHER side lacks it (no hash to compare — the
    # exemption keeps pre-#299 stores resumable), so an overwrite proceeds past
    # a comparison that never happened. Warn on BOTH directions of that
    # exemption (fold review: the guard was asymmetric where the exemption is
    # symmetric), because they fail differently:
    #
    # * existing has no hash, this run does  -> the existing DATA's semantics are
    #   unverifiable. The store is coherent going forward (the rewrite stamps
    #   this run's hash, and every re-dispatched leaf is re-templated wholesale
    #   — ``emit_shard_template`` clears the leaf prefix first) but leaves this
    #   run does not rewrite may carry the old schema.
    # * existing HAS a hash, this run does not -> the overwrite strips the
    #   recorded hash from a hashed store, un-provenancing a #299 store rather
    #   than merely failing to verify one. Strictly the worse direction, and the
    #   one no warning covered.
    #
    # ``build_manifest`` always stamps a hash, so the second direction is not
    # reachable from a zagg-built manifest today — it is the "older zagg (or a
    # hand-assembled manifest) writes into a newer store" shape, one refactor of
    # ``build_manifest`` away from being live.
    hash_exempted = existing is not None and (existing.get("semantic_hash") is None) != (
        manifest.get("semantic_hash") is None
    )
    if overwrite and frozen_matches and hash_exempted:
        listing = obstore.list_with_delimiter(store)
        children = [p.rstrip("/").split("/")[-1] for p in listing["common_prefixes"]]
        if any(_is_base_component(c) for c in children):
            if existing.get("semantic_hash") is None:
                detail = (
                    "the existing manifest predates semantic hashing (issue #299), so "
                    "semantic compatibility of the existing shard data cannot be "
                    "verified; re-dispatched leaves are re-templated wholesale, but "
                    "leaves this run does not rewrite may carry the old schema"
                )
            else:
                detail = (
                    "this run's manifest carries NO semantic hash while the existing "
                    "one does (issue #299), so the overwrite DROPS the recorded hash "
                    "from a hashed store — the existing shard data becomes "
                    "un-provenanced, not merely unverified"
                )
            logger.warning(f"overwriting {MANIFEST_NAME} at {store_root}: {detail}")
    return existing


def ensure_manifest(
    store_root: str,
    manifest: dict,
    *,
    overwrite: bool = False,
    config=None,
    **store_kwargs,
):
    """Write the root manifest; verify it on reruns (idempotent).

    Lifecycle (issue #252 hybrid): the PRIMARY write runs at init — the local
    dispatcher writes directly pre-dispatch; the lambda dispatcher fires the
    ``mode="setup"`` hive branch as a fire-and-forget Event invoke right
    after the ping — so the manifest typically lands within seconds of init
    (best-effort: the Event invoke shares worker concurrency and runs
    retries-0, deferring to the finalize backstop under throttling or a
    dropped invoke) and a reader can consume completed leaves while the store
    builds. Finalize calls this
    again as an idempotent BACKSTOP (an existing frozen-key-matching manifest
    is accepted — no second PUT): worker Event invokes run with retries 0, so
    a lost async init write self-heals at finalize. The fail-fast half of the
    guard is exposed separately as :func:`validate_manifest` (the ping's
    read-only precheck), which this function runs first.

    A retry into an existing hive store must be able to proceed (that is the
    D4 debris/retry model), so an existing manifest is accepted — but only if
    its FROZEN keys match the run's own (:data:`_FROZEN_MANIFEST_KEYS`: orders
    + identity + schedule — the flat path's ``_check_signature`` analogue).
    ``generated_at`` and ``pyramid`` are excluded: the pyramid block is
    populated/updated by the §7 sweep by design (D11), so comparing it would
    brick every resume after the first sweep.

    ``overwrite=True`` replaces the MANIFEST ONLY — it never clears data. To
    guard against the silent-corruption footgun (committed leaves from the old
    orders would survive a "re-template" and be indistinguishable from legal
    mixed-order data, D2), an overwrite that CHANGES the frozen keys refuses
    when the digit tree already has children (one delimiter-LIST); clear the
    store root first. Returns the manifest now in effect.
    """
    import obstore

    store = open_object_store(store_root, **store_kwargs)
    existing = validate_manifest(store_root, manifest, overwrite=overwrite, **store_kwargs)
    if existing is not None and not overwrite:
        return existing
    obstore.put(store, MANIFEST_NAME, json.dumps(manifest, indent=1).encode())
    if config is not None:
        write_semantic_core(store_root, config, **store_kwargs)
    return manifest


def write_semantic_core(store_root: str, config, **store_kwargs) -> None:
    """PUT the canonical semantic core as ``aggregation.yaml`` (D19).

    A derived convenience next to the manifest (D9 cache class): sorted-key,
    deterministic YAML of the output-defining subset, so product names stay
    human-inspectable without a registry. FAIL-OPEN — the manifest's
    ``semantic_hash`` is the truth, and the core is regenerable straight from
    the config (rewriting it is a single :func:`write_semantic_core` call), so
    a failed PUT must not fail the run. The D22 pyramid sweep is the intended
    owner of the rewrite once it lands; there is no regenerator today.
    """
    import obstore
    import yaml

    from zagg.semantics import semantic_core

    try:
        payload = yaml.safe_dump(semantic_core(config), sort_keys=True)
        obstore.put(
            open_object_store(store_root, **store_kwargs),
            AGGREGATION_CORE_NAME,
            payload.encode(),
        )
    except Exception as e:
        logger.warning(f"{AGGREGATION_CORE_NAME} write failed (fail-open, D9 cache class): {e}")


#: Manifest keys the resume match-check compares (orders + identity + split
#: and temporal schedules — a windowing change re-partitions the leaf names,
#: so it must refuse resume exactly like an orders change). ``generated_at``
#: (a timestamp) and ``pyramid`` (populated by the §7 sweep, D11) are mutable
#: by design and excluded. ``temporal`` projects to ``None`` on both sides for
#: pre-#246 manifests, so existing stores resume unchanged.
_FROZEN_MANIFEST_KEYS = (
    "spec",
    "dataset",
    "semantic_hash",
    "cell_order",
    "shard_order",
    "split_schedule",
    "path_grouping",
    "temporal",
)


def _frozen(manifest: dict) -> dict:
    """The frozen-key projection of a manifest (resume/overwrite match-check).

    ``path_grouping`` normalizes ``absent -> 1`` (D21: existing stores are
    retroactively ``1``), so appends to pre-D21 manifests never refuse on it.
    """
    frozen = {k: manifest.get(k) for k in _FROZEN_MANIFEST_KEYS}
    frozen["path_grouping"] = manifest.get("path_grouping") or 1
    return frozen


def _frozen_matches(existing: dict | None, manifest: dict) -> bool:
    """Whether two manifests agree on every frozen key (issue #299).

    ``semantic_hash`` is compared only when BOTH sides declare it: a
    pre-#299 manifest lacks the key, and refusing every append to an
    existing store on its absence would brick resumes — the orders/schedule
    keys still guard those stores. Two hash-carrying manifests must match
    exactly (D19: the hash is a frozen key).
    """
    if existing is None:
        return False
    fa, fb = _frozen(existing), _frozen(manifest)
    if fa["semantic_hash"] is None or fb["semantic_hash"] is None:
        fa.pop("semantic_hash")
        fb.pop("semantic_hash")
    return fa == fb


def read_manifest(store_root: str, **store_kwargs) -> dict | None:
    """Read ``morton_hive.json`` from a store root; ``None`` when absent."""
    return _read_json(open_object_store(store_root, **store_kwargs), MANIFEST_NAME)
