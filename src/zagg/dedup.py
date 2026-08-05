"""Per-shard dedup status: ``has_run`` (issue #299 phase 4, D19).

Answers "has this exact product already produced this shard?" from durable
store state, for the estimate/skip flows (#298) to consume. Three signals, in
strictly increasing cost:

1. **The commit stamp** (D4): a leaf whose root attrs lack the stamp — or no
   leaf at all — is a plain ``"miss"`` (debris is invisible).
2. **The D20 stats sidecar**: the recorded ``semantic_hash`` (intended
   identity, D19) and ``granules_sha256`` (catalog identity — the output is
   ``f(template, shard, catalog snapshot)``, and ATL03 is a living
   collection). A stamped leaf whose sidecar is missing, records a different
   or absent ``semantic_hash``, or hashes a different granule set is
   ``"stale"`` — present but not provably this product over this catalog;
   a catalog-grown shard is stale, **never** a hit.
3. **The O11 content hashes**, surfaced (not recomputed) when the sidecar
   carries them: the *verifier* — "intended identical" (semantic hash) vs
   "actually byte-identical" (per-array decoded-value hashes) — for callers
   that go on to compare two stores.

Statuses are conservative by construction: every ambiguity degrades toward
recompute (``stale``/``miss``), never toward a false ``hit`` — a wrong "skip"
silently ships wrong data; a wrong "recompute" costs one shard's work.

:func:`classify_leaf_identity` (issue #388) is the worker-side flavor of the
same comparison: pure (the caller supplies the recorded sidecar), with the
id-set reading — equal / expansion / contraction / mixed — that the leaf
skip-if-current and contraction-guard decisions key on.
"""

from __future__ import annotations

import logging

from zagg.hive import read_commit, read_manifest, shard_leaf_path
from zagg.store import open_store
from zagg.telemetry import granules_sha256, read_sidecar

logger = logging.getLogger(__name__)

#: ``has_run`` statuses: complete + verified for THIS product and catalog
#: snapshot / present but unverifiable-or-outdated / not (completely) there.
STATUSES = ("hit", "stale", "miss")


def shard_status(
    store_root: str,
    shard_key,
    *,
    semantic_hash: str,
    granule_ids=None,
    window: str | None = None,
    spec: str | None = None,
    **store_kwargs,
) -> dict:
    """Dedup status of ONE (shard, window) leaf; see :func:`has_run`.

    ``granule_ids`` is the shard's CURRENT catalog snapshot in the same id
    space the sidecars record (resolved granule URLs on the aggregation
    path, STAC item ids/datetimes for raster — cf.
    :func:`zagg.telemetry.granules_sha256`); ``None`` skips the catalog
    check (identity match alone then gates the hit).

    .. warning::
       The catalog check is an EXACT ``granules_sha256`` comparison, so
       ``granule_ids`` must be the same id space the writer recorded — a
       caller that supplies a different space (bare short-names vs resolved
       URLs, sorted vs dispatch order) makes EVERY stamped shard report
       ``stale`` on ``catalog_match`` rather than a genuine catalog change.
       A ``"catalog grown/changed"`` stale carries both
       ``granules_sha256_recorded`` and ``granules_sha256_current`` so a
       mis-spaced caller can see the mismatch is systematic (every shard
       differs), not a real snapshot drift.
    """
    leaf = shard_leaf_path(store_root, int(shard_key), window=window)
    stamp = read_commit(open_store(leaf, **store_kwargs))
    if stamp is None:
        return {"status": "miss"}
    sidecar = read_sidecar(leaf, spec=spec, **store_kwargs)
    if sidecar is None:
        # Stamped but unverifiable (pre-#297 leaf, or a lost fail-open PUT):
        # never a hit — the leaf may be any product/catalog vintage.
        return {"status": "stale", "reason": "no stats sidecar"}
    detail: dict = {
        "semantic_hash_match": sidecar.get("semantic_hash") == semantic_hash,
        "catalog_match": None,
    }
    if content := sidecar.get("content_hashes"):
        # O11 verifier, surfaced when recorded (never recomputed here).
        detail["content_hashes"] = content
    if granule_ids is not None:
        recorded = sidecar.get("granules_sha256")
        current = granules_sha256(granule_ids)
        detail["catalog_match"] = recorded == current
    if not detail["semantic_hash_match"]:
        return {"status": "stale", "reason": "semantic_hash mismatch or unrecorded", **detail}
    if detail["catalog_match"] is False:
        # Surface both digests so a systematic id-space mismatch (every shard
        # differs) is distinguishable from a genuine per-shard catalog change.
        return {
            "status": "stale",
            "reason": "catalog grown/changed",
            "granules_sha256_recorded": recorded,
            "granules_sha256_current": current,
            **detail,
        }
    return {"status": "hit", **detail}


def has_run(
    store_root: str,
    config,
    shards,
    *,
    window: str | None = None,
    spec: str | None = None,
    **store_kwargs,
) -> dict:
    """Per-shard dedup status for a prospective run (issue #299 phase 4).

    Parameters
    ----------
    store_root : str
        The PRODUCT root (apply :func:`zagg.hive.effective_store_root` /
        :func:`zagg.hive.product_root` first for multi-product stores).
    config : PipelineConfig
        The prospective run's config; its ``semantic_hash`` is the identity
        compared against each sidecar.
    shards : mapping or iterable
        ``{shard_key: granule_ids}`` (current catalog snapshot per shard —
        the same id space the sidecars record) or a bare iterable of shard
        keys (catalog check skipped).
    window : str, optional
        Window label for windowed stores (one status per (shard, window)).
    spec : str, optional
        The store's naming spec for sidecar keys; default: read once from
        the manifest (``None`` on a manifest-less root = the ``/1`` legacy
        names).

    Returns
    -------
    dict
        ``{int(shard_key): {"status": "hit"|"stale"|"miss", ...detail}}``.
    """
    from zagg.semantics import semantic_hash as _semantic_hash

    want = _semantic_hash(config)
    if spec is None:
        spec = (read_manifest(store_root, **store_kwargs) or {}).get("spec")
    items = shards.items() if hasattr(shards, "items") else ((k, None) for k in shards)
    return {
        int(key): shard_status(
            store_root,
            key,
            semantic_hash=want,
            granule_ids=ids,
            window=window,
            spec=spec,
            **store_kwargs,
        )
        for key, ids in items
    }


#: Skip-if-current actions (issue #388): what the worker does with one unit.
IDENTITY_ACTIONS = ("skip", "rewrite", "refuse")


def classify_leaf_identity(recorded, *, semantic_hash, planned_ids) -> dict:
    """Classify one planned unit against its leaf's recorded identity (#388).

    Identity is the ruled PAIR: the run's ``semantic_hash`` (what/how, D19)
    x the unit's planned granule-id set (over what). The recorded side is
    the leaf's D20 stats sidecar (``semantic_hash``, ``granules_sha256``,
    and — from issue #388 on — ``granule_ids``).

    Fast path: ``granules_sha256`` equality (one compare — the common rerun
    case). The id-set difference runs ONLY on a mismatch, classifying:

    - ``skip`` / ``"equal"`` — both identity halves match: the leaf is
      current; the caller no-ops the fold and touches the leaf.
    - ``refuse`` / ``"contraction"`` or ``"mixed"`` — the recorded set has
      ids the planned set no longer lists (``recorded - planned != {}``,
      the ruled predicate — deliberately NOT strict-subset, so the mixed
      add-and-drop signature of an upstream purge behind a fresh catalog
      query trips it even when the planned set grew). ``missing`` names the
      dropped ids; the caller proceeds only under ``allow_contraction``.
      Judged on the id sets alone — a semantic mismatch does not excuse
      dropping inputs.
    - ``rewrite`` / everything else — expansion (``planned >= recorded``),
      a semantic change over covered inputs, or an unverifiable leaf (no
      sidecar; or a pre-#388 sidecar without recorded ids): wholesale D4
      rewrite exactly as today. Conservative like :func:`shard_status`:
      every ambiguity degrades toward recompute, never toward a false skip.

    Parameters
    ----------
    recorded : dict or None
        The leaf's D20 sidecar record (:func:`zagg.telemetry.read_sidecar`),
        or ``None`` when absent.
    semantic_hash : str or None
        The run's semantic-core hash (D19). ``None`` never skips.
    planned_ids : iterable of str, or None
        The unit's planned granule ids, in the id space the sidecars record
        (resolved granule URLs on the aggregation path, STAC item
        ids/datetimes for raster — cf. :func:`zagg.telemetry.granules_sha256`,
        and the id-space warning on :func:`shard_status`).

    Returns
    -------
    dict
        ``{"action", "classification", "missing"}`` — ``action`` one of
        :data:`IDENTITY_ACTIONS`; ``missing`` the sorted ``recorded -
        planned`` difference (non-empty exactly on ``refuse``).
    """
    planned = [str(g) for g in planned_ids] if planned_ids else []
    if recorded is None:
        return {"action": "rewrite", "classification": "no-sidecar", "missing": []}
    rec_hash = recorded.get("granules_sha256")
    semantic_match = semantic_hash is not None and recorded.get("semantic_hash") == semantic_hash
    if rec_hash is not None and rec_hash == granules_sha256(planned) and semantic_match:
        return {"action": "skip", "classification": "equal", "missing": []}
    rec_ids = recorded.get("granule_ids")
    if rec_ids is None:
        # Pre-#388 sidecar (or a cross-leaf rollup, where the field collapses
        # to None): something mismatched, but there is no recorded set to
        # diff — undecidable, so today's rewrite.
        return {"action": "rewrite", "classification": "unrecorded-ids", "missing": []}
    recorded_set = {str(g) for g in rec_ids}
    missing = sorted(recorded_set - set(planned))
    if missing:
        added = set(planned) - recorded_set
        return {
            "action": "refuse",
            "classification": "mixed" if added else "contraction",
            "missing": missing,
        }
    if set(planned) == recorded_set:
        # Same id set, yet the fast path failed: a semantic-core change over
        # identical inputs, an unrecorded semantic hash, or duplicate/order
        # drift in the hash's multiset. Not current, not a contraction.
        classification = "id-multiset-drift" if semantic_match else "semantic-mismatch"
        return {"action": "rewrite", "classification": classification, "missing": []}
    return {"action": "rewrite", "classification": "expansion", "missing": []}


__all__ = ["IDENTITY_ACTIONS", "STATUSES", "classify_leaf_identity", "has_run", "shard_status"]
