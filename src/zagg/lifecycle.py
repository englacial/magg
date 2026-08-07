"""Lifecycle touch for skip-if-current leaf units (issue #388 phase 3).

A skipped unit writes nothing — correct for the store, but invisible to
bucket lifecycle policies: sliderule-public purges older objects, so a
re-request that no-ops the pipeline must still reset the purge clock on the
products it just certified current. The touch refreshes ``LastModified``
across the unit's WHOLE footprint (lifecycle rules act per object):

- the leaf ``.zarr`` tree — commit stamp, arrays, and the in-leaf
  ``coverage.moc`` sidecar included (they are objects under the prefix);
- the D20 stats sidecar and D22 sub-map SIBLINGS of the leaf
  (:func:`zagg.telemetry.sidecar_key` / :func:`zagg.sweep.submap_key`, the
  spec-keyed naming seams);
- the issue #383 leaf column tree plus its own stats sidecar, when the run
  declares one — the identity gate has already verified declaration and
  artifact agree (``column-drift`` never reaches the touch), so the caller
  simply passes the column path on the declared arm and ``None`` otherwise.

Mechanism: local stores get ``os.utime``; S3 gets a server-side self-copy
(``CopyObject`` onto itself with ``MetadataDirective="REPLACE"`` — S3
rejects an identity copy without it; already a ``boto3`` dependency, and
obstore's ``copy`` exposes no metadata directive). The self-copy preserves
content (and the ETag for non-multipart objects) but REPLACES user metadata
with none — zagg's writers attach none, so nothing is lost. Documented
caveats, not solved here: a versioned bucket mints a new object version per
touch, and an object >= 5 GB would need a multipart copy and simply counts
as failed (leaf objects never approach it).

The unit footprint is not the whole store: the STORE-ROOT objects have no
unit that owns them, and an all-skip run re-PUTs none of them
(:func:`zagg.hive.ensure_manifest` early-returns on a frozen-key match, and
every skip-capable run has ``overwrite=False``). :func:`touch_store_root`
covers them once per run — the D6 ``morton_hive.json``, the
``aggregation.yaml`` semantic core, and the root ``coverage.moc`` — and the
local backend calls it from the same post-units wrap-up that already unions
the root MOC (that process is also the worker; the D8 orchestrator-no-write
rule constrains only the Lambda paths, which therefore do NOT get this
today — see the PR #397 question).

Everything here is BEST-EFFORT and fail-open (the D9 posture): a failed
touch logs and counts, and the run degrades to today's behavior — it never
fails the unit, never un-skips it, and :func:`touch_current_unit` never
raises out of the seam. An ABSENT path is neither touched nor failed: a
unit legitimately has no sub-map (non-HEALPix grids, id-less entries).

Documented gaps, not solved here: the §7 sweep's ancestor-node overviews
and ``pyramid.json`` envelopes belong to no unit footprint and are not
store-root objects either, and a skip produces zero sweep dirtiness by
construction — so a skip-only store ages its pyramid above the leaves while
the leaves and the root survive. Walking them would mean a store-wide walk
this module deliberately does not own (PR #397 question).
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def touch_current_unit(
    leaf_path,
    *,
    column_path=None,
    sidecar_spec=None,
    store_kwargs=None,
) -> dict:
    """The issue #388 skip-path touch for ONE ``(shard, window)`` unit.

    Assembles the unit's footprint from its leaf path — the leaf tree, the
    stats-sidecar and sub-map siblings named by ``sidecar_spec`` (the
    manifest spec in effect), and ``column_path``'s tree + sidecar when the
    caller passes one — then touches it via :func:`touch_unit_footprint`.
    Returns the counts dict; NEVER raises (assembly errors count as one
    failure and the touch is skipped).
    """
    try:
        from zagg.sweep import submap_key
        from zagg.telemetry import sidecar_key

        prefix, _, name = str(leaf_path).rstrip("/").rpartition("/")
        trees = [str(leaf_path)]
        objects = [
            f"{prefix}/{sidecar_key(name, sidecar_spec)}",
            f"{prefix}/{submap_key(name, sidecar_spec)}",
        ]
        if column_path:
            trees.append(str(column_path))
            # The column's D20 sidecar: its own stem + ``.stats.json``
            # (``zagg.column._sidecar_name``'s grammar).
            objects.append(str(column_path).removesuffix(".zarr") + ".stats.json")
    except Exception as e:
        logger.warning(f"lifecycle touch skipped — footprint unresolved (fail-open): {e}")
        return {"touched": 0, "failed": 1}
    return touch_unit_footprint(trees, objects, store_kwargs=store_kwargs)


def touch_store_root(store_root, *, store_kwargs=None) -> dict:
    """Touch the store-ROOT objects no unit footprint covers (issue #388).

    A run whose every unit skipped writes nothing at the root either:
    :func:`zagg.hive.ensure_manifest` accepts a frozen-key-matching manifest
    with no second PUT (and ``aggregation.yaml`` is written only inside that
    PUT branch), and every skip-capable run has ``overwrite=False``. So the
    REQUIRED reader-facing ``morton_hive.json`` (D6) would expire under a
    lifecycle rule while 100% of the data objects it indexes are fresh.
    Three objects, O(1) requests, all fail-open (an absent root
    ``coverage.moc`` — ``output.coverage_moc`` off — is not a failure).
    """
    try:
        from zagg.hive import AGGREGATION_CORE_NAME, MANIFEST_NAME, ROOT_COVERAGE_NAME

        root = str(store_root).rstrip("/")
        names = (MANIFEST_NAME, AGGREGATION_CORE_NAME, ROOT_COVERAGE_NAME)
    except Exception as e:
        logger.warning(f"store-root touch skipped — names unresolved (fail-open): {e}")
        return {"touched": 0, "failed": 1}
    return touch_unit_footprint([], [f"{root}/{name}" for name in names], store_kwargs=store_kwargs)


def touch_unit_footprint(trees, objects, *, store_kwargs=None) -> dict:
    """Touch every object under each of ``trees`` + each single ``objects`` path.

    Paths are either local filesystem paths or ``s3://`` URLs; ``store_kwargs``
    is the writers' store-kwargs dict (``region`` / ``credentials`` /
    ``endpoint_url``) and keys the S3 client. Returns ``{"touched": n,
    "failed": m}`` and never raises — an unexpected error (client build, LIST
    fault) counts one failure and abandons the remainder (best-effort).
    """
    counts = {"touched": 0, "failed": 0}
    box: dict = {}
    try:
        for tree in trees:
            if _is_s3(tree):
                _touch_s3_tree(_client(box, store_kwargs), tree, counts)
            else:
                _touch_local_tree(tree, counts)
        for obj in objects:
            if _is_s3(obj):
                bucket, key = _split_s3(obj)
                _touch_s3_object(_client(box, store_kwargs), bucket, key, counts)
            else:
                _touch_local_object(obj, counts)
    except Exception as e:
        counts["failed"] += 1
        logger.warning(f"lifecycle touch aborted mid-footprint (fail-open, issue #388): {e}")
    return counts


def _is_s3(path) -> bool:
    return str(path).startswith("s3://")


def _split_s3(path) -> tuple:
    bucket, _, key = str(path)[len("s3://") :].partition("/")
    return bucket, key


def _client(box: dict, store_kwargs):
    """One lazily built S3 client per touch call (never built for local stores)."""
    if "s3" not in box:
        box["s3"] = _s3_client(store_kwargs or {})
    return box["s3"]


def _s3_client(store_kwargs: dict):
    """boto3 S3 client from the writers' store-kwargs (camelCase credentials)."""
    import boto3

    kwargs: dict = {}
    if store_kwargs.get("region"):
        kwargs["region_name"] = store_kwargs["region"]
    if store_kwargs.get("endpoint_url"):
        kwargs["endpoint_url"] = store_kwargs["endpoint_url"]
    creds = store_kwargs.get("credentials")
    if creds:
        kwargs["aws_access_key_id"] = creds["accessKeyId"]
        kwargs["aws_secret_access_key"] = creds["secretAccessKey"]
        if creds.get("sessionToken"):
            kwargs["aws_session_token"] = creds["sessionToken"]
    return boto3.client("s3", **kwargs)


def _touch_local_tree(root, counts) -> None:
    if not os.path.isdir(root):
        return
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            _touch_local_object(os.path.join(dirpath, name), counts)


def _touch_local_object(path, counts) -> None:
    if not os.path.isfile(path):
        return
    try:
        os.utime(path, None)
        counts["touched"] += 1
    except OSError as e:
        counts["failed"] += 1
        logger.warning(f"lifecycle touch failed for {path} (fail-open): {e}")


def _touch_s3_tree(s3, path, counts) -> None:
    bucket, key = _split_s3(path)
    prefix = key.rstrip("/") + "/"
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for entry in page.get("Contents") or []:
            _touch_s3_object(s3, bucket, entry["Key"], counts)


def _touch_s3_object(s3, bucket, key, counts) -> None:
    try:
        s3.copy_object(
            Bucket=bucket,
            Key=key,
            CopySource={"Bucket": bucket, "Key": key},
            MetadataDirective="REPLACE",
        )
        counts["touched"] += 1
    except Exception as e:
        code = ""
        response = getattr(e, "response", None)
        if isinstance(response, dict):
            code = (response.get("Error") or {}).get("Code", "")
        if code in ("NoSuchKey", "404"):
            return  # an absent sibling (e.g. no sub-map) is not a failure
        counts["failed"] += 1
        logger.warning(f"lifecycle touch failed for s3://{bucket}/{key} (fail-open): {e}")


__all__ = ["touch_current_unit", "touch_store_root", "touch_unit_footprint"]
