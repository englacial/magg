"""Unified second-pass rollup sweep (issue #300, design §7 / D22).

One idempotent bottom-up pass over a hive store's digit tree that folds leaf
artifacts into interior-node rollups, per registered **artifact family**
(D22): stats sidecars (the :func:`zagg.telemetry.merge` fold), MOC regen,
sub-shardmap rollups (leaf ShardMap JSON folded via the #294 exact coarsen
regroup), overview zarrs (issue #201; :mod:`zagg.sweep_overview`), and
optional debris collection (stubbed). Everything the sweep
writes is a **regenerable cache, never truth** (D9): deleting every rollup
leaves all leaf reads intact, and the leaf sidecars/stamps remain the durable
ground truth.

Discovery is from **run records, never a recursive LIST** (D22): callers pass
the leaves a run touched — ``(shard_key, window)`` pairs from the dispatcher's
run report (the end-of-run hook) or from the run-level stats parquets at the
store root (the manual CLI). The walk visits only the ancestor paths of those
leaves; untouched siblings contribute through their existing stored rollups
(read, never recomputed), so incremental runs accumulate exactly.

Every rollup is stamped with **generation info** — merged-leaf count + max
leaf timestamp (D22) — making staleness *detectable, not prevented*: a leaf
re-run bumps its artifact timestamp past every earlier stamp, so ancestors'
stored generations no longer match and the next sweep rewrites exactly that
chain. Skip-if-current is a **two-part test**: the generation stamp is the
fast path (a matching count + max timestamp), backstopped by a
**payload-equality** check for same-second rewrites. Both timestamp sources
resolve to whole seconds (``timespec="seconds"``), so a leaf re-run within one
wall-clock second carries an unchanged stamp; every node therefore recomputes
its merged payload BEFORE the skip decision and PUTs whenever the stored
payload differs, so a same-second content change still rewrites the whole
chain (shard nodes re-read their leaf sidecars each pass; interior nodes fold
the freshly computed child payloads, so the rewrite cascades up). A second
sweep over an unchanged tree recomputes but PUTs nothing.

Rollup objects are JSON sidecars at digit nodes, named ``{family}.rollup.json``
— deliberately DISTINCT from the leaf sidecar names (``stats.json`` /
``coverage.moc``): under D24 mixed-order stores a node can be leaf and
interior at once, so sharing the leaf names would self-clobber; distinct names
also keep the walker's closed name set unambiguous (they list as objects,
never as digit-shaped prefixes, so the §5 discovery walk is unaffected).
"""

from __future__ import annotations

import json
import logging

# Facade re-exports (issue #330 phase 3). Plain imports are the names this
# module's own code calls; the ``x as x`` redundant aliases are the explicit
# re-export marker for the rest of the pre-split surface — ``zagg.runner`` and
# ``zagg.client`` import ``submap_emittable``/``write_leaf_submap`` off
# ``zagg.sweep``, and ``tests/test_sweep.py`` imports ``SWEEP_SPEC``,
# ``submap_key``, ``_ReprojectTarget`` and ``_node_rel`` the same way.
# READ compat only: patch the owning submodule, not the re-export.
from zagg.sweep_families import (
    DEFAULT_FAMILIES,
    FAMILIES,
    get_family,
)
from zagg.sweep_families import SUBMAP_NAME as SUBMAP_NAME
from zagg.sweep_families import SWEEP_SPEC as SWEEP_SPEC
from zagg.sweep_families import DebrisFamily as DebrisFamily
from zagg.sweep_families import MocFamily as MocFamily
from zagg.sweep_families import OverviewFamily as OverviewFamily
from zagg.sweep_families import StatsFamily as StatsFamily
from zagg.sweep_families import SubmapFamily as SubmapFamily
from zagg.sweep_families import SweepFamily as SweepFamily
from zagg.sweep_families import _moc_payload as _moc_payload
from zagg.sweep_families import _ReprojectTarget as _ReprojectTarget
from zagg.sweep_families import _warn_unsupported_submap as _warn_unsupported_submap
from zagg.sweep_families import _warned_unsupported_submap as _warned_unsupported_submap
from zagg.sweep_families import submap_emittable as submap_emittable
from zagg.sweep_families import submap_key as submap_key
from zagg.sweep_families import write_leaf_submap as write_leaf_submap
from zagg.sweep_rollup import _ancestor, _node_rel, _rollup_interior, _rollup_shard_node
from zagg.sweep_rollup import _generation as _generation
from zagg.sweep_rollup import _merged as _merged
from zagg.sweep_rollup import _put_rollup as _put_rollup
from zagg.sweep_rollup import _read_rollup as _read_rollup
from zagg.sweep_rollup import _rollup_key as _rollup_key

logger = logging.getLogger(__name__)


def run_sweep(store_root: str, leaves, *, families=None, store_kwargs: dict | None = None) -> dict:
    """One sweep pass: fold leaf artifacts up-tree for each family (D22).

    ``leaves`` is the run-record-derived work set — an iterable of
    ``(shard_key, window)`` pairs (or bare shard keys, meaning unwindowed):
    the leaves whose ancestors may be stale. The walk visits ONLY those
    ancestor paths; siblings contribute via their stored rollups. Leaves at a
    non-manifest order are skipped with a warning (mixed-order stores are
    unsupported this round, matching ``refresh_root_coverage``).

    Idempotent: a rollup whose stored generation (merged-leaf count + max
    leaf timestamp) AND stored payload both match the freshly computed ones is
    left untouched, so a second pass over an unchanged tree writes nothing; the
    payload compare is the same-second backstop the generation stamp cannot see
    (module docstring). Returns a summary with
    per-family ``written`` / ``current`` (skip-if-current) / ``empty`` (no
    artifact found) / ``failed`` (unmergeable, logged) counts.
    """
    from zagg.hive import MANIFEST_NAME, read_manifest
    from zagg.store import open_object_store

    store_kwargs = dict(store_kwargs or {})
    manifest = read_manifest(store_root, **store_kwargs)
    if manifest is None:
        raise ValueError(f"no {MANIFEST_NAME} at {store_root} — not a hive store root")
    shard_order = int(manifest["shard_order"])
    fams = [get_family(n) for n in (DEFAULT_FAMILIES if families is None else families)]
    by_shard, skipped = _normalize_leaves(leaves, shard_order)
    store = open_object_store(store_root, **store_kwargs)
    summary: dict = {
        "store_root": store_root,
        "shard_order": shard_order,
        "n_leaves": sum(len(w) for w in by_shard.values()),
        "skipped_leaves": skipped,
        "families": {},
    }
    for fam in fams:
        runner = getattr(fam, "sweep_store", None)
        if runner is not None:
            summary["families"][fam.name] = runner(store_root, manifest, by_shard, store_kwargs)
        else:
            summary["families"][fam.name] = _sweep_family(
                store_root, store, fam, by_shard, shard_order, manifest.get("spec"), store_kwargs
            )
    return summary


def _normalize_leaves(leaves, shard_order: int):
    """``{shard_decimal: {window, ...}}`` from run-record leaf refs."""
    from zagg.grids.morton import morton_decimal
    from zagg.hive import _decimal_order

    by_shard: dict[str, set] = {}
    skipped: list[str] = []
    for ref in leaves:
        key, window = ref if isinstance(ref, (tuple, list)) else (ref, None)
        decimal = morton_decimal(int(key))
        if _decimal_order(decimal) != shard_order:
            logger.warning(
                f"sweep: skipping leaf {decimal} at order {_decimal_order(decimal)} under a "
                f"shard_order-{shard_order} manifest (mixed-order stores are unsupported)"
            )
            skipped.append(decimal)
            continue
        by_shard.setdefault(decimal, set()).add(None if window is None else str(window))
    return by_shard, skipped


def _sweep_family(store_root, store, fam, by_shard, shard_order, spec, store_kwargs) -> dict:
    """Bottom-up fold of one family over the dirty ancestor paths."""
    counts = {"written": 0, "current": 0, "empty": 0, "failed": 0}
    computed: dict[str, dict | None] = {}
    for decimal in sorted(by_shard):
        computed[decimal] = _rollup_shard_node(
            store_root,
            store,
            fam,
            decimal,
            by_shard[decimal],
            shard_order,
            spec,
            store_kwargs,
            counts,
        )
    frontier = [d for d in sorted(by_shard) if computed[d] is not None]
    for _order in range(shard_order - 1, -1, -1):
        parents = sorted({a for d in frontier if (a := _ancestor(d)) is not None})
        frontier = []
        for node in parents:
            computed[node] = _rollup_interior(store, fam, node, computed, counts)
            if computed[node] is not None:
                frontier.append(node)
        if not frontier:
            break
    tops = [computed[d] for d in frontier]
    result = dict(counts)
    result.update(fam.finish(store_root, tops, shard_order, store_kwargs))
    return result


# ---------------------------------------------------------------------------
# Trigger surfaces (issue #300 phases 4-5): run-record discovery, the manual
# CLI, and the end-of-run hook wrapper.
# ---------------------------------------------------------------------------


def sweep_after_run(
    store_root: str, leaves, *, families=None, store_kwargs: dict | None = None
) -> dict | None:
    """End-of-run hook: fail-open wrapper around :func:`run_sweep` (D22).

    Off the critical path by contract: any failure — missing manifest, no
    store write access, a family blowing up — logs one warning and returns
    ``None``; the run result is untouched. Rollups are regenerable caches
    (D9), so a skipped sweep costs one later CLI pass, never a wrong answer.
    Called in-process by the LOCAL dispatchers only (they are the workers and
    hold the user's store credentials); the Lambda dispatchers never PUT (the
    D8 standing rule) and post a fire-and-forget ``mode="sweep"`` worker
    Event invoke instead, whose handler calls :func:`run_sweep` directly.
    """
    try:
        summary = run_sweep(store_root, leaves, families=families, store_kwargs=store_kwargs)
        logger.info(f"Post-run sweep: {summary['families']}")
        return summary
    except Exception as e:
        logger.warning(f"post-run sweep failed (fail-open, D9/D22 — rollups are caches): {e}")
        return None


def leaves_from_stats_records(records) -> list:
    """``(shard_key, window)`` work-set pairs from per-shard stats records.

    The dispatcher-side bridge from a run report to the sweep: every
    successful unit's record (envelope- or meta-ridden) names its leaf via
    ``shard_key`` + the issue #300 ``window`` field. Records without a window
    key (older workers) map to the unwindowed leaf name — on a windowed store
    that read simply misses (fail-open; the CLI backstops). Failure and
    ``None`` records are skipped; pairs are deduplicated and sorted.
    """
    refs = {
        (int(r["shard_key"]), r.get("window"))
        for r in records
        if r and r.get("success") and r.get("shard_key") is not None
    }
    return sorted(refs, key=lambda p: (p[0], p[1] is not None, p[1] or ""))


def discover_leaves(store_root: str, *, store_kwargs: dict | None = None) -> list:
    """Leaf refs from the run-record parquets at the product root (D22).

    One shallow delimiter LIST of the product root finds the
    ``stats_*.parquet`` run records (both the timestamp-first D20 names and
    the older ``stats_{run_id}_{ts}`` form); their success rows give the work
    set — discovery is from run records, never a tree enumeration. Windowed
    leaves resolve through the records' ``window`` column; every shard that
    contributed a pre-column (window-less) row on a windowed store falls back
    to one delimiter LIST of that shard's node — regardless of whether some
    other record already named one of its windows, since the LIST + set dedup
    is idempotent and a mixed-deployment store can hold a window only the
    sidecar knows about (bounded by the run-record shard set). Rows whose
    ``shard_key`` came back float-typed (pre-fix parquets mixing keys with
    failure-row nulls) are skipped past 2^53 with a warning — those keys are
    inexact by construction; :func:`zagg.telemetry.write_run_parquet` now
    writes the column nullable-UInt64 so new records round-trip exactly.
    Returns sorted, deduplicated ``(shard_key, window)`` pairs.
    """
    import re
    import tempfile

    import obstore

    from zagg.hive import MANIFEST_NAME, read_manifest
    from zagg.store import open_object_store

    store_kwargs = dict(store_kwargs or {})
    manifest = read_manifest(store_root, **store_kwargs)
    if manifest is None:
        raise ValueError(f"no {MANIFEST_NAME} at {store_root} — not a hive store root")
    spec = manifest.get("spec")
    windowed = manifest.get("temporal") is not None
    store = open_object_store(store_root, **store_kwargs)
    listing = obstore.list_with_delimiter(store)
    names = sorted(
        o["path"].rsplit("/", 1)[-1]
        for o in listing["objects"]
        if re.fullmatch(r"stats_.+\.parquet", o["path"].rsplit("/", 1)[-1])
    )
    refs: set = set()
    fallback_keys: set[int] = set()
    warned_float = False
    for name in names:
        import pandas as pd

        with tempfile.NamedTemporaryFile(suffix=".parquet") as tmp:
            tmp.write(bytes(obstore.get(store, name).bytes()))
            tmp.flush()
            try:
                df = pd.read_parquet(tmp.name, engine="fastparquet")
            except Exception as e:
                logger.warning(f"sweep discovery: unreadable run record {name}; skipping ({e})")
                continue
        if "shard_key" not in df.columns:
            logger.warning(f"sweep discovery: run record {name} has no shard_key; skipping")
            continue
        ok = df[df["shard_key"].notna() & df.get("success", True)]
        has_window = "window" in df.columns
        for _idx, row in ok.iterrows():
            key = row["shard_key"]
            if isinstance(key, float):
                if key != int(key) or key >= 2**53:
                    if not warned_float:
                        warned_float = True
                        logger.warning(
                            f"sweep discovery: {name} stores shard_key as float64 (a "
                            f"pre-fix run record); keys past 2^53 are inexact and are "
                            f"skipped — re-run those shards or sweep them by hand"
                        )
                    continue
            key = int(key)
            if has_window:
                window = row["window"]
                refs.add((key, None if window is None or pd.isna(window) else str(window)))
            elif windowed:
                fallback_keys.add(key)  # pre-column record: resolve below
            else:
                refs.add((key, None))
    # Pre-``window``-column records on a windowed store: the row can't name
    # its leaf, so resolve each shard's windows with ONE delimiter LIST of its
    # node — scoped by the run-record shard set, never a tree walk. Every
    # pre-column shard is LISTed even if another record already named one of
    # its windows: the LIST + set dedup is idempotent, and on a mixed
    # deployment the sidecar can hold a window no post-column record names.
    from zagg.grids.morton import morton_decimal

    for key in sorted(fallback_keys):
        node_listing = obstore.list_with_delimiter(store, _node_rel(morton_decimal(key)) + "/")
        for obj in node_listing["objects"]:
            window = _sidecar_window(obj["path"].rsplit("/", 1)[-1], spec)
            if window is not _NO_SIDECAR:
                refs.add((key, window))
    return sorted(refs, key=lambda r: (r[0], r[1] is not None, r[1] or ""))


#: Sentinel: "this object is not a stats sidecar" (``None`` means unwindowed).
_NO_SIDECAR = object()


def _sidecar_window(name: str, spec: str | None):
    """The window label a stats-sidecar object name encodes, else the sentinel."""
    from zagg.telemetry import SIDECAR_NAME, SPEC_V3
    from zagg.windows import validate_label

    try:
        if spec == SPEC_V3:
            if not name.endswith(".stats.json"):
                return _NO_SIDECAR
            stem = name.removesuffix(".stats.json")
            validate_label(stem)
            return stem
        if name == SIDECAR_NAME:
            return None
        stem, ext = SIDECAR_NAME.rsplit(".", 1)
        if name.startswith(f"{stem}_") and name.endswith(f".{ext}"):
            window = name[len(stem) + 1 : -(len(ext) + 1)]
            validate_label(window)
            return window
    except ValueError:
        return _NO_SIDECAR
    return _NO_SIDECAR


def main(argv=None) -> int:
    """Manual CLI: ``python -m zagg.sweep <store_root>`` (issue #300, D22)."""
    import argparse

    parser = argparse.ArgumentParser(
        description="zagg unified rollup sweep: fold leaf artifacts into interior-node "
        "rollups (issue #300). Work is discovered from the store's run records."
    )
    parser.add_argument("store_root", help="Hive store root (local path or s3://bucket/prefix)")
    parser.add_argument(
        "--families",
        default=None,
        help=f"Comma-separated families (default: {','.join(DEFAULT_FAMILIES)}; "
        f"registered: {', '.join(sorted(FAMILIES))})",
    )
    parser.add_argument("--region", default="us-west-2", help="AWS region (default: us-west-2)")
    parser.add_argument(
        "--output-creds",
        default=None,
        metavar="PATH",
        help="Path to a JSON credentials file for the store (same format as python -m zagg)",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    store_kwargs: dict = {"region": args.region}
    if args.output_creds:
        from zagg.runner import normalize_output_credentials

        with open(args.output_creds) as f:
            credentials = normalize_output_credentials(json.load(f))
        store_kwargs["credentials"] = credentials
        store_kwargs["endpoint_url"] = credentials.get("endpointUrl")
    families = [f.strip() for f in args.families.split(",")] if args.families else None
    leaves = discover_leaves(args.store_root, store_kwargs=store_kwargs)
    if not leaves:
        print("No completed leaves found in the store's run records; nothing to sweep.")
        return 0
    summary = run_sweep(args.store_root, leaves, families=families, store_kwargs=store_kwargs)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
