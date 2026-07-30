"""Morton-hive store layout: leaf paths, manifest, and commit stamp (issue #199).

Phase 2 of the layout migration (``docs/design/sparse_coverage.md`` §2-§3).
Under ``output.store_layout: hive`` each shard is its own **self-describing
leaf zarr** under a morton digit tree::

    {store_root}/
      morton_hive.json               <- static manifest (§3); root-only exception
      {sign+base}/{d1}/.../{d_n}/    <- one decimal digit per level (D2)
        {full_id}.zarr/              <- vanilla zarr v3 leaf (D3)

- Ids are morton decimal strings (D1); the leaf path is computed by mortie's
  ``hive_path`` (the convention is owned by the mortie spec) and re-checked
  here against the node invariant.
- **Node invariant (D5)**: below the root a node contains only digit children
  (``[1-4]/``, or the ``{sign+base}`` component at the first level) and
  ``*.zarr`` objects — zero zarr metadata above the leaf, so 2,000 workers
  share no mutable state and a delimiter-LIST with no digit children is a
  definitive "nothing finer exists".
- **Manifest (D6)**: ``morton_hive.json`` is written once — at finalize since
  issue #252 (reader-facing only; workers never read it) — and never touched
  during a run; with it every shard path is computable with zero requests.
  The convention is versioned (``morton-hive/1``) from day one.
- **Commit stamp (D4)**: the shard's FINAL write is a root
  ``group.attrs.update(...)`` marking completion (plus cell count, timestamp,
  granule count). A ``.zarr/`` prefix whose root metadata lacks the stamp is
  debris — incomplete, ignorable, safe to overwrite on retry. This is NOT
  consolidated metadata: one small PUT on an object that must exist anyway.

- **Coverage (§4, issue #200)**: the stamp carries a ``coverage`` payload —
  tier 0 is the shard's morton box, the canonical <= 4-member cover of its
  occupied cells (:func:`zagg.grids.morton.morton_box`), padded to exactly
  four decimal-string slots with JSON-null sentinels. Zero extra requests
  (it rides the stamp PUT) and debris semantics are inherited: a torn
  worker's coverage never becomes visible. Exact cell-order occupancy is a
  zstd-compressed bitmap SIDECAR inside the leaf (``coverage.moc`` — the O8
  resolution; the one recorded exception to the vanilla-v3 leaf: data reads
  are unaffected, but member enumeration warns and skips it), written
  before the stamp and pointed to from the envelope; attrs stay lean and the
  extra GET is paid only by readers that pass the box test. The optional end-of-run
  root ``coverage.moc`` (issue #200 phase 3, default-on for hive) is a
  shard-order ranges MOC at the store root — the second root-only object,
  written fail-open by the dispatcher (locally) or a fire-and-forget worker
  invoke (Lambda), and a regenerable cache under D9. The manifest's
  ``pyramid`` block declares the overview schedule at template time and the
  §7 sweep populates its actuals (D11/D22 — issue #201).

Package layout (issue #330 phase 1 — the module passed the CLAUDE.md §4
~1000-line limit at 1,393). Pure moves behind this facade; every name the
single-file module exposed still resolves as ``zagg.hive.<name>``:

- :mod:`zagg.hive.layout` — spec versions, the D19 product-name grammar, D2/D3
  leaf-path arithmetic, the D5 node invariant, and the shared JSON/clock helpers.
- :mod:`zagg.hive.manifest` — the §3/D6 ``morton_hive.json`` builder, the
  frozen-key resume guard, the D19 semantic core, and root-form discrimination.
- :mod:`zagg.hive.coverage` — the D4 commit stamp, §4 tier-0 coverage plus its
  bitmap sidecar, and the store-root ranges MOC rollup.

The per-shard write path (:func:`process_and_write_hive`) stays here, mirroring
the ratified plan's runner facade which keeps ``agg()``: it is the entry point
callers reach through ``zagg.hive``, and tests patch the stamp/sidecar helpers
on this module to assert leaf write ORDER, which only resolves if the call site
reads them from this namespace.
"""

from __future__ import annotations

import time

import numpy as np

# Facade re-exports (issue #330 phase 1). Plain imports carry the ``__all__``
# surface; the ``x as x`` redundant aliases are the explicit re-export marker
# for everything else the pre-split single-file ``zagg.hive`` exposed — public
# helpers other zagg modules import by name, plus the private helpers
# ``zagg.coverage``, ``zagg.sweep`` and ``zagg.sweep_overview`` reach for.
# ``__all__`` itself stays byte-identical to pre-split.
from zagg.hive.coverage import _ZSTD_LEVEL as _ZSTD_LEVEL
from zagg.hive.coverage import (
    COMMIT_ATTR,
    COVERAGE_BOX_SLOTS,
    COVERAGE_SIDECAR,
    COVERAGE_SPEC,
    ROOT_COVERAGE_NAME,
    build_coverage,
    build_root_coverage,
    decode_coverage_bitmap,
    encode_coverage_bitmap,
    read_commit,
    read_coverage,
    read_coverage_bitmap,
    read_root_coverage,
    root_coverage_words,
    stamp_commit,
    write_coverage_sidecar,
    write_root_coverage,
)
from zagg.hive.coverage import _cell_ranks as _cell_ranks
from zagg.hive.coverage import _decimal_base as _decimal_base
from zagg.hive.coverage import _decimal_order as _decimal_order
from zagg.hive.coverage import _decimal_rank as _decimal_rank
from zagg.hive.coverage import _rank_tail as _rank_tail
from zagg.hive.layout import _PRODUCT_NAME_RE as _PRODUCT_NAME_RE
from zagg.hive.layout import (
    HIVE_SPEC,
    HIVE_SPEC_V2,
    check_node_invariant,
    shard_leaf_path,
)
from zagg.hive.layout import PRODUCT_NAME_MAX as PRODUCT_NAME_MAX
from zagg.hive.layout import _is_base_component as _is_base_component
from zagg.hive.layout import _is_valid_product_name as _is_valid_product_name
from zagg.hive.layout import _read_json as _read_json
from zagg.hive.layout import _utcnow as _utcnow
from zagg.hive.layout import effective_store_root as effective_store_root
from zagg.hive.layout import product_root as product_root
from zagg.hive.layout import validate_product_name as validate_product_name
from zagg.hive.manifest import _FROZEN_MANIFEST_KEYS as _FROZEN_MANIFEST_KEYS
from zagg.hive.manifest import AGGREGATION_CORE_NAME as AGGREGATION_CORE_NAME
from zagg.hive.manifest import (
    MANIFEST_NAME,
    build_manifest,
    ensure_manifest,
    read_manifest,
)
from zagg.hive.manifest import _frozen as _frozen
from zagg.hive.manifest import _frozen_matches as _frozen_matches
from zagg.hive.manifest import classify_store_root as classify_store_root
from zagg.hive.manifest import list_products as list_products
from zagg.hive.manifest import validate_manifest as validate_manifest
from zagg.hive.manifest import write_semantic_core as write_semantic_core


def leaf_block_index(grid, block_index, shard_key) -> tuple:
    """Leaf-LOCAL storage block for a chunk in a hive leaf (issue #199 phase 2).

    The hive leaf's arrays are sized to one shard, so a chunk's block index is
    its position WITHIN the shard, not the global block ``iter_chunks`` yields.
    Derived from the existing ``shard_local_region`` seam (the sharded path's
    within-shard placement): the region's start divided by the chunk extent.
    At K==1 this is always ``(0,)``.
    """
    region = grid.shard_local_region(block_index, shard_key)
    return tuple(int(s.start) // int(c) for s, c in zip(region, grid.chunk_shape))


def process_and_write_hive(
    shard_key,
    granule_urls,
    grid,
    s3_creds,
    store_root,
    config,
    *,
    store_kwargs,
    driver=None,
    handoff="arrow",
    aoi_payload=None,
    profile=False,
    window=None,
):
    """Process one shard into its own hive leaf store (issue #199 phase 2).

    The SHARED per-shard write path for both backends (phase 3): the local
    runner's ``_cell_work`` and the Lambda handler's hive branch both call
    this, so leaf templating, chunk placement, ragged layout, and stamp
    ordering cannot drift between dispatchers. The shard's output is a
    self-describing leaf zarr at :func:`shard_leaf_path` ``(store_root,
    shard_key)`` (D3), with dense chunks written at leaf-LOCAL block indices
    and — as the shard's FINAL write — the D4 commit stamp on the leaf's root
    group. The leaf template is emitted lazily on the first chunk write
    (mirroring the Lambda handler's lazy store open), so a no-data shard never
    creates the ``.zarr/`` prefix; a worker that dies mid-shard leaves an
    UNSTAMPED prefix — debris, overwritten wholesale on retry
    (``overwrite=True`` on the leaf template makes the retry idempotent).
    When ``grid.sharded`` (issue #236) the dense chunks are not streamed:
    the K carriers accumulate and the whole leaf is written once
    (``write_leaf_to_zarr`` — one ShardingCodec object per array), mirroring
    the flat sharded switch in ``runner._process_and_write``.
    Phase timings are always collected (issue #297; formerly the opt-in
    ``profile`` gate of issues #100/#249): ``process_shard`` fills
    ``metadata["phase_timings"]`` with read/index/aggregate, and the leaf
    write work — interleaved with the stream (or a single post-stream pass
    when sharded), plus the ragged/coverage/stamp finalize — accumulates into
    an additive ``write`` phase alongside them. Unlike the flat path, the
    lazy leaf template emission counts as write: in hive the worker owns its
    leaf's template PUTs (there is no dispatcher-side template), so excluding
    them would hide real write-out cost. ``profile`` is retained and
    forwarded (it still gates dispatcher-side rollup verbosity).

    ``window`` (issue #246, D13/D15) is one dispatch unit's time window:
    ``{"label", "start", "end"}``, bounds half-open in DATASET units
    (converted once at dispatch). It selects the windowed leaf name, injects
    the observation-level ``time_field`` filter (the temporal analog of
    ``aoi_mask`` — see :func:`zagg.config.windowed_cell_config`), stamps the
    window label + the ACTUAL written ISO-UTC time range (also returned as
    ``metadata["time_range"]`` for the root-summary union), and arms the D14
    popcount (``encoding: "full"``). ``None`` is byte-identical to
    pre-windowing behavior.
    """
    from zagg.processing import (
        process_shard,
        write_dataframe_to_zarr,
        write_leaf_to_zarr,
        write_ragged_leaf_to_zarr,
    )
    from zagg.store import open_store

    windowing = None
    time_range_of = None
    if window is not None:
        from zagg.config import windowed_cell_config

        # Inject the window's observation filter into a per-unit config copy
        # (the issue #43 machinery — see windowed_cell_config).
        config, windowing = windowed_cell_config(config, window)
        time_range_of = windowing["time_field"]

    leaf_path = shard_leaf_path(store_root, shard_key, window=window["label"] if window else None)
    box: dict = {}
    _write_elapsed = 0.0

    def _leaf():
        if "store" not in box:
            store = open_store(leaf_path, **store_kwargs)
            # overwrite=True: any existing prefix here is either debris from a
            # torn run (D4) or a prior committed write being redone — both are
            # replaced wholesale; per-leaf state never blocks a retry. Since
            # issue #341 the template DELETES the leaf prefix up front, so the
            # wholesale claim is literal: retired members of a narrowed schema
            # (and the prior attempt's coverage sidecar) are gone before the
            # new template lands, and no enumeration ever walks stale/orphan
            # member dirs (the pre-#341 walk warned on the sidecar and could
            # die on an orphan array dir).
            #
            # This DOES widen the redundant-duplicate-writer window (fold
            # review), and the change is deliberate. ``dispatch._LAMBDA_RETRYABLE``
            # classifies off the exception string of the ``Invoke`` call itself,
            # and for ``InvocationType=Event`` a request Lambda accepted whose
            # HTTP response timed out is indistinguishable from one it never
            # got — so a retry can produce a second live worker for one shard.
            # Before: A wrote + stamped, B templated over the top, and if B died
            # the leaf still held A's objects under A's stamp (stale-but-complete,
            # certified). After: B's clear removes A's committed leaf first, so a
            # B that dies mid-write leaves the leaf EMPTY and unstamped. Nothing
            # is silently corrupt — the stamp is written last, so the leaf reads
            # as debris and is re-dispatchable (test_leaf_clear_under_a_live_
            # writer_leaves_debris_not_corruption) — but a redundant retry can now
            # destroy a leaf that had already succeeded, which write-over could
            # not. Refusing the clear on a valid stamp is the lever if that
            # trade stops being acceptable; it is not taken here because D4 makes
            # "replaced wholesale" the contract and a stamp must never block a
            # retry.
            grid.emit_shard_template(store, overwrite=True)
            box["store"] = store
        return box["store"]

    # Sharded leaf output (issue #236): the sharded leaf template bundles each
    # dense array's K inner chunks into ONE ShardingCodec object, so the
    # per-chunk streaming write below would read-modify-write that object K
    # times (the same failure the flat sharded path warns about in
    # ``runner._process_and_write``). Mirror the flat switch: accumulate the K
    # carriers via ``chunk_results`` and write the whole leaf once after the
    # stream (``write_leaf_to_zarr`` — dense + ragged, one object each). The
    # re-added O(shard) dense term is ~1.3 MB at production geometry
    # (parent 11 / child 19: 65,536 cells x ~20 B across the dense arrays) —
    # trivial next to the ragged accumulation below, which this path folds
    # into the same single-write pass.
    sharded = getattr(grid, "sharded", False)
    chunk_results: list | None = [] if sharded else None

    # Ragged fields accumulate across the streamed chunks (leaf-LOCAL blocks)
    # and are written ONCE after the stream (issue #209): the leaf's ragged
    # vlen array is a single ShardingCodec object spanning the shard, so a
    # per-chunk write here would read-modify-write that object K times.
    # Memory bound (review, PR #211): this re-adds an O(shard-payload) term to
    # the otherwise O(chunk) streaming path (issue #91) — at the o8 t-digest
    # scale this fix exists to unlock (sparse NEON o8: 17.6 M centroids × 8 B
    # ≈ 141 MB of held payload; ~200 MB peak through the single write, once
    # per-cell ``bytes``-object overhead and the assembled ~60 MB shard object
    # are counted). Accepted deliberately: workers run 4 GB (issue #193), the
    # dense side keeps its O(chunk) stream-and-free bound, and the
    # accumulation is what deletes the ~K×7-object PUT storm that was ~1/3 of
    # shard wall at CONUS scale (issue #209). If 88S-scale shards or the #148
    # streaming budget ever say otherwise, the escape valve is spilling the
    # ragged field back to per-inner-chunk writes against a regular-chunked
    # vlen array (the unsharded flat layout) — named here, not built.
    ragged_chunks: list = []

    def _write_chunk(block_index, carrier, ragged):
        nonlocal _write_elapsed
        _t0 = time.time()
        store = _leaf()
        local = leaf_block_index(grid, block_index, shard_key)
        write_dataframe_to_zarr(carrier, store, grid=grid, chunk_idx=local)
        if ragged:
            ragged_chunks.append((local, ragged))
        _write_elapsed += time.time() - _t0

    # Occupied-cell sink (issue #200): the worker already holds the shard's
    # populated cell words; collect them here to derive the stamp's coverage.
    occupied: list = []
    _df_out, metadata = process_shard(
        grid,
        int(shard_key),
        granule_urls,
        s3_credentials=s3_creds,
        config=config,
        driver=driver,
        handoff=handoff,
        aoi_payload=aoi_payload,
        chunk_results=chunk_results,
        write_chunk=None if sharded else _write_chunk,
        occupied_out=occupied,
        time_range_of=time_range_of,
        profile=profile,
    )
    # Windowed stamp truth (D15): convert the worker's dataset-unit extent to
    # ISO-8601 UTC once, here — the same strings feed the stamp below and the
    # dispatcher's root-summary union (via the returned metadata).
    time_range = None
    if window is not None and metadata.get("time_range") is not None:
        from zagg.windows import iso_time_range

        time_range = iso_time_range(metadata["time_range"], windowing)
        metadata["time_range"] = time_range
    # Sharded leaf: ONE whole-leaf write per array (dense + ragged together,
    # issue #236), after the stream. The leaf template is emitted here (still
    # lazily, via ``_leaf``), so a shard that produced no chunks never creates
    # the ``.zarr/`` prefix — same contract as the streaming path's first
    # ``_write_chunk``.
    if sharded and chunk_results and not metadata.get("error"):
        _t0 = time.time()
        write_leaf_to_zarr(chunk_results, _leaf(), grid=grid, shard_key=int(shard_key))
        _write_elapsed += time.time() - _t0
    # Stamp ONLY a fully-written leaf: an errored shard (or one that streamed
    # no chunks) stays unstamped — debris by definition (D4). The stamp is the
    # last write, so its presence certifies everything before it landed — the
    # box payload rides it (zero extra requests), the exact-occupancy bitmap
    # sidecar is PUT just before it (issue #200 phase 2), and both inherit
    # its debris semantics: a torn worker's coverage never becomes visible.
    # The leaf write order is pinned: dense (streamed, or one object each when
    # sharded) -> ragged (one object, issue #209) -> coverage sidecar -> stamp.
    if "store" in box and not metadata.get("error"):
        _t0 = time.time()
        if not sharded:
            write_ragged_leaf_to_zarr(ragged_chunks, box["store"], grid=grid)
        words = np.concatenate(occupied) if occupied else None
        if words is not None and words.size == 0:
            words = None
        bitmap = None
        # D14 popcount (issue #246): a fully-occupied subtree stamps
        # ``encoding: "full"``, no sidecar. Gated on windowing (/2 stores
        # only) so schedule-none output stays object-for-object identical to
        # pre-#246 runs (the mortie spec files "full" under /2).
        depth = int(grid.child_order) - int(grid.parent_order)
        full = window is not None and words is not None and np.unique(words).size == 4**depth
        # Depth 0 (child_order == parent_order, a legal one-cell-per-shard
        # config) skips the sidecar: a 1-bit bitmap says nothing the stamp
        # itself doesn't, and encode would raise AFTER the chunk writes,
        # leaving the shard permanently unstampable debris (review finding,
        # PR #208 round 2). The envelope simply omits the pointer — box only.
        if words is not None and not full and depth > 0:
            bitmap = encode_coverage_bitmap(shard_key, words, grid.child_order)
            write_coverage_sidecar(leaf_path, bitmap, **store_kwargs)
        stamp_commit(
            box["store"],
            cells_with_data=metadata.get("cells_with_data", 0),
            granule_count=metadata.get("granule_count", 0),
            coverage=build_coverage(shard_key, words, grid.child_order, bitmap=bitmap, full=full),
            window=window["label"] if window else None,
            time_range=time_range,
        )
        _write_elapsed += time.time() - _t0
    # Write-phase split (issue #249): read/index/aggregate come from
    # ``process_shard``; ``write`` is the leaf write-out above (template +
    # dense chunks + ragged + coverage sidecar + stamp). Same gate as the flat
    # Lambda handler's issue #100 write bracket: only a clean, actually-written
    # shard carries it, so a time-to-failure never lands as a write duration
    # and a no-data shard (no leaf) stays write-less.
    if not metadata.get("error") and "phase_timings" in metadata and "store" in box:
        metadata["phase_timings"]["write"] = _write_elapsed
    return metadata


__all__ = [
    "COMMIT_ATTR",
    "COVERAGE_BOX_SLOTS",
    "COVERAGE_SIDECAR",
    "COVERAGE_SPEC",
    "HIVE_SPEC",
    "HIVE_SPEC_V2",
    "MANIFEST_NAME",
    "ROOT_COVERAGE_NAME",
    "build_coverage",
    "build_manifest",
    "build_root_coverage",
    "check_node_invariant",
    "decode_coverage_bitmap",
    "encode_coverage_bitmap",
    "ensure_manifest",
    "leaf_block_index",
    "process_and_write_hive",
    "read_commit",
    "read_coverage",
    "read_coverage_bitmap",
    "read_manifest",
    "read_root_coverage",
    "root_coverage_words",
    "shard_leaf_path",
    "stamp_commit",
    "write_coverage_sidecar",
    "write_root_coverage",
]
