"""Notebook-first dispatch facade: per-shard futures over Lambda (issue #326).

:class:`Run` owns the composition that ``.github/scripts/run_benchmark.py``
and the dispatch scratch scripts do imperatively today — config-load →
shardmap → dispatch → harvest — as library API::

    from zagg.client import Run

    run = Run.from_config("atl03_tdigest_healpix_o9_hive.yaml",
                          shardmap="shardmap.json", store="s3://bucket/out.zarr")
    handle = run.dispatch()                    # returns after the setup handshake
    for fut in handle.progress():              # tqdm over as_completed
        r = fut.result()                       # worker envelope / raises ShardError
    handle.status()                            # {"pending", "ok", "failed"}

Draining the harvest iterator to exhaustion (``as_completed`` / ``progress`` /
``results`` / a clean ``raise_first``) also joins the post-run tail — the
finalize backstop and the coverage/stats/sweep rollup invokes — so the loop
above IS the whole run and :meth:`RunHandle.wait` is only needed for explicit
control (a timeout, or a handle harvested some other way). Abandoning a handle
without draining it is the one flow that truncates the tail: the finisher is a
daemon thread, so an undrained handle cannot wedge interpreter exit (review
finding, PR #333).

**v1 transport** (ratified on issue #265): a ``ThreadPoolExecutor`` over the
existing synchronous ``RequestResponse`` invokes — each shard is one
:func:`zagg.runner._invoke_lambda_cell` call and its pool future is the
shard's :class:`concurrent.futures.Future`, wrapping exactly what the worker
returns today (timings, error payloads). Zero worker-side change; the cost is
one held connection per in-flight shard and the client staying alive. The v2
transport (``Event`` invoke + status-object resolver, issue #327) swaps the
resolver under this same public surface.

The dispatcher-never-writes invariant (D8) is untouched: every store write —
template setup, shard output, finalize, coverage/stats/sweep rollups — rides a
worker invoke; this module only holds futures over those invokes.

Scope (v1): the spatial point path on the ``lambda`` backend. Temporal,
raster, and windowed configs are refused with a pointer to
:func:`zagg.runner.agg` / :func:`zagg.notebook.run`, which already run them.

tqdm is an *optional* import (``analysis`` extra, espg-ratified on issue
#265): only :meth:`RunHandle.progress` touches it, so importing this module —
and everything else in zagg — works without it.
"""

from __future__ import annotations

import logging
import threading
import uuid
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from concurrent.futures import wait as futures_wait
from dataclasses import asdict
from typing import Any, Iterator

from zagg.config import (
    PipelineConfig,
    get_child_order,
    get_consolidate_metadata,
    get_coverage_moc,
    get_driver,
    get_handoff,
    get_output_endpoint_url,
    get_output_region,
    get_parent_order,
    get_pipeline_type,
    get_store_layout,
    get_store_path,
    get_sweep,
    get_windowing,
    load_config,
    load_config_from_dict,
    validate_config,
)
from zagg.dispatch import LAMBDA_ARCH, max_cost_usd
from zagg.grids import from_config as grid_from_config
from zagg.grids.morton import morton_word
from zagg.hive import effective_store_root
from zagg.notebook import _BENIGN_ERRORS

logger = logging.getLogger(__name__)

#: Fan-out width when ``dispatch(max_workers=...)`` is not given. Deliberately
#: modest: the v1 sync transport holds one connection per in-flight shard and
#: this path runs no account-concurrency probe (unlike ``agg``'s preflight,
#: which clamps against account limits before sizing the pool) — fine at
#: notebook/NEON scale; pass ``max_workers`` explicitly for bigger fan-outs.
_DEFAULT_MAX_WORKERS = 64


class ShardError(RuntimeError):
    """A shard's worker invoke failed; raised from that shard's ``future.result()``.

    Carries the full per-shard result dict (``payload``) the dispatch loop
    accumulates on the ``agg`` path — ``{shard_key, status_code, body,
    wall_time, lambda_duration, error, retries, ...}`` — so timings and the
    worker's error body survive into the exception.
    """

    def __init__(self, message: str, *, shard_key: int, label: str, payload: dict):
        super().__init__(message)
        self.shard_key = shard_key
        self.label = label
        self.payload = payload


class RunHandle:
    """Live handle over one dispatched fan-out: per-shard futures + rollup.

    ``futures`` maps the shardmap's shard key (int) to that shard's
    :class:`~concurrent.futures.Future`. A future resolves to the worker's
    result dict; a failed shard raises :class:`ShardError` from ``.result()``.
    Benign no-work outcomes (``"No granules found"`` / ``"No data after
    filtering"``) resolve normally and count as ``ok`` — matching how the
    ``agg`` path counts them (neither data nor error).

    A background finisher thread waits for every shard, then runs the same
    worker-invoke tail as ``agg``'s lambda path: the finalize backstop, plus
    the fail-open coverage/stats/sweep rollup invokes (D8: all worker-side).
    Draining any of the harvest iterators (:meth:`as_completed`,
    :meth:`progress`, :meth:`results`, a clean :meth:`raise_first`) joins that
    finisher before returning and surfaces a finalize failure, so a completed
    harvest loop means a completed run; :meth:`wait` is the explicit form (and
    the only one taking a timeout). Only a handle nobody drains truncates the
    tail — the finisher is a daemon so that case cannot hang exit.
    """

    def __init__(self, futures: dict[int, Future], *, store_path: str):
        self.futures = futures
        self.store_path = store_path
        self._finisher: threading.Thread | None = None
        self._finalize_error: BaseException | None = None
        self._tail_error: BaseException | None = None

    def __len__(self) -> int:
        return len(self.futures)

    def __repr__(self) -> str:
        s = self.status()
        return (
            f"RunHandle({len(self)} shards: {s['pending']} pending, "
            f"{s['ok']} ok, {s['failed']} failed)"
        )

    def as_completed(self, timeout: float | None = None) -> Iterator[Future]:
        """Yield each shard future as it completes (``concurrent.futures`` order).

        Exhausting this iterator joins the post-run tail (see :meth:`wait`) and
        re-raises a finalize failure, so the documented harvest loop finishes a
        complete run without calling :meth:`wait` (review finding, PR #333).
        Breaking out early skips that join — call :meth:`wait` if you do.
        """
        yield from as_completed(self.futures.values(), timeout=timeout)
        self._join_tail()

    def status(self) -> dict[str, int]:
        """Non-blocking snapshot: ``{"pending": n, "ok": n, "failed": n}``."""
        counts = {"pending": 0, "ok": 0, "failed": 0}
        for fut in self.futures.values():
            if not fut.done():
                counts["pending"] += 1
            elif fut.exception() is not None:
                counts["failed"] += 1
            else:
                counts["ok"] += 1
        return counts

    def results(self, *, return_exceptions: bool = False) -> dict[int, Any]:
        """Block until every shard completes; return ``{shard_key: result}``.

        With ``return_exceptions=False`` (default) the first failed shard's
        :class:`ShardError` propagates (key order); with ``True`` failures
        appear as the exception object in the mapping instead — the
        ``asyncio.gather`` convention. Either way the post-run tail is joined
        first (see :meth:`as_completed`), so a shard failure does not truncate
        the rollups.
        """
        out: dict[int, Any] = {}
        first_exc: BaseException | None = None
        for key, fut in self.futures.items():
            exc = fut.exception()
            if exc is None:
                out[key] = fut.result()
            else:
                out[key] = exc
                if first_exc is None:
                    first_exc = exc
        self._join_tail()
        if first_exc is not None and not return_exceptions:
            raise first_exc
        return out

    def raise_first(self) -> None:
        """Block until the first failure (raising it) or all shards complete.

        A clean run drains :meth:`as_completed`, so it joins the post-run tail;
        raising on a failure leaves the tail in flight (:meth:`wait` joins it).
        """
        for fut in self.as_completed():
            exc = fut.exception()
            if exc is not None:
                raise exc

    def progress(self, **tqdm_kwargs) -> Iterator[Future]:
        """:meth:`as_completed` under a ``tqdm.auto`` bar (one tick per shard).

        tqdm is optional (the ``analysis`` extra, espg-ratified on issue
        #265); without it this raises ``ImportError`` with the install hint —
        :meth:`as_completed` is the bar-free equivalent.
        """
        try:
            from tqdm.auto import tqdm
        except ImportError as e:
            raise ImportError(
                "RunHandle.progress() needs tqdm (`pip install 'zagg[analysis]'`, "
                "issue #326); handle.as_completed() is the bar-free equivalent"
            ) from e
        kwargs: dict[str, Any] = {"total": len(self), "desc": "shards", "unit": "shard"}
        kwargs.update(tqdm_kwargs)
        return tqdm(self.as_completed(), **kwargs)

    def wait(self, timeout: float | None = None) -> None:
        """Block until every shard AND the post-run tail finish.

        Re-raises a failed finalize backstop (the manifest is required
        reader-facing schema, D6); the coverage/stats/sweep rollups are
        fail-open by design and never raise here, but an exception in the
        tail's own plumbing does surface (it would otherwise vanish into the
        finisher thread). Raises ``TimeoutError`` when the tail is still
        running at ``timeout``.

        Draining a harvest iterator already joins the tail, so ``wait`` is for
        explicit control.
        """
        if self._finisher is not None:
            self._finisher.join(timeout)
            if self._finisher.is_alive():
                raise TimeoutError(f"run tail still in flight after {timeout}s")
        self._raise_tail_error()

    def _join_tail(self) -> None:
        """Join the post-run finisher, then re-raise what it recorded.

        Idempotent (joining a finished thread is a no-op), so draining a
        harvest iterator and then calling :meth:`wait` is safe.
        """
        if self._finisher is not None:
            self._finisher.join()
        self._raise_tail_error()

    def _raise_tail_error(self) -> None:
        # Finalize first: it is the D6 manifest backstop, a strictly more
        # load-bearing failure than a crashed fail-open rollup leg.
        if self._finalize_error is not None:
            raise self._finalize_error
        if self._tail_error is not None:
            raise self._tail_error


class Run:
    """A configured-but-not-dispatched zagg Lambda run (issue #326).

    Build one with :meth:`from_config`; :meth:`dispatch` fans the shards out
    and returns a :class:`RunHandle` of per-shard futures. One ``Run`` may be
    dispatched more than once (each dispatch is an independent fan-out with
    its own run id and pool).
    """

    def __init__(
        self,
        config: PipelineConfig,
        catalog_data: dict,
        *,
        store: str,
        function_name: str,
        region: str,
        lambda_client=None,
        driver: str,
        handoff: str,
        profile: bool = False,
        max_retries: int = 3,
        overwrite: bool = False,
        output_credentials: dict | None = None,
        output_endpoint_url: str | None = None,
        source_credentials: dict | None = None,
    ):
        from zagg import runner

        self.config = config
        self.catalog_data = catalog_data
        self.store = store
        self.function_name = function_name
        self.region = region
        self.driver = driver
        self.handoff = handoff
        self.profile = profile
        self.max_retries = max_retries
        self.overwrite = overwrite
        self._lambda_client = lambda_client
        self._output_credentials = output_credentials
        self._output_endpoint_url = output_endpoint_url
        self._source_credentials = source_credentials

        grid_type = (config.output.get("grid") or {}).get("type", "healpix")
        self._grid_type = grid_type
        self._parent_order = get_parent_order(config) if grid_type == "healpix" else None
        self._child_order = get_child_order(config) if grid_type == "healpix" else None
        self.grid = grid_from_config(config)
        runner._check_signature(self.grid, catalog_data)

    def __repr__(self) -> str:
        return (
            f"Run({len(self.catalog_data['shard_keys'])} shards -> {self.store!r}, "
            f"function={self.function_name!r}, region={self.region!r})"
        )

    def __len__(self) -> int:
        return len(self.catalog_data["shard_keys"])

    @classmethod
    def from_config(
        cls,
        config: PipelineConfig | dict | str,
        *,
        shardmap: dict | str | None = None,
        store: str | None = None,
        function_name: str | None = None,
        region: str = "us-west-2",
        lambda_client=None,
        driver: str | None = None,
        handoff: str | None = None,
        profile: bool = False,
        max_retries: int = 3,
        overwrite: bool = False,
        output_credentials: dict | None = None,
        output_endpoint_url: str | None = None,
        source_credentials: dict | None = None,
    ) -> "Run":
        """Build a :class:`Run` from a config plus a shard map.

        Parameters
        ----------
        config : PipelineConfig, dict, or str
            A loaded :class:`~zagg.config.PipelineConfig`, a plain config
            dict, or a path to a YAML config file. Dicts and paths are
            validated on load.
        shardmap : dict or str, optional
            A loaded ShardMap manifest dict or a path to its JSON. Falls back
            to the config's ``catalog:`` key. The map's grid signature must
            match the config's grid (same guard as ``agg``).
        store : str, optional
            Output store; falls back to ``output.store:``. Must be ``s3://``
            (this facade is Lambda-only; the local backend stays on ``agg``).
        function_name : str, optional
            Lambda function name; default resolves ``ZAGG_LAMBDA_FUNCTION_NAME``
            plus the config ``worker:`` variant suffix, exactly like ``agg``.
        region : str
            AWS region (default ``us-west-2``; a config ``output.region``
            overrides the default, mirroring ``agg``).
        lambda_client : optional
            An injected boto3 Lambda client (testability / custom botocore
            config). Default ``None`` builds one per dispatch, sized to the
            fan-out (``read_timeout`` above the 900 s function ceiling). With
            an injected client no STS caller-identity probe runs, so worker
            stats records carry a null ``invoked_by`` (fail-open, issue #297).
        driver, handoff, profile, max_retries, overwrite,
        output_credentials, output_endpoint_url
            Same semantics as the matching :func:`zagg.runner.agg` kwargs.
        source_credentials : dict, optional
            Explicit source-read S3 credentials (camelCase ``accessKeyId`` /
            ``secretAccessKey`` / ``sessionToken``). Default ``None`` resolves
            the config's credentials provider (NSIDC by default) at dispatch
            time, exactly like ``agg``.
        """
        from zagg import runner

        if isinstance(config, str):
            config = load_config(config)
        elif isinstance(config, dict):
            config = load_config_from_dict(config)
            validate_config(config)

        # v1 scope gate: the spatial point path only. The other pipelines
        # already run through agg()/zagg.notebook.run; refusing here beats a
        # wrong fan-out (windowed units are (shard, window) pairs, not shards).
        kind = get_pipeline_type(config)
        if kind != "spatial":
            raise NotImplementedError(
                f"zagg.client v1 covers the spatial point path (got pipeline "
                f"type {kind!r}); temporal runs go through zagg.runner.agg(events=...)"
            )
        if (config.data_source or {}).get("reader") == "raster":
            raise NotImplementedError(
                "zagg.client v1 covers the spatial point path (got reader: "
                "raster); raster runs go through zagg.runner.agg / zagg.notebook.run"
            )
        if get_windowing(config) is not None:
            raise NotImplementedError(
                "zagg.client v1 dispatches one future per shard; windowed "
                "configs fan out (shard, window) units — use zagg.runner.agg / "
                "zagg.notebook.run until the v2 transport (issue #327)"
            )

        if isinstance(shardmap, dict):
            catalog_data = shardmap
            if not all(k in catalog_data for k in ("shard_keys", "granules", "grid_signature")):
                raise ValueError(
                    "shardmap dict is not a Phase-5 ShardMap (needs shard_keys "
                    "+ granules + grid_signature)"
                )
        else:
            catalog_path = shardmap or config.catalog
            if not catalog_path:
                raise ValueError("No shardmap specified (pass shardmap= or set catalog: in config)")
            catalog_data = runner._load_catalog(catalog_path)

        store_path = store or get_store_path(config)
        if not store_path:
            raise ValueError("No store path specified (pass store= or set output.store: in config)")
        store_path = effective_store_root(store_path, config)
        if not store_path.startswith("s3://"):
            raise ValueError(f"Lambda dispatch requires an s3:// store path, got: {store_path}")

        config_region = get_output_region(config)
        if config_region and region == "us-west-2":
            region = config_region

        return cls(
            config,
            catalog_data,
            store=store_path,
            function_name=runner._resolve_function_name(config, function_name),
            region=region,
            lambda_client=lambda_client,
            driver=driver or get_driver(config),
            handoff=handoff if handoff is not None else get_handoff(config),
            profile=profile,
            max_retries=max_retries,
            overwrite=overwrite,
            output_credentials=output_credentials,
            # Non-secret endpoint: runtime kwarg > config (mirrors agg).
            output_endpoint_url=output_endpoint_url or get_output_endpoint_url(config),
            source_credentials=source_credentials,
        )

    # -- shard selection ----------------------------------------------------

    def _resolve_key(self, key) -> int:
        """One requested shard key -> the shardmap's raw int key.

        Ints pass through as raw shardmap keys (packed morton words for
        HEALPix); strings are the external form — the decimal morton string
        for HEALPix (issue #199), stringified ints otherwise.
        """
        if isinstance(key, str):
            if self._grid_type == "healpix":
                return morton_word(key)
            return int(key)
        return int(key)

    def _select(self, shard_keys) -> list[tuple]:
        """(shard_key, records) pairs to dispatch, biggest-work-first ordered."""
        from zagg import runner

        pairs = runner._select_cells(self.catalog_data)
        if shard_keys is not None:
            wanted = {self._resolve_key(k) for k in shard_keys}
            missing = wanted - {int(k) for k, _ in pairs}
            if missing:
                raise ValueError(f"shard key(s) not in shardmap: {sorted(missing)}")
            pairs = [(k, recs) for k, recs in pairs if int(k) in wanted]
        return runner._lambda_dispatch_order(pairs)

    # -- dispatch -----------------------------------------------------------

    def dispatch(self, shard_keys=None, max_workers: int | None = None) -> RunHandle:
        """Fan the shards out; return a :class:`RunHandle` of per-shard futures.

        Returns as soon as the setup handshake completes (worker-side template
        / manifest write — one short synchronous invoke; hive additionally
        fail-fast-pings first): the fan-out itself runs on a background pool
        and each shard's outcome arrives through its future.

        Parameters
        ----------
        shard_keys : iterable, optional
            Subset of shards to dispatch — raw int shardmap keys or external
            string labels (decimal morton for HEALPix). Default all shards.
        max_workers : int, optional
            Pool width, clamped to the shard count. Default
            :data:`_DEFAULT_MAX_WORKERS`; no account-concurrency probe runs
            on this path (see the constant's note).
        """
        from zagg import runner

        cells = self._select(shard_keys)
        if not cells:
            raise ValueError("no shards selected")
        n = len(cells)
        workers = min(max_workers or _DEFAULT_MAX_WORKERS, n)

        client = self._lambda_client
        invoked_by = None
        if client is None:
            import boto3
            from botocore.config import Config

            session = boto3.Session()
            # read_timeout must exceed the 900 s function ceiling (same
            # rationale as agg's preflight-built client, issue #148); the
            # connection pool tracks the fan-out width.
            client = session.client(
                "lambda",
                region_name=self.region,
                config=Config(
                    read_timeout=960,
                    connect_timeout=10,
                    retries={"max_attempts": 0},
                    max_pool_connections=workers,
                ),
            )
            invoked_by = runner._resolve_invoked_by(session, self.region)

        # Pre-invoke cost ceiling (issue #298): displayed, never gated — the
        # notebook path never blocks on cost (ratified on issue #298).
        memory_gb = runner._worker_memory_gb(self.config)
        ceiling = max_cost_usd(n, memory_gb, timeout_s=runner._DEFAULT_FUNCTION_TIMEOUT_S)
        logger.info(
            f"Max cost ceiling: ~${ceiling:.2f} ({n} units x {memory_gb:g} GB x "
            f"{runner._DEFAULT_FUNCTION_TIMEOUT_S}s, {LAMBDA_ARCH})"
        )

        s3_creds = self._source_credentials or runner._resolve_source_credentials(self.config)
        config_dict = asdict(self.config)
        output_creds_event = runner._build_output_creds_event(
            self._output_credentials, self._output_endpoint_url, self.region
        )
        run_id = uuid.uuid4().hex

        # Setup handshake, synchronous and load-bearing (worker-side writes
        # only — D8). Hive: fail-fast ping + fire-and-forget manifest write;
        # finalize below is the idempotent backstop (issue #252 hybrid). Flat:
        # the template write itself.
        dataset = None
        if get_store_layout(self.config) == "hive":
            md = self.catalog_data.get("metadata") or {}
            dataset = {"short_name": md.get("short_name"), "version": md.get("version")}
            runner._invoke_lambda_ping(
                client,
                self.function_name,
                self.store,
                config_dict=config_dict,
                dataset=dataset,
                parent_order=self._parent_order,
                overwrite=self.overwrite,
                output_creds_event=output_creds_event,
            )
            runner._invoke_lambda_setup_async(
                client,
                self.function_name,
                self.store,
                config_dict=config_dict,
                dataset=dataset,
                parent_order=self._parent_order,
                overwrite=self.overwrite,
                output_creds_event=output_creds_event,
            )
        else:
            runner._invoke_lambda_setup(
                client,
                self.function_name,
                self.store,
                parent_order=self._parent_order,
                child_order=self._child_order,
                overwrite=self.overwrite,
                config_dict=config_dict,
                output_creds_event=output_creds_event,
            )

        aoi_by_shard = runner._aoi_payload_map(self.catalog_data)
        pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="zagg-client")
        futures: dict[int, Future] = {}
        for key, records in cells:
            futures[int(key)] = pool.submit(
                self._shard_work,
                client,
                int(key),
                records,
                config_dict=config_dict,
                s3_creds=s3_creds,
                output_creds_event=output_creds_event,
                aoi_payload=aoi_by_shard.get(int(key)),
                invoked_by=invoked_by,
                run_id=run_id,
                max_workers=workers,
            )

        handle = RunHandle(futures, store_path=self.store)
        finisher = threading.Thread(
            target=self._post_run,
            args=(handle, client, pool, config_dict, dataset, output_creds_event, run_id),
            name="zagg-client-finish",
            daemon=True,
        )
        handle._finisher = finisher
        finisher.start()
        return handle

    def _shard_work(
        self,
        client,
        shard_key: int,
        records: list,
        *,
        config_dict: dict,
        s3_creds: dict,
        output_creds_event: dict | None,
        aoi_payload,
        invoked_by: dict | None,
        run_id: str,
        max_workers: int,
    ) -> dict:
        """One synchronous shard invoke; the pool future over this IS the API.

        Mirrors ``_run_lambda``'s ``_cell_work`` payload construction (labels,
        granule-worker clamp, leaf sub-map) minus the async result channel —
        v1 is the legacy sync ``RequestResponse`` path by design.
        """
        from zagg import runner

        label = runner._safe_label(self.grid, shard_key)
        granule_urls = runner._resolve_urls(records, self.driver)
        ds = runner._clamped_data_source(dict(self.config.data_source), len(granule_urls))
        cell_config = {**config_dict, "data_source": ds} if ds is not None else config_dict
        submap = {
            "grid_signature": self.catalog_data["grid_signature"],
            "metadata": self.catalog_data.get("metadata"),
            "granules": records,
        }
        result = runner._invoke_lambda_cell(
            client,
            self.grid.block_index(shard_key),
            shard_key,
            self._parent_order,
            self._child_order,
            granule_urls,
            self.store,
            s3_creds,
            function_name=self.function_name,
            config_dict=cell_config,
            output_creds_event=output_creds_event,
            max_retries=self.max_retries,
            max_workers=max_workers,
            handoff=self.handoff,
            profile=self.profile,
            aoi_payload=aoi_payload,
            label=label,
            invoked_by=invoked_by,
            run_id=run_id,
            submap=submap,
        )
        error = result.get("error")
        if result.get("status_code") == 200 and not error:
            return result
        if error in _BENIGN_ERRORS:
            # "Nothing to do" is a normal outcome, not a failure — matching
            # _run_lambda's error counting (neither with_data nor error).
            return result
        detail = error or f"status {result.get('status_code')}"
        raise ShardError(
            f"shard {label}: {detail}",
            shard_key=shard_key,
            label=label,
            payload=result,
        )

    def _post_run(
        self,
        handle: RunHandle,
        client,
        pool: ThreadPoolExecutor,
        config_dict: dict,
        dataset: dict | None,
        output_creds_event: dict | None,
        run_id: str,
    ) -> None:
        """Finisher-thread entry point: run the tail, never lose its failure.

        Anything the tail raises outside its own fail-open guards would
        otherwise die in ``threading.excepthook`` while ``wait()`` reported a
        clean run, so it is recorded on the handle as ``_tail_error`` (kept
        distinct from the D6-load-bearing ``_finalize_error``) and the pool is
        shut down either way (review finding, PR #333).
        """
        try:
            self._run_tail(handle, client, config_dict, dataset, output_creds_event, run_id)
        except BaseException as e:
            handle._tail_error = e
            logger.warning(f"post-run tail failed (surfaced via handle.wait()): {e}")
        finally:
            pool.shutdown(wait=False)

    def _run_tail(
        self,
        handle: RunHandle,
        client,
        config_dict: dict,
        dataset: dict | None,
        output_creds_event: dict | None,
        run_id: str,
    ) -> None:
        """After every shard settles: the same worker-invoke tail as ``agg``.

        Finalize backstop (hive always — the idempotent manifest self-heal;
        flat only under ``consolidate_metadata``), then the fail-open rollups:
        root ``coverage.moc``, the run-stats record, and the sweep trigger —
        each a fire-and-forget worker invoke (D8), each swallowed on failure
        exactly like ``_run_lambda``'s tail. A finalize failure is recorded on
        the handle and re-raised from :meth:`RunHandle.wait`.

        Every shard contributes exactly one result dict, so every shard gets a
        run-stats row: a worker envelope, a :class:`ShardError` payload, or —
        for a transport-level exception that never produced an envelope (the
        payload-cap ``ValueError``, an fd-exhaustion re-raise, a botocore error
        escaping the retry loop) — a synthesized failure dict, which
        ``_lambda_result_rows`` turns into a ``failure_record`` row rather than
        dropping the shard from telemetry (review finding, PR #333).
        """
        from zagg import runner

        futures = list(handle.futures.values())
        futures_wait(futures)
        results = []
        for key, fut in handle.futures.items():
            exc = fut.exception()
            if exc is None:
                results.append(fut.result())
            elif isinstance(exc, ShardError):
                results.append(exc.payload)
            else:
                results.append(
                    {
                        "shard_key": key,
                        "status_code": None,
                        "body": {},
                        "error": f"{type(exc).__name__}: {exc}",
                        "retries": None,
                    }
                )
        layout = get_store_layout(self.config)
        try:
            if layout == "hive":
                runner._invoke_lambda_finalize(
                    client,
                    self.function_name,
                    self.store,
                    output_creds_event=output_creds_event,
                    config_dict=config_dict,
                    dataset=dataset,
                    parent_order=self._parent_order,
                    overwrite=self.overwrite,
                )
            elif get_consolidate_metadata(self.config):
                runner._invoke_lambda_finalize(
                    client,
                    self.function_name,
                    self.store,
                    output_creds_event=output_creds_event,
                )
        except Exception as e:
            handle._finalize_error = e
            logger.warning(f"finalize invoke failed (surfaced via handle.wait()): {e}")

        ok_results = [r for r in results if r.get("status_code") == 200 and not r.get("error")]
        if layout == "hive" and get_coverage_moc(self.config):
            try:
                from zagg.hive import build_root_coverage
                from zagg.windows import union_time_range

                done = [r["shard_key"] for r in ok_results]
                # hive is HEALPix-only (validated), so parent_order is set here.
                if done and self._parent_order is not None:
                    envelope = build_root_coverage(
                        done,
                        int(self._parent_order),
                        time_range=union_time_range(
                            *(r.get("body", {}).get("time_range") for r in ok_results)
                        ),
                    )
                    runner._invoke_lambda_coverage(
                        client,
                        self.function_name,
                        self.store,
                        envelope,
                        output_creds_event=output_creds_event,
                    )
            except Exception as e:
                logger.warning(f"root coverage.moc dispatch failed (fail-open, D9): {e}")

        # Run-record + sweep, both fail-open (issues #297/#313/#300): sync
        # transport -> no status prefix, so oversized row sets skip inside
        # _dispatch_run_stats exactly like agg's invocation="sync" path.
        stats_rows, stats_inline = runner._lambda_result_rows(results, run_id=run_id)
        runner._dispatch_run_stats(
            client,
            self.function_name,
            self.store,
            stats_rows,
            run_id=run_id,
            result_prefix=None,
            output_creds_event=output_creds_event,
            store_kwargs=runner._output_store_kwargs(output_creds_event, self.region),
            inline_rows=stats_inline,
        )
        if layout == "hive" and get_sweep(self.config):
            try:
                from zagg.sweep import leaves_from_stats_records

                leaves = leaves_from_stats_records(
                    [(r.get("body") or {}).get("stats") for r in ok_results]
                )
                if leaves:
                    runner._invoke_lambda_sweep(
                        client,
                        self.function_name,
                        self.store,
                        leaves,
                        output_creds_event=output_creds_event,
                    )
            except Exception as e:
                logger.warning(f"rollup sweep dispatch failed (fail-open, D9): {e}")
