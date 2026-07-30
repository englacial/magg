"""Tests for the v2 Event transport (issue #327): status objects + poller.

No AWS, no network: Lambda traffic goes through a stub client whose Event
"workers" write status objects into an ``obstore`` MemoryStore — through the
REAL ``build_shard_status`` (the worker half), so the two halves of the
channel cannot drift — and the poller reads them back through the same
obstore functions the production path uses.
"""

import json
import threading
import time
from concurrent.futures import Future

import obstore
import pytest
from obstore.store import MemoryStore

import zagg.client_transport as ct
from zagg.client import Run, ShardError
from zagg.config import default_config

# Mirrors tests/test_client.py: order-6 shardmap over three packed morton words.
_ATL06_SIG = {
    "type": "healpix",
    "indexing_scheme": "nested",
    "parent_order": 6,
    "child_order": 12,
    "layout": "fullsphere",
}
_WORDS = [11828422946311897094, 11828141471335186438, 11827859996358475782]
_LABELS = ["-4211324", "-4211323", "-4211322"]
_CREDS = {"accessKeyId": "AK", "secretAccessKey": "SK", "sessionToken": "TK"}
_STORE = "s3://test-bucket/out.zarr"


def _rec(n):
    return {"id": f"g{n}", "s3": f"s3://bucket/granule{n}.h5", "https": f"https://h/granule{n}.h5"}


@pytest.fixture
def catalog():
    return {
        "metadata": {"short_name": "ATL06", "version": "006"},
        "grid_signature": dict(_ATL06_SIG),
        "shard_keys": list(_WORDS),
        "granules": [[_rec(4), _rec(5), _rec(6)], [_rec(3)], [_rec(1), _rec(2)]],
    }


@pytest.fixture(autouse=True)
def _fast_poll(monkeypatch):
    monkeypatch.setattr(ct, "_POLL_INITIAL_INTERVAL_S", 0.01)
    monkeypatch.setattr(ct, "_POLL_MAX_INTERVAL_S", 0.02)


@pytest.fixture(autouse=True)
def _no_stats_verify(monkeypatch):
    from zagg import runner

    monkeypatch.setattr(runner, "_RUN_STATS_VERIFY_WINDOW_S", 0)


@pytest.fixture
def status_store(monkeypatch):
    """One in-memory status store shared by stub workers and the poller."""
    store = MemoryStore()
    monkeypatch.setattr(ct, "open_status_store", lambda prefix, kwargs: store)
    return store


def _envelope(body: dict, status=200) -> dict:
    return {"statusCode": status, "body": json.dumps(body)}


class EventStubLambdaClient:
    """Stub Lambda whose Event 'workers' write status objects (issue #327).

    A per-unit Event invoke computes the worker's canned response envelope and
    records it as a status object via the REAL worker half
    (``build_shard_status``), exactly as the deployed handler seam does. Mode
    invokes (ping/setup/finalize/...) answer synchronously like the v1 stub.
    """

    def __init__(self, status_store, *, fail=(), benign=(), drop=(), fail_times=None):
        self.status_store = status_store
        self.events: list[tuple[str, str, dict]] = []
        self._lock = threading.Lock()
        self._fail = set(fail)  # deterministic failures (every attempt)
        self._benign = set(benign)
        self._drop = set(drop)  # invoke accepted, no status ever lands
        self._fail_times = dict(fail_times or {})  # key -> failures before success
        self.status_writes = 0

    def invoke(self, **kwargs):
        event = json.loads(kwargs["Payload"])
        with self._lock:
            self.events.append((kwargs["FunctionName"], kwargs["InvocationType"], event))
        if event.get("mode") is not None:
            return {
                "Payload": type(
                    "P", (), {"read": lambda self: b'{"statusCode": 200, "body": "{}"}'}
                )(),
                "FunctionError": None,
            }
        assert kwargs["InvocationType"] == "Event", "v2 cell invokes are fire-and-forget"
        key = event["shard_key"]
        if key in self._drop:
            return {"StatusCode": 202}
        if key in self._fail:
            response = _envelope({"error": "boom", "shard_key": key}, status=500)
        elif self._fail_times.get(key, 0) > 0:
            with self._lock:
                self._fail_times[key] -= 1
            response = _envelope({"error": "boom", "shard_key": key}, status=500)
        elif key in self._benign:
            response = _envelope({"error": "No granules found", "shard_key": key}, status=500)
        else:
            response = _envelope(
                {
                    "shard_key": key,
                    "total_obs": 7,
                    "duration_s": 1.5,
                    "phase_timings": {"read": 1.0},
                    "stats": {"schema_version": 1, "shard_key": key, "run_id": event.get("run_id")},
                }
            )
        self._write_status(event, response)
        return {"StatusCode": 202}

    def _write_status(self, event, response):
        obj_key, obj = ct.build_shard_status(event, response)
        with self._lock:
            self.status_writes += 1
        obstore.put(self.status_store, obj_key, json.dumps(obj).encode())

    def cell_events(self):
        return [(n, t, e) for n, t, e in self.events if e.get("mode") is None]

    def modes(self):
        return [e.get("mode") for _, _, e in self.events]


def _run(catalog, *, client, config=None, **kwargs):
    return Run.from_config(
        config if config is not None else default_config("atl06"),
        shardmap=catalog,
        store=_STORE,
        function_name="process-shard-test",
        lambda_client=client,
        source_credentials=_CREDS,
        **kwargs,
    )


# -- Run.dispatch(transport="event") -------------------------------------------


class TestEventDispatch:
    def test_all_shards_resolve_from_status_objects(self, catalog, status_store):
        stub = EventStubLambdaClient(status_store)
        handle = _run(catalog, client=stub).dispatch(transport="event")
        results = handle.results()
        assert set(results) == set(_WORDS)
        assert all(r["body"]["total_obs"] == 7 for r in results.values())
        assert all(r["status_code"] == 200 for r in results.values())
        assert handle.status() == {"pending": 0, "ok": 3, "failed": 0}
        cells = stub.cell_events()
        assert len(cells) == 3
        assert all(t == "Event" for _, t, _ in cells)
        # v2 needs no per-shard mirror target: statuses are always-on.
        assert all("result_url" not in e for _, _, e in cells)

    def test_event_payloads_match_sync_modulo_invocation_type(self, catalog, status_store):
        from test_client import StubLambdaClient

        sync_stub = StubLambdaClient()
        _run(catalog, client=sync_stub).dispatch(transport="sync").results()
        sync_events = {e["shard_key"]: e for _, _, e in sync_stub.cell_events()}

        stub = EventStubLambdaClient(status_store)
        _run(catalog, client=stub).dispatch(transport="event").results()
        event_events = {e["shard_key"]: e for _, _, e in stub.cell_events()}

        assert set(event_events) == set(sync_events) == set(_WORDS)
        for key in _WORDS:
            mine, theirs = dict(event_events[key]), dict(sync_events[key])
            assert mine.pop("run_id") and theirs.pop("run_id")  # fresh per dispatch
            assert mine == theirs  # byte-identical construction (issue #327)

    def test_benign_no_data_counts_ok(self, catalog, status_store):
        stub = EventStubLambdaClient(status_store, benign={_WORDS[2]})
        handle = _run(catalog, client=stub).dispatch(transport="event")
        assert handle.results()[_WORDS[2]]["error"] == "No granules found"
        assert handle.status() == {"pending": 0, "ok": 3, "failed": 0}

    def test_failed_status_redispatches_then_raises_shard_error(self, catalog, status_store):
        stub = EventStubLambdaClient(status_store, fail={_WORDS[1]})
        handle = _run(catalog, client=stub, max_retries=2).dispatch(transport="event")
        with pytest.raises(ShardError, match="boom") as excinfo:
            handle.futures[_WORDS[1]].result(timeout=10)
        err = excinfo.value
        assert err.shard_key == _WORDS[1]
        assert err.label == _LABELS[1]
        assert err.payload["retries"] == 1  # second (last) attempt
        handle.results(return_exceptions=True)
        # The migrated retry policy actually re-dispatched: two Event invokes.
        attempts = [e for _, _, e in stub.cell_events() if e["shard_key"] == _WORDS[1]]
        assert len(attempts) == 2

    def test_redispatch_recovers_a_transiently_failing_shard(self, catalog, status_store):
        stub = EventStubLambdaClient(status_store, fail_times={_WORDS[0]: 1})
        handle = _run(catalog, client=stub, max_retries=3).dispatch(transport="event")
        result = handle.futures[_WORDS[0]].result(timeout=10)
        assert result["body"]["total_obs"] == 7
        assert result["retries"] == 1  # succeeded on the second attempt
        handle.results()
        assert handle.status() == {"pending": 0, "ok": 3, "failed": 0}

    def test_post_run_tail_matches_v1(self, catalog, status_store):
        stub = EventStubLambdaClient(status_store)
        handle = _run(catalog, client=stub).dispatch(transport="event")
        handle.results()  # drains + joins the tail
        modes = stub.modes()
        first_cell = modes.index(None)
        assert modes[:first_cell] == ["ping", "setup"]
        assert "finalize" in modes and "coverage" in modes and "stats" in modes

    def test_unknown_transport_refused(self, catalog, status_store):
        run = _run(catalog, client=EventStubLambdaClient(status_store))
        with pytest.raises(ValueError, match="transport"):
            run.dispatch(transport="carrier-pigeon")


# -- StatusPoller unit behavior --------------------------------------------------


class TestStatusPoller:
    @staticmethod
    def _poller(store, **kwargs):
        kwargs.setdefault("drop_timeout_s", 1050.0)
        return ct.StatusPoller(lambda: store, **kwargs)

    @staticmethod
    def _put_status(store, shard_key, status="ok", attempt_id=None, **fields):
        obj = {
            "schema_version": 1,
            "status": status,
            "attempt_id": attempt_id or f"aid-{time.time_ns()}",
            "shard": str(shard_key),
            "timings": None,
            "error": fields.pop("error", None),
            "status_code": fields.pop("status_code", 200),
            "body": fields.pop("body", {}),
        }
        obstore.put(store, ct.shard_status_key(shard_key), json.dumps(obj).encode())

    def test_pacing_window_bounds_in_flight_dispatch(self):
        # max_in_flight=1: the second shard's Event must not fire until the
        # first resolves — load-bearing under the fleet's 60 s max event age.
        store = MemoryStore()
        fired: list[int] = []
        poller = self._poller(store, max_in_flight=1)
        f1 = poller.register(1, "1", dispatch=lambda: fired.append(1))
        f2 = poller.register(2, "2", dispatch=lambda: fired.append(2))
        poller.start()
        deadline = time.time() + 5
        while not fired and time.time() < deadline:
            time.sleep(0.005)
        time.sleep(0.05)  # a few idle ticks: the window must still hold
        assert fired == [1]
        self._put_status(store, 1)
        assert f1.result(timeout=5)["status_code"] == 200
        deadline = time.time() + 5
        while len(fired) < 2 and time.time() < deadline:
            time.sleep(0.005)
        assert fired == [1, 2]
        self._put_status(store, 2)
        assert f2.result(timeout=5)["status_code"] == 200
        poller.shutdown(wait=True)

    def test_attempt_id_dedupe_acts_once_per_execution(self):
        # A duplicate poll of the SAME attempt's object must not double-count
        # against the retry budget; only a NEW attempt_id is consumed.
        store = MemoryStore()
        fired: list[float] = []
        poller = self._poller(store, max_retries=3)
        fut = poller.register(1, "1", dispatch=lambda: fired.append(time.time()))
        self._put_status(store, 1, status="failed", attempt_id="a1", error="boom")
        poller.start()
        deadline = time.time() + 5
        while len(fired) < 2 and time.time() < deadline:
            time.sleep(0.005)
        assert len(fired) == 2  # initial dispatch + ONE re-dispatch for a1
        # Ticks keep seeing the stale a1 object: no further consumption.
        time.sleep(0.1)
        assert len(fired) == 2
        # The retry's execution lands with a fresh id and resolves the future.
        self._put_status(store, 1, status="ok", attempt_id="a2")
        assert fut.result(timeout=5)["retries"] == 1
        poller.shutdown(wait=True)

    def test_invoke_fault_burns_an_attempt_and_retries(self):
        store = MemoryStore()
        calls = {"n": 0}

        def flaky():
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("TooManyRequestsException")

        poller = self._poller(store, max_retries=3)
        fut = poller.register(1, "1", dispatch=flaky)
        poller.start()
        deadline = time.time() + 5
        while calls["n"] < 2 and time.time() < deadline:
            time.sleep(0.005)
        assert calls["n"] == 2
        self._put_status(store, 1)
        assert fut.result(timeout=5)["retries"] == 1
        poller.shutdown(wait=True)

    def test_default_failure_resolution_is_a_result_dict(self):
        # agg's mode: no on_failed -> the terminal failure resolves as the
        # v1-shaped result dict (the accumulator records it), never an exception.
        store = MemoryStore()
        poller = self._poller(store, max_retries=1)
        fut = poller.register(7, "7", dispatch=lambda: None)
        self._put_status(store, 7, status="failed", error="boom", status_code=500)
        poller.start()
        result = fut.result(timeout=5)
        assert result["error"] == "boom"
        assert result["status_code"] == 500
        assert result["shard_key"] == 7
        poller.shutdown(wait=True)

    def test_transient_list_fault_does_not_kill_the_poller(self):
        store = MemoryStore()
        calls = {"n": 0}
        real_list = obstore.list

        def flaky_list(s, *a, **k):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("503 slow down")
            return real_list(s, *a, **k)

        poller = self._poller(store)
        fut = poller.register(1, "1", dispatch=lambda: None)
        self._put_status(store, 1)
        try:
            obstore.list = flaky_list
            poller.start()
            assert fut.result(timeout=5)["status_code"] == 200
        finally:
            obstore.list = real_list
        poller.shutdown(wait=True)

    def test_observe_only_entry_never_redispatches(self):
        # attach's mode (issue #327 phase 5): dispatch=None -> a failed status
        # resolves immediately through on_failed, no retry budget to spend.
        store = MemoryStore()
        failures: list = []
        poller = self._poller(
            store,
            max_retries=3,
            on_failed=lambda e, r: (failures.append(r), e.future.set_result(r)),
        )
        fut = poller.register(1, "1", dispatch=None)
        self._put_status(store, 1, status="failed", error="boom")
        poller.start()
        assert fut.result(timeout=5)["error"] == "boom"
        assert len(failures) == 1
        poller.shutdown(wait=True)


# -- keys / prefix ---------------------------------------------------------------


class TestKeys:
    def test_run_status_prefix_is_a_store_sibling(self):
        assert ct.run_status_prefix("s3://b/out.zarr", "abc") == "s3://b/out.zarr.status/run-abc"
        assert ct.run_status_prefix("s3://b/out.zarr/", "abc").endswith("out.zarr.status/run-abc")

    def test_shard_status_key_decimal_and_window(self):
        assert ct.shard_status_key(12345) == "shard-12345.json"
        assert ct.shard_status_key(12345, window="2020") == "shard-12345_2020.json"
        with pytest.raises(ValueError, match="grammar"):
            ct.shard_status_key(1, window="../escape")

    def test_drop_timeout_formula(self):
        # function timeout + 60 s max event age + margin (issue #327 (a')).
        assert ct.drop_timeout_s(900) == 900 + 60 + 90


# -- Future plumbing -------------------------------------------------------------


def test_register_returns_unresolved_future():
    poller = ct.StatusPoller(lambda: MemoryStore(), drop_timeout_s=10.0)
    fut = poller.register(1, "1", dispatch=lambda: None)
    assert isinstance(fut, Future)
    assert not fut.done()
