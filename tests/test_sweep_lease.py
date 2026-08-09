"""The sweep-admission lease (issue #384): control plane, per-store, atomic.

Standing claims: admission is one conditional PUT (``If-None-Match: *``); a
live foreign intent refuses NAMING the runner; an expired heartbeat is
claimable and the claim is verified by read-back; the holder heartbeats;
release deletes only the holder's own intent. No data object is ever locked.
"""

from __future__ import annotations

import json

import pytest

from zagg.sweep_lease import (
    DEFAULT_TTL_S,
    LEASE_NAME,
    SweepRefusedError,
    acquire_lease,
    heartbeat_lease,
    read_lease,
    release_lease,
)


def _root(tmp_path):
    root = tmp_path / "store"
    root.mkdir()
    return str(root)


class TestAcquire:
    def test_acquire_writes_the_intent(self, tmp_path):
        root = _root(tmp_path)
        lease = acquire_lease(root, run_id="A", scope=["-42113"])
        assert lease["run_id"] == "A" and lease["ttl_s"] == DEFAULT_TTL_S
        stored = read_lease(root)
        assert stored["run_id"] == "A" and stored["scope"] == ["-42113"]

    def test_live_foreign_intent_refuses_naming_the_runner(self, tmp_path):
        root = _root(tmp_path)
        acquire_lease(root, run_id="A")
        with pytest.raises(SweepRefusedError, match="'A'"):
            acquire_lease(root, run_id="B")

    def test_reacquire_by_the_holder_is_idempotent(self, tmp_path):
        root = _root(tmp_path)
        acquire_lease(root, run_id="A")
        assert acquire_lease(root, run_id="A")["run_id"] == "A"

    def test_expired_heartbeat_is_claimable(self, tmp_path):
        root = _root(tmp_path)
        lease = acquire_lease(root, run_id="A")
        stale = dict(lease, heartbeat_at="2020-01-01T00:00:00+00:00")
        (tmp_path / "store" / LEASE_NAME).write_text(json.dumps(stale))
        claim = acquire_lease(root, run_id="B")
        assert claim["run_id"] == "B" and claim["claimed_from"] == "A"
        assert read_lease(root)["run_id"] == "B"

    def test_debris_is_claimable(self, tmp_path):
        root = _root(tmp_path)
        (tmp_path / "store" / LEASE_NAME).write_text("not json")
        assert acquire_lease(root, run_id="B")["run_id"] == "B"


class TestHeartbeatAndRelease:
    def test_heartbeat_moves(self, tmp_path):
        root = _root(tmp_path)
        lease = acquire_lease(root, run_id="A")
        stale = dict(lease, heartbeat_at="2020-01-01T00:00:00+00:00")
        (tmp_path / "store" / LEASE_NAME).write_text(json.dumps(stale))
        fresh = heartbeat_lease(root, lease)
        assert fresh["heartbeat_at"] > "2020-01-01"

    def test_heartbeat_refuses_a_stolen_lease(self, tmp_path):
        root = _root(tmp_path)
        lease = acquire_lease(root, run_id="A")
        stolen = dict(lease, run_id="B")
        (tmp_path / "store" / LEASE_NAME).write_text(json.dumps(stolen))
        with pytest.raises(SweepRefusedError, match="another run claimed"):
            heartbeat_lease(root, lease)

    def test_release_deletes_only_our_own(self, tmp_path):
        root = _root(tmp_path)
        acquire_lease(root, run_id="A")
        assert release_lease(root, run_id="B") is False
        assert read_lease(root)["run_id"] == "A"
        assert release_lease(root, run_id="A") is True
        assert read_lease(root) is None
        assert release_lease(root, run_id="A") is False  # already gone
