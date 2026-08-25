"""Live-S3 half of the issue #522 canned-ACL harness.

``tests/test_store_acl_signing.py`` pins what obstore puts on the wire; this
module pins what real S3 does about it, because "the header is unsigned" is
only a bug because S3 answers that shape with a 403. It reproduces the fleet
failure WITHOUT touching Source Cooperative: the issue #495 treatment (an
``x-amz-acl`` obstore default header) is forced onto an IN-ACCOUNT handle, and
the signing physics are identical, so the LIST fails the same way.

Skipped unless ``ZAGG_LIVE_S3_PREFIX`` names a writable ``s3://bucket/prefix``
this account owns -- and skipped too if it names a published bucket, which is
checked, not assumed -- so CI (which has no AWS credentials) never sees it::

    AWS_PROFILE=nasa ZAGG_LIVE_S3_PREFIX=s3://sliderule-public/zagg-demo/_acl_repro_20260825 \\
        pytest tests/test_store_acl_live.py -v

Every object it writes lives under that prefix and is deleted on teardown.
"""

import os
import uuid

import pytest

LIVE_PREFIX = os.environ.get("ZAGG_LIVE_S3_PREFIX")


def _names_a_published_bucket() -> bool:
    """Is ``ZAGG_LIVE_S3_PREFIX`` pointed at a Source Cooperative bucket?

    Makes this module's in-account precondition executable rather than a
    docstring promise. A published prefix would have the repro writing
    ACL-bearing objects into the one bucket where stray objects are most
    expensive, so skip instead of running.
    """
    if not LIVE_PREFIX:
        return False
    from zagg.store import _PUBLISHED_BUCKETS, parse_s3_path

    return parse_s3_path(LIVE_PREFIX)[0] in _PUBLISHED_BUCKETS


pytestmark = [
    pytest.mark.skipif(
        not LIVE_PREFIX, reason="set ZAGG_LIVE_S3_PREFIX to an owned s3:// prefix to run"
    ),
    pytest.mark.skipif(
        _names_a_published_bucket(),
        reason="ZAGG_LIVE_S3_PREFIX must name an IN-ACCOUNT bucket, not a published one",
    ),
]

ACL = "bucket-owner-full-control"
ACL_HEADERS = {"x-amz-acl": ACL}


@pytest.fixture
def live_prefix():
    """A unique child of ``ZAGG_LIVE_S3_PREFIX``, emptied afterwards."""
    import obstore

    from zagg.store import open_object_store

    prefix = f"{LIVE_PREFIX.rstrip('/')}/{uuid.uuid4().hex[:12]}"
    yield prefix
    # The cleanup handle can list even on a published bucket: since phase 2
    # ``open_object_store`` always returns the CLEAN store and the canned ACL
    # rides a twin reached only through ``put_object``, so this LIST is an
    # ordinary signed request and cannot re-enter the bug under test. The
    # ``finally`` is belt-and-braces on top of that: whatever was listed before
    # a mid-stream error still gets deleted, rather than the whole cleanup
    # being skipped and objects left behind.
    store = open_object_store(prefix)
    keys: list[str] = []
    try:
        for batch in obstore.list(store):
            keys.extend(entry["path"] for entry in batch)
    finally:
        for key in keys:
            obstore.delete(store, key)


def _forced_acl_store(prefix):
    """An in-account handle wearing the issue #495 treatment.

    Built directly rather than through ``zagg.store`` on purpose: the point is
    the treatment, not the bucket, so the repro survives whatever
    ``_PUBLISHED_BUCKETS`` happens to contain.
    """
    from obstore.auth.boto3 import Boto3CredentialProvider
    from obstore.store import S3Store

    from zagg.store import parse_s3_path

    bucket, key_prefix = parse_s3_path(prefix)
    return S3Store(
        bucket,
        prefix=key_prefix,
        region="us-west-2",
        credential_provider=Boto3CredentialProvider(),
        client_options={"default_headers": ACL_HEADERS},
    )


class TestForcedAclDefaultHeader:
    """The production failure, reproduced in-account (issue #522)."""

    def test_list_403s_with_headers_not_signed(self, live_prefix):
        import obstore
        import obstore.exceptions

        # The concrete type, not a bare ``Exception``: a credential-chain
        # failure, a region redirect or a bucket typo would otherwise reach the
        # substring assertions below and fail there, pointing at the wrong
        # thing. obstore wraps the LIST 403 as ``GenericError`` (measured
        # offline in ``test_store_acl_signing.py``'s stand-in by answering the
        # ListObjectsV2 with the real 403 body).
        store = _forced_acl_store(live_prefix)
        with pytest.raises(obstore.exceptions.GenericError) as excinfo:
            list(obstore.list(store))
        message = str(excinfo.value)
        # The exact shape 2,726/2,726 fleet workers died on.
        assert "403" in message
        assert "AccessDenied" in message
        assert "not signed" in message
        assert "x-amz-acl" in message

    def test_put_through_the_same_handle_succeeds(self, live_prefix):
        # The other half of the asymmetry, and why the bug survived issue
        # #496's validation: object-creating requests sign the default header,
        # so the write probe passed while every read was already broken.
        import obstore

        store = _forced_acl_store(live_prefix)
        obstore.put(store, "probe.txt", b"issue 522 live repro")
        # Read it back through a handle WITHOUT the treatment: the object
        # landed, so the PUT really was accepted rather than silently dropped.
        from zagg.store import open_object_store

        clean = open_object_store(live_prefix)
        assert bytes(obstore.get(clean, "probe.txt").bytes()) == b"issue 522 live repro"

    def test_a_clean_handle_lists_the_same_prefix(self, live_prefix):
        # Control: nothing about the prefix or the credentials is at fault --
        # remove the header and the identical LIST succeeds.
        import obstore

        from zagg.store import open_object_store

        obstore.put(_forced_acl_store(live_prefix), "probe.txt", b"x")
        clean = open_object_store(live_prefix)
        keys = [entry["path"] for batch in obstore.list(clean) for entry in batch]
        assert keys == ["probe.txt"]


@pytest.fixture
def published_live(monkeypatch, live_prefix):
    """Treat the in-account test bucket as a published one, against real S3.

    The regression half of this module: the repro above shows the pre-fix
    treatment 403ing, and this shows the shipped seam surviving the identical
    request against the identical bucket, with the ACL still on the writes.
    """
    import zagg.store as store_mod

    bucket = store_mod.parse_s3_path(live_prefix)[0]
    monkeypatch.setattr(store_mod, "_PUBLISHED_BUCKETS", frozenset({bucket}))
    store_mod._OBJECT_STORE_CACHE.clear()
    yield live_prefix
    store_mod._OBJECT_STORE_CACHE.clear()


class TestFixedSeamAgainstRealS3:
    """Issue #522, the shipped fix, exercised on the bucket that rejects it wrong."""

    def test_the_raw_handle_writes_then_lists(self, published_live):
        import obstore

        from zagg.store import acl_write_store, open_object_store, put_object

        store = open_object_store(published_live)
        # The handle really is treated as external -- otherwise this test would
        # pass by never having an ACL in play at all.
        twin = acl_write_store(store)
        assert twin is not store
        # ...and the twin really carries the header, which "a twin exists" does
        # not say. Construction only: against an IN-ACCOUNT bucket S3 cannot
        # confirm it from the other side, because a same-account
        # ``bucket-owner-full-control`` grant is indistinguishable from the
        # default owner grant in ``GetObjectAcl``. Bytes because obstore
        # normalizes header values (``tests/test_store.py`` pins the same form).
        assert twin.client_options["default_headers"]["x-amz-acl"] == b"bucket-owner-full-control"
        assert store.client_options is None

        put_object(store, "probe.txt", b"issue 522 fixed seam")
        keys = [entry["path"] for batch in obstore.list(store) for entry in batch]
        assert keys == ["probe.txt"]
        assert bytes(obstore.get(store, "probe.txt").bytes()) == b"issue 522 fixed seam"

    def test_a_zarr_store_round_trips(self, published_live):
        # The worker's own shape: create an array, write a chunk, read it back.
        # Every one of those steps lists or heads the store, which is what the
        # fleet died on.
        import numpy as np
        import zarr

        from zagg.store import _AclWriteObjectStore, open_store

        zstore = open_store(f"{published_live}/leaf.zarr")
        assert isinstance(zstore, _AclWriteObjectStore)
        array = zarr.create_array(
            store=zstore, name="temp", shape=(4,), chunks=(2,), dtype="i4", zarr_format=3
        )
        array[:] = np.arange(4, dtype="i4")

        read_back = zarr.open_array(store=open_store(f"{published_live}/leaf.zarr"), path="temp")
        assert read_back[:].tolist() == [0, 1, 2, 3]

    def test_the_status_poller_lists_a_published_channel(self, published_live):
        # The client-side failure: the event transport's poller listing a run's
        # `.status/` channel. Pre-fix this LIST was the 403.
        from zagg.client_transport import StatusPoller
        from zagg.store import open_object_store, put_object

        prefix = f"{published_live}/out.zarr.status/run-e1ebd1c0"
        put_object(open_object_store(prefix), "shard-0.json", b"{}")

        poller = StatusPoller(store_factory=lambda: open_object_store(prefix), drop_timeout_s=1.0)
        assert poller._list_keys() == {"shard-0.json"}
