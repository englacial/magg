"""The issue #522 dual-handle split, end to end through the zagg seam.

``tests/test_store_acl_signing.py`` pins obstore's behaviour; this module pins
zagg's use of it, by driving real ``zagg.store`` handles against the same
in-memory S3 stand-in and reading the captured wire bytes. Nothing here mocks
obstore, so an assertion about ``SignedHeaders`` is an assertion about the bytes
a fleet worker would send.

The stand-in is reached through ``endpoint_url``, which
:func:`zagg.store._external_target` deliberately excludes (R2/MinIO do not
implement canned ACLs), so each test forces the external-target verdict on and
lets the rest of the seam run for real.
"""

import pytest
from test_store_acl_signing import ACL, FakeS3

STORE_PATH = "s3://us-west-2.opendata.source.coop/englacial/zagg/demo.zarr"
CREDS = {"accessKeyId": "AKIAEXAMPLE", "secretAccessKey": "secret"}


@pytest.fixture
def fake_s3():
    server = FakeS3()
    try:
        yield server
    finally:
        server.close()


@pytest.fixture
def published(monkeypatch, fake_s3):
    """Open zagg stores against the stand-in as if it were a published bucket.

    Returns a callable taking ``zagg.store`` opener kwargs.
    """
    import zagg.store as store_mod

    monkeypatch.setattr(store_mod, "_external_target", lambda *a, **k: True)
    store_mod._OBJECT_STORE_CACHE.clear()

    def _open(opener, path=STORE_PATH, **kwargs):
        kwargs.setdefault("client_options", {"allow_http": True})
        return opener(path, credentials=CREDS, endpoint_url=fake_s3.endpoint, **kwargs)

    yield _open
    store_mod._OBJECT_STORE_CACHE.clear()


def _puts(fake_s3):
    return [r for r in fake_s3.requests if r.method == "PUT"]


def _lists(fake_s3):
    return [r for r in fake_s3.requests if r.method == "GET" and "list-type" in r.path]


class TestOwnershipSemanticsUnchanged:
    """Phase 3 of issue #522: the split must not cost the canned ACL.

    Source Cooperative's direct-IAM upload path requires
    ``bucket-owner-full-control`` on every object we publish, so the fix is only
    a fix if the ACL still rides every object-creating request -- and rides it
    inside the signature, which is where obstore already puts a default header
    on that request shape.
    """

    def test_a_real_zarr_write_signs_the_acl_on_every_put(self, published, fake_s3):
        # Driven through zarr, not through obstore directly: the write path
        # that publishes shard bytes is zarr's obstore adapter, and it is the
        # path issue #495's per-PUT alternative could not reach.
        import numpy as np
        import zarr

        from zagg.store import _AclWriteObjectStore, open_store

        zstore = published(open_store)
        assert isinstance(zstore, _AclWriteObjectStore)
        array = zarr.create_array(
            store=zstore, name="temp", shape=(4,), chunks=(2,), dtype="i4", zarr_format=3
        )
        array[:] = np.arange(4, dtype="i4")

        puts = _puts(fake_s3)
        assert puts, "the zarr write issued no PUT"
        for put in puts:
            assert put.acl == ACL, f"{put.path} published with no canned ACL"
            assert put.acl_signed, f"{put.path} carries an unsigned ACL: {put.signed_headers}"

    def test_put_object_signs_the_acl_on_a_side_channel_write(self, published, fake_s3):
        # Status envelopes, hive manifests and stats sidecars are real objects
        # in the published bucket, so they hand over ownership too.
        from zagg.store import open_object_store, put_object

        put_object(published(open_object_store), "run1.json", b"{}")

        (put,) = _puts(fake_s3)
        assert put.acl == ACL
        assert put.acl_signed

    def test_the_handle_the_caller_holds_never_sends_an_acl(self, published, fake_s3):
        # The other half of the contract: reads and lists carry no ACL at all,
        # signed or otherwise, so there is nothing for S3 to reject.
        import obstore

        from zagg.store import open_object_store, put_object

        store = published(open_object_store)
        put_object(store, "run1.json", b"{}")
        fake_s3.clear()
        obstore.get(store, "run1.json").bytes()
        list(obstore.list(store))

        assert fake_s3.requests
        for request in fake_s3.requests:
            assert request.acl is None, f"{request.method} {request.path} still carries the ACL"
