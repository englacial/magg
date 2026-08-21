"""Tests for the store factory."""

from pathlib import Path
from unittest import mock

import pytest
from zarr.storage import LocalStore

from zagg.store import open_object_store, open_store, parse_s3_path


@pytest.fixture
def mock_s3(monkeypatch):
    """Patch obstore S3Store / Boto3CredentialProvider / zarr ObjectStore.

    Returns the (S3Store, Boto3CredentialProvider) mocks so tests can assert
    on the kwargs ``_open_s3_store`` passes to ``S3Store``.
    """
    s3_cls = mock.MagicMock(name="S3Store")
    prov_cls = mock.MagicMock(name="Boto3CredentialProvider")
    obj_cls = mock.MagicMock(name="ObjectStore")
    monkeypatch.setattr("obstore.store.S3Store", s3_cls)
    monkeypatch.setattr("obstore.auth.boto3.Boto3CredentialProvider", prov_cls)
    monkeypatch.setattr("zarr.storage.ObjectStore", obj_cls)
    return s3_cls, prov_cls


@pytest.fixture(autouse=True)
def _clear_object_store_cache():
    """Reset the process-global ambient store cache (issue #287) around every
    test so one test's cached store can't leak into another's assertions."""
    from zagg import store

    store._OBJECT_STORE_CACHE.clear()
    yield
    store._OBJECT_STORE_CACHE.clear()


class TestOpenStore:
    def test_local_absolute_path(self, tmp_path):
        store = open_store(str(tmp_path / "test.zarr"))
        assert isinstance(store, LocalStore)

    def test_local_relative_path(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        store = open_store("./output.zarr")
        assert isinstance(store, LocalStore)
        assert Path(str(store.root)).is_absolute()

    def test_local_read_only(self, tmp_path):
        p = tmp_path / "test.zarr"
        p.mkdir()
        store = open_store(str(p), read_only=True)
        assert isinstance(store, LocalStore)


class TestOpenS3Store:
    """Assert the credential/endpoint branching in ``_open_s3_store``."""

    def test_no_creds_uses_credential_provider(self, mock_s3):
        s3_cls, prov_cls = mock_s3
        open_store("s3://bucket/prefix.zarr")
        _, kwargs = s3_cls.call_args
        # Ambient path: credential_provider set, no explicit keys.
        assert kwargs["credential_provider"] is prov_cls.return_value
        assert "access_key_id" not in kwargs
        assert "secret_access_key" not in kwargs
        assert "session_token" not in kwargs
        assert "endpoint" not in kwargs
        # Default addressing unchanged (no path-style forced).
        assert "virtual_hosted_style_request" not in kwargs

    def test_explicit_creds(self, mock_s3):
        s3_cls, prov_cls = mock_s3
        creds = {
            "accessKeyId": "AKIA",
            "secretAccessKey": "secret",
            "sessionToken": "tok",
        }
        open_store("s3://us-west-2.opendata.source.coop/foo.zarr", credentials=creds)
        _, kwargs = s3_cls.call_args
        assert kwargs["access_key_id"] == "AKIA"
        assert kwargs["secret_access_key"] == "secret"
        assert kwargs["session_token"] == "tok"
        assert "credential_provider" not in kwargs
        # Path-style addressing for dotted bucket names / external targets.
        assert kwargs["virtual_hosted_style_request"] is False
        prov_cls.assert_not_called()

    def test_explicit_creds_no_session_token(self, mock_s3):
        s3_cls, _ = mock_s3
        creds = {"accessKeyId": "AKIA", "secretAccessKey": "secret"}
        open_store("s3://bucket/foo.zarr", credentials=creds)
        _, kwargs = s3_cls.call_args
        assert "session_token" not in kwargs

    def test_custom_endpoint(self, mock_s3):
        s3_cls, prov_cls = mock_s3
        creds = {"accessKeyId": "AKIA", "secretAccessKey": "secret"}
        open_store(
            "s3://bucket/foo.zarr",
            credentials=creds,
            endpoint_url="https://acct.r2.cloudflarestorage.com",
        )
        _, kwargs = s3_cls.call_args
        assert kwargs["endpoint"] == "https://acct.r2.cloudflarestorage.com"
        assert kwargs["virtual_hosted_style_request"] is False

    def test_endpoint_only_no_creds(self, mock_s3):
        # endpoint_url alone (no creds) still takes the explicit branch.
        s3_cls, prov_cls = mock_s3
        open_store("s3://bucket/foo.zarr", endpoint_url="https://minio.local")
        _, kwargs = s3_cls.call_args
        assert kwargs["endpoint"] == "https://minio.local"
        assert kwargs["virtual_hosted_style_request"] is False
        assert "credential_provider" not in kwargs


class TestBucketOwnerAcl:
    """Issue #495: writes to a target we do not own must hand ownership to the
    bucket owner. S3 object ownership follows the WRITING account, so a
    cross-account PUT without ``x-amz-acl: bucket-owner-full-control`` leaves
    objects the bucket owner cannot manage -- Source Cooperative's in-region
    upload grant requires the canned ACL for exactly that reason. obstore has no
    ACL config key, so it rides as a default request header."""

    HEADER = {"x-amz-acl": "bucket-owner-full-control"}
    CREDS = {"accessKeyId": "ASIA", "secretAccessKey": "secret", "sessionToken": "tok"}

    def test_external_target_sets_canned_acl(self, mock_s3):
        s3_cls, _ = mock_s3
        open_store(
            "s3://us-west-2.opendata.source.coop/englacial/zagg/d.zarr", credentials=self.CREDS
        )
        _, kwargs = s3_cls.call_args
        assert kwargs["client_options"]["default_headers"] == self.HEADER

    def test_side_channel_objects_get_the_header_too(self, mock_s3):
        # Status envelopes, hive manifests, stats sidecars and the temporal
        # tabular object are real writes to the same external store, so the
        # raw-obstore route must carry the ACL as well.
        s3_cls, _ = mock_s3
        open_object_store(
            "s3://us-west-2.opendata.source.coop/englacial/zagg/d.zarr.status/run1",
            credentials=self.CREDS,
        )
        _, kwargs = s3_cls.call_args
        assert kwargs["client_options"]["default_headers"] == self.HEADER

    def test_ambient_write_to_a_published_bucket_sends_the_header(self, mock_s3):
        # Phase 3 (issue #495) made publishing AMBIENT: the execution role
        # reaches source.coop with no injected credentials. Keying the ACL on
        # "did the caller pass credentials?" would therefore stop sending it on
        # exactly the path it exists for, and the fleet would publish objects
        # Source Cooperative cannot manage -- silently, since the PUT succeeds.
        # The destination decides instead (review finding on PR #496).
        s3_cls, _ = mock_s3
        open_store("s3://us-west-2.opendata.source.coop/englacial/zagg/demo/d.zarr")
        _, kwargs = s3_cls.call_args
        assert kwargs["client_options"]["default_headers"] == self.HEADER
        # ...and through the raw-obstore route too (status envelopes, hive
        # manifests, stats sidecars), including its ambient per-process cache.
        open_object_store(
            "s3://us-west-2.opendata.source.coop/englacial/zagg/demo/d.zarr.status/run1"
        )
        _, kwargs = s3_cls.call_args
        assert kwargs["client_options"]["default_headers"] == self.HEADER

    def test_in_account_write_sends_no_acl_header(self, mock_s3):
        # Ambient writes to a bucket we OWN: nothing to hand over, and no
        # client_options are introduced at all. Scoped to ownership, not to
        # ambience -- the header requires s3:PutObjectAcl on the target, which
        # the execution role holds on the published prefix only, so sending it
        # to every AWS endpoint would 403 self-hosters' own output buckets and
        # sliderule-public-cors (whose bucket policy is not ours to change).
        s3_cls, _ = mock_s3
        open_store("s3://our-bucket/out.zarr")
        _, kwargs = s3_cls.call_args
        assert "client_options" not in kwargs
        open_object_store("s3://our-bucket/out.zarr.status/run1")
        _, kwargs = s3_cls.call_args
        assert "client_options" not in kwargs
        open_store("s3://sliderule-public-cors/zagg-examples/d.zarr")
        _, kwargs = s3_cls.call_args
        assert "client_options" not in kwargs

    def test_anonymous_read_sends_no_acl_header(self, mock_s3):
        s3_cls, _ = mock_s3
        open_store("s3://public-bucket/demo.zarr", read_only=True, skip_signature=True)
        _, kwargs = s3_cls.call_args
        assert "client_options" not in kwargs

    def test_read_only_published_store_sends_no_acl_header(self, mock_s3):
        # The read_only exclusion is unchanged by the destination-keyed
        # predicate: reading the published store back is not a write.
        s3_cls, _ = mock_s3
        open_store("s3://us-west-2.opendata.source.coop/englacial/zagg/demo/d.zarr", read_only=True)
        _, kwargs = s3_cls.call_args
        assert "client_options" not in kwargs

    def test_custom_endpoint_sends_no_acl_header(self, mock_s3):
        # Canned ACLs are an AWS-S3 concept; the S3-compatible stores behind
        # endpoint_url (R2, MinIO) do not implement them.
        s3_cls, _ = mock_s3
        open_store(
            "s3://bucket/foo.zarr",
            credentials=self.CREDS,
            endpoint_url="https://acct.r2.cloudflarestorage.com",
        )
        _, kwargs = s3_cls.call_args
        assert "client_options" not in kwargs
        # ...and the exclusion still wins over the destination, which is what
        # keeps the retired data.source.coop proxy hop (an endpoint-routed AWS
        # target) out of the header path.
        open_store(
            "s3://us-west-2.opendata.source.coop/englacial/zagg/demo/d.zarr",
            endpoint_url="https://data.source.coop",
        )
        _, kwargs = s3_cls.call_args
        assert "client_options" not in kwargs

    def test_read_with_input_credentials_sends_no_acl_header(self, mock_s3):
        # The issue #223 consumer-INPUT channel: explicit credentials that read
        # somebody else's bucket (temporal.open_dataset's .zarr branch), not a
        # write target of ours. read_only is consumed by _s3_object_store and
        # never reaches S3Store.
        s3_cls, _ = mock_s3
        open_store("s3://someones-input/mask.zarr", read_only=True, credentials=self.CREDS)
        _, kwargs = s3_cls.call_args
        assert "client_options" not in kwargs
        assert "read_only" not in kwargs

    def test_object_store_reads_are_the_documented_exception(self, mock_s3):
        # open_object_store has no read_only concept, so temporal's NetCDF
        # branch (a pure GET of an input bucket) still carries the header. Inert
        # -- S3 reads x-amz-acl only on object-creating requests -- and pinned
        # here so the exception stays visible.
        s3_cls, _ = mock_s3
        open_object_store("s3://someones-input", credentials=self.CREDS, region="us-west-2")
        _, kwargs = s3_cls.call_args
        assert kwargs["client_options"]["default_headers"] == self.HEADER

    def test_caller_client_options_are_preserved(self, mock_s3):
        # Additive merge: unrelated client options survive, and a caller who
        # sets x-amz-acl explicitly wins over the default.
        s3_cls, _ = mock_s3
        open_store(
            "s3://external/foo.zarr",
            credentials=self.CREDS,
            client_options={
                "timeout": "30s",
                "default_headers": {"x-amz-acl": "private", "x-custom": "1"},
            },
        )
        _, kwargs = s3_cls.call_args
        assert kwargs["client_options"]["timeout"] == "30s"
        assert kwargs["client_options"]["default_headers"] == {
            "x-amz-acl": "private",
            "x-custom": "1",
        }

    def test_caller_mixed_case_acl_header_still_wins(self, mock_s3):
        # obstore lowercases header keys, so a raw setdefault cannot see
        # ``X-Amz-Acl`` and the collision resolves inside obstore in OUR
        # favour -- a caller who deliberately asked for ``private`` would be
        # silently overridden (review finding). Lowercasing the merge fixes it.
        s3_cls, _ = mock_s3
        open_store(
            "s3://external/foo.zarr",
            credentials=self.CREDS,
            client_options={"default_headers": {"X-Amz-Acl": "private", "X-Custom": "1"}},
        )
        _, kwargs = s3_cls.call_args
        assert kwargs["client_options"]["default_headers"] == {
            "x-amz-acl": "private",
            "x-custom": "1",
        }

    def test_caller_can_strip_the_header_with_an_explicit_none(self, mock_s3):
        # The escape hatch for an external AWS target that must send no ACL:
        # neither obstore-legal value expresses absence (None is rejected, ""
        # is a live empty header S3 rejects), so a None value means REMOVE.
        s3_cls, _ = mock_s3
        open_store(
            "s3://external/foo.zarr",
            credentials=self.CREDS,
            client_options={"default_headers": {"x-amz-acl": None, "x-custom": "1"}},
        )
        _, kwargs = s3_cls.call_args
        assert kwargs["client_options"]["default_headers"] == {"x-custom": "1"}

    def test_a_real_obstore_store_accepts_and_normalizes_the_header(self):
        # Deliberately UNMOCKED (no network: construction only). Every other
        # test here mocks S3Store, so they pin what zagg passes and never what
        # obstore accepts -- client_options/default_headers validate only at
        # real construction (a typo raises ValueError: Invalid key), so an
        # obstore rename would leave this class green while the fleet wrote
        # owner-less objects into someone else's bucket. Note the bytes:
        # obstore normalizes header values.
        from zagg.store import _s3_object_store

        s3 = _s3_object_store("s3://external/foo.zarr", credentials=self.CREDS)
        assert s3.client_options["default_headers"]["x-amz-acl"] == b"bucket-owner-full-control"

    def test_a_real_obstore_read_only_store_carries_no_header(self):
        # The issue #223 consumer-input shape, unmocked: no client_options at
        # all reach obstore, so nothing rides the GET.
        from zagg.store import _s3_object_store

        s3 = _s3_object_store(
            "s3://someones-input/mask.zarr", credentials=self.CREDS, read_only=True
        )
        assert s3.client_options is None

    def test_caller_client_options_are_not_mutated(self, mock_s3):
        s3_cls, _ = mock_s3
        options = {"default_headers": {"x-custom": "1"}}
        open_store("s3://external/foo.zarr", credentials=self.CREDS, client_options=options)
        assert options == {"default_headers": {"x-custom": "1"}}


class TestS3RetryConfig:
    """Issue #186: S3 stores get a paced retry policy by default. obstore's
    default (10 retries, 100 ms init backoff, uniform jitter) exhausts its whole
    budget in ~2-4 s under a sustained 503 SlowDown burst — near-immediate
    retries that amplify the throttle instead of riding it out."""

    def test_default_retry_config_ambient(self, mock_s3):
        from zagg.store import _S3_RETRY_CONFIG

        s3_cls, _ = mock_s3
        open_store("s3://bucket/prefix.zarr")
        _, kwargs = s3_cls.call_args
        assert kwargs["retry_config"] == _S3_RETRY_CONFIG

    def test_default_retry_config_explicit_creds(self, mock_s3):
        from zagg.store import _S3_RETRY_CONFIG

        s3_cls, _ = mock_s3
        creds = {"accessKeyId": "AKIA", "secretAccessKey": "secret"}
        open_object_store("s3://bucket/foo.zarr.status/run1", credentials=creds)
        _, kwargs = s3_cls.call_args
        assert kwargs["retry_config"] == _S3_RETRY_CONFIG

    def test_caller_override_wins(self, mock_s3):
        s3_cls, _ = mock_s3
        custom = {"max_retries": 2}
        open_store("s3://bucket/prefix.zarr", retry_config=custom)
        _, kwargs = s3_cls.call_args
        assert kwargs["retry_config"] == custom

    def test_explicit_none_gets_default(self, mock_s3):
        """``retry_config=None`` means the zagg default, not obstore's
        fast-burn default (a bare ``setdefault`` would let None through)."""
        from zagg.store import _S3_RETRY_CONFIG

        s3_cls, _ = mock_s3
        open_store("s3://bucket/prefix.zarr", retry_config=None)
        _, kwargs = s3_cls.call_args
        assert kwargs["retry_config"] == _S3_RETRY_CONFIG

    def test_read_only_gets_short_policy(self, mock_s3):
        """``read_only=True`` (the interactive population) fails fast on a
        dead endpoint: the shorter read-only policy, not the write one."""
        from zagg.store import _S3_READONLY_RETRY_CONFIG

        s3_cls, _ = mock_s3
        open_store("s3://bucket/prefix.zarr", read_only=True)
        _, kwargs = s3_cls.call_args
        assert kwargs["retry_config"] == _S3_READONLY_RETRY_CONFIG
        # No aliasing, nested backoff included (mirrors test_default_not_aliased).
        assert kwargs["retry_config"] is not _S3_READONLY_RETRY_CONFIG
        assert kwargs["retry_config"]["backoff"] is not _S3_READONLY_RETRY_CONFIG["backoff"]

    def test_read_only_explicit_none_gets_short_policy(self, mock_s3):
        """``retry_config=None`` on a read-only store means the read-only
        default (mirrors ``test_explicit_none_gets_default`` on the write
        path — the guard is ``is None``, not key-absence)."""
        from zagg.store import _S3_READONLY_RETRY_CONFIG

        s3_cls, _ = mock_s3
        open_store("s3://bucket/prefix.zarr", read_only=True, retry_config=None)
        _, kwargs = s3_cls.call_args
        assert kwargs["retry_config"] == _S3_READONLY_RETRY_CONFIG

    def test_read_only_override_wins(self, mock_s3):
        s3_cls, _ = mock_s3
        custom = {"max_retries": 7}
        open_store("s3://bucket/prefix.zarr", read_only=True, retry_config=custom)
        _, kwargs = s3_cls.call_args
        assert kwargs["retry_config"] == custom

    def test_anonymous_read_skips_credential_provider(self, mock_s3):
        """``skip_signature=True`` (public bucket, e.g. the example notebooks
        on binder) must not construct ``Boto3CredentialProvider`` — it raises
        without ambient AWS credentials, which anonymous environments lack."""
        s3_cls, prov_cls = mock_s3
        open_store("s3://bucket/public.zarr", read_only=True, skip_signature=True)
        _, kwargs = s3_cls.call_args
        assert kwargs["skip_signature"] is True
        assert "credential_provider" not in kwargs
        prov_cls.assert_not_called()
        # The anonymous branch still inherits the read-only retry policy.
        from zagg.store import _S3_READONLY_RETRY_CONFIG

        assert kwargs["retry_config"] == _S3_READONLY_RETRY_CONFIG

    def test_skip_signature_with_credentials_takes_credentialed_branch(self, mock_s3):
        """Explicit credentials win over ``skip_signature``: the credentialed
        branch is taken (signed client config), ``skip_signature`` rides
        through kwargs for obstore to honor."""
        s3_cls, prov_cls = mock_s3
        creds = {"accessKeyId": "AKIA", "secretAccessKey": "secret"}
        open_store("s3://bucket/x.zarr", credentials=creds, skip_signature=True)
        _, kwargs = s3_cls.call_args
        assert kwargs["access_key_id"] == "AKIA"
        assert kwargs["skip_signature"] is True
        prov_cls.assert_not_called()

    def test_default_not_aliased(self, mock_s3):
        """Each store gets its own copy — mutating one store's config must
        not edit the module-level default (nested ``backoff`` included)."""
        from zagg.store import _S3_RETRY_CONFIG

        s3_cls, _ = mock_s3
        open_store("s3://bucket/prefix.zarr")
        _, kwargs = s3_cls.call_args
        assert kwargs["retry_config"] is not _S3_RETRY_CONFIG
        assert kwargs["retry_config"]["backoff"] is not _S3_RETRY_CONFIG["backoff"]

    def test_default_paces_retries(self, tmp_path):
        """End-to-end through ``open_object_store`` against a local always-503
        endpoint: retries are paced by real exponential backoff (a small
        override keeps the test fast; the default values are pinned above).
        With obstore's default policy the same request burns 10 retries with
        sub-second gaps."""
        import http.server
        import threading
        import time
        from datetime import timedelta

        import obstore

        times: list[float] = []

        class SlowDown(http.server.BaseHTTPRequestHandler):
            def do_PUT(self):
                times.append(time.monotonic())
                body = (
                    b"<Error><Code>SlowDown</Code>"
                    b"<Message>Please reduce your request rate.</Message></Error>"
                )
                self.send_response(503)
                self.send_header("Content-Type", "application/xml")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), SlowDown)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            store = open_object_store(
                "s3://bkt/prefix",
                credentials={"accessKeyId": "x", "secretAccessKey": "y"},
                endpoint_url=f"http://127.0.0.1:{srv.server_address[1]}",
                client_options={"allow_http": True},
                retry_config={
                    "max_retries": 2,
                    "retry_timeout": timedelta(seconds=30),
                    "backoff": {
                        "init_backoff": timedelta(milliseconds=400),
                        "max_backoff": timedelta(seconds=2),
                        "base": 2,
                    },
                },
            )
            with pytest.raises(Exception, match="SlowDown|503|Server"):
                obstore.put(store, "zarr.json", b"{}")
        finally:
            srv.shutdown()
            srv.server_close()
        # 1 initial request + exactly max_retries retries: the override reached
        # the client, which is the plumbing under test (backoff jitter is
        # uniform from zero, so per-gap timing floors would flake).
        assert len(times) == 3
        assert times[-1] - times[0] < 10


class TestOpenObjectStore:
    """Raw obstore store for side-channel objects (issue #151): same path and
    credential handling as ``open_store``, but no Zarr wrapper."""

    def test_local_roundtrip_creates_dir(self, tmp_path):
        import obstore

        store = open_object_store(str(tmp_path / "x.zarr.status" / "run1"))
        obstore.put(store, "12345.json", b'{"statusCode": 200}')
        got = bytes(obstore.get(store, "12345.json").bytes())
        assert got == b'{"statusCode": 200}'

    def test_local_missing_object_raises_not_found(self, tmp_path):
        import obstore
        from obstore.exceptions import NotFoundError

        store = open_object_store(str(tmp_path / "status"))
        with pytest.raises((FileNotFoundError, NotFoundError)):
            obstore.get(store, "nope.json")

    def test_s3_returns_bare_store_with_ambient_creds(self, mock_s3):
        s3_cls, prov_cls = mock_s3
        store = open_object_store("s3://bucket/prefix.zarr.status/run1")
        # The raw S3Store, not wrapped in zarr's ObjectStore.
        assert store is s3_cls.return_value
        _, kwargs = s3_cls.call_args
        assert kwargs["credential_provider"] is prov_cls.return_value

    def test_s3_explicit_creds_and_endpoint(self, mock_s3):
        s3_cls, prov_cls = mock_s3
        creds = {"accessKeyId": "AKIA", "secretAccessKey": "secret", "sessionToken": "tok"}
        open_object_store(
            "s3://bucket/foo.zarr.status/run1",
            credentials=creds,
            endpoint_url="https://minio.local",
        )
        _, kwargs = s3_cls.call_args
        assert kwargs["access_key_id"] == "AKIA"
        assert kwargs["endpoint"] == "https://minio.local"
        assert kwargs["virtual_hosted_style_request"] is False
        prov_cls.assert_not_called()


class TestObjectStoreCache:
    """Ambient ``s3://`` stores are memoized per process (issue #287): the
    sidecar manifest fetch calls ``open_object_store(self.store)`` once per
    granule, and each call previously rebuilt a ``Boto3CredentialProvider``
    (~300 ms credential-chain walk) on the read hot path."""

    def test_ambient_store_built_once_and_reused(self, mock_s3):
        s3_cls, prov_cls = mock_s3
        p = "s3://sliderule-public-cors/zagg-index/ATL03/007"
        stores = [open_object_store(p) for _ in range(5)]
        # One construction total; the credential chain is walked once, not 5x.
        assert s3_cls.call_count == 1
        assert prov_cls.call_count == 1
        # Every caller gets the same shared store.
        assert all(s is stores[0] for s in stores)

    def test_distinct_paths_get_distinct_stores(self, mock_s3):
        s3_cls, _ = mock_s3
        open_object_store("s3://bucket/a")
        open_object_store("s3://bucket/b")
        assert s3_cls.call_count == 2

    def test_explicit_credentials_bypass_cache(self, mock_s3):
        s3_cls, _ = mock_s3
        creds = {"accessKeyId": "AKIA", "secretAccessKey": "secret"}
        open_object_store("s3://bucket/a", credentials=creds)
        open_object_store("s3://bucket/a", credentials=creds)
        # A statically-supplied token must never be cached (it would freeze on a
        # warm worker) -- each call builds fresh.
        assert s3_cls.call_count == 2

    def test_endpoint_bypasses_cache(self, mock_s3):
        s3_cls, _ = mock_s3
        open_object_store("s3://bucket/a", endpoint_url="https://minio.local")
        open_object_store("s3://bucket/a", endpoint_url="https://minio.local")
        assert s3_cls.call_count == 2

    def test_kwargs_bypass_cache(self, mock_s3):
        """A caller passing any kwargs (e.g. the read-only retry config on the
        temporal path) falls through to a fresh build -- the cache is scoped to
        the sidecar's bare ``open_object_store(path)`` call only."""
        s3_cls, _ = mock_s3
        from zagg.store import _S3_READONLY_RETRY_CONFIG

        open_object_store("s3://bucket/a", retry_config=_S3_READONLY_RETRY_CONFIG)
        open_object_store("s3://bucket/a", retry_config=_S3_READONLY_RETRY_CONFIG)
        assert s3_cls.call_count == 2

    def test_local_paths_not_cached(self, tmp_path):
        s1 = open_object_store(str(tmp_path / "s"))
        s2 = open_object_store(str(tmp_path / "s"))
        # Local stores are cheap and unaffected -- fresh each call, not cached.
        assert s1 is not s2


class TestParseS3Path:
    def test_bucket_and_prefix(self):
        assert parse_s3_path("s3://mybucket/some/prefix.zarr") == ("mybucket", "some/prefix.zarr")

    def test_bucket_only(self):
        assert parse_s3_path("s3://mybucket") == ("mybucket", "")

    def test_not_s3_raises(self):
        with pytest.raises(ValueError, match="Not an S3 path"):
            parse_s3_path("./local/path.zarr")
