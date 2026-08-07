"""Lifecycle touch for skip-if-current units — issue #388 phase 3.

Pins ``zagg.lifecycle``: the unit-footprint assembly (leaf tree + stats
sidecar + sub-map siblings + declared column tree/sidecar), the local
``os.utime`` mechanism, the S3 self-copy (``CopyObject`` onto itself with
``MetadataDirective="REPLACE"``) against a mocked client, and the fail-open
counting contract (a failed or absent object never raises, never un-skips).
"""

import os

import pytest
from botocore.exceptions import ClientError

from zagg import lifecycle

LEAF = "-5112333.zarr"


def _make_unit(tmp_path, *, submap=True, column=False):
    """A committed-looking local unit: leaf tree + sidecar siblings."""
    node = tmp_path / "store" / "-5" / "1"
    leaf = node / LEAF
    (leaf / "m" / "c").mkdir(parents=True)
    (leaf / "zarr.json").write_bytes(b"{}")
    (leaf / "m" / "c" / "0").write_bytes(b"\x00\x01")
    (leaf / "coverage.moc").write_bytes(b"moc")
    (node / "stats.json").write_bytes(b"{}")
    if submap:
        (node / "shardmap.json").write_bytes(b"{}")
    column_path = None
    if column:
        column_path = node / "all.pyramid.zarr"
        (column_path / "h").mkdir(parents=True)
        (column_path / "zarr.json").write_bytes(b"{}")
        (column_path / "h" / "c").write_bytes(b"\x02")
        (node / "all.pyramid.stats.json").write_bytes(b"{}")
    return node, leaf, column_path


def _age(root, epoch=10_000):
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            os.utime(os.path.join(dirpath, name), (epoch, epoch))
    return epoch * 10**9


def _mtimes(root):
    out = {}
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            p = os.path.join(dirpath, name)
            out[os.path.relpath(p, root)] = os.stat(p).st_mtime_ns
    return out


class TestTouchLocal:
    def test_touches_the_whole_unit_footprint(self, tmp_path):
        node, leaf, _column = _make_unit(tmp_path)
        aged = _age(node)
        counts = lifecycle.touch_current_unit(str(leaf))
        after = _mtimes(node)
        # Leaf tree (coverage.moc included) + stats.json + shardmap.json.
        assert counts == {"touched": 5, "failed": 0}
        assert all(mtime > aged for mtime in after.values())

    def test_column_tree_and_sidecar_ride_the_footprint(self, tmp_path):
        node, leaf, column = _make_unit(tmp_path, column=True)
        aged = _age(node)
        counts = lifecycle.touch_current_unit(str(leaf), column_path=str(column))
        after = _mtimes(node)
        # + the column's two objects and its own stats sidecar.
        assert counts == {"touched": 8, "failed": 0}
        assert all(mtime > aged for mtime in after.values())

    def test_absent_column_is_left_out(self, tmp_path):
        # column_path=None (nothing declared): the column family is not part
        # of the footprint — nothing to touch, nothing to fail.
        node, leaf, _column = _make_unit(tmp_path)
        counts = lifecycle.touch_current_unit(str(leaf), column_path=None)
        assert counts["failed"] == 0 and counts["touched"] == 5

    def test_missing_submap_is_neither_touched_nor_failed(self, tmp_path):
        # A unit legitimately has no sub-map (non-HEALPix, id-less entries).
        _node, leaf, _column = _make_unit(tmp_path, submap=False)
        counts = lifecycle.touch_current_unit(str(leaf))
        assert counts == {"touched": 4, "failed": 0}

    def test_a_failing_utime_counts_and_never_raises(self, tmp_path, monkeypatch):
        node, leaf, _column = _make_unit(tmp_path)
        real = os.utime
        victim = str(leaf / "zarr.json")

        def flaky(path, *a, **k):
            if str(path) == victim:
                raise OSError("EACCES")
            return real(path, *a, **k)

        monkeypatch.setattr(lifecycle.os, "utime", flaky)
        counts = lifecycle.touch_current_unit(str(leaf))
        assert counts == {"touched": 4, "failed": 1}

    def test_footprint_assembly_failure_is_fail_open(self):
        # A leaf name the spec-keyed naming seam refuses (not a .zarr name)
        # cannot raise out of the touch: one counted failure, nothing touched.
        counts = lifecycle.touch_current_unit("/nowhere/not-a-leaf")
        assert counts == {"touched": 0, "failed": 1}


class TestTouchStoreRoot:
    """The root objects no unit footprint covers (review finding, phase 3)."""

    def _root(self, tmp_path, *, moc=True):
        from zagg.hive import AGGREGATION_CORE_NAME, MANIFEST_NAME, ROOT_COVERAGE_NAME

        root = tmp_path / "store"
        root.mkdir()
        (root / MANIFEST_NAME).write_bytes(b"{}")
        (root / AGGREGATION_CORE_NAME).write_bytes(b"a: 1\n")
        if moc:
            (root / ROOT_COVERAGE_NAME).write_bytes(b"{}")
        return root

    def test_touches_the_manifest_semantic_core_and_root_moc(self, tmp_path):
        # ensure_manifest early-returns on a frozen-key match and every
        # skip-capable run is overwrite=False, so nothing else moves these.
        root = self._root(tmp_path)
        aged = _age(root)
        counts = lifecycle.touch_store_root(str(root))
        assert counts == {"touched": 3, "failed": 0}
        assert all(mtime > aged for mtime in _mtimes(root).values())

    def test_an_absent_root_moc_is_neither_touched_nor_failed(self, tmp_path):
        # output.coverage_moc off: the root MOC was never written.
        root = self._root(tmp_path, moc=False)
        assert lifecycle.touch_store_root(str(root)) == {"touched": 2, "failed": 0}

    def test_a_missing_store_root_is_fail_open(self, tmp_path):
        assert lifecycle.touch_store_root(str(tmp_path / "nope")) == {"touched": 0, "failed": 0}

    def test_s3_root_objects_are_exact_keys(self, monkeypatch):
        from unittest.mock import MagicMock

        client = MagicMock()
        client.head_object.return_value = {}
        monkeypatch.setattr(lifecycle, "_s3_client", lambda kw: client)
        counts = lifecycle.touch_store_root("s3://bkt/store/")
        assert {c.kwargs["Key"] for c in client.copy_object.call_args_list} == {
            "store/morton_hive.json",
            "store/aggregation.yaml",
            "store/coverage.moc",
        }
        # No LIST: the root is three named objects, not a tree walk (a walk
        # would re-touch every leaf in the store).
        client.get_paginator.assert_not_called()
        assert counts == {"touched": 3, "failed": 0}


class TestTouchS3:
    BUCKET = "bkt"
    TREE_KEYS = [
        f"store/-5/1/{LEAF}/zarr.json",
        f"store/-5/1/{LEAF}/m/c/0",
        f"store/-5/1/{LEAF}/coverage.moc",
    ]

    def _client(self, monkeypatch, *, copy_error=None, listed=None, head=None):
        from unittest.mock import MagicMock

        client = MagicMock()
        paginator = MagicMock()
        contents = listed if listed is not None else [{"Key": k} for k in self.TREE_KEYS]
        paginator.paginate.return_value = [{"Contents": contents}]
        client.get_paginator.return_value = paginator
        # S3 omits StorageClass for STANDARD in both LIST and HEAD.
        client.head_object.return_value = head if head is not None else {}
        if copy_error is not None:
            client.copy_object.side_effect = copy_error
        monkeypatch.setattr(lifecycle, "_s3_client", lambda kw: client)
        return client

    def test_self_copy_with_metadata_replace_over_the_footprint(self, monkeypatch):
        client = self._client(monkeypatch)
        counts = lifecycle.touch_current_unit(
            f"s3://{self.BUCKET}/store/-5/1/{LEAF}",
            store_kwargs={"region": "us-west-2", "credentials": None, "endpoint_url": None},
        )
        # The tree is enumerated with one delimiter-less LIST under the leaf
        # prefix; the siblings are direct copies.
        client.get_paginator.assert_called_once_with("list_objects_v2")
        assert client.get_paginator.return_value.paginate.call_args.kwargs == {
            "Bucket": self.BUCKET,
            "Prefix": f"store/-5/1/{LEAF}/",
        }
        calls = client.copy_object.call_args_list
        assert {c.kwargs["Key"] for c in calls} == set(self.TREE_KEYS) | {
            "store/-5/1/stats.json",
            "store/-5/1/shardmap.json",
        }
        for c in calls:
            # The identity self-copy: S3 rejects it without REPLACE.
            assert c.kwargs["CopySource"] == {"Bucket": self.BUCKET, "Key": c.kwargs["Key"]}
            assert c.kwargs["MetadataDirective"] == "REPLACE"
            # LIST/HEAD omit the class for STANDARD; the copy must still name
            # one, or REPLACE resets whatever the object had.
            assert c.kwargs["StorageClass"] == "STANDARD"
        assert counts == {"touched": 5, "failed": 0}

    def test_the_storage_class_is_preserved_not_reset_to_standard(self, monkeypatch):
        # MetadataDirective=REPLACE covers SYSTEM metadata, so a copy with no
        # StorageClass silently promotes a STANDARD_IA/GLACIER_IR object back
        # to STANDARD — defeating a lifecycle TRANSITION policy and re-paying
        # the transition on every skip run (review finding).
        client = self._client(
            monkeypatch,
            listed=[
                {"Key": self.TREE_KEYS[0], "StorageClass": "GLACIER_IR"},
                {"Key": self.TREE_KEYS[1], "StorageClass": "STANDARD_IA"},
            ],
            head={"StorageClass": "STANDARD_IA"},
        )
        counts = lifecycle.touch_current_unit(f"s3://{self.BUCKET}/store/-5/1/{LEAF}")
        classes = {
            c.kwargs["Key"]: c.kwargs["StorageClass"] for c in client.copy_object.call_args_list
        }
        assert classes == {
            self.TREE_KEYS[0]: "GLACIER_IR",
            self.TREE_KEYS[1]: "STANDARD_IA",
            # The siblings have no LIST entry: one HEAD each buys the class.
            "store/-5/1/stats.json": "STANDARD_IA",
            "store/-5/1/shardmap.json": "STANDARD_IA",
        }
        assert client.head_object.call_count == 2  # named objects only
        assert counts == {"touched": 4, "failed": 0}

    def test_absent_sibling_is_neither_touched_nor_failed(self, monkeypatch):
        # copy of a missing key (e.g. no sub-map was ever written) NoSuchKey-s;
        # absence is not a failure.
        err = ClientError({"Error": {"Code": "NoSuchKey"}}, "CopyObject")
        self._client(monkeypatch, copy_error=err)
        counts = lifecycle.touch_current_unit(f"s3://{self.BUCKET}/store/-5/1/{LEAF}")
        assert counts == {"touched": 0, "failed": 0}

    def test_a_failing_copy_counts_and_never_raises(self, monkeypatch):
        err = ClientError({"Error": {"Code": "AccessDenied"}}, "CopyObject")
        self._client(monkeypatch, copy_error=err)
        counts = lifecycle.touch_current_unit(f"s3://{self.BUCKET}/store/-5/1/{LEAF}")
        assert counts == {"touched": 0, "failed": 5}

    def test_a_list_fault_aborts_fail_open(self, monkeypatch):
        client = self._client(monkeypatch)
        client.get_paginator.return_value.paginate.side_effect = RuntimeError("boom")
        counts = lifecycle.touch_current_unit(f"s3://{self.BUCKET}/store/-5/1/{LEAF}")
        assert counts["failed"] == 1 and counts["touched"] == 0

    def test_client_is_keyed_by_the_store_kwargs(self, monkeypatch):
        seen = {}

        def fake_boto3_client(service, **kwargs):
            from unittest.mock import MagicMock

            seen["service"] = service
            seen.update(kwargs)
            client = MagicMock()
            client.get_paginator.return_value.paginate.return_value = []
            return client

        import boto3

        monkeypatch.setattr(boto3, "client", fake_boto3_client)
        lifecycle.touch_current_unit(
            f"s3://{self.BUCKET}/store/-5/1/{LEAF}",
            store_kwargs={
                "region": "eu-west-1",
                "endpoint_url": "https://r2.example",
                "credentials": {
                    "accessKeyId": "AK",
                    "secretAccessKey": "SK",
                    "sessionToken": "ST",
                },
            },
        )
        assert seen == {
            "service": "s3",
            "region_name": "eu-west-1",
            "endpoint_url": "https://r2.example",
            "aws_access_key_id": "AK",
            "aws_secret_access_key": "SK",
            "aws_session_token": "ST",
        }

    def test_local_unit_builds_no_client(self, tmp_path, monkeypatch):
        # A local store must never construct a boto3 client (no AWS reach).
        def explode(kw):
            raise AssertionError("S3 client built for a local store")

        monkeypatch.setattr(lifecycle, "_s3_client", explode)
        _node, leaf, _column = _make_unit(tmp_path)
        counts = lifecycle.touch_current_unit(str(leaf))
        assert counts["failed"] == 0


@pytest.mark.parametrize(
    "path,expected",
    [
        ("s3://b/k/x.zarr", ("b", "k/x.zarr")),
        ("s3://bucket-only", ("bucket-only", "")),
    ],
)
def test_split_s3(path, expected):
    assert lifecycle._split_s3(path) == expected
