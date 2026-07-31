"""§5 O11 content-hash recipe port: parity + raise-gate tests (issue #342).

Phase-1 gates: byte-identical parity against the committed conformance
fixtures' PINNED ``content_hashes`` (recomputed from the fixture stores —
per-array AND combined), the §5.2 element→bytes normalization including its
"anything else → raise" gate, and an optional cross-check against the
moczarr reference implementation (``moczarr.stats``) when it is importable.
The recipe is not zagg's to adjust: a parity failure here is a spec/fixture
finding to raise on issue #342, never something to patch around.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import zarr
from zarr.storage import LocalStore

from zagg.content_hash import (
    combined_hash,
    content_hashes_record,
    hash_array,
    hash_arrays,
)

SPEC_DATA = Path(__file__).parent / "data" / "spec"
FIXTURES = ("minimal", "kitchen_sink")


def _expected(name: str) -> dict:
    return json.loads((SPEC_DATA / f"{name}.expected.json").read_text())


def _leaf_group(name: str, exp: dict):
    leaf = SPEC_DATA / name / exp["leaf"]
    return zarr.open_group(LocalStore(str(leaf)), mode="r", zarr_format=3)


class TestFixtureParity:
    """The pinned fixture hashes (spec §7) recomputed through the port."""

    @pytest.mark.parametrize("name", FIXTURES)
    def test_per_array_hashes_match_pinned(self, name):
        exp = _expected(name)
        assert hash_arrays(_leaf_group(name, exp)) == exp["content_hashes"]["arrays"]

    @pytest.mark.parametrize("name", FIXTURES)
    def test_combined_hash_matches_pinned(self, name):
        exp = _expected(name)
        hashes = hash_arrays(_leaf_group(name, exp))
        assert combined_hash(hashes) == exp["content_hashes"]["combined"]

    @pytest.mark.parametrize("name", FIXTURES)
    def test_record_shape_matches_pinned_and_is_key_sorted(self, name):
        exp = _expected(name)
        record = content_hashes_record(hash_arrays(_leaf_group(name, exp)))
        assert record == exp["content_hashes"]
        assert list(record["arrays"]) == sorted(record["arrays"])

    @pytest.mark.parametrize("name", FIXTURES)
    def test_moczarr_reference_agrees(self, name):
        """Optional cross-check against the reference implementation."""
        moczarr_stats = pytest.importorskip("moczarr.stats")
        exp = _expected(name)
        theirs = moczarr_stats.hash_arrays(str(SPEC_DATA / name), exp["leaf"])
        ours = hash_arrays(_leaf_group(name, exp))
        assert theirs == ours
        assert moczarr_stats.combined_hash(theirs) == combined_hash(ours)


class TestVlenRecipe:
    """§5.2 element→bytes normalization, one test per contract row."""

    def _obj(self, elements):
        values = np.empty(len(elements), dtype=object)
        values[:] = elements
        return values

    def test_length_prefix_is_injective(self):
        # Without the u64le prefix these two cell grids would collide.
        assert hash_array(self._obj([b"ab", b"c"])) != hash_array(self._obj([b"a", b"bc"]))

    def test_none_hashes_as_empty_payload(self):
        # An unwritten cell may decode as None, not b"" — identical by design.
        assert hash_array(self._obj([None, b"x"])) == hash_array(self._obj([b"", b"x"]))

    def test_str_is_utf8(self):
        assert hash_array(self._obj(["hé"])) == hash_array(self._obj(["hé".encode()]))

    def test_ndarray_element_is_le_c_order_bytes(self):
        # A typed /2 cell decodes to an ndarray whose LE C-order bytes are the
        # /1 cell's payload bytes (spec §6.2) — the same hash, no recipe change.
        payload = np.array([[1.5, 2.0], [3.0, 4.0]], dtype="<f4")
        assert hash_array(self._obj([payload])) == hash_array(self._obj([payload.tobytes()]))

    def test_big_endian_ndarray_element_is_byteswapped(self):
        le = np.array([1, 2, 3], dtype="<u8")
        assert hash_array(self._obj([le.astype(">u8")])) == hash_array(self._obj([le]))

    def test_unsupported_element_raises(self):
        # The "anything else → raise" gate: never hash a repr/pointer buffer.
        with pytest.raises(ValueError, match="no O11 byte recipe"):
            hash_array(self._obj([3.5]))


class TestDenseRecipe:
    def test_big_endian_array_hashes_as_little_endian(self):
        values = np.array([1, 2, 3], dtype="<u4")
        assert hash_array(values.astype(">u4")) == hash_array(values)

    def test_non_contiguous_hashes_as_c_order(self):
        values = np.arange(12, dtype="<f8").reshape(3, 4)
        assert hash_array(values.T) == hash_array(np.ascontiguousarray(values.T))

    def test_combined_is_order_immune_but_value_sensitive(self):
        hashes = {"a": "00", "b": "ff"}
        assert combined_hash(hashes) == combined_hash({"b": "00", "a": "ff"})
        assert combined_hash(hashes) != combined_hash({"a": "00", "b": "fe"})
