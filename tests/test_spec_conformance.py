"""Spec-conformance tests against the committed golden fixtures (issue #340).

``docs/specification.md`` §7: the fixtures under ``tests/data/spec/`` (two
tiny hive stores written by ``tools/generate_spec_fixtures.py`` through the
production write path) are part of the store contract. Every test here binds
one of the spec's normative claims to those committed bytes, two ways:

- **through the shipping readers** (``zagg.readers.tdigest_tensor`` +
  ``zagg.stats.composition``), so the reader cannot drift from the fixtures;
- **through spec-text-only decoders** (struct + zstd + hashlib — no zagg
  read path), so the spec's byte recipes (§1.4 wire framing, §1.5 shard
  index, §5 O11 hashes) are proven sufficient to decode the store without
  importing zagg — the moczarr acceptance criterion.

The expected values in ``*.expected.json`` were computed from the fixture
generator's INPUTS (the arrays handed to the writers), so writer, reader,
and spec text are pinned against each other through the committed bytes.
"""

import hashlib
import json
import struct
from pathlib import Path

import numpy as np
import pytest
import zarr
from numcodecs import Zstd
from zarr.storage import LocalStore

from zagg.readers.tdigest_tensor import read_cell, read_locations
from zagg.stats.composition import counts_from_composition, unpack_composition

SPEC_DATA = Path(__file__).parent / "data" / "spec"
FIXTURES = ("minimal", "kitchen_sink")
#: (fixture, ragged field, element dtype, inner shape) — every committed
#: ``zagg-ragged/1`` array, payload and located siblings alike.
RAGGED_ARRAYS = [
    ("minimal", "h_tdigest", "float32", (2,)),
    ("kitchen_sink", "h_tdigest_signal", "float32", (2,)),
    ("kitchen_sink", "h_tdigest_noise", "float32", (2,)),
    ("kitchen_sink", "h_tdigest_signal_locations", "uint64", ()),
    ("kitchen_sink", "h_tdigest_noise_locations", "uint64", ()),
]
SENTINEL = 2**64 - 1

#: FROZEN §5 combined digests, as literals — the anti-circularity pin (the
#: ``FROZEN_COMBINED_4111`` precedent, espg/moczarr ``ba804e6``). Both the
#: fixture generator and every recomputation here run the §5 recipe, so a
#: changed length-prefix width, joiner, hash function, or key set would agree
#: with itself and sail through. These literals are what such a change has to
#: get past — and §5 is the recipe moczarr adopts verbatim, so a silent zagg
#: change is a silent cross-implementation divergence.
FROZEN_COMBINED = {
    "minimal": "2f4ff37de621de05962ab720cec05fd643757977f1afbd0e859ca588a143b72e",
    "kitchen_sink": "57c413489b33fd82187072504a51b76dbb6455d357c4702d9bf92f77bcafab31",
}
#: FROZEN per-array digests, one per §5.2 element kind: a vlen digest payload
#: array, a vlen uint64 locations sibling, and two fixed-width arrays.
FROZEN_ARRAYS = {
    ("minimal", "6/h_tdigest"): "ba43774865b69f60aaf42875c9c4a72edaca6058bd9a4fb95760d43f203c9d4c",
    ("minimal", "6/morton"): "f6282635e373d534ef4d91166d306441049e7bbed3d7d2ec8add306f62274d06",
    (
        "kitchen_sink",
        "6/h_tdigest_signal_locations",
    ): "ffd8bcc3be76c43184fd9ce6ddef16e24dba6c5615e302d6a8ad3b3cb603734a",
    (
        "kitchen_sink",
        "6/composition",
    ): "04886a9dfb60c8a53de48202dc3d5ac694698864825426f084be961aa678acd5",
}


def _decode_shard(path: Path, k: int) -> dict:
    """``{chunk_ordinal: raw framed bytes}`` per the §1.5 index recipe."""
    blob = path.read_bytes()
    index = np.frombuffer(blob[-(16 * k + 4) : -4], dtype="<u8").reshape(k, 2)
    chunks = {}
    for ordinal, (offset, nbytes) in enumerate(index):
        if int(offset) == SENTINEL:
            assert int(nbytes) == SENTINEL
            continue
        chunks[ordinal] = bytes(Zstd().decode(blob[int(offset) : int(offset) + int(nbytes)]))
    return chunks


def _decode_framing(raw: bytes, n_cells: int) -> list:
    """The §1.4 recipe: u32 cell count, then u32 length + payload per cell."""
    (count,) = struct.unpack_from("<I", raw, 0)
    assert count == n_cells
    payloads, pos = [], 4
    for _ in range(count):
        (length,) = struct.unpack_from("<I", raw, pos)
        pos += 4
        payloads.append(raw[pos : pos + length])
        pos += length
    assert pos == len(raw)
    return payloads


def _expected(name):
    return json.loads((SPEC_DATA / f"{name}.expected.json").read_text())


def _leaf_dir(name, exp) -> Path:
    return SPEC_DATA / name / exp["leaf"]


def _leaf_store(name, exp):
    return LocalStore(str(_leaf_dir(name, exp)))


def _array_meta(name, exp, field) -> dict:
    """The array's raw ``zarr.json`` — unnormalized, straight off disk."""
    return json.loads((_leaf_dir(name, exp) / exp["group"] / field / "zarr.json").read_text())


def _digest_expectations(exp):
    """Yield ``(field, cell_index, expected (k, 2) float32 digest)``."""
    for cell in exp["cells"]:
        for field in cell:
            if field.startswith("h_tdigest") and not field.endswith("_locations"):
                want = np.array(cell[field], dtype=np.float32).reshape(-1, 2)
                yield field, cell["index"], want


class TestRaggedAttrs:
    """§1.2 — the self-describing ``ragged`` attrs block."""

    @pytest.mark.parametrize(("name", "field", "dtype", "inner"), RAGGED_ARRAYS)
    def test_spec_marker_and_element(self, name, field, dtype, inner):
        exp = _expected(name)
        block = _array_meta(name, exp, field)["attributes"]["ragged"]
        assert block["spec"] == "zagg-ragged/1"
        assert block["element"] == {"dtype": dtype, "shape": [-1, *inner]}

    def test_locations_binding_declared_not_named(self):
        exp = _expected("kitchen_sink")
        for stratum in ("signal", "noise"):
            payload = _array_meta("kitchen_sink", exp, f"h_tdigest_{stratum}")["attributes"]
            assert payload["ragged"]["locations"] == f"h_tdigest_{stratum}_locations"
            # Provenance attrs ride the payload array only (§1.2/§3.3)...
            assert payload["stratum"] == stratum
            assert payload["signal_threshold"] == 2
            sibling = _array_meta("kitchen_sink", exp, f"h_tdigest_{stratum}_locations")
            # ...and the sibling carries no user attrs and no locations key.
            assert set(sibling["attributes"]) == {"ragged"}
            assert "locations" not in sibling["attributes"]["ragged"]

    def test_unlocated_field_records_nothing(self):
        exp = _expected("minimal")
        assert "locations" not in _array_meta("minimal", exp, "h_tdigest")["attributes"]["ragged"]


class TestRaggedCodecChain:
    """§1.3/§1.5 — codec chain and the sharded storage geometry."""

    @pytest.mark.parametrize(("name", "field", "dtype", "inner"), RAGGED_ARRAYS)
    def test_sharded_vlen_zstd3_chain(self, name, field, dtype, inner):
        exp = _expected(name)
        meta = _array_meta(name, exp, field)
        assert meta["fill_value"] == ""
        assert meta["data_type"] in ("variable_length_bytes", "bytes")
        cells = exp["cells_per_chunk"] * exp["chunks_per_shard"]
        assert meta["chunk_grid"]["configuration"]["chunk_shape"] == [cells]
        (sharding,) = meta["codecs"]
        assert sharding["name"] == "sharding_indexed"
        cfg = sharding["configuration"]
        assert cfg["chunk_shape"] == [exp["cells_per_chunk"]]
        assert [c["name"] for c in cfg["codecs"]] == ["vlen-bytes", "zstd"]
        assert cfg["codecs"][1]["configuration"] == {"level": 3, "checksum": False}
        assert [c["name"] for c in cfg["index_codecs"]] == ["bytes", "crc32c"]
        assert cfg["index_location"] == "end"


class TestWireFraming:
    """§1.4/§1.5 — spec-text-only decode of the committed shard objects.

    No zagg read path: the shard index suffix, the zstd inner chunks, and
    the u32 vlen framing are parsed exactly as the spec describes, and the
    decoded payloads must equal the committed expectations.
    """

    @pytest.mark.parametrize(("name", "field", "dtype", "inner"), RAGGED_ARRAYS)
    def test_payloads_decode_per_spec(self, name, field, dtype, inner):
        exp = _expected(name)
        per_chunk = exp["cells_per_chunk"]
        chunks = _decode_shard(
            _leaf_dir(name, exp) / exp["group"] / field / "c" / "0", exp["chunks_per_shard"]
        )
        # The empty chunk is absent from the shard index (the §1.5 sentinel).
        assert exp["empty_chunk"] not in chunks
        decoded: dict[int, np.ndarray] = {}
        for ordinal, raw in chunks.items():
            for local, payload in enumerate(_decode_framing(raw, per_chunk)):
                cell = ordinal * per_chunk + local
                decoded[cell] = np.frombuffer(payload, dtype=f"<{np.dtype(dtype).str[1:]}")
                decoded[cell] = decoded[cell].reshape(-1, *inner)
        source = field.removesuffix("_locations")
        for cell in exp["cells"]:
            key = field if field.endswith("_locations") else source
            want = cell.get(key if key in cell else field)
            if want is None:
                continue
            got = decoded.get(cell["index"], np.empty((0, *inner), dtype=dtype))
            if dtype == "uint64":
                np.testing.assert_array_equal(got, np.array([int(w) for w in want], np.uint64))
            else:
                np.testing.assert_array_equal(
                    got, np.array(want, dtype=np.float32).reshape(-1, *inner)
                )

    def test_populated_chunks_only_no_ghost_payloads(self):
        # Cells outside the expected set decode as b"" (the fill) — §1.1.
        exp = _expected("minimal")
        chunks = _decode_shard(
            _leaf_dir("minimal", exp) / "6" / "h_tdigest" / "c" / "0", exp["chunks_per_shard"]
        )
        populated = {c["index"] for c in exp["cells"]}
        for ordinal, raw in chunks.items():
            for local, payload in enumerate(_decode_framing(raw, exp["cells_per_chunk"])):
                cell = ordinal * exp["cells_per_chunk"] + local
                assert (len(payload) > 0) == (cell in populated)


class TestDigestPayload:
    """§2 — centroid semantics through the shipping reader."""

    @pytest.mark.parametrize("name", FIXTURES)
    def test_read_cell_matches_expected(self, name):
        exp = _expected(name)
        store = _leaf_store(name, exp)
        for field, cell_index, want in _digest_expectations(exp):
            got = read_cell(store, f"{exp['group']}/{field}", cell_index)
            np.testing.assert_array_equal(got, want)
            assert got.dtype == np.float32

    @pytest.mark.parametrize("name", FIXTURES)
    def test_absent_cell_decodes_zero_length(self, name):
        exp = _expected(name)
        store = _leaf_store(name, exp)
        field = "h_tdigest" if name == "minimal" else "h_tdigest_signal"
        empty_cell = exp["empty_chunk"] * exp["cells_per_chunk"]
        assert read_cell(store, f"{exp['group']}/{field}", empty_cell).shape == (0, 2)

    @pytest.mark.parametrize("name", FIXTURES)
    def test_means_sorted_and_weights_are_exact_counts(self, name):
        # §2.1 over the recorded expectations; bound to the store through
        # test_read_cell_matches_expected above, which reads the same values
        # back with the shipping reader.
        exp = _expected(name)
        for cell in exp["cells"]:
            counts = {"h_tdigest": cell["count"]}
            if name == "kitchen_sink":
                counts = {
                    "h_tdigest_signal": cell["n_signal"],
                    "h_tdigest_noise": cell["count"] - cell["n_signal"],
                }
            for field, n in counts.items():
                digest = np.array(cell.get(field, []), dtype=np.float32).reshape(-1, 2)
                assert np.all(np.diff(digest[:, 0]) >= 0), f"{field} means not ascending"
                assert digest[:, 1].sum() == n, f"{field} weights != exact count"
                assert np.all(digest[:, 1] >= 1) or digest.size == 0

    def test_locations_row_aligned_with_payload(self):
        exp = _expected("kitchen_sink")
        store = _leaf_store("kitchen_sink", exp)
        seen = 0
        for stratum in ("signal", "noise"):
            field = f"{exp['group']}/h_tdigest_{stratum}"
            by_cell = {
                c["index"]: np.array(
                    [int(w) for w in c[f"h_tdigest_{stratum}_locations"]], dtype=np.uint64
                )
                for c in exp["cells"]
            }
            for cell in exp["cells"]:
                got = read_cell(store, f"{field}_locations", cell["index"])
                np.testing.assert_array_equal(got.reshape(-1), by_cell[cell["index"]])
                assert got.reshape(-1).shape[0] == len(cell[f"h_tdigest_{stratum}"])
                seen += 1
        assert seen == 2 * len(exp["cells"])

    def test_read_locations_binds_through_attrs(self):
        exp = _expected("kitchen_sink")
        store = _leaf_store("kitchen_sink", exp)
        rows = list(read_locations(store, f"{exp['group']}/h_tdigest_signal"))
        got = sorted(np.concatenate([locs for _w, _rc, locs in rows]).tolist())
        want = sorted(int(w) for c in exp["cells"] for w in c["h_tdigest_signal_locations"])
        assert got == want


class TestCoordinateAndDense:
    """§1.1 — the `morton` coordinate and the dense `count` field, off the store.

    The fixtures *record* `morton` and `count` per populated cell, and §7 tells
    an external reader to reproduce them; without these asserts a change to the
    coordinate (ordering, dtype, nested-vs-ring) or to a dense write would
    leave the recorded values wrong and this suite green — while moczarr, whose
    whole addressing path keys on `morton`, would break (review finding).
    """

    @staticmethod
    def _open(name, exp, field):
        return zarr.open_array(_leaf_store(name, exp), path=f"{exp['group']}/{field}", mode="r")

    @pytest.mark.parametrize("name", FIXTURES)
    def test_morton_coordinate_matches_expected(self, name):
        exp = _expected(name)
        words = [int(w) for w in np.asarray(self._open(name, exp, "morton")[:])]
        assert self._open(name, exp, "morton").dtype == np.uint64
        assert len(words) == exp["cells_per_chunk"] * exp["chunks_per_shard"]
        for cell in exp["cells"]:
            assert words[cell["index"]] == int(cell["morton"])

    @pytest.mark.parametrize("name", FIXTURES)
    def test_morton_ascends_per_written_chunk_and_is_fill_where_absent(self, name):
        # The coordinate array gets the same §1.5 sub-shard sparsity as every
        # other array: the empty inner chunk's slots hold the 0 fill, so a
        # reader MUST NOT assume `morton` is dense across the shard.
        exp = _expected(name)
        per_chunk = exp["cells_per_chunk"]
        words = [int(w) for w in np.asarray(self._open(name, exp, "morton")[:])]
        for ordinal in range(exp["chunks_per_shard"]):
            chunk = words[ordinal * per_chunk : (ordinal + 1) * per_chunk]
            if ordinal == exp["empty_chunk"]:
                assert chunk == [0] * per_chunk
            else:
                assert all(b > a for a, b in zip(chunk, chunk[1:], strict=False))

    @pytest.mark.parametrize("name", FIXTURES)
    def test_count_field_matches_expected(self, name):
        exp = _expected(name)
        arr = self._open(name, exp, "count")
        assert arr.dtype == np.int32
        counts = np.asarray(arr[:])
        by_cell = {c["index"]: c["count"] for c in exp["cells"]}
        for cell in range(counts.shape[0]):
            assert int(counts[cell]) == by_cell.get(cell, 0)


class TestComposition:
    """§3 — the packed composition word on the committed store."""

    def _array(self):
        exp = _expected("kitchen_sink")
        store = _leaf_store("kitchen_sink", exp)
        return exp, zarr.open_array(store, path=f"{exp['group']}/composition", mode="r")

    def test_attrs_block(self):
        exp, arr = self._array()
        assert arr.attrs["composition"] == {
            "spec": "zagg-composition/1",
            "lanes": ["land", "ocean", "sea_ice", "land_ice", "inland_water", "low", "med", "high"],
            "of": "h_tdigest_signal",
            "threshold": 2,
        }

    def test_attrs_block_binds_to_the_writer_constants(self):
        # §3.3: spec/lanes are writer-stamped, so the committed block must
        # equal the module constants — not just the literal above (which the
        # generator's own config could otherwise have dictated).
        from zagg.stats.composition import COMPOSITION_SPEC, LANES

        _exp, arr = self._array()
        block = arr.attrs["composition"]
        assert block["spec"] == COMPOSITION_SPEC
        assert block["lanes"] == list(LANES)

    def test_words_match_expected_and_empty_cells_zero(self):
        exp, arr = self._array()
        words = np.asarray(arr[:])
        by_cell = {c["index"]: int(c["composition"]) for c in exp["cells"]}
        for cell in range(words.shape[0]):
            assert int(words[cell]) == by_cell.get(cell, 0)

    def test_golden_word_single_photon(self):
        # §3.1: one signal photon, confs (4, -1, 0, 3, 1) at threshold 2.
        exp, arr = self._array()
        (golden,) = [c for c in exp["cells"] if c["n_signal"] == 1 and c["count"] == 1]
        word = int(arr[golden["index"]])
        assert word == 0xFF000000FF0000FF
        np.testing.assert_array_equal(unpack_composition(word), [255, 0, 0, 255, 0, 0, 0, 255])
        np.testing.assert_array_equal(counts_from_composition(word, 1), [1, 0, 0, 1, 0, 0, 0, 1])

    def test_n_signal_is_the_of_digests_total_weight(self):
        # §3.3: the ``of`` linkage — count recovery keys on the signal
        # digest's summed weights, which equal the recorded n_signal.
        exp, _arr = self._array()
        store = _leaf_store("kitchen_sink", exp)
        for cell in exp["cells"]:
            digest = read_cell(store, f"{exp['group']}/h_tdigest_signal", cell["index"])
            assert int(digest[:, 1].sum()) == cell["n_signal"]


class TestContentHashes:
    """§5 — the O11 recipe, reimplemented from spec text only.

    Three legs, because the recipe is what moczarr adopts verbatim: the
    decoded-value hasher below (recomputed over the written store), an
    independent recomputation of the vlen digests from the shard objects
    alone, and the :data:`FROZEN_COMBINED` / :data:`FROZEN_ARRAYS` literals —
    the only leg that can fail when a recipe change is made consistently on
    both sides.
    """

    @staticmethod
    def _element_bytes(element) -> bytes:
        """The §5.2 element→bytes normalization, from the spec table.

        ``None`` (an unwritten vlen cell may decode as ``None``) is
        zero-length; a `/1` cell is bytes as-is; a typed `/2` ndarray cell is
        C-contiguous little-endian bytes at its dtype; anything else raises
        rather than hashing something wrong.
        """
        if element is None:
            return b""
        if isinstance(element, bytes | bytearray | memoryview):
            return bytes(element)
        if isinstance(element, str):
            return element.encode()
        if isinstance(element, np.ndarray):
            values = np.ascontiguousarray(element)
            if values.dtype.byteorder == ">":
                values = values.astype(values.dtype.newbyteorder("<"))
            return values.tobytes()
        raise ValueError(f"vlen element of type {type(element).__name__} has no O11 recipe")

    @staticmethod
    def _hash_leaf(leaf: Path) -> dict:
        group = zarr.open_group(LocalStore(str(leaf)), mode="r", zarr_format=3)
        hashes = {}
        for key, node in group.members(max_depth=None):
            if not isinstance(node, zarr.Array):
                continue
            values = np.ascontiguousarray(node[...])
            if values.dtype.kind == "O":
                digest = hashlib.sha256()
                for element in values.ravel(order="C"):
                    payload = TestContentHashes._element_bytes(element)
                    digest.update(len(payload).to_bytes(8, "little"))
                    digest.update(payload)
                hashes[key] = digest.hexdigest()
                continue
            if values.dtype.byteorder == ">":
                values = values.astype(values.dtype.newbyteorder("<"))
            hashes[key] = hashlib.sha256(values.tobytes()).hexdigest()
        return hashes

    @staticmethod
    def _vlen_digest_from_shard_bytes(name, exp, field) -> str:
        """The §5.2 vlen digest with NO zarr and no shared helper.

        Payloads come from the §1.4/§1.5 byte recipes (``_decode_shard`` /
        ``_decode_framing``), so this is a second, independent implementation
        of the recipe — an absent inner chunk (the §1.5 sentinel) contributes
        its cells' zero lengths, which is what makes the digest cover the
        cell *grid* rather than only the payloads.
        """
        chunks = _decode_shard(
            _leaf_dir(name, exp) / exp["group"] / field / "c" / "0", exp["chunks_per_shard"]
        )
        per_chunk = exp["cells_per_chunk"]
        digest = hashlib.sha256()
        for ordinal in range(exp["chunks_per_shard"]):
            raw = chunks.get(ordinal)
            payloads = _decode_framing(raw, per_chunk) if raw is not None else [b""] * per_chunk
            for payload in payloads:
                digest.update(len(payload).to_bytes(8, "little"))
                digest.update(payload)
        return digest.hexdigest()

    @pytest.mark.parametrize("name", FIXTURES)
    def test_per_array_hashes_match_golden(self, name):
        exp = _expected(name)
        assert self._hash_leaf(_leaf_dir(name, exp)) == exp["content_hashes"]["arrays"]

    @pytest.mark.parametrize(("name", "field", "dtype", "inner"), RAGGED_ARRAYS)
    def test_vlen_digest_from_shard_bytes_only(self, name, field, dtype, inner):
        # The §5.2 vlen recipe reproduced from the shard objects alone — the
        # decoded-value hasher above and the generator both go through zarr,
        # so this is the leg that proves the spec text suffices.
        exp = _expected(name)
        want = exp["content_hashes"]["arrays"][f"{exp['group']}/{field}"]
        assert self._vlen_digest_from_shard_bytes(name, exp, field) == want

    @pytest.mark.parametrize("name", FIXTURES)
    def test_combined_hash_recipe(self, name):
        exp = _expected(name)
        arrays = exp["content_hashes"]["arrays"]
        combined = hashlib.sha256("\n".join(sorted(arrays.values())).encode()).hexdigest()
        assert combined == exp["content_hashes"]["combined"]

    @pytest.mark.parametrize("name", FIXTURES)
    def test_recorded_arrays_map_is_key_sorted(self, name):
        # Regeneration must be diff-clean: member enumeration yields
        # concurrently, so the generator sorts the map before recording it.
        arrays = _expected(name)["content_hashes"]["arrays"]
        assert list(arrays) == sorted(arrays)

    def test_element_bytes_normalization(self):
        # §5.2's table: None ≡ b"", a typed /2 ndarray cell normalizes to
        # C-contiguous little-endian bytes, anything else raises.
        norm = TestContentHashes._element_bytes
        assert norm(None) == b""
        assert norm(bytearray(b"ab")) == b"ab"
        assert norm(memoryview(b"ab")) == b"ab"
        assert norm("ab") == b"ab"
        little = np.array([[1.5, 2.0]], dtype="<f4")
        assert norm(np.asfortranarray(little.astype(">f4"))) == little.tobytes()
        with pytest.raises(ValueError, match="no O11 recipe"):
            norm(3.5)

    @pytest.mark.parametrize("name", FIXTURES)
    def test_frozen_digests_pin_the_recipe(self, name):
        # Not self-certified: the literals are the pin. Every other assertion
        # here (and the generator's) runs the same recipe on both sides, so
        # only a frozen value catches a recipe change made consistently.
        exp = _expected(name)
        assert exp["content_hashes"]["combined"] == FROZEN_COMBINED[name]
        for (fixture, key), frozen in FROZEN_ARRAYS.items():
            if fixture == name:
                assert exp["content_hashes"]["arrays"][key] == frozen


class TestStoreEnvelope:
    """The fixture leaves are complete, stamped stores (D4; §4.3 debris rule)."""

    @pytest.mark.parametrize("name", FIXTURES)
    def test_leaf_is_stamped_and_unclassified(self, name):
        exp = _expected(name)
        attrs = json.loads((_leaf_dir(name, exp) / "zarr.json").read_text())["attributes"]
        assert attrs["morton_hive_commit"]["complete"] is True
        # Source leaves carry no role key — absence means source (§4.3).
        assert "role" not in attrs

    @pytest.mark.parametrize("name", FIXTURES)
    def test_manifest_spec_marker(self, name):
        manifest = json.loads((SPEC_DATA / name / "morton_hive.json").read_text())
        assert manifest["spec"] == "morton-hive/1"
