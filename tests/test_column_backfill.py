"""Column backfill for pre-column stores (issue #520): the /1 -> /2 bridge.

Standing claims:

- **byte identity** — a column recomputed from a leaf's STORED bytes is
  byte-identical to the one the leaf worker wrote at build time, per leaf,
  per array (phase 1); and the whole ``/1 -> /2`` upgrade of a pyramid-OFF
  store lands a ladder byte-equal to a twin built pyramid-ON from identical
  inputs (phase 4);
- the backfill is a **sweep family**, so it rides the already-wired
  ``mode: "sweep"`` transport with partitioning and the lease for free;
- it is **declaration-driven**: a store still declaring ``/1``, declared-off,
  or ``class: none`` on every field refuses loudly and says re-declare first;
- **idempotent**: a second pass writes nothing, and a moved declaration or a
  re-run leaf is not current.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest
import zarr

from zagg.grids.morton import morton_word
from zagg.store import open_store

GENERATOR = pathlib.Path(__file__).parent.parent / "tools" / "generate_spec_fixtures.py"

#: Four leaves under one order-4 shard tree, spread over two base cells so the
#: ladder above them has something to k-way merge rather than relay.
SHARDS = ("11213", "11214", "11223", "21213")


def _generator():
    """The spec fixture generator, loaded as a module (test_column precedent)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("zagg_backfill_fixture_generator", GENERATOR)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _build_store(root, monkeypatch, *, pyramid=None, shards=SHARDS, kitchen_sink=False):
    """A real hive store: every shard through ``process_and_write_hive``.

    ``pyramid=None`` takes the issue #384 default flip (``/2`` at the grid's
    chunk order, columns written at birth); ``pyramid=False`` is the
    pre-column store this issue exists to upgrade. The leaf ARRAYS are
    identical either way — ``output.pyramid`` changes only which sibling
    artifacts the unit writes — which is what makes the phase 4 twins
    comparable.
    """
    import zagg.processing as processing
    from zagg import hive
    from zagg.grids import HealpixGrid

    gen = _generator()
    cfg = gen._config(kitchen_sink=kitchen_sink, pyramid=pyramid)
    grid = HealpixGrid(4, 6, layout="fullsphere", config=cfg, chunk_inner=5, sharded=True)
    root.mkdir(parents=True, exist_ok=True)
    hive.ensure_manifest(
        str(root), hive.build_manifest(grid, dataset={"short_name": "COL_TEST", "version": "1"})
    )
    for decimal in shards:
        shard = morton_word(decimal)
        by_chunk, _cells = gen._build_cells(grid, shard, kitchen_sink=kitchen_sink)
        inner = gen._fake_process_shard(grid, by_chunk, kitchen_sink=kitchen_sink)

        def fake(*args, _inner=inner, **kwargs):
            if kwargs.get("chunk_results") is None:
                kwargs["chunk_results"] = []
            df, meta = _inner(*args, **kwargs)
            meta["phase_timings"] = {"read": 0.0, "index": 0.0, "aggregate": 0.0}
            return df, meta

        monkeypatch.setattr(processing, "process_shard", fake)
        meta = hive.process_and_write_hive(
            shard, ["s3://fixture/a.h5"], grid, {}, str(root), cfg, store_kwargs={}
        )
        assert meta.get("error") is None, meta.get("error")
    return cfg, grid


def _column_arrays(path) -> dict:
    """``{(group, array): bytes}`` for every array under a column or overview."""
    store = open_store(str(path), read_only=True)
    root = zarr.open_group(store, path="", mode="r", zarr_format=3)
    out = {}
    for res, group in root.groups():
        for name, arr in group.arrays():
            values = arr[:]
            out[(res, name)] = (
                [bytes(p or b"") for p in values] if values.dtype == object else values.tobytes()
            )
    return out


def _objects(prefix: pathlib.Path) -> dict:
    """``{relative key: bytes}`` for every object under a store prefix."""
    return {
        str(f.relative_to(prefix)): f.read_bytes() for f in sorted(prefix.rglob("*")) if f.is_file()
    }


def _sans_timestamps(raw: bytes) -> dict:
    """A column root ``zarr.json``, minus the two keys a rewrite always moves."""
    meta = json.loads(raw)
    attrs = meta.get("attributes", {})
    attrs.get("zagg_column", {}).pop("generated_at", None)
    attrs.get("morton_hive_commit", {}).pop("written_at", None)
    return meta


def _column_path(root, decimal, window=None):
    from zagg.column import column_name
    from zagg.hive import shard_leaf_path

    leaf = shard_leaf_path(str(root), morton_word(decimal), window=window)
    return pathlib.Path(leaf).parent / column_name(window)


def _plan(root):
    from zagg.column import manifest_column_plan
    from zagg.hive import read_manifest

    return manifest_column_plan(read_manifest(str(root)))


# ---------------------------------------------------------------------------
# Phase 1: the recipe recomputed from stored bytes IS the build-time column.
# ---------------------------------------------------------------------------


class TestStoredLeafParity:
    @pytest.mark.parametrize("kitchen_sink", [False, True])
    def test_recomputed_column_is_byte_identical_per_leaf(
        self, tmp_path, monkeypatch, kitchen_sink
    ):
        from zagg.column import column_from_leaf, write_column
        from zagg.hive import read_commit, shard_leaf_path

        root = tmp_path / "on"
        _build_store(root, monkeypatch, kitchen_sink=kitchen_sink)
        plan = _plan(root)
        assert plan.resolutions == [5, 4]
        for decimal in SHARDS:
            built = _column_arrays(_column_path(root, decimal))
            assert built, f"the pyramid-ON build wrote no column at {decimal}"
            folded = column_from_leaf(
                str(root),
                morton_word(decimal),
                plan.fields,
                node_order=plan.node_order,
                cell_order=plan.cell_order,
                resolutions=plan.resolutions,
            )
            # Re-write it into a scratch store rather than comparing slabs in
            # memory: the pin is the stored BYTES, which is what a /2 reader
            # and the staged sweep consume.
            scratch = tmp_path / f"scratch-{decimal}"
            leaf = shard_leaf_path(str(root), morton_word(decimal))
            stamp = read_commit(open_store(leaf, read_only=True))
            write_column(
                str(scratch),
                morton_word(decimal),
                folded,
                plan.fields,
                node_order=plan.node_order,
                cell_order=plan.cell_order,
                granule_count=stamp["granule_count"],
            )
            recomputed = _column_path(scratch, decimal)
            assert _column_arrays(recomputed) == built
            # Object-level identity: every object under the prefix is byte-equal
            # except the root ``zarr.json``, which carries the two provenance
            # timestamps a rewrite always moves (spec §4.6).
            a, b = _objects(_column_path(root, decimal)), _objects(recomputed)
            assert set(a) == set(b)
            for key in a:
                if key == "zarr.json":
                    assert _sans_timestamps(a[key]) == _sans_timestamps(b[key])
                else:
                    assert a[key] == b[key], key

    def test_stored_slabs_match_the_staged_sink(self, tmp_path, monkeypatch):
        """The read-back adapter returns the writer's own in-memory values."""
        from zagg.column import stored_leaf_slabs
        from zagg.hive import shard_leaf_path

        captured: dict = {}
        import zagg.column as column_mod

        real = column_mod.leaf_slabs

        def spy(staged, fields, **kwargs):
            out = real(staged, fields, **kwargs)
            captured.update(out)
            return out

        monkeypatch.setattr(column_mod, "leaf_slabs", spy)
        root = tmp_path / "on"
        _build_store(root, monkeypatch, shards=SHARDS[:1], kitchen_sink=True)
        assert captured, "the worker seam never folded a column"
        plan = _plan(root)
        stored = stored_leaf_slabs(
            shard_leaf_path(str(root), morton_word(SHARDS[0])),
            plan.fields,
            cell_order=plan.cell_order,
            n_cells=4 ** (plan.cell_order - plan.node_order),
        )
        assert set(stored) == set(captured)
        for name, slab in stored.items():
            want = captured[name]
            if slab.dtype == object:
                assert [bytes(p or b"") for p in slab] == [bytes(p or b"") for p in want]
            else:
                assert np.array_equal(slab, want, equal_nan=slab.dtype.kind == "f")

    def test_mixed_order_leaf_refuses(self, tmp_path, monkeypatch):
        from zagg.column import stored_leaf_slabs
        from zagg.hive import shard_leaf_path

        root = tmp_path / "on"
        _build_store(root, monkeypatch, shards=SHARDS[:1])
        plan = _plan(root)
        with pytest.raises(ValueError, match="mixed-order source leaves"):
            stored_leaf_slabs(
                shard_leaf_path(str(root), morton_word(SHARDS[0])),
                plan.fields,
                cell_order=plan.cell_order,
                n_cells=17,
            )

    def test_absent_declared_field_reads_as_fill(self, tmp_path, monkeypatch):
        """Schema evolution: a field the leaf predates folds as the sink's fill."""
        from zagg.column import stored_leaf_slabs
        from zagg.hive import shard_leaf_path

        root = tmp_path / "on"
        _build_store(root, monkeypatch, shards=SHARDS[:1])
        plan = _plan(root)
        fields = {
            **plan.fields,
            "later": {"class": "exact", "method": "sum", "dtype": "int32", "fill_value": 0},
        }
        stored = stored_leaf_slabs(
            shard_leaf_path(str(root), morton_word(SHARDS[0])),
            fields,
            cell_order=plan.cell_order,
            n_cells=4 ** (plan.cell_order - plan.node_order),
        )
        assert stored["later"].tolist() == [0] * 4 ** (plan.cell_order - plan.node_order)

    def test_mismatched_weights_declaration_refuses(self, tmp_path, monkeypatch):
        from zagg.column import stored_leaf_slabs
        from zagg.hive import shard_leaf_path

        root = tmp_path / "on"
        _build_store(root, monkeypatch, shards=SHARDS[:1])
        plan = _plan(root)
        fields = {n: dict(m) for n, m in plan.fields.items()}
        fields["h_tdigest"]["weights"] = "flux"
        with pytest.raises(ValueError, match="weights declaration"):
            stored_leaf_slabs(
                shard_leaf_path(str(root), morton_word(SHARDS[0])),
                fields,
                cell_order=plan.cell_order,
                n_cells=4 ** (plan.cell_order - plan.node_order),
            )
