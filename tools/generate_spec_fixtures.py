"""Generate the committed spec-conformance fixtures with zagg's REAL writers.

The `docs/specification.md` §7 fixtures (issue #340), under
``tests/data/spec/``: two tiny hive stores, each one shard leaf written by
the production write path — ``hive.ensure_manifest`` +
``hive.process_and_write_hive`` (leaf template, sharded dense + ragged
writes, coverage sidecar, commit stamp) — plus one MANIFEST-ONLY pyramid
declaration written by the production declaration paths (issue #382, no
leaf beneath it). Each carries a committed ``*.expected.json`` recording the
decoded values and the §5 O11 content hashes.
``tests/test_spec_conformance.py`` asserts
the fixtures against the shipping readers AND against spec-text-only
decoders, so the spec, the fixtures, and the reader cannot drift apart
silently. moczarr vendors the same fixtures for its parity gates
(espg/moczarr#19/#20).

Run from a zagg checkout::

    uv run python tools/generate_spec_fixtures.py

Geometry (both leaf fixtures): parent order 4 / chunk order 5 / cell order 6 —
16 cells per shard, K = 4 inner chunks of 4 cells, `sharded` (the D17 hive
default), deliberately tiny. Chunk ordinal 2 is left EMPTY to pin the
shard-index absence sentinel (§1.5); populated chunks carry empty cells to
pin the ``b""`` fill (§1.1). Observations are synthetic and deterministic
(seeded rng); regeneration reproduces the same logical values, though stamp
timestamps and compressed bytes may differ across zstd versions — the
conformance tests assert decoded values, never object bytes.

- ``minimal/`` — one UNLOCATED digest field (`h_tdigest`) + `count`.
- ``kitchen_sink/`` — located signal/noise strata + `composition` + `count`
  (the `atl03_tdigest_strata_healpix.yaml` field shapes), including a
  single-photon cell that packs the §3.1 golden word `0xFF000000FF0000FF`
  and a noise-only cell whose signal payload is the empty ``(0, 2)`` array.
- ``column/`` — the ``minimal`` inputs plus an explicit
  ``output.pyramid.overviews: 5`` knob, so the SAME worker invocation also
  writes the §4.6 leaf column (issue #383): ``all.pyramid.zarr`` beside the
  leaf, groups {5, 4} (declared base + the node-order member), `role:
  column` + ``zagg_column`` attrs, its own commit stamp and O11 record.
- ``pyramid/`` — MANIFEST ONLY: the §4.5 ``zagg-pyramid/2`` declaration
  (issue #382, collapsed grammar), produced by the production declaration
  paths end to end (``/1`` template -> production sweep bookkeeping ->
  ``declare_pyramid`` retrofit to ``/2``). Its grid is shard order 3 (see
  :data:`PYRAMID_GRID`) so one manifest carries every ``/2`` reading a
  decoder must tell apart: a multi-resolution leaf entry, the fixed
  every-order ladder rooted at node 0, the #376 fold keys, and the
  preserved ``/1``-era ``materialized`` actuals — plus, through the
  ``pyramid.expected.json`` record of the raw config knob, the leaf-list
  form of the declaration the expansion was derived from. No store beneath
  it on purpose — the block is a template-time artifact, decodable from
  ``morton_hive.json`` alone; the ``/2`` artifacts a fleet writes are the
  ``column/`` fixture's job (issue #383 — sweep-side levels are issue #384).

STALE BY DESIGN: ``minimal/`` and ``kitchen_sink/`` were committed before
issue #382 and their ``morton_hive.json`` still carries the pre-#382
``pyramid`` block (``{"orders": [], "aggregation": {}}``). Running this
script refreshes them to the current block shape — real churn in files no
test asserts. When regenerating for one fixture only, ``git checkout --``
the others; a deliberate refresh of the older-era manifests is its own
commit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

#: Order-4 shard both fixtures write (decimal morton key; northern base 1).
SHARD_KEY = "11213"
#: Chunk ordinal (0..3) deliberately left empty — the §1.5 sentinel pin.
EMPTY_CHUNK = 2
#: t-digest compression budget: small enough that the 300-obs cell merges
#: observations into weight>1 centroids (exercising §2.2 common ancestors).
DELTA = 16
#: The §3.1 golden-word photon: per-surface confidences at threshold 2 pack
#: lanes [255, 0, 0, 255, 0, 0, 0, 255] = 0xFF000000FF0000FF.
GOLDEN_CONF = (4, -1, 0, 3, 1)
#: The ``pyramid/`` fixture's ``output.pyramid`` knob (§4.5, issue #382,
#: collapsed grammar): leaf cell resolutions only. Two members exercise the
#: multi-resolution leaf entry; everything above the shard is the fixed
#: every-order ladder, expanded by the production path into the manifest.
PYRAMID_KNOB = {"overviews": [5, 4]}
#: The pyramid fixture's grid: shard order 3 (not the leaf fixtures' 4) so
#: the leaf window (parent_order, child_order) = (3, 6) has TWO interior
#: resolutions — a multi-member leaf entry is impossible on the 4/6 window.
PYRAMID_GRID = {
    "type": "healpix",
    "parent_order": 3,
    "child_order": 6,
    "chunk_inner": 5,
    "sharded": True,
}
#: Fields the ``pyramid/`` fixture declares beyond the ``minimal`` pair, so
#: one manifest carries every composability class: exact (``count`` +
#: ``h_min``), approximate (``h_tdigest``), and ``none`` (``h_mean`` — no
#: exact merge law, D24).
PYRAMID_EXTRA_VARIABLES = {
    "h_min": {"function": "nanmin", "source": "h", "dtype": "float32", "fill_value": 0},
    "h_mean": {"function": "mean", "source": "h", "dtype": "float32", "fill_value": 0},
}
#: The ``/1``-era sweep actuals the retrofit must preserve (§4.5): the
#: ``{order: fold_source}`` shape the production bookkeeping writer takes.
PYRAMID_V1_ACTUALS = {1: "leaves", 0: "cascade"}


def _cell_photons(rng, n, *, signal_frac=0.6):
    """Synthetic photons: heights + 5 confidence columns, ``n`` rows."""
    h = np.round(rng.normal(30.0, 5.0, n), 3).astype(np.float64)
    conf = np.full((n, 5), -1, dtype=np.int64)
    n_sig = int(round(n * signal_frac))
    for i in range(n):
        if i < n_sig:  # signal: strongest confidence 2..4 on 1-2 surfaces
            surfaces = rng.choice(5, size=int(rng.integers(1, 3)), replace=False)
            conf[i, surfaces] = rng.integers(2, 5)
        else:  # noise: everything below threshold
            conf[i] = rng.integers(-2, 2, 5)
    return h, conf


def _point_words(grid, cell_word, n, rng):
    """Order-29 point-kind words jittered around the cell's center."""
    from mortie import MortonIndexArray

    from zagg.grids.morton import morton_words

    lat, lon = grid.cell_centers(np.array([cell_word], dtype=np.uint64))
    lat0, lon0 = float(np.asarray(lat).ravel()[0]), float(np.asarray(lon).ravel()[0])
    lats = np.clip(lat0 + rng.uniform(-1e-5, 1e-5, n), -89.9, 89.9)
    lons = lon0 + rng.uniform(-1e-5, 1e-5, n)
    return morton_words(MortonIndexArray.from_latlon(lats, lons, points=True))


def _config(kitchen_sink: bool, pyramid: dict | None = None):
    from zagg.config import PipelineConfig

    variables: dict = {
        "count": {"function": "len", "source": "h", "dtype": "int32", "fill_value": 0}
    }
    if kitchen_sink:
        for stratum in ("signal", "noise"):
            variables[f"h_tdigest_{stratum}"] = {
                "kind": "ragged",
                "function": "zagg.stats.tdigest.build_tdigest_where",
                "source": "h",
                "location": "leaf_id",
                "inner_shape": [2],
                "dtype": "float32",
                "fill_value": 0,
                "params": {"delta": DELTA},
                "attrs": {"stratum": stratum, "signal_threshold": 2},
            }
        variables["composition"] = {
            "function": "zagg.stats.composition.pack_composition",
            "source": "h",
            "dtype": "uint64",
            "fill_value": 0,
            "params": {"threshold": 2},
            "attrs": {
                "composition": {
                    "spec": "zagg-composition/1",
                    "lanes": [
                        "land",
                        "ocean",
                        "sea_ice",
                        "land_ice",
                        "inland_water",
                        "low",
                        "med",
                        "high",
                    ],
                    "of": "h_tdigest_signal",
                    "threshold": 2,
                }
            },
        }
    else:
        variables["h_tdigest"] = {
            "kind": "ragged",
            "function": "zagg.stats.tdigest.build_tdigest",
            "source": "h",
            "inner_shape": [2],
            "dtype": "float32",
            "fill_value": 0,
            "params": {"delta": DELTA},
        }
    output: dict = {
        "store_layout": "hive",
        "grid": {
            "type": "healpix",
            "parent_order": 4,
            "child_order": 6,
            "chunk_inner": 5,
            "sharded": True,
        },
    }
    if pyramid is not None:
        output["pyramid"] = pyramid
    return PipelineConfig(
        data_source={"groups": ["g"]},
        aggregation={
            "coordinates": {"morton": {"dtype": "uint64", "fill_value": 0}},
            "variables": variables,
        },
        output=output,
    )


def _build_cells(grid, shard, kitchen_sink: bool):
    """Per-cell synthetic inputs and their expected decoded values.

    Returns ``(by_chunk, expected_cells)`` where ``by_chunk`` maps a chunk
    ordinal to ``{local_cell: field_values}`` ready for the write path.
    """
    from zagg.stats.composition import pack_composition
    from zagg.stats.tdigest import build_tdigest, build_tdigest_where

    rng = np.random.default_rng(340)
    children = grid.children(shard)
    # (chunk ordinal, local cell, n obs, kind) — kind only used kitchen-sink.
    plan = [
        (0, 0, 40, "mixed"),
        (0, 2, 1, "golden"),
        (1, 1, 5, "noise_only"),
        (3, 3, 300, "mixed"),
    ]
    by_chunk: dict = {}
    expected = []
    for chunk, local, n, kind in plan:
        cell_index = chunk * grid.cells_per_chunk + local
        cell_word = int(children[cell_index])
        h, conf = _cell_photons(rng, n, signal_frac={"mixed": 0.6}.get(kind, 0.0))
        if kind == "golden":
            conf = np.array([GOLDEN_CONF], dtype=np.int64)
        if kind == "noise_only":
            conf[:] = rng.integers(-2, 2, conf.shape)
        record: dict = {"index": cell_index, "morton": str(cell_word), "count": n}
        fields: dict = {"count": n}
        if kitchen_sink:
            words = _point_words(grid, cell_word, n, rng)
            signal = (conf >= 2).any(axis=1)
            for stratum, mask in (("signal", signal), ("noise", ~signal)):
                digest, locs = build_tdigest_where(h, DELTA, where=mask, locations=words)
                fields[f"h_tdigest_{stratum}"] = (digest, locs)
                record[f"h_tdigest_{stratum}"] = [[float(m), float(w)] for m, w in digest]
                record[f"h_tdigest_{stratum}_locations"] = [str(int(w)) for w in locs]
            word = pack_composition(
                h,
                conf_land=conf[:, 0],
                conf_ocean=conf[:, 1],
                conf_sea_ice=conf[:, 2],
                conf_land_ice=conf[:, 3],
                conf_inland_water=conf[:, 4],
                threshold=2,
            )
            fields["composition"] = word
            record["composition"] = str(word)
            record["n_signal"] = int(signal.sum())
        else:
            digest = build_tdigest(h, DELTA)
            fields["h_tdigest"] = digest
            record["h_tdigest"] = [[float(m), float(w)] for m, w in digest]
        by_chunk.setdefault(chunk, {})[local] = fields
        expected.append(record)
    return by_chunk, expected


def _fake_process_shard(grid, by_chunk, kitchen_sink: bool):
    """A ``process_shard`` stand-in feeding the REAL sharded leaf write path."""

    def fake(g, shard_key, urls, **kwargs):
        occupied = []
        for ordinal, (block, children) in enumerate(grid.iter_chunks(int(shard_key))):
            cells = by_chunk.get(ordinal, {})
            if not cells:
                kwargs["chunk_results"].append((block, pd.DataFrame(), {}))
                continue
            n = grid.cells_per_chunk
            df = pd.DataFrame({"morton": np.asarray(children, dtype=np.uint64)})
            df["count"] = np.array(
                [cells.get(i, {}).get("count", 0) for i in range(n)], dtype=np.int32
            )
            ragged: dict = {}
            if kitchen_sink:
                df["composition"] = np.array(
                    [cells.get(i, {}).get("composition", 0) for i in range(n)],
                    dtype=np.uint64,
                )
                for field in ("h_tdigest_signal", "h_tdigest_noise"):
                    ids = sorted(i for i, f in cells.items() if field in f)
                    ragged[field] = (
                        [cells[i][field][0] for i in ids],
                        ids,
                        [cells[i][field][1] for i in ids],
                    )
            else:
                ids = sorted(cells)
                ragged["h_tdigest"] = ([cells[i]["h_tdigest"] for i in ids], ids)
            kwargs["chunk_results"].append((block, df, ragged))
            occupied.extend(int(children[i]) for i in sorted(cells))
        kwargs["occupied_out"].append(np.asarray(occupied, dtype=np.uint64))
        return pd.DataFrame(), {
            "shard_key": int(shard_key),
            "cells_with_data": len(occupied),
            "total_obs": 346,
            "granule_count": 1,
            "files_processed": 1,
            "duration_s": 0.0,
            "error": None,
        }

    return fake


def _element_bytes(element) -> bytes:
    """One vlen cell's payload bytes, per the §5.2 normalization.

    ``None`` (an unwritten cell may decode as ``None``, not ``b""``) is
    zero-length; bytes are as-is; a typed ``/2`` ndarray cell normalizes to
    C-contiguous little-endian bytes. Anything else RAISES — a silently wrong
    digest is worse than no digest.
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
    raise ValueError(f"vlen element of type {type(element).__name__} has no O11 byte recipe")


def _o11_hashes(leaf_path: str) -> dict:
    """The §5 O11 recipe over every array beneath the leaf root."""
    import zarr

    group = zarr.open_group(zarr.storage.LocalStore(leaf_path), mode="r", zarr_format=3)
    hashes: dict[str, str] = {}
    for key, node in group.members(max_depth=None):
        if not isinstance(node, zarr.Array):
            continue
        values = np.ascontiguousarray(node[...])
        if values.dtype.kind == "O":  # vlen: length-prefixed payloads, C order
            digest = hashlib.sha256()
            for element in values.ravel(order="C"):
                payload = _element_bytes(element)
                digest.update(len(payload).to_bytes(8, "little"))
                digest.update(payload)
            hashes[key] = digest.hexdigest()
            continue
        if values.dtype.byteorder == ">":
            values = values.astype(values.dtype.newbyteorder("<"))
        hashes[key] = hashlib.sha256(values.tobytes()).hexdigest()
    combined = hashlib.sha256("\n".join(sorted(hashes.values())).encode()).hexdigest()
    # Key-sorted so regeneration is diff-clean: ``group.members()`` yields
    # concurrently, so insertion order varies per run. ``combined`` is immune
    # either way (it sorts the DIGESTS), which is a real robustness property
    # of the recipe — but the recorded map would otherwise churn every run.
    return {"arrays": dict(sorted(hashes.items())), "combined": combined}


def build(out: Path, kitchen_sink: bool, pyramid: dict | None = None) -> None:
    import zagg.processing as processing
    from zagg import hive
    from zagg.grids import HealpixGrid
    from zagg.grids.morton import morton_word

    cfg = _config(kitchen_sink, pyramid=pyramid)
    grid = HealpixGrid(4, 6, layout="fullsphere", config=cfg, chunk_inner=5, sharded=True)
    shard = morton_word(SHARD_KEY)
    by_chunk, expected_cells = _build_cells(grid, shard, kitchen_sink)
    assert EMPTY_CHUNK not in by_chunk

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    root = str(out)
    hive.ensure_manifest(
        root,
        hive.build_manifest(grid, dataset={"short_name": "SPEC_FIXTURE", "version": "1"}),
    )
    original = processing.process_shard
    processing.process_shard = _fake_process_shard(grid, by_chunk, kitchen_sink)
    try:
        meta = hive.process_and_write_hive(
            shard,
            ["s3://fixture/a.h5"],
            grid,
            {},
            root,
            cfg,
            store_kwargs={},
        )
    finally:
        processing.process_shard = original
    assert meta.get("error") is None, meta

    leaf_rel = hive.shard_leaf_path("", shard).lstrip("/")
    expected = {
        "shard": SHARD_KEY,
        "leaf": leaf_rel,
        "group": grid.group_path,
        "shard_order": 4,
        "chunk_order": 5,
        "cell_order": 6,
        "cells_per_chunk": grid.cells_per_chunk,
        "chunks_per_shard": grid.chunks_per_shard,
        "empty_chunk": EMPTY_CHUNK,
        "delta": DELTA,
        "cells": expected_cells,
        "content_hashes": _o11_hashes(str(out / leaf_rel)),
    }
    if pyramid is not None:
        # The §4.6 column fixture (issue #383): the SAME worker invocation
        # wrote the leaf's column; record the raw knob, the decoded groups,
        # and the column's own O11 hashes so a spec-text decoder can be
        # asserted against committed bytes.
        # ``meta["leaf_column"]`` is the seam's own key (issue #388 renamed it
        # from the SQL-reserved ``column``); the fixture's ``column`` block
        # below is the fixture schema's name and stays as published.
        assert meta.get("leaf_column"), meta
        expected["declared"] = pyramid
        expected["column"] = _column_expected(
            out / leaf_rel.rsplit("/", 1)[0] / meta["leaf_column"], meta["leaf_column"]
        )
    expected_path = out.parent / f"{out.name}.expected.json"
    expected_path.write_text(json.dumps(expected, indent=1) + "\n")
    print(f"{out.name}: leaf {leaf_rel}, {len(expected_cells)} populated cells")


def _column_expected(column_dir: Path, basename: str) -> dict:
    """The committed §4.6 column record: attrs, decoded groups, O11 hashes."""
    import zarr

    store = zarr.storage.LocalStore(str(column_dir))
    attrs = dict(zarr.open_group(store, mode="r", zarr_format=3).attrs)
    fields = attrs["zagg_column"]["fields"]
    groups: dict = {}
    for res in sorted(attrs["zagg_column"]["groups"], key=int, reverse=True):
        group = zarr.open_group(store, path=res, mode="r", zarr_format=3)
        record: dict = {"morton": [str(w) for w in group["morton"][:]]}
        for name, meta in fields.items():
            # Element typing comes from the attrs block (§4.6 carries dtype +
            # inner_shape for exactly this), never hardcoded: a kitchen-sink
            # column's digest field need not be float32/(-1, 2), and its
            # ``none``-class fields are absent from the artifact entirely.
            if meta["class"] == "approximate":
                dtype = np.dtype(meta["dtype"]).newbyteorder("<")
                inner = tuple(meta["inner_shape"])
                record[name] = [
                    np.frombuffer(bytes(p), dtype).reshape((-1, *inner)).tolist()
                    for p in group[name][:]
                ]
            elif meta["class"] == "exact":
                record[name] = [v.item() for v in group[name][:]]
        groups[res] = record
    return {
        "object": basename,
        "role": attrs["role"],
        "zagg_column": attrs["zagg_column"],
        "commit": {
            k: v for k, v in attrs["morton_hive_commit"].items() if k != "written_at"
        },  # timestamps move on regeneration; the spec pins fields, not clocks
        "groups": groups,
        "content_hashes": _o11_hashes(str(column_dir)),
    }


def build_pyramid(out: Path) -> None:
    """The manifest-only fixture: the §4.5 ``zagg-pyramid/2`` declaration.

    Every byte of the committed manifest comes from a production declaration
    path, in the order a real store would live it: templated under ``/1``
    (``hive.build_manifest`` — no ``overviews`` knob), given sweep actuals by
    the production bookkeeping writer
    (``sweep_overview._update_manifest_pyramid``, the function the real ``/1``
    sweep calls), then retrofitted to ``/2`` with ``declare_pyramid`` and the
    :data:`PYRAMID_KNOB` config — which must preserve those ``/1``-era
    actuals verbatim (§4.5). No leaf exists on purpose (the declaration is a
    template-time artifact; ``declare_pyramid``'s field probe skips loudly),
    so this fixture writes exactly one object: ``morton_hive.json``.

    The expectations are derived HERE from the generator's INPUTS
    (:data:`PYRAMID_KNOB` expanded by the §4.5 leaf-entry rule and the §4.4
    fixed-ladder law, slab lengths by the §4.4 rule), never read back out of
    the written manifest — the same discipline the leaf fixtures' cell
    values follow.
    """
    from zagg import hive
    from zagg.grids import HealpixGrid
    from zagg.sweep_overview import _update_manifest_pyramid, declare_pyramid

    cfg_v1 = _config(False)
    cfg_v2 = _config(False, pyramid=PYRAMID_KNOB)
    for cfg in (cfg_v1, cfg_v2):
        cfg.aggregation["variables"].update(PYRAMID_EXTRA_VARIABLES)
        cfg.output["grid"] = dict(PYRAMID_GRID)
    grid = HealpixGrid(3, 6, layout="fullsphere", config=cfg_v1, chunk_inner=5, sharded=True)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    hive.ensure_manifest(
        str(out),
        hive.build_manifest(grid, dataset={"short_name": "SPEC_FIXTURE", "version": "1"}),
    )
    assert _update_manifest_pyramid(str(out), dict(PYRAMID_V1_ACTUALS), {})
    summary = declare_pyramid(str(out), cfg_v2)
    assert summary["updated"] is True, summary
    # The fully expanded (node, cells) list, spelled from the KNOB by the
    # §4.5 leaf-entry rule plus the §4.4 fixed-ladder law (d = base - shard;
    # one member per order from shard - 1 down to 0), and §4.4's slab rule.
    s = PYRAMID_GRID["parent_order"]
    resolutions = list(PYRAMID_KNOB["overviews"])
    d = resolutions[-1] - s
    levels = [{"node": s, "cells": resolutions}] + [
        {"node": k, "cells": [k + d]} for k in range(s - 1, -1, -1)
    ]
    chunk = PYRAMID_GRID["chunk_inner"]
    expected = {
        "shard_order": s,
        "chunk_order": chunk,
        "cell_order": PYRAMID_GRID["child_order"],
        "declared": PYRAMID_KNOB,
        "overviews": levels,
        "slabs": [[4 ** (r - e["node"]) for r in e["cells"]] for e in levels],
        "fold_source": "cascade",
        "exact_levels": 1,
        "materialized": {
            "orders": sorted(PYRAMID_V1_ACTUALS),
            "fold_sources": {str(k): v for k, v in PYRAMID_V1_ACTUALS.items()},
        },
        "fields": {
            "count": "exact",
            "h_tdigest": "approximate",
            "h_min": "exact",
            "h_mean": "none",
        },
        # The §4.5 omitted-knob default for this geometry ([chunk_order] at
        # the leaf, then the same fixed ladder), pinned so zagg's derivation
        # cannot drift from the formula on the page.
        "default_overviews": [{"node": s, "cells": [chunk]}]
        + [{"node": k, "cells": [k + (chunk - s)]} for k in range(s - 1, -1, -1)],
    }
    (out.parent / f"{out.name}.expected.json").write_text(json.dumps(expected, indent=1) + "\n")
    print(f"{out.name}: manifest-only, {len(levels)} level entries, /1 actuals preserved")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "tests" / "data" / "spec",
    )
    args = parser.parse_args()
    build(args.out / "minimal", kitchen_sink=False)
    build(args.out / "kitchen_sink", kitchen_sink=True)
    build(args.out / "column", kitchen_sink=False, pyramid={"overviews": 5})
    build_pyramid(args.out / "pyramid")


if __name__ == "__main__":
    main()
