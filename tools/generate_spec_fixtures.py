"""Generate the committed spec-conformance fixtures with zagg's REAL writers.

The `docs/specification.md` §7 fixtures (issue #340): two tiny hive stores
under ``tests/data/spec/``, each one shard leaf written by the production
write path — ``hive.ensure_manifest`` + ``hive.process_and_write_hive``
(leaf template, sharded dense + ragged writes, coverage sidecar, commit
stamp) — plus a committed ``*.expected.json`` recording the decoded values
and the §5 O11 content hashes. ``tests/test_spec_conformance.py`` asserts
the fixtures against the shipping readers AND against spec-text-only
decoders, so the spec, the fixtures, and the reader cannot drift apart
silently. moczarr vendors the same fixtures for its parity gates
(espg/moczarr#19/#20).

Run from a zagg checkout::

    uv run python tools/generate_spec_fixtures.py

Geometry (both fixtures): parent order 4 / chunk order 5 / cell order 6 —
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


def _config(kitchen_sink: bool):
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
    return PipelineConfig(
        data_source={"groups": ["g"]},
        aggregation={
            "coordinates": {"morton": {"dtype": "uint64", "fill_value": 0}},
            "variables": variables,
        },
        output={
            "store_layout": "hive",
            "grid": {
                "type": "healpix",
                "parent_order": 4,
                "child_order": 6,
                "chunk_inner": 5,
                "sharded": True,
            },
        },
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
                payload = b"" if element is None else bytes(element)
                digest.update(len(payload).to_bytes(8, "little"))
                digest.update(payload)
            hashes[key] = digest.hexdigest()
            continue
        if values.dtype.byteorder == ">":
            values = values.astype(values.dtype.newbyteorder("<"))
        hashes[key] = hashlib.sha256(values.tobytes()).hexdigest()
    combined = hashlib.sha256("\n".join(sorted(hashes.values())).encode()).hexdigest()
    return {"arrays": hashes, "combined": combined}


def build(out: Path, kitchen_sink: bool) -> None:
    import zagg.processing as processing
    from zagg import hive
    from zagg.grids import HealpixGrid
    from zagg.grids.morton import morton_word

    cfg = _config(kitchen_sink)
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
    expected_path = out.parent / f"{out.name}.expected.json"
    expected_path.write_text(json.dumps(expected, indent=1) + "\n")
    print(f"{out.name}: leaf {leaf_rel}, {len(expected_cells)} populated cells")


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


if __name__ == "__main__":
    main()
