"""Spatial-fidelity goldens for the reader deinterleave — issue #336.

The tensor readers place a cell at the bit deinterleave of its chunk-local
nested rank (``readers._layout``), pinned to the normative convention in
mortie's ``docs/specification.md`` §8 (healpy ``pix2xyf`` face-local frame;
gridlook ``bit_combine(j, i)`` texture orientation: row = y, col = x). The
golden vectors are IMPORTED from mortie's merged test suite
(``mortie/tests/test_rank_xy.py``, espg/mortie#150) rather than re-derived,
so zagg cannot drift from the spec by construction.
"""

import numpy as np
import pytest
from test_readers import _KEY_A, _build_store, _sharded_store
from zarr.storage import MemoryStore

from zagg.grids.morton import morton_word
from zagg.readers._layout import rank_to_rowcol, rowcol_to_rank
from zagg.readers.tdigest_tensor import read_tensors

# Golden (rank, x, y) triples copied verbatim from mortie's merged test file
# mortie/tests/test_rank_xy.py (espg/mortie#150) — provenance: healpy 1.20.0
# `pix2xyf(2**depth, rank, nest=True)` (face 0), rng seed 149; the normative
# convention pinned by mortie docs/specification.md §8. Depth 6 = the 64x64
# inner chunk, depth 8 = the 256x256 shard.
GOLDEN_DEPTH6 = [
    (0, 0, 0),
    (1, 1, 0),
    (2, 0, 1),
    (3, 1, 1),
    (335, 27, 3),
    (907, 17, 27),
    (1069, 35, 6),
    (1265, 45, 12),
    (1873, 61, 16),
    (2048, 0, 32),
    (2135, 15, 33),
    (2511, 27, 43),
    (3024, 28, 56),
    (3369, 49, 38),
    (3592, 32, 50),
    (4095, 63, 63),
]
GOLDEN_DEPTH8 = [
    (0, 0, 0),
    (1, 1, 0),
    (2, 0, 1),
    (3, 1, 1),
    (3149, 43, 34),
    (13006, 74, 91),
    (30094, 242, 75),
    (32094, 254, 99),
    (32768, 0, 128),
    (35193, 29, 166),
    (36422, 42, 177),
    (45487, 83, 207),
    (58565, 171, 200),
    (61371, 181, 255),
    (65052, 230, 242),
    (65535, 255, 255),
]


class TestLayoutGoldens:
    """The helper pair matches the mortie spec §8 golden vectors exactly."""

    @pytest.mark.parametrize("depth,golden", [(6, GOLDEN_DEPTH6), (8, GOLDEN_DEPTH8)])
    def test_rank_to_rowcol_golden(self, depth, golden):
        ranks, xs, ys = (np.array(col, dtype=np.uint64) for col in zip(*golden))
        row, col = rank_to_rowcol(ranks, depth)
        # Orientation contract: row = y, col = x (gridlook bit_combine(j, i)).
        np.testing.assert_array_equal(row, ys)
        np.testing.assert_array_equal(col, xs)

    @pytest.mark.parametrize("depth,golden", [(6, GOLDEN_DEPTH6), (8, GOLDEN_DEPTH8)])
    def test_rowcol_to_rank_golden(self, depth, golden):
        ranks, xs, ys = (np.array(col, dtype=np.uint64) for col in zip(*golden))
        np.testing.assert_array_equal(rowcol_to_rank(ys, xs, depth), ranks)

    @pytest.mark.parametrize("depth", [1, 2, 6, 8, 13])
    def test_round_trip(self, depth):
        rng = np.random.default_rng(336 + depth)
        n = int(min(4**depth, 2048))
        ranks = rng.integers(0, 4**depth, size=n, dtype=np.uint64)
        row, col = rank_to_rowcol(ranks, depth)
        assert row.max() < 2**depth and col.max() < 2**depth
        np.testing.assert_array_equal(rowcol_to_rank(row, col, depth), ranks)
        row2, col2 = rank_to_rowcol(rowcol_to_rank(row, col, depth), depth)
        np.testing.assert_array_equal(row2, row)
        np.testing.assert_array_equal(col2, col)


def _tensor_for(cell_to_values, **kwargs):
    """One 64x64 chunk's float32 tensor from a real store write of _KEY_A."""
    store, _grid, _words = _build_store({_KEY_A: cell_to_values})
    out = list(read_tensors(store, "12/h_tdigest", dtype="float32", **kwargs))
    assert len(out) == 1
    tensor, morton = out[0]
    assert morton == morton_word(_KEY_A)
    return tensor


class TestTensorPlacement:
    """Golden-driven placement through the production write + read path."""

    def test_golden_ranks_land_at_golden_xy(self):
        # One distinguishable digest per golden rank: cell i holds 10*(i+1)
        # samples, so the per-cell tensor mass identifies which digest landed
        # where. All cells share one value range → one shared z-window.
        rng = np.random.default_rng(1)
        counts = {rank: 10 * (i + 1) for i, (rank, _x, _y) in enumerate(GOLDEN_DEPTH6)}
        cells = {rank: rng.uniform(10.0, 30.0, n) for rank, n in counts.items()}
        t = _tensor_for(cells, bottom=0.0, top=1.0)
        mass = t.sum(axis=2)
        for rank, x, y in GOLDEN_DEPTH6:
            assert mass[y, x] == pytest.approx(counts[rank], rel=0.01)
        # Exactly the golden positions are populated.
        rows, cols = np.nonzero(mass)
        assert {(int(r), int(c)) for r, c in zip(rows, cols)} == {
            (y, x) for _rank, x, y in GOLDEN_DEPTH6
        }

    def test_sphere_adjacent_cells_land_tensor_adjacent(self):
        """The case the row-major reshape fails (issue #336): ranks 0..3 are
        one nested quad — a 2x2 block of mutually adjacent cells on the
        sphere (mortie spec §8.1: child tuples order (x, y) = (0,0), (1,0),
        (0,1), (1,1)) — so they must land as a 2x2 block in the tensor.
        ``divmod(rank, side)`` strung them along row 0 as (0,0)..(0,3),
        tearing the quad's two sphere-adjacent rows 64 columns apart."""
        rng = np.random.default_rng(2)
        counts = {rank: 10 * (rank + 1) for rank in range(4)}
        cells = {rank: rng.uniform(10.0, 30.0, n) for rank, n in counts.items()}
        mass = _tensor_for(cells, bottom=0.0, top=1.0).sum(axis=2)
        rows, cols = np.nonzero(mass)
        assert {(int(r), int(c)) for r, c in zip(rows, cols)} == {
            (0, 0),
            (0, 1),
            (1, 0),
            (1, 1),
        }
        # And in the spec-§8.1 order: rank 1 east of rank 0, rank 2 north.
        for rank, (row, col) in [(0, (0, 0)), (1, (0, 1)), (2, (1, 0)), (3, (1, 1))]:
            assert mass[row, col] == pytest.approx(counts[rank], rel=0.01)

    def test_reported_rowcol_matches_tensor_position(self):
        """read_raw_values / read_locations report the SAME (row, col) the
        tensor places the cell at — one convention across the readers."""
        from conftest import point_words

        from zagg.readers.tdigest_tensor import read_locations, read_raw_values

        ranks = [rank for rank, _x, _y in GOLDEN_DEPTH6]
        vals = {rank: np.array([float(rank), float(rank) + 1.0]) for rank in ranks}
        locs = {rank: point_words(2, rank + 1) for rank in ranks}
        store, _grid, _words = _build_store({_KEY_A: vals}, located_locs={_KEY_A: locs})
        expected = {(y, x) for _rank, x, y in GOLDEN_DEPTH6}
        raw_positions = {rc for _m, rc, _v in read_raw_values(store, "12/h_tdigest")}
        loc_positions = {rc for _m, rc, _v in read_locations(store, "12/h_tdigest")}
        assert raw_positions == loc_positions == expected
        # Values identify the cell: rank recovered from (row, col) round-trips.
        for _m, (row, col), v in read_raw_values(store, "12/h_tdigest"):
            assert v[0] == float(rowcol_to_rank(row, col, 6))


class TestBlockAssembly:
    """``block_order=`` assembles inner-chunk tensors into one block tensor
    (issue #336 phase 3). Geometry: order-6 shard of K=16 order-8 chunks
    (16×16 cells each, cell order 12), so block_order 7 = 4 chunks (32×32)
    and block_order 6 = the whole shard (64×64). Each populated chunk holds
    one digest at cell rank 11 → (row, col) = (3, 1) within its tile."""

    def _store(self, populate, values=None):
        store = MemoryStore()
        _grid, shard6, _targets = _sharded_store(store, populate=populate, values=values)
        return store, shard6

    def test_block_bit_identical_to_per_chunk_tiles(self):
        # Identical values in every populated chunk → each chunk's window ==
        # the shared block window → block tiles are bit-identical to the
        # per-chunk tensors (the acceptance pin for the assembly path).
        vals = np.random.default_rng(3).uniform(5.0, 25.0, 400)
        populate = {0, 3, 7, 12}
        store, shard6 = self._store(populate, values={k: vals for k in populate})

        per_chunk = list(read_tensors(store, "12/h_tdigest"))
        assert len(per_chunk) == len(populate)
        ((block, word),) = read_tensors(store, "12/h_tdigest", block_order=6)
        assert word == shard6
        assert block.shape == (64, 64, 128)

        seen = np.zeros((4, 4), dtype=bool)
        for local, (chunk_t, _cw) in zip(sorted(populate), per_chunk):
            trow, tcol = rank_to_rowcol(local, 2)  # tile position at depth 2
            tile = block[trow * 16 : (trow + 1) * 16, tcol * 16 : (tcol + 1) * 16, :]
            np.testing.assert_array_equal(tile, chunk_t)
            seen[trow, tcol] = True
        # Every other tile is all zero (unpopulated chunks contribute nothing).
        for trow in range(4):
            for tcol in range(4):
                if not seen[trow, tcol]:
                    assert (
                        block[trow * 16 : (trow + 1) * 16, tcol * 16 : (tcol + 1) * 16].sum() == 0
                    )

    def test_block_order_7_groups_and_coarsens_morton(self):
        from mortie import generate_morton_children

        store, shard6 = self._store({0, 1, 2, 3, 5})
        out = list(read_tensors(store, "12/h_tdigest", block_order=7))
        assert [t.shape for t, _m in out] == [(32, 32, 128)] * 2
        # Block ids are the order-7 children of the shard, in nested order.
        kids = [int(k) for k in np.asarray(generate_morton_children(shard6, 7))]
        assert [m for _t, m in out] == kids[:2]
        # Block 0: chunks 0..3 populated at cell rank 11 → tile (r, c) of the
        # 2x2 chunk grid, cell (3, 1) within each 16x16 tile.
        mass0 = out[0][0].sum(axis=2)
        assert {tuple(map(int, p)) for p in zip(*np.nonzero(mass0))} == {
            (3, 1),
            (3, 17),
            (19, 1),
            (19, 17),
        }
        # Block 1: only chunk 5 (local rank 1 within the block → tile (0, 1)).
        mass1 = out[1][0].sum(axis=2)
        assert {tuple(map(int, p)) for p in zip(*np.nonzero(mass1))} == {(3, 17)}

    def test_window_reconciled_block_wide(self):
        rng = np.random.default_rng(4)
        # Two chunks whose ranges only fit ONE 64 m window jointly anchored
        # at the global floor: [0, 20] and [40, 60].
        values = {0: rng.uniform(0.0, 20.0, 300), 1: rng.uniform(40.0, 60.0, 300)}
        store, _shard6 = self._store({0, 1}, values=values)
        ((block, _w),) = read_tensors(
            store, "12/h_tdigest", block_order=7, bottom=0.0, top=1.0, dtype="float32"
        )
        mass = block.sum(axis=2)
        # Both cells' full populations are in the one shared window.
        assert mass[3, 1] == pytest.approx(300, rel=0.01)
        assert mass[3, 17] == pytest.approx(300, rel=0.01)

    def test_fit_raise_decided_block_wide(self):
        rng = np.random.default_rng(5)
        # Each chunk fits a 64 m window alone, but jointly they span ~100 m —
        # the block-wide decision must raise where per-chunk reads pass.
        values = {0: rng.uniform(0.0, 20.0, 300), 1: rng.uniform(80.0, 100.0, 300)}
        store, _shard6 = self._store({0, 1}, values=values)
        assert len(list(read_tensors(store, "12/h_tdigest", bottom=0.0, top=1.0))) == 2
        with pytest.raises(ValueError, match="exceeds the fixed window"):
            list(read_tensors(store, "12/h_tdigest", block_order=7, bottom=0.0, top=1.0))

    def test_block_order_out_of_range_raises(self):
        store, _shard6 = self._store({0})
        for bad in (9, -1):  # chunk order is 8
            with pytest.raises(ValueError, match="block_order .* out of range"):
                list(read_tensors(store, "12/h_tdigest", block_order=bad))

    def test_chunk_order_block_matches_default(self):
        store, _shard6 = self._store({0, 5})
        default = list(read_tensors(store, "12/h_tdigest"))
        explicit = list(read_tensors(store, "12/h_tdigest", block_order=8))
        assert [m for _t, m in default] == [m for _t, m in explicit]
        for (t_d, _m1), (t_e, _m2) in zip(default, explicit):
            np.testing.assert_array_equal(t_d, t_e)
