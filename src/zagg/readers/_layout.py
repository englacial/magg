"""Chunk-local nested rank ↔ 2-D ``(row, col)`` tensor layout (issue #336).

A depth-``d`` HEALPix subtree holds ``4**d`` cells whose ascending nested
(morton) order traces a Z-order curve over a ``2**d × 2**d`` block — so a
row-major reshape of the 1-D cells axis scrambles the block spatially beyond
2-cell runs. The faithful mapping is the pure bit deinterleave that mortie
0.9.3 ships as ``rank_to_xy``/``xy_to_rank`` (espg/mortie#150; normative
contract in mortie's ``docs/specification.md`` §8, frozen for mortie 1.x):
``x`` gathers the rank's even bits, ``y`` its odd bits, with ``(0, 0)`` at the
subtree's **south corner**, ``x`` increasing toward the north-east edge and
``y`` toward the north-west edge — exactly healpy's face-local
``pix2xyf``/``xyf2pix`` (nest) convention.

Row/col orientation (pinned here, once)
---------------------------------------
Tensor axis 0 (row) is ``y`` and axis 1 (col) is ``x``. This matches
gridlook's texture convention (``src/ui/grids/Healpix.vue``,
``getUnshuffleIndex``): ``texture[i*size + j] = data[bit_combine(j, i)]``,
i.e. the texture's slow axis ``i`` feeds ``bit_combine``'s second (odd-bit,
``y``) argument and the fast axis ``j`` its first (even-bit, ``x``) argument
— so a zagg tensor uploads to a gridlook texture with no further shuffle.
``tensor[0, 0]`` is the subtree's south corner; rows advance toward the
north-west edge, columns toward the north-east edge.
"""

from __future__ import annotations

from mortie import rank_to_xy, xy_to_rank

__all__ = ["rank_to_rowcol", "rowcol_to_rank"]


def rank_to_rowcol(rank, depth: int):
    """``(row, col)`` tensor position of a chunk-local nested rank.

    ``rank`` is the cell's position ``0..4**depth - 1`` within its depth-
    ``depth`` subtree (scalar or array; the same position the ragged writers
    use on the cells axis). Returns ``(row, col) = (y, x)`` per the module
    orientation contract (mortie spec §8 / gridlook ``bit_combine(j, i)``).
    """
    x, y = rank_to_xy(rank, depth)
    return y, x


def rowcol_to_rank(row, col, depth: int):
    """Chunk-local nested rank at a ``(row, col)`` tensor position.

    Inverse of :func:`rank_to_rowcol`: ``row`` is ``y``, ``col`` is ``x``
    (scalars or arrays; values must be ``< 2**depth``).
    """
    return xy_to_rank(col, row, depth)
