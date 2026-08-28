"""Shared viewers for the reader-only demo notebooks.

`hhdc_viewer.ipynb` and `waveform_viewer.ipynb` both open the same paired
stores and differ only in how they draw what they find, so the drawing lives
here and the notebooks stay about the READ path -- polygon, coverage, shards,
leaves, digests.

Everything here is reader-side: `mortie` for the geometry, `moczarr` for the
store. The one zagg import (`cdf_from_tdigest`) is lazy and confined to the
waveform view -- moczarr imports the t-digest algebra from zagg rather than
vendoring it (moczarr issue #19), which is what the `moczarr[zagg]` extra
carries and what `read_tensors` has always needed.
"""

from __future__ import annotations

import time

import matplotlib.pyplot as plt
import moczarr as mz
import numpy as np
from IPython.display import display
from ipywidgets import (
    Checkbox,
    Dropdown,
    FloatText,
    HBox,
    IntSlider,
    VBox,
    interactive_output,
)
from matplotlib.colors import LogNorm
from moczarr.hhdc import block_rank, rank_to_rowcol, rowcol_to_rank
from mortie import generate_morton_children, toc2time

#: The o12 block the views frame everything in. Cells across a block edge are
#: ``2**(cell_order - BLOCK_ORDER)`` -- 128 for ATL03's o19, 64 for GEDI's o18.
BLOCK_ORDER = 12

#: o12 block side in metres (equal-area, square-equivalent).
SIDE = float(np.sqrt(4 * np.pi / (12 * 4**BLOCK_ORDER)) * 6_371_000)

#: Points drawn per sensor per block. A leaf holds millions; the 3-D view
#: subsamples deterministically so rotation stays smooth.
CAP = 20_000

__all__ = ["BLOCK_ORDER", "CAP", "SIDE", "View", "grid_xy", "load", "view3d", "waveform_view"]


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #
def grid_xy(words, block_order: int = BLOCK_ORDER):
    """Word -> (x, y) metres inside its o12 block. Three library calls, no bits.

    ``block_rank`` recovers each word's block-local nested rank and its own
    order -- normalizing POINT words to their order-29 area twins on the way,
    which is the step whose absence used to make the level-28/29 digits decode
    out of range. ``rank_to_rowcol`` is the bit deinterleave (mortie spec
    section 8): it returns ``(row, col) = (y, x)`` with ``[0, 0]`` at the
    subtree's south corner, and this returns them as ``(x, y) = (col, row)``
    so the picture matches the tensors this notebook exports.

    Those axes are NOT east and north. zagg's ``readers/_layout.py`` pins the
    semantics: "``tensor[0, 0]`` is the subtree's south corner; rows advance
    toward the north-WEST edge, columns toward the north-EAST edge." A HEALPix
    face is a diamond, so its two local axes run along the face's edges, not
    along the compass. ``view3d`` labels them for what they are.

    Note the tensors ``read_tensors`` (and so ``hhdc_viewer``'s export cells)
    hand back are indexed ``(row, col, bin)``: the picture's x is the tensor's
    SECOND axis. :func:`_binned_pts` applies the same swap, so the binned and
    exact views agree with each other.

    A leaf mixes orders -- o29 located words beside coarser merged-centroid
    words, and GEDI's cell words are o18 -- so the ranks are grouped by depth
    and handed over one vectorized call per depth.
    """
    rank, order = block_rank(words, block_order)
    depth = order - block_order
    row = np.zeros(len(rank))
    col = np.zeros(len(rank))
    for d in np.unique(depth):
        m = np.flatnonzero(depth == d)
        r, c = rank_to_rowcol(rank[m], int(d))
        row[m] = np.asarray(r, dtype=float)
        col[m] = np.asarray(c, dtype=float)
    side = 2.0 ** depth.astype(float)  # cells along a block edge
    return (col + 0.5) / side * SIDE, (row + 0.5) / side * SIDE


def load(store, field, block_order: int = BLOCK_ORDER) -> dict:
    """One sensor's centroids: z, weight, xy, o12 block, acquisition days.

    xy comes from the located sibling's word where the field declares one
    (ATL03 -- point-exact) and from the cell word where it does not (GEDI
    flux, whose shots are unlocated by design, so a centroid renders at its
    cell's centre). Both land in the block's own lattice frame.
    """
    arr, _ = mz.open_ragged(store, field)
    attrs = dict(arr.attrs)
    locname = (attrs.get("ragged") or {}).get("locations")
    tname = attrs.get("times")
    zs, wts, cells, locw, seq = [], [], [], [], []
    for row in mz.read_ragged(store, field, locations=bool(locname)):
        seq.append(row[0])
        v = np.asarray(row[1])
        zs.append(v[:, 0])
        wts.append(v[:, 1])
        cells.append(np.full(len(v), row[0], dtype=np.uint64))
        if locname:
            locw.append(np.asarray(row[2], dtype=np.uint64))
    z, wt = np.concatenate(zs), np.concatenate(wts)
    cells = np.concatenate(cells)
    sh = np.uint64(6 + 2 * (27 - block_order))
    blocks = ((cells >> sh) << sh) | np.uint64(block_order)  # o12 ancestor by truncation
    x, y = grid_xy(np.concatenate(locw) if locname else cells, block_order)
    t = None
    if tname:
        tmap = {
            int(r[0]): np.asarray(r[1], dtype=np.uint64).ravel()
            for r in mz.read_ragged(store, field.rsplit("/", 1)[0] + "/" + tname)
        }
        tw = np.concatenate([tmap[int(c)] for c in seq])
        ns2018 = float(
            (np.datetime64("2018-01-01") - np.datetime64("1850-01-01")) // np.timedelta64(1, "ns")
        )
        t = (np.asarray(toc2time(tw)[0], dtype="float64") - ns2018) / 86.4e12
    return {
        "z": z,
        "wt": wt,
        "x": x,
        "y": y,
        "blocks": blocks,
        "t": t,
        "xy_note": "exact xy" if locname else "cell-center xy",
    }


# --------------------------------------------------------------------------- #
# the 3-D view
# --------------------------------------------------------------------------- #
def _binned_pts(tensor, offset, gain, cap: int = CAP):
    """Occupied voxels of one block tensor -> (x, y, z, weight), subsampled.

    ``x``/``y`` come back as FRACTIONS of the block edge (the tensor's own
    ``side`` is its xy shape), so the caller scales them by :data:`SIDE` the
    same way the exact path does, and ``x`` is the tensor's COL axis and ``y``
    its ROW axis, the same round as :func:`grid_xy`.

    ``np.nonzero`` gives cell INDICES, so the ``+ 0.5`` that moves a point off
    the cell corner onto its centre is the same one :func:`grid_xy` applies --
    without it, toggling **binned** shifts the whole cloud by half a cell, and
    by a DIFFERENT half-cell per sensor (6.2 m for ATL03's o19 cells, 12.4 m
    for GEDI's o18).
    """
    side = tensor.shape[0]
    rows, cols, zs = np.nonzero(tensor)  # the tensor is indexed (row, col, bin)
    counts = tensor[rows, cols, zs].astype("float64")
    if len(rows) > cap:
        keep = np.random.default_rng(0).choice(len(rows), cap, replace=False)
        rows, cols, zs, counts = rows[keep], cols[keep], zs[keep], counts[keep]
    return (cols + 0.5) / side, (rows + 0.5) / side, offset + zs * gain, counts


class View:
    """Holds what is on screen so `export` knows which block you mean."""

    shard = None
    block = None


def view3d(
    handles,
    shard,
    block_order: int = BLOCK_ORDER,
    cap: int = CAP,
    n_bins: int = 256,
    resolution: float = 1.0,
) -> View:
    """Interactive paired 3-D view. Drag to rotate -- the two panes stay linked.

    ``handles`` is ``{sensor: (leaf_store, field)}``. Two ways to see the same
    block, switched by the **binned** box:

    * **binned** (default) -- the fixed ``read_tensors`` voxels, so xy AND z
      are on the store's own grid and the two sensors share a z window.
    * **exact** -- the digests read straight through, z the stored float32
      centroid elevation and xy decoded from the located sibling's word where
      the field carries one (ATL03, point-exact) or the cell word where it
      does not (GEDI flux, footprint scale).

    Tensors are read once up front (the default view needs them) and REDUCED as
    they arrive; the exact centroids are swept lazily on the first unbinned
    draw, so a reader who never unticks the box never pays for them.

    Returns the :class:`View` the widgets write to, so a later cell can export
    whatever is on screen.
    """
    view = View()
    view.shard = shard
    names = list(handles)

    # One sweep per sensor, reduced block by block as it goes. A block tensor is
    # up to 16 MiB and a shard holds ~64 of them per sensor, but the view draws
    # ONE block at a time and never more than `cap` of its voxels -- so keep the
    # drawable cloud and the occupancy count, not the tensor. That is ~1.3 GiB
    # resident versus ~80 MiB, and mybinder.org caps the WHOLE container at 2 GB.
    print("binning tensors (once, whole shard, both sensors)...", flush=True)
    t0 = time.perf_counter()
    voxels: dict = {}
    for name, (store, field) in handles.items():
        voxels[name] = {
            b[3]: {
                "pts": _binned_pts(b[0], *b[2], cap),
                "cells": int(b[1].astype(bool).sum()),
                "gain": b[2][1],
            }
            for b in mz.read_tensors(
                store,
                field,
                n_bins=n_bins,
                resolution=resolution,
                block_order=block_order,
                fit="degrade_resolution",
            )
        }
    print(
        "  "
        + ", ".join(f"{len(voxels[n]):,} {n} blocks" for n in handles)
        + f" in {time.perf_counter() - t0:.0f}s",
        flush=True,
    )
    exact: dict = {}  # filled on the first unbinned draw

    def _exact():
        if not exact:
            # One whole-shard sweep per sensor, millions of centroids -- say so,
            # or the first untick reads as a frozen notebook.
            print("sweeping exact centroids (once, whole shard, both sensors)...", flush=True)
            t0 = time.perf_counter()
            for name, (store, field) in handles.items():
                exact[name] = load(store, field, block_order)
            print(
                "  "
                + ", ".join(f"{len(exact[n]['z']):,} {n}" for n in handles)
                + f" in {time.perf_counter() - t0:.0f}s",
                flush=True,
            )
        return exact

    # Blocks both sensors populate, SORTED, each labelled with how much it holds.
    joint = sorted(set.intersection(*(set(v) for v in voxels.values())))
    if not joint:
        raise ValueError(f"shard {shard}: no block is populated in every store")
    options = [
        (
            f"{mz.morton_decimal(w)}  ("
            + ", ".join(f"{voxels[n][w]['cells']:,} {n} cells" for n in names)
            + ")",
            w,
        )
        for w in joint
    ]

    dd = Dropdown(options=options, value=joint[0], description="block")
    zmode = Dropdown(
        options=[("independent z", "auto"), *[(f"pin z to {n}", n) for n in names]],
        value="auto",
        description="z extent",
    )
    elev_cb = Checkbox(value=False, description="color by elevation (shared)")
    time_cb = Checkbox(value=False, description="color by time (shared)")
    bin_cb = Checkbox(value=True, description="binned (fixed tensors)")

    def _panes(block, binned):
        if binned:
            out = []
            for n in names:
                rec = voxels[n][block]
                x, y, z, wt = rec["pts"]
                out.append(
                    {
                        "x": x * SIDE,
                        "y": y * SIDE,
                        "z": z,
                        "wt": wt,
                        "t": None,
                        "label": n,
                        "xy_note": f"binned ({rec['gain']:g} m z)",
                    }
                )
            return out
        data = _exact()
        out = []
        rng = np.random.default_rng(0)
        for n in names:
            d = data[n]
            m = np.flatnonzero(d["blocks"] == np.uint64(block))
            if len(m) > cap:
                m = m[rng.choice(len(m), cap, replace=False)]
            out.append(
                {
                    **{k: d[k][m] for k in ("z", "wt", "x", "y")},
                    "t": None if d["t"] is None else d["t"][m],
                    "label": n,
                    "xy_note": d["xy_note"],
                }
            )
        return out

    live: dict = {}  # the one figure this view keeps open

    def draw(block, by_elev, by_time, binned, zmode):
        view.block = block
        panes = _panes(block, binned)
        zlim = (
            None
            if zmode == "auto"
            else (lambda p: (p["z"].min(), p["z"].max()))(panes[names.index(zmode)])
        )
        shared_elev = (
            plt.Normalize(min(p["z"].min() for p in panes), max(p["z"].max() for p in panes))
            if by_elev and not by_time
            else None
        )
        tv = [p["t"] for p in panes if p["t"] is not None]
        shared_time = (
            plt.Normalize(min(t.min() for t in tv), max(t.max() for t in tv))
            if (by_time and tv)
            else None
        )
        ticks = [0.0, SIDE / 2, SIDE]
        labels = [f"{v:.0f} m" for v in ticks]

        # `draw` reruns on EVERY widget change, and `clear_output(wait=True)`
        # clears the Output widget's display, not matplotlib's figure registry.
        # An ipympl figure is a live canvas -- it holds its websocket comm and
        # the `_sync` handler connected below for the kernel's lifetime -- so
        # without this the 21st interaction warns and every orphan keeps
        # handling mouse events. Close the PREVIOUS figure, not this one: the
        # widget backend needs the current canvas alive to render it.
        if (prev := live.pop("fig", None)) is not None:
            plt.close(prev)
        fig = plt.figure(figsize=(11, 5.2))
        live["fig"] = fig
        axes = []
        for k, (p, cmap) in enumerate(zip(panes, ("viridis", "plasma"))):
            ax = fig.add_subplot(1, 2, k + 1, projection="3d")
            axes.append(ax)
            note = f" — {p['xy_note']}"
            alpha = np.clip(p["wt"] / max(np.percentile(p["wt"], 98), 1e-9), 0.08, 1.0)
            if by_time and p["t"] is not None:
                pts = ax.scatter(
                    p["x"],
                    p["y"],
                    p["z"],
                    c=p["t"],
                    s=1.5,
                    cmap="turbo",
                    norm=shared_time,
                    alpha=alpha,
                )
                fig.colorbar(pts, shrink=0.55, pad=0.10, label="days since 2018-01-01")
            elif by_time:
                ax.scatter(p["x"], p["y"], p["z"], color="#9498a0", s=1.5, alpha=0.15)
                note += " — no temporal channel" + (" in binned tensors" if binned else "")
            elif by_elev:
                pts = ax.scatter(
                    p["x"],
                    p["y"],
                    p["z"],
                    c=p["z"],
                    s=1.5,
                    cmap="viridis",
                    norm=shared_elev,
                    alpha=alpha,
                )
                fig.colorbar(pts, shrink=0.55, pad=0.10, label="elevation (m)")
            else:
                pts = ax.scatter(
                    p["x"],
                    p["y"],
                    p["z"],
                    c=p["wt"],
                    s=1.5,
                    cmap=cmap,
                    norm=LogNorm(),
                    alpha=alpha,
                )
                fig.colorbar(pts, shrink=0.55, pad=0.10, label="weight")
            if zlim is not None:
                ax.set_zlim(*zlim)
            ax.set_title(p["label"] + note, fontsize=9)
            ax.set_zlabel("elevation (m)")
            ax.set_xticks(ticks, labels, fontsize=7)
            ax.set_yticks(ticks, labels, fontsize=7)
            # Face-local axes from the block's SOUTH corner: col runs toward the
            # north-EAST edge, row toward the north-WEST (zagg readers/_layout.py).
            # A HEALPix face is a diamond -- these are not compass east/north.
            ax.set_xlabel("→ NE edge", fontsize=8, labelpad=-2)
            ax.set_ylabel("→ NW edge", fontsize=8, labelpad=-2)

        # Linked rotation: only while DRAGGING, and only when the angles moved --
        # a bare hover must not trigger a redraw.
        def _sync(event):
            if event.button is None or event.inaxes not in axes:
                return
            src = event.inaxes
            for other in axes:
                if other is not src and (other.elev != src.elev or other.azim != src.azim):
                    other.view_init(elev=src.elev, azim=src.azim)
                    fig.canvas.draw_idle()

        fig.canvas.mpl_connect("motion_notify_event", _sync)
        fig.suptitle(
            f"shard {shard} — block {mz.morton_decimal(block)}"
            + ("" if binned else " — exact centroids"),
            fontsize=11,
        )
        plt.show()

    out = interactive_output(
        draw,
        {"block": dd, "by_elev": elev_cb, "by_time": time_cb, "binned": bin_cb, "zmode": zmode},
    )
    display(VBox([HBox([dd, zmode]), HBox([elev_cb, time_cb, bin_cb]), out]))
    return view


# --------------------------------------------------------------------------- #
# the waveform view
# --------------------------------------------------------------------------- #
def _mixture(digest, z, sigma):
    """Digest -> density on `z`: each centroid a Gaussian of width `sigma`."""
    mu, wt = digest[:, 0], digest[:, 1]
    pdf = (wt[None, :] * np.exp(-0.5 * ((z[:, None] - mu[None, :]) / sigma) ** 2)).sum(axis=1)
    return pdf / max(wt.sum(), 1e-9) / (sigma * np.sqrt(2 * np.pi))


def _atl03_digests(store, field, w, r, c, chunk_side, chunk_order, block_order=BLOCK_ORDER):
    """The 2x2 finer digests under one coarser cell `(r, c)` of block `w`.

    GEDI's chunk grid puts one o12 block in one chunk; ATL03's chunks sit at
    `chunk_order` (o13 in these stores), so the o19 cell has to be resolved to
    the right chunk before `cell_index` can address it.

    `chunk_side` is ATL03's cells across a CHUNK edge, ``2**(19 - 13) = 64``.
    That is deliberately NOT the caller's `gside`, which is GEDI's cells across
    a BLOCK edge, ``2**(18 - 12)`` -- also 64, but only for this store pair.
    Sharing one name would make a change to either sensor's cell order, or to
    the store's chunk order, address the wrong cell silently rather than raise,
    and the `except` below would report the damage as "no photons here".
    """
    kids = generate_morton_children(int(w), chunk_order)
    depth = chunk_order - block_order
    out = []
    for dr in (0, 1):
        for dc in (0, 1):
            rr, cc = 2 * r + dr, 2 * c + dc
            chunk = int(kids[rowcol_to_rank(rr // chunk_side, cc // chunk_side, depth=depth)])
            try:
                out.append(
                    mz.read_cell(
                        store,
                        field,
                        mz.cell_index(store, field, chunk, rr % chunk_side, cc % chunk_side),
                    )
                )
            except (KeyError, ValueError):
                pass  # that quarter holds no photons
    return np.concatenate([k for k in out if len(k)]) if out else np.empty((0, 2))


def waveform_view(
    stores, fields, blocks, pairs, shard, gside: int = 64, atl03_chunk_order: int = 13
):
    """Interactive coincident-waveform view.

    ``stores``/``fields`` are ``{sensor: ...}``; ``blocks`` is
    ``{sensor: {block_word: read_tensors tuple}}``; ``pairs`` is the sorted
    ``(word, joint_mask, A2, G2)`` list the notebook builds.

    ``gside`` is GEDI's cells across an o12 BLOCK edge. ``atl03_chunk_order``
    is where ATL03's chunks sit in these stores; ATL03's cells across a CHUNK
    edge is derived from it and the cell order the field's path already
    carries (``19/…``), because that is a different quantity that merely also
    equals 64 here -- see :func:`_atl03_digests`.
    """
    from zagg.stats.tdigest import cdf_from_tdigest  # moczarr[zagg]; see module docstring

    acell_order = int(fields["atl03"].split("/", 1)[0])
    achunk_side = 2 ** (acell_order - atl03_chunk_order)

    live: dict = {}  # the one figure this view keeps open

    # `binw` is a pure rendering knob, but it shares the `interactive_output`
    # callback with `pair`/`nth` -- so without this, typing in the bin box
    # re-runs five S3 round trips (one GEDI `read_cell` plus up to four ATL03)
    # on digests already in memory. Only `pair` and `nth` change what is READ.
    cache: dict = {}

    def _digests(pair, r, c, w):
        key = (pair, r, c)
        if key not in cache:
            if len(cache) > 64:  # a session's worth; these are a few KB each
                cache.clear()
            cache[key] = (
                mz.read_cell(
                    stores["gedi"],
                    fields["gedi"],
                    mz.cell_index(stores["gedi"], fields["gedi"], int(w), r, c),
                ),
                _atl03_digests(
                    stores["atl03"], fields["atl03"], w, r, c, achunk_side, atl03_chunk_order
                ),
            )
        return cache[key]

    def paired_waveform(pair=0, nth=0, binw=1.0):
        w, joint, a2, g2 = pairs[pair]  # a2/g2: the notebook's A2/G2, lowercased for N806
        _, _, (_aoff, ag), _ = blocks["atl03"][w]
        _, _, (_goff, gg), _ = blocks["gedi"][w]
        # Rank the joint cells by the weaker side. a2 is ATL03 PHOTON counts and
        # g2 GEDI PHOTOELECTRONS -- incommensurate units that run two to three
        # orders of magnitude apart on this store pair, so a raw `np.minimum`
        # would collapse to a2 on essentially every cell and "the weaker member"
        # would really mean "the ATL03 count". Scale each side by its own
        # maximum over the JOINT cells first, so the min picks out whichever
        # sensor is relatively weaker. (What makes a pick *coincident* at all is
        # the `joint` mask itself -- both sensors populated -- not this min.)
        a = a2 / max(int(a2[joint].max()), 1)
        g = g2 / max(int(g2[joint].max()), 1)
        rank = np.minimum(a, g) * joint
        order = np.argsort(rank.ravel())[::-1]
        r, c = np.unravel_index(int(order[min(nth, int(joint.sum()) - 1)]), rank.shape)

        gdigest, adigest = _digests(pair, int(r), int(c), w)

        # `_atl03_digests` can legitimately return nothing: the `joint` mask is
        # built from the TENSORS (a finite z window, possibly degraded), while
        # this reads the RAW digests through a different addressing path, and
        # the two are not guaranteed to agree cell for cell -- which is the case
        # the `except` in `_atl03_digests` was written for. Make that guard real
        # by drawing the GEDI side alone and saying so, rather than letting an
        # empty `.min()` turn "no photons here" into a traceback in the widget.
        zmu = [gdigest[:, 0]] + ([adigest[:, 0]] if len(adigest) else [])
        lo = min(a.min() for a in zmu) - 5
        hi = max(a.max() for a in zmu) + 5
        z = np.linspace(lo, hi, 700)
        amu, awt = adigest[:, 0], adigest[:, 1]

        # Same figure-registry bound as `view3d.draw`: this is an
        # `interactive_output` callback too, so close the previous figure
        # rather than accumulating one per widget change.
        if (prev := live.pop("fig", None)) is not None:
            plt.close(prev)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.6), sharey=True)
        live["fig"] = fig
        ax1.plot(
            _mixture(gdigest, z, gg),
            z,
            color="#7b3294",
            lw=2,
            label=f"GEDI flux ({gdigest[:, 1].sum():.0f} pe)",
        )
        if len(adigest):
            ax1.plot(
                _mixture(adigest, z, ag),
                z,
                color="#008837",
                lw=2,
                label=f"ATL03 signal ({awt.sum():.0f} ph)",
            )
        ax1.set_xlabel("normalized density")
        ax1.set_ylabel("elevation (m)")

        # Top axis: the RAW ATL03 photons -- binned bars, or one dot per centroid
        # when the width is 0 (cells under the store's delta budget are loss-free,
        # so centroids ~ photons there; a merged centroid's weight sets its size).
        ax1t = ax1.twiny()
        if binw and binw > 0:
            edges = np.arange(lo, hi + binw, binw)
            counts, _ = np.histogram(amu, bins=edges, weights=awt)
            ax1t.barh(
                edges[:-1] + binw / 2,
                counts,
                height=binw * 0.9,
                color="#008837",
                alpha=0.25,
                zorder=0,
            )
            ax1t.set_xlabel(f"ATL03 photons / {binw:g} m bin", fontsize=9)
        else:
            ax1t.scatter(
                np.zeros(len(amu)),
                amu,
                s=np.clip(awt * 8, 8, 40),
                color="#008837",
                alpha=0.45,
                zorder=0,
            )
            ax1t.set_xlim(-0.05, 1.0)
            ax1t.set_xlabel("ATL03 photons (unbinned)", fontsize=9)
        ax1.set_zorder(ax1t.get_zorder() + 1)
        ax1.patch.set_visible(False)
        ax1.set_title(
            f"cell ({r},{c}) @o18 — GEDI {len(gdigest)} centroids "
            + (
                f"vs ATL03 {len(adigest)} (2×2 @o19)"
                if len(adigest)
                else "— no ATL03 digest under this cell"
            ),
            fontsize=9,
        )
        ax1.legend(fontsize=8)

        # cdf_from_tdigest returns CUMULATIVE WEIGHT (pe for GEDI, photons for
        # ATL03) -- normalize each by its own total so both share the axis honestly.
        ax2.plot(
            cdf_from_tdigest(gdigest, z) / max(gdigest[:, 1].sum(), 1e-9), z, color="#7b3294", lw=2
        )
        if len(adigest):
            ax2.plot(
                cdf_from_tdigest(adigest, z) / max(adigest[:, 1].sum(), 1e-9),
                z,
                color="#008837",
                lw=2,
            )
        ax2.set_xlim(0, 1)
        ax2.set_xlabel("CDF (probability)")
        ax2.set_title("cumulative", fontsize=10)
        for ax in (ax1, ax2):
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(alpha=0.25, lw=0.5)
        fig.suptitle(f"shard {shard} — block {mz.morton_decimal(w)}", fontsize=11)
        plt.tight_layout()
        plt.show()

    dd = Dropdown(
        options=[
            (f"{mz.morton_decimal(w)}  ({int(j.sum()):,} joint cells)", i)
            for i, (w, j, _, _) in enumerate(pairs)
        ],
        value=0,
        description="block",
    )
    nth = IntSlider(min=0, max=40, value=0, description="nth joint")
    binw = FloatText(value=1.0, step=0.5, description="bin (m)")
    display(
        VBox(
            [
                HBox([dd, nth, binw]),
                interactive_output(paired_waveform, {"pair": dd, "nth": nth, "binw": binw}),
            ]
        )
    )
