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
    subtree's south corner, so x reads the row and y the col.

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
    return (row + 0.5) / side * SIDE, (col + 0.5) / side * SIDE


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
    same way the exact path does.
    """
    side = tensor.shape[0]
    xs, ys, zs = np.nonzero(tensor)
    counts = tensor[xs, ys, zs].astype("float64")
    if len(xs) > cap:
        keep = np.random.default_rng(0).choice(len(xs), cap, replace=False)
        xs, ys, zs, counts = xs[keep], ys[keep], zs[keep], counts[keep]
    return xs / side, ys / side, offset + zs * gain, counts


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

    Tensors are read once up front (the default view needs them); the exact
    centroids are swept lazily on the first unbinned draw, so a reader who
    never unticks the box never pays for them.

    Returns the :class:`View` the widgets write to, so a later cell can export
    whatever is on screen.
    """
    view = View()
    view.shard = shard
    names = list(handles)

    tensors = {
        name: {
            b[3]: b
            for b in mz.read_tensors(
                store,
                field,
                n_bins=n_bins,
                resolution=resolution,
                block_order=block_order,
                fit="degrade_resolution",
            )
        }
        for name, (store, field) in handles.items()
    }
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
    joint = sorted(set.intersection(*(set(t) for t in tensors.values())))
    if not joint:
        raise ValueError(f"shard {shard}: no block is populated in every store")
    options = [
        (
            f"{mz.morton_decimal(w)}  ("
            + ", ".join(f"{int(tensors[n][w][1].astype(bool).sum()):,} {n} cells" for n in names)
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
                tensor, _mask, (off, gain), _w = tensors[n][block]
                x, y, z, wt = _binned_pts(tensor, off, gain, cap)
                out.append(
                    {
                        "x": x * SIDE,
                        "y": y * SIDE,
                        "z": z,
                        "wt": wt,
                        "t": None,
                        "label": n,
                        "xy_note": f"binned ({gain:g} m z)",
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

        fig = plt.figure(figsize=(11, 5.2))
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
            ax.set_xlabel("east", fontsize=8, labelpad=-2)
            ax.set_ylabel("north", fontsize=8, labelpad=-2)

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


def _atl03_digests(store, field, w, r, c, gside, block_order=BLOCK_ORDER):
    """The 2x2 finer digests under one coarser cell `(r, c)` of block `w`.

    GEDI's chunk grid puts one o12 block in one chunk; ATL03's chunks sit at
    o13, so the o19 cell has to be resolved to the right o13 child before
    `cell_index` can address it.
    """
    kids = generate_morton_children(int(w), block_order + 1)
    out = []
    for dr in (0, 1):
        for dc in (0, 1):
            rr, cc = 2 * r + dr, 2 * c + dc
            chunk = int(kids[rowcol_to_rank(rr // gside, cc // gside, depth=1)])
            try:
                out.append(
                    mz.read_cell(
                        store, field, mz.cell_index(store, field, chunk, rr % gside, cc % gside)
                    )
                )
            except (KeyError, ValueError):
                pass  # that quarter holds no photons
    return np.concatenate([k for k in out if len(k)]) if out else np.empty((0, 2))


def waveform_view(stores, fields, blocks, pairs, shard, gside: int = 64):
    """Interactive coincident-waveform view.

    ``stores``/``fields`` are ``{sensor: ...}``; ``blocks`` is
    ``{sensor: {block_word: read_tensors tuple}}``; ``pairs`` is the sorted
    ``(word, joint_mask, A2, G2)`` list the notebook builds.
    """
    from zagg.stats.tdigest import cdf_from_tdigest  # moczarr[zagg]; see module docstring

    def paired_waveform(pair=0, nth=0, binw=1.0):
        w, joint, a2, g2 = pairs[pair]  # a2/g2: the notebook's A2/G2, lowercased for N806
        _, _, (_aoff, ag), _ = blocks["atl03"][w]
        _, _, (_goff, gg), _ = blocks["gedi"][w]
        rank = np.minimum(a2, g2) * joint  # rank joint cells by the weaker side
        order = np.argsort(rank.ravel())[::-1]
        r, c = np.unravel_index(int(order[min(nth, int(joint.sum()) - 1)]), rank.shape)

        gdigest = mz.read_cell(
            stores["gedi"],
            fields["gedi"],
            mz.cell_index(stores["gedi"], fields["gedi"], int(w), int(r), int(c)),
        )
        adigest = _atl03_digests(stores["atl03"], fields["atl03"], w, int(r), int(c), gside)

        lo = min(gdigest[:, 0].min(), adigest[:, 0].min()) - 5
        hi = max(gdigest[:, 0].max(), adigest[:, 0].max()) + 5
        z = np.linspace(lo, hi, 700)
        amu, awt = adigest[:, 0], adigest[:, 1]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.6), sharey=True)
        ax1.plot(
            _mixture(gdigest, z, gg),
            z,
            color="#7b3294",
            lw=2,
            label=f"GEDI flux ({gdigest[:, 1].sum():.0f} pe)",
        )
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
            f"vs ATL03 {len(adigest)} (2×2 @o19)",
            fontsize=9,
        )
        ax1.legend(fontsize=8)

        # cdf_from_tdigest returns CUMULATIVE WEIGHT (pe for GEDI, photons for
        # ATL03) -- normalize each by its own total so both share the axis honestly.
        ax2.plot(
            cdf_from_tdigest(gdigest, z) / max(gdigest[:, 1].sum(), 1e-9), z, color="#7b3294", lw=2
        )
        ax2.plot(
            cdf_from_tdigest(adigest, z) / max(adigest[:, 1].sum(), 1e-9), z, color="#008837", lw=2
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
