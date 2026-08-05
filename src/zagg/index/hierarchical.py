"""The default ``hierarchical`` index backend (issue #160, phase 1).

Today's read path — coarse geolocation read + ``plan_read`` + h5coro
hyperslices (with the flat / full-read fallbacks) — refactored *behind* the
:class:`~zagg.index.VirtualIndex` protocol rather than replaced. Pure
delegation: output is byte-identical to calling
``zagg.processing._read_group`` directly, which is the conformance bar every
other backend is then held to (PR #150's byte-equality gate).
"""

from __future__ import annotations

from zagg.index import VirtualIndex


class HierarchicalIndex(VirtualIndex):
    """Selection via the coarse spatial index, addressing via h5coro hyperslices."""

    name = "hierarchical"
    # No backend-specific keys: today's path is configured by the existing
    # ``data_source.read_plan`` / ``levels`` surface, not the index block.

    def read_group(
        self,
        h5obj,
        group,
        data_source,
        shard_key,
        grid,
        arrow=False,
        granule_url=None,
        io_stats=None,
    ):
        # Resolve through the package namespace at call time (not an import-time
        # binding) so tests that ``monkeypatch.setattr("zagg.processing._read_group",
        # ...)`` keep intercepting the worker's reads, exactly as before the seam.
        import zagg.processing as _processing

        # ``granule_url`` is forwarded only when set (the a-priori arm), so
        # monkeypatched ``_read_group`` fakes keep their existing signature on
        # every other path -- mirroring the worker's presence-gated kwarg.
        # ``io_stats`` (issue #374) is NOT gated: it is always-on
        # instrumentation, the worker always passes a dict (``worker.py:422``,
        # ``worker.py:436``), and ``_record_obs_read`` no-ops on ``None`` for
        # the direct callers that omit it -- so gating it would protect nothing
        # while diverging from ``InlineIndex.read_group``, which forwards it
        # unconditionally (review finding).
        kwargs = {"arrow": arrow, "io_stats": io_stats}
        if granule_url is not None:
            kwargs["granule_url"] = granule_url
        return _processing._read_group(h5obj, group, data_source, shard_key, grid, **kwargs)
