"""Packaged example data: demo AOI polygons, loadable from any install.

sklearn-style accessors (issue #328): the demo notebooks resolve their AOIs
from the installed package rather than a repo checkout, so they run from any
working directory -- pip, binder, colab. Mirrors ``default_config`` for the
packaged YAML templates and mortie's packaged basin polygons
(:func:`zagg.catalog.load_antarctic_basins`).
"""

from __future__ import annotations

from importlib import resources

#: name -> packaged GeoJSON. ``serc`` is the NEON SERC AOP flight box
#: (Edgewater, MD); ``neon_aop`` is every NEON AOP flight box, domains
#: D01-D20 (61 polygon parts, Alaska to Puerto Rico); ``california`` is the
#: state outline (Natural Earth 1:10m). Each file's ``properties`` block
#: records its provenance.
_AOIS = {
    "serc": "serc.geojson",
    "neon_aop": "neon_aop.geojson",
    "california": "california.geojson",
}


def demo_aoi(name: str) -> str:
    """Filesystem path of a packaged demo AOI GeoJSON.

    Parameters
    ----------
    name : str
        One of ``"serc"``, ``"neon_aop"``, ``"california"``.

    Returns
    -------
    str
        Path usable anywhere a GeoJSON path is accepted (``Query.region``,
        :func:`zagg.catalog.load_polygon`, ...).
    """
    if name not in _AOIS:
        raise KeyError(f"unknown demo AOI {name!r} (available: {sorted(_AOIS)})")
    return str(resources.files(__package__) / _AOIS[name])


__all__ = ["demo_aoi"]
