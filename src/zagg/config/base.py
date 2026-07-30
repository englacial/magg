"""Config dataclass, typed blocks and the shared scalar helpers (issue #330).

Split out of the single-file ``zagg.config`` (issue #330 phase 4, which passed
the CLAUDE.md §4 ~1000-line limit at 2,747); the public surface is unchanged
and re-exported from :mod:`zagg.config`. The dependency-free bottom of the
package: :class:`PipelineConfig`, the ``TypedDict`` shapes, the filter-operator
and worker-size vocabularies, and the few leaf helpers the validators and
accessors both use.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import NotRequired, TypedDict

import numpy as np


class LinkDict(TypedDict):
    """Per-level link to the next coarser level (issue #43, Phase B).

    A *link* describes a contiguous-range parent->child tiling: each parent segment
    ``p`` covers base-rate indices ``[index_beg[p] - index_base, ...`` for
    ``count[p]`` children.  ``index_base`` shifts the raw ``index_beg`` values so
    that Python 0-based indexing into the base array is straightforward.

    ``reference_index`` is a reserved slot for a future explicit-index-array variant
    (non-contiguous children per parent); leave it ``None`` for the contiguous case.
    """

    to: str  # key of the coarser level in ``levels``
    index_beg: str  # HDF5 path for the per-parent start index array
    count: str  # HDF5 path for the per-parent child count array
    index_base: NotRequired[int]  # subtracted from index_beg values (default 0)
    reference_index: NotRequired[str | None]  # reserved; must be None


class LevelDict(TypedDict):
    """One hierarchical level in a multi-rate HDF5 source (issue #43, Phase B).

    A source may have several rates (e.g. ATL03 ``photons`` and ``segments``).
    Each level declares its own ``path``, ``coordinates``, and ``variables``,
    plus an optional ``link`` to a coarser parent level.  The flat single-level
    form (no ``levels``/``base_level`` keys in ``data_source``) stays first-class.
    """

    path: str  # HDF5 group path template (may contain ``{group}``)
    coordinates: list[str]  # coordinate dataset names within ``path``
    # ``variables`` has two forms: a documentation-only ``list[str]`` of names, or
    # (non-base levels, issue #30) a ``{name: path-template}`` mapping declaring a
    # *readable* segment-level variable. The mapping form is read at coarse rate and
    # broadcast to the base (photon) rows via ``link`` so e.g. ``dem_h`` (one value
    # per ~100 photons) becomes a per-photon column the aggregation can reduce.
    variables: list[str] | dict[str, str]
    link: NotRequired[LinkDict | None]


class DataSourceDict(TypedDict):
    """Type hints for the ``data_source`` section of a pipeline config."""

    reader: str
    groups: list[str]
    coordinates: dict[str, str]
    variables: dict[str, str]
    quality_filter: NotRequired[dict]
    filters: NotRequired[list[dict]]
    # Hierarchical multi-level form (issue #43, Phase B). When present, the flat
    # ``coordinates``/``variables`` keys are still accepted for the base level but
    # ``levels`` + ``base_level`` take precedence for the read path.
    levels: NotRequired[dict[str, LevelDict]]
    base_level: NotRequired[str]
    # Virtual chunk-index backend block (issue #160). Absent → the default
    # ``hierarchical`` path, byte-identical. ``backend`` names a registered
    # backend (builtin or ``zagg.index_backends`` entry point); the remaining
    # keys are backend-specific and validated against the backend's declared
    # ``config_keys`` — irrelevant keys are config errors, not ignored.
    index: NotRequired[dict]
    # Credential-provider registry name for source-data S3 reads (issue #213
    # Phase 4/6): built-ins ``nsidc``/``gesdisc``; plugins may register others.
    # Absent → the spatial default (NSIDC); temporal events may also carry
    # per-event ``s3_credentials``, which win.
    credentials_provider: NotRequired[str]


# Structured-predicate comparison operators (issue #43). ``in``/``not_in`` take a
# ``values`` list; the rest take a scalar ``value``. These are the only
# pushdown-eligible filter language; an ``expression`` filter is a base-level-only,
# aggregation-time escape hatch that forfeits pushdown.
_SCALAR_OPS = frozenset({"eq", "ne", "ge", "le", "lt", "gt"})
_SET_OPS = frozenset({"in", "not_in"})
FILTER_OPS = _SCALAR_OPS | _SET_OPS


_PIPELINE_TYPES = frozenset({"spatial", "temporal", "event"})

# Memory sizes (MB) of the pre-provisioned Lambda worker-size variants
# (issue #235). Must match template.yaml's WorkerMemorySizes parameter — the
# runner resolves ``worker:`` to a ``<base>-<memory>[-disk]`` function name,
# so an unlisted size would dispatch to a function that does not exist.
WORKER_MEMORIES = frozenset({2048, 4096, 8192})


@dataclass
class PipelineConfig:
    """Full pipeline configuration.

    Parameters
    ----------
    data_source : DataSourceDict
        Reader, groups, coordinates, variables, quality filter.
    aggregation : dict
        Coordinate and variable aggregation definitions.
    output : dict
        Grid spec, store path, and indexing details.
    catalog : str or None
        Optional path to granule catalog JSON.
    bounds : dict or None
        Optional temporal/spatial bounds for filtering.
    pipeline : dict
        Pipeline kind selector (issue #12). ``{"type": "spatial"}`` (default)
        runs the point-cloud->grid aggregation path; ``"temporal"`` /
        ``"event"`` route to the event-streaming engines added in later
        phases. Absent ``pipeline`` key in YAML defaults to ``spatial`` for
        backward compatibility with every existing config.
    worker : dict or None
        Optional Lambda worker-size selector (issue #235):
        ``{"memory": 2048|4096|8192, "extra_disk": bool}``. The runner
        resolves it to a pre-provisioned function-name suffix
        (``-<memory>``, plus ``-disk`` when ``extra_disk`` is true) on the
        lambda backend; an explicit ``function_name`` kwarg wins over it.
        Absent block -> the unsuffixed default function, byte-identical
        prior behavior. Ignored by the local backend.
    """

    data_source: DataSourceDict = field(default_factory=dict)
    aggregation: dict = field(default_factory=dict)
    output: dict = field(default_factory=dict)
    catalog: str | None = None
    bounds: dict | None = None
    pipeline: dict = field(default_factory=lambda: {"type": "spatial"})
    worker: dict | None = None


def get_pipeline_type(config: PipelineConfig) -> str:
    """Return the pipeline kind, defaulting to ``"spatial"``.

    Centralised so dispatch / strategy selection has a single source of truth.
    """
    if not isinstance(config.pipeline, dict):
        raise ValueError("pipeline must be a mapping with a 'type' key")
    t = config.pipeline.get("type", "spatial")
    if t not in _PIPELINE_TYPES:
        raise ValueError(f"pipeline.type must be one of {sorted(_PIPELINE_TYPES)} (got {t!r})")
    return t


def _normalize_filter(f: dict) -> dict:
    """Normalize one raw filter dict into canonical form (see :func:`get_filters`)."""
    if "expression" in f:
        return {"level": f.get("level"), "expression": f["expression"]}
    op = f["op"]
    out = {
        "level": f.get("level"),
        "dataset": f["dataset"],
        "column": f.get("column"),
        "op": op,
        "keep": bool(f.get("keep", True)),
    }
    if op in _SET_OPS:
        out["values"] = list(f["values"])
    else:
        out["value"] = f["value"]
    return out


def _segment_variable_names(data_source: dict) -> set[str]:
    """Names of readable segment-level (non-base) variables (issue #30).

    A non-base level may declare ``variables`` as a ``{name: path-template}``
    mapping; each name becomes a per-photon column once broadcast at read time
    (:func:`zagg.processing._read_segment_broadcasts`). The documentation-only
    ``list[str]`` form contributes nothing. Empty when no level declares the
    mapping form, so plain configs are unaffected.
    """
    levels = data_source.get("levels")
    base_level = data_source.get("base_level")
    if not isinstance(levels, dict) or base_level is None:
        return set()
    names: set[str] = set()
    for name, lvl in levels.items():
        if name == base_level or not isinstance(lvl, dict):
            continue
        lvl_vars = lvl.get("variables")
        if isinstance(lvl_vars, dict):
            names |= set(lvl_vars)
    return names


def _is_numeric(s: str) -> bool:
    """Check if a string is a numeric literal."""
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False


def _validate_expression_columns(var_name: str, expr: str, ds_vars: set[str]) -> None:
    """Check that identifiers in an expression that look like column names exist."""
    # Extract bare identifiers
    tokens = set(re.findall(r"\b([a-zA-Z_]\w*)\b", expr))
    # Remove known safe names
    safe = {"np", "numpy", "len", "sum", "sqrt", "abs", "log", "exp", "float", "int"}
    for tok in tokens - safe:
        if tok in ds_vars:
            continue
        # If it's an attribute (e.g. np.sqrt) the parent object handles it
        # Only flag tokens that could plausibly be columns but aren't
        if tok not in dir(np) and not hasattr(np, tok):
            raise ValueError(
                f"Variable '{var_name}': expression references '{tok}' "
                f"which is not in data_source.variables or numpy namespace"
            )
