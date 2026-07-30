"""Filter and expression accessors for a pipeline config (issue #330).

Split out of the single-file ``zagg.config`` (issue #330 phase 4); the public
surface is unchanged and re-exported from :mod:`zagg.config`. The filter
vocabulary readers, level lookups, capability resolution, and the sandboxed
expression evaluators.
"""

from __future__ import annotations

import importlib
import logging
from collections.abc import Callable
from typing import Any

import numpy as np

from zagg.config.base import PipelineConfig, _normalize_filter

logger = logging.getLogger(__name__)


def get_filters(config: PipelineConfig) -> list[dict]:
    """Return the ordered list of normalized data-source filters (issue #43).

    Two filter languages coexist:

    - **Structured predicates** ``{level?, dataset, column?, op, value|values,
      keep?}`` are machine-inspectable and are the only kind eligible for read
      pushdown (Phase C). ``op`` is one of :data:`FILTER_OPS`; ``in``/``not_in``
      take ``values`` (a list), the rest take a scalar ``value``. ``column`` is an
      integer selector into an N-D flag array (e.g. ATL03 ``signal_conf_ph``).
      ``keep`` (default ``True``) keeps matching rows; ``keep: false`` drops them.
    - **Expression** filters ``{expression: "<py expr>"}`` are a base-level-only,
      aggregation-time escape hatch that forfeits pushdown (opaque to the planner).

    The flat ``quality_filter: {dataset, value}`` is sugar synthesizing one
    base-level ``op: eq`` structured filter, so the ATL06 path is unchanged. An
    explicit ``filters:`` list, when present, is used as-is (the flat
    ``quality_filter`` is then ignored).

    Each returned filter carries a normalized ``level`` (``None`` means the base
    level) and, for structured predicates, an explicit ``keep`` bool.

    Parameters
    ----------
    config : PipelineConfig

    Returns
    -------
    list[dict]
    """
    return filters_from_data_source(config.data_source)


def filters_from_data_source(data_source: dict) -> list[dict]:
    """Normalize the filter list from a raw ``data_source`` dict.

    Shared by :func:`get_filters` and the read path (which only holds the
    ``data_source`` mapping). See :func:`get_filters` for the schema.
    """
    explicit = data_source.get("filters")
    if explicit is not None:
        return [_normalize_filter(f) for f in explicit]
    qf = data_source.get("quality_filter")
    if qf is not None:
        return [
            {
                "level": None,
                "dataset": qf["dataset"],
                "column": None,
                "op": "eq",
                "value": qf["value"],
                "keep": True,
            }
        ]
    return []


def get_levels(config: "PipelineConfig") -> dict | None:
    """Return the ``levels`` mapping from the data source, or ``None`` if flat.

    Parameters
    ----------
    config : PipelineConfig

    Returns
    -------
    dict or None
    """
    return config.data_source.get("levels")


def get_base_level(config: "PipelineConfig") -> str | None:
    """Return the ``base_level`` key from the data source, or ``None`` if flat.

    Parameters
    ----------
    config : PipelineConfig

    Returns
    -------
    str or None
    """
    return config.data_source.get("base_level")


def resolve_function(name: str) -> Callable:
    """Resolve a function name to a callable.

    Resolution rules:
    - ``"len"`` or ``"count"`` -> builtin ``len``
    - No dot (e.g. ``"min"``) -> ``np.<name>``
    - Dotted path (e.g. ``"np.quantile"``) -> importlib resolution

    Parameters
    ----------
    name : str
        Function name or dotted path.

    Returns
    -------
    Callable

    Raises
    ------
    ValueError
        If the name cannot be resolved to a callable.
    """
    if name in ("len", "count"):
        return len

    # Normalize np. prefix to numpy lookup
    if name.startswith("np."):
        name = name[3:]

    if "." not in name:
        # numpy shorthand
        func = getattr(np, name, None)
        if func is not None and callable(func):
            return func
        raise ValueError(f"Cannot resolve '{name}' as numpy function")

    # Dotted path (e.g. numpy.quantile)
    parts = name.rsplit(".", 1)
    try:
        mod = importlib.import_module(parts[0])
        func = getattr(mod, parts[1])
    except (ImportError, AttributeError) as e:
        raise ValueError(f"Cannot resolve '{name}': {e}") from e

    if not callable(func):
        raise ValueError(f"'{name}' is not callable")
    return func


def _eval_expression_raw(expression: str, columns: dict[str, np.ndarray]) -> Any:
    """Evaluate an expression string in a restricted namespace, uncoerced.

    Returns the expression's native value (a scalar, an ndarray, ...). Used by
    vector ``expression`` fields (issue #29), which coerce the result through
    ``_coerce_field_value`` rather than casting to ``float``.

    Parameters
    ----------
    expression : str
        Python expression using numpy and column variables.
    columns : dict[str, np.ndarray]
        Mapping of column names to arrays.

    Returns
    -------
    Any
        Whatever the expression evaluates to.
    """
    ns = {
        "__builtins__": {},
        "np": np,
        "numpy": np,
        "len": len,
        "float": float,
        "int": int,
        "abs": abs,
        "sum": sum,
        **columns,
    }
    return eval(expression, ns)  # noqa: S307


def evaluate_expression(expression: str, columns: dict[str, np.ndarray]) -> float:
    """Evaluate an expression string in a restricted namespace.

    Parameters
    ----------
    expression : str
        Python expression using numpy and column variables.
    columns : dict[str, np.ndarray]
        Mapping of column names to arrays.

    Returns
    -------
    float
    """
    return float(_eval_expression_raw(expression, columns))


def evaluate_filter_expression(expression: str, columns: dict[str, np.ndarray]) -> np.ndarray:
    """Evaluate a boolean filter expression to a per-row mask (issue #43).

    Like :func:`evaluate_expression` but returns the raw boolean array rather than
    a scalar float — the base-level ``expression`` filter escape hatch (e.g.
    ``"(h_li > 0) & (s_li < 1)"``). Uses the same restricted namespace.

    Parameters
    ----------
    expression : str
        Python boolean expression over numpy and column variables.
    columns : dict[str, np.ndarray]
        Mapping of column names to arrays.

    Returns
    -------
    numpy.ndarray
        Boolean mask.
    """
    ns = {
        "__builtins__": {},
        "np": np,
        "numpy": np,
        "len": len,
        "float": float,
        "int": int,
        "abs": abs,
        "sum": sum,
        **columns,
    }
    return np.asarray(eval(expression, ns), dtype=bool)  # noqa: S307
