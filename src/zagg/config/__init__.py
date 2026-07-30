"""YAML-driven pipeline configuration for zagg.

Package layout (issue #330 phase 4 — the module passed the CLAUDE.md §4
~1000-line limit at 2,747). Pure moves behind this facade; every name the
single-file module exposed still resolves as ``zagg.config.<name>``:

- :mod:`zagg.config.base` — :class:`PipelineConfig`, the ``TypedDict`` block
  shapes, the filter/worker vocabularies, and the shared leaf helpers.
- :mod:`zagg.config.expressions` — filter readers, level lookups, capability
  resolution, and the sandboxed expression evaluators.
- :mod:`zagg.config.accessors` — the ``get_*`` family.
- :mod:`zagg.config.validate_source` — ``data_source`` block validators
  (temporal specs, raster, collections).
- :mod:`zagg.config.validate_output` — ``output`` block validators (store
  layout, windowing, chunk precompute, output kinds, pyramid).
- :mod:`zagg.config.validate` — :func:`validate_config` and the cross-block
  schema checks.

The loaders stay here, mirroring the ratified plan's runner facade which keeps
``agg()``: they are the entry points callers reach through ``zagg.config``, and
keeping them above :mod:`zagg.config.validate` is what makes the package a DAG
(a loader calls ``validate_config``, which reads the dataclass the loader
builds).
"""

from __future__ import annotations

import logging
from importlib import resources

import yaml

import zagg.configs
from zagg.config.accessors import _variable_entry as _variable_entry
from zagg.config.accessors import collection_options as collection_options
from zagg.config.accessors import get_agg_fields as get_agg_fields
from zagg.config.accessors import get_aoi_mask as get_aoi_mask
from zagg.config.accessors import get_child_order as get_child_order
from zagg.config.accessors import get_chunk_precompute as get_chunk_precompute
from zagg.config.accessors import get_consolidate_metadata as get_consolidate_metadata
from zagg.config.accessors import get_coords as get_coords
from zagg.config.accessors import get_coverage_moc as get_coverage_moc
from zagg.config.accessors import get_data_vars as get_data_vars
from zagg.config.accessors import get_driver as get_driver
from zagg.config.accessors import get_emit_cell_ids as get_emit_cell_ids
from zagg.config.accessors import get_handoff as get_handoff
from zagg.config.accessors import get_layout as get_layout
from zagg.config.accessors import get_output_endpoint_url as get_output_endpoint_url
from zagg.config.accessors import get_output_region as get_output_region
from zagg.config.accessors import get_output_signature as get_output_signature
from zagg.config.accessors import get_parent_order as get_parent_order
from zagg.config.accessors import get_product_name as get_product_name
from zagg.config.accessors import get_pyramid as get_pyramid
from zagg.config.accessors import get_raster_bands as get_raster_bands
from zagg.config.accessors import get_sharded as get_sharded
from zagg.config.accessors import get_store_layout as get_store_layout
from zagg.config.accessors import get_store_path as get_store_path
from zagg.config.accessors import get_sweep as get_sweep
from zagg.config.accessors import get_windowing as get_windowing
from zagg.config.accessors import output_field_signature as output_field_signature
from zagg.config.accessors import window_time_filters as window_time_filters
from zagg.config.accessors import windowed_cell_config as windowed_cell_config
from zagg.config.base import _PIPELINE_TYPES as _PIPELINE_TYPES
from zagg.config.base import _SCALAR_OPS as _SCALAR_OPS
from zagg.config.base import _SET_OPS as _SET_OPS
from zagg.config.base import FILTER_OPS as FILTER_OPS
from zagg.config.base import WORKER_MEMORIES as WORKER_MEMORIES

# Facade re-exports (issue #330 phase 4). ``zagg.config`` declared no
# ``__all__`` before the split, so the whole pre-split surface is re-exported
# with the ``x as x`` redundant-alias marker except the two names the loaders
# below call directly. READ compat only: patch the owning submodule, not the
# re-export.
from zagg.config.base import DataSourceDict as DataSourceDict
from zagg.config.base import LevelDict as LevelDict
from zagg.config.base import LinkDict as LinkDict
from zagg.config.base import PipelineConfig
from zagg.config.base import _is_numeric as _is_numeric
from zagg.config.base import _normalize_filter as _normalize_filter
from zagg.config.base import _segment_variable_names as _segment_variable_names
from zagg.config.base import _validate_expression_columns as _validate_expression_columns
from zagg.config.base import get_pipeline_type as get_pipeline_type
from zagg.config.expressions import _eval_expression_raw as _eval_expression_raw
from zagg.config.expressions import evaluate_expression as evaluate_expression
from zagg.config.expressions import evaluate_filter_expression as evaluate_filter_expression
from zagg.config.expressions import filters_from_data_source as filters_from_data_source
from zagg.config.expressions import get_base_level as get_base_level
from zagg.config.expressions import get_filters as get_filters
from zagg.config.expressions import get_levels as get_levels
from zagg.config.expressions import resolve_function as resolve_function
from zagg.config.validate import _validate_composition_attrs as _validate_composition_attrs
from zagg.config.validate import _validate_ds_variables as _validate_ds_variables
from zagg.config.validate import _validate_filter_levels as _validate_filter_levels
from zagg.config.validate import _validate_filters as _validate_filters
from zagg.config.validate import _validate_index as _validate_index
from zagg.config.validate import _validate_levels as _validate_levels
from zagg.config.validate import _validate_worker as _validate_worker
from zagg.config.validate import validate_config
from zagg.config.validate_output import _NAN_AMBIGUOUS_REDUCTIONS as _NAN_AMBIGUOUS_REDUCTIONS
from zagg.config.validate_output import OUTPUT_KINDS as OUTPUT_KINDS
from zagg.config.validate_output import OUTPUT_RESOLUTIONS as OUTPUT_RESOLUTIONS
from zagg.config.validate_output import _is_nan_fill as _is_nan_fill
from zagg.config.validate_output import _validate_chunk_precompute as _validate_chunk_precompute
from zagg.config.validate_output import _validate_output_kind as _validate_output_kind
from zagg.config.validate_output import _validate_pyramid as _validate_pyramid
from zagg.config.validate_output import _validate_store_layout_keys as _validate_store_layout_keys
from zagg.config.validate_output import _validate_trailing_shape as _validate_trailing_shape
from zagg.config.validate_output import _validate_windowing as _validate_windowing
from zagg.config.validate_output import _validate_windowing_windows as _validate_windowing_windows
from zagg.config.validate_output import (
    _warn_nan_ambiguous_reductions as _warn_nan_ambiguous_reductions,
)
from zagg.config.validate_source import _RESAMPLE_HOWS as _RESAMPLE_HOWS
from zagg.config.validate_source import _TEMPORAL_SPEC_KEYS as _TEMPORAL_SPEC_KEYS
from zagg.config.validate_source import _validate_collection_options as _validate_collection_options
from zagg.config.validate_source import _validate_raster_config as _validate_raster_config
from zagg.config.validate_source import _validate_temporal_config as _validate_temporal_config

logger = logging.getLogger(__name__)


def load_config(path: str) -> PipelineConfig:
    """Load a YAML config file and return a validated PipelineConfig.

    Parameters
    ----------
    path : str
        Path to YAML file.

    Returns
    -------
    PipelineConfig
    """
    with open(path) as f:
        d = yaml.safe_load(f)
    cfg = load_config_from_dict(d)
    validate_config(cfg)
    return cfg


def load_config_from_dict(d: dict) -> PipelineConfig:
    """Build a PipelineConfig from a plain dict (e.g. Lambda JSON payload).

    Parameters
    ----------
    d : dict
        Dictionary with keys ``data_source``, ``aggregation``, ``output``.

    Returns
    -------
    PipelineConfig
    """
    return PipelineConfig(
        data_source=d.get("data_source", {}),
        aggregation=d.get("aggregation", {}),
        output=d.get("output", {}),
        catalog=d.get("catalog"),
        bounds=d.get("bounds"),
        pipeline=d.get("pipeline", {"type": "spatial"}),
        worker=d.get("worker"),
    )


def default_config(name: str = "atl06") -> PipelineConfig:
    """Load a built-in YAML config shipped with the package.

    Parameters
    ----------
    name : str
        Config name (without ``.yaml`` extension). Default ``"atl06"``.

    Returns
    -------
    PipelineConfig

    Raises
    ------
    FileNotFoundError
        If the named config does not exist.
    """
    ref = resources.files(zagg.configs).joinpath(f"{name}.yaml")
    if not ref.is_file():
        raise FileNotFoundError(f"No built-in config named '{name}'")
    text = ref.read_text(encoding="utf-8")
    d = yaml.safe_load(text)
    cfg = load_config_from_dict(d)
    validate_config(cfg)
    return cfg
