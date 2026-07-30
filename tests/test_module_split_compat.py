"""Import-compat guards for the issue #330 module splits.

Every split in the series is a PURE MOVE behind a re-export facade: the
pre-split import path must keep resolving each top-level name the single-file
module defined — the documented public API *and* the private helpers other
zagg modules import by name (``zagg.coverage`` takes ``_read_json`` off
``zagg.hive``; ``zagg.sweep``/``zagg.sweep_overview`` take ``_utcnow`` and the
``_decimal_*`` family). These tests freeze that surface, so a name a facade
drops fails here instead of at some importer's call site.

The frozen tuples are the AST census of names each module DEFINED at its
pre-split commit. Growing a submodule's surface is fine; shrinking the facade's
is not. Names a pre-split module merely imported (``json``, ``zarr``,
``zagg.windows.union_time_range``, ``zagg.store.open_object_store``, …) are
deliberately NOT frozen: they were import artifacts, never API, and re-exporting
a dependency on the facade would be a trap — a ``monkeypatch.setattr`` against
the facade binding would silently miss the submodule that actually calls it.

For the same reason a facade re-export is **read** compat only. Tests that
patch an internal (``zagg.hive``'s ``open_object_store``,
``zagg.processing.raster``'s ``_STORE_CACHE``) patch the owning SUBMODULE, since
that is where the calling code reads the global from.
"""

import importlib
import pathlib

import pytest

#: CLAUDE.md §4 module-size ceiling — the reason for the splits.
LINE_LIMIT = 1000

#: Top-level names ``src/zagg/hive.py`` defined before the phase-1 split (the
#: module was 1,393 lines).
HIVE_NAMES = (
    "AGGREGATION_CORE_NAME",
    "COMMIT_ATTR",
    "COVERAGE_BOX_SLOTS",
    "COVERAGE_SIDECAR",
    "COVERAGE_SPEC",
    "HIVE_SPEC",
    "HIVE_SPEC_V2",
    "MANIFEST_NAME",
    "PRODUCT_NAME_MAX",
    "ROOT_COVERAGE_NAME",
    "_FROZEN_MANIFEST_KEYS",
    "_PRODUCT_NAME_RE",
    "_ZSTD_LEVEL",
    "_cell_ranks",
    "_decimal_base",
    "_decimal_order",
    "_decimal_rank",
    "_frozen",
    "_frozen_matches",
    "_is_base_component",
    "_is_valid_product_name",
    "_rank_tail",
    "_read_json",
    "_utcnow",
    "build_coverage",
    "build_manifest",
    "build_root_coverage",
    "check_node_invariant",
    "classify_store_root",
    "decode_coverage_bitmap",
    "effective_store_root",
    "encode_coverage_bitmap",
    "ensure_manifest",
    "leaf_block_index",
    "list_products",
    "process_and_write_hive",
    "product_root",
    "read_commit",
    "read_coverage",
    "read_coverage_bitmap",
    "read_manifest",
    "read_root_coverage",
    "root_coverage_words",
    "shard_leaf_path",
    "stamp_commit",
    "validate_manifest",
    "validate_product_name",
    "write_coverage_sidecar",
    "write_root_coverage",
    "write_semantic_core",
)

#: Top-level names ``src/zagg/processing/raster.py`` defined before the phase-2
#: split (the module was 1,266 lines).
RASTER_NAMES = (
    "_DEFAULT_SHARD_WORKERS",
    "_DEFAULT_WRITE_BUFFER",
    "_DTYPES",
    "_S3_VHOST",
    "_STORE_CACHE",
    "_STORE_LOCK",
    "_TIME_ATTRS",
    "_build_store",
    "_check_raster_grid",
    "_chord2",
    "_combine_by_ownership",
    "_geo_from_ifd",
    "_iso_us",
    "_raster_array_spec",
    "_raster_center_lonlat",
    "_raster_members",
    "_run_sync",
    "_sample_one",
    "_shard_cell_range",
    "_shard_workers",
    "_store_and_path",
    "_us_iso",
    "_write_buffer",
    "emit_raster_leaf_template",
    "emit_raster_template",
    "new_stage_stats",
    "process_and_write_raster_hive",
    "process_raster_shard",
    "raster_group_spec",
    "raster_leaf_spec",
    "raster_time_index",
    "sample_asset",
    "sample_asset_async",
    "sample_item_async",
    "write_raster_coords",
    "write_raster_leaf_slab",
    "write_raster_slab",
)

#: Top-level names ``src/zagg/sweep.py`` defined before the phase-3 split (the
#: module was 1,034 lines). Split into SIBLING modules rather than a package:
#: ``zagg.sweep`` is a CLI entry point (``python -m zagg.sweep``), which a
#: package ``__init__`` would not serve, and ``zagg.sweep_overview`` is the
#: tree's existing precedent for splitting this module family.
SWEEP_NAMES = (
    "DEFAULT_FAMILIES",
    "DebrisFamily",
    "FAMILIES",
    "MocFamily",
    "OverviewFamily",
    "SUBMAP_NAME",
    "SWEEP_SPEC",
    "StatsFamily",
    "SubmapFamily",
    "SweepFamily",
    "_NO_SIDECAR",
    "_ReprojectTarget",
    "_ancestor",
    "_generation",
    "_merged",
    "_moc_payload",
    "_node_rel",
    "_normalize_leaves",
    "_put_rollup",
    "_read_rollup",
    "_rollup_interior",
    "_rollup_key",
    "_rollup_shard_node",
    "_sidecar_window",
    "_sweep_family",
    "_warn_unsupported_submap",
    "_warned_unsupported_submap",
    "discover_leaves",
    "get_family",
    "leaves_from_stats_records",
    "main",
    "run_sweep",
    "submap_emittable",
    "submap_key",
    "sweep_after_run",
    "write_leaf_submap",
)

#: Top-level names ``src/zagg/config.py`` defined before the phase-4 split (the
#: module was 2,747 lines — the second-worst offender).
CONFIG_NAMES = (
    "DataSourceDict",
    "FILTER_OPS",
    "LevelDict",
    "LinkDict",
    "OUTPUT_KINDS",
    "OUTPUT_RESOLUTIONS",
    "PipelineConfig",
    "WORKER_MEMORIES",
    "_NAN_AMBIGUOUS_REDUCTIONS",
    "_PIPELINE_TYPES",
    "_RESAMPLE_HOWS",
    "_SCALAR_OPS",
    "_SET_OPS",
    "_TEMPORAL_SPEC_KEYS",
    "_eval_expression_raw",
    "_is_nan_fill",
    "_is_numeric",
    "_normalize_filter",
    "_segment_variable_names",
    "_validate_chunk_precompute",
    "_validate_collection_options",
    "_validate_composition_attrs",
    "_validate_ds_variables",
    "_validate_expression_columns",
    "_validate_filter_levels",
    "_validate_filters",
    "_validate_index",
    "_validate_levels",
    "_validate_output_kind",
    "_validate_pyramid",
    "_validate_raster_config",
    "_validate_store_layout_keys",
    "_validate_temporal_config",
    "_validate_trailing_shape",
    "_validate_windowing",
    "_validate_windowing_windows",
    "_validate_worker",
    "_variable_entry",
    "_warn_nan_ambiguous_reductions",
    "collection_options",
    "default_config",
    "evaluate_expression",
    "evaluate_filter_expression",
    "filters_from_data_source",
    "get_agg_fields",
    "get_aoi_mask",
    "get_base_level",
    "get_child_order",
    "get_chunk_precompute",
    "get_consolidate_metadata",
    "get_coords",
    "get_coverage_moc",
    "get_data_vars",
    "get_driver",
    "get_emit_cell_ids",
    "get_filters",
    "get_handoff",
    "get_layout",
    "get_levels",
    "get_output_endpoint_url",
    "get_output_region",
    "get_output_signature",
    "get_parent_order",
    "get_pipeline_type",
    "get_product_name",
    "get_pyramid",
    "get_raster_bands",
    "get_sharded",
    "get_store_layout",
    "get_store_path",
    "get_sweep",
    "get_windowing",
    "load_config",
    "load_config_from_dict",
    "output_field_signature",
    "resolve_function",
    "validate_config",
    "window_time_filters",
    "windowed_cell_config",
)

#: ``{facade module: (pre-split names, submodules it re-exports from)}``.
SPLITS = {
    "zagg.config": (
        CONFIG_NAMES,
        ("base", "expressions", "accessors", "validate_source", "validate_output", "validate"),
    ),
    "zagg.hive": (HIVE_NAMES, ("layout", "manifest", "coverage")),
    "zagg.processing.raster": (RASTER_NAMES, ("decode", "template", "write")),
    "zagg.sweep": (SWEEP_NAMES, ("zagg.sweep_families", "zagg.sweep_rollup")),
}


def _submodules(facade):
    """Import every module the facade re-exports from.

    A split is either a PACKAGE (``zagg.hive`` -> ``zagg.hive.layout``, named
    relatively) or SIBLING modules (``zagg.sweep`` -> ``zagg.sweep_families``,
    named absolutely because ``zagg.sweep`` must stay a module to keep working
    as ``python -m zagg.sweep``).
    """
    for sub in SPLITS[facade][1]:
        yield importlib.import_module(sub if "." in sub else f"{facade}.{sub}")


@pytest.mark.parametrize("facade", sorted(SPLITS))
class TestSplitFacade:
    """The old import path keeps its whole pre-split name surface."""

    def test_facade_resolves_every_pre_split_name(self, facade):
        module = importlib.import_module(facade)
        missing = [name for name in SPLITS[facade][0] if not hasattr(module, name)]
        assert missing == [], f"{facade} facade dropped {missing}"

    def test_from_import_works_for_every_pre_split_name(self, facade):
        """``from <facade> import <name>`` — how importers actually reach these."""
        names = SPLITS[facade][0]
        module = __import__(facade, fromlist=list(names))
        assert [name for name in names if not hasattr(module, name)] == []

    def test_all_stays_within_the_pre_split_surface(self, facade):
        module = importlib.import_module(facade)
        if not hasattr(module, "__all__"):
            pytest.skip(f"{facade} declared no __all__ before the split either")
        assert set(module.__all__) <= set(SPLITS[facade][0])
        assert [name for name in module.__all__ if not hasattr(module, name)] == []

    def test_facade_names_are_the_submodule_objects(self, facade):
        """Re-export, not a copy — identity, so ``is`` comparisons still hold."""
        module = importlib.import_module(facade)
        seen = 0
        for submodule in _submodules(facade):
            for name in SPLITS[facade][0]:
                if name in vars(submodule):
                    assert getattr(module, name) is getattr(submodule, name)
                    seen += 1
        assert seen, f"{facade} re-exports nothing from {SPLITS[facade][1]}"

    def test_every_split_module_is_under_the_line_limit(self, facade):
        paths = [pathlib.Path(m.__file__) for m in _submodules(facade)]
        facade_path = pathlib.Path(importlib.import_module(facade).__file__)
        paths.append(facade_path)
        if facade_path.name == "__init__.py":  # a package: catch stragglers too
            paths.extend(facade_path.parent.glob("*.py"))
        oversize = {
            path.name: len(path.read_text().splitlines())
            for path in sorted(set(paths))
            if len(path.read_text().splitlines()) > LINE_LIMIT
        }
        assert oversize == {}, f"{facade} modules over {LINE_LIMIT} lines: {oversize}"


class TestHiveSplitSeams:
    """Each pre-split name landed in the submodule its seam says it should."""

    def test_layout_owns_paths_and_the_product_grammar(self):
        from zagg.hive import layout

        for name in ("shard_leaf_path", "check_node_invariant", "validate_product_name"):
            assert name in vars(layout)

    def test_manifest_owns_the_manifest_and_its_resume_guard(self):
        from zagg.hive import manifest

        for name in ("build_manifest", "validate_manifest", "ensure_manifest"):
            assert name in vars(manifest)

    def test_coverage_owns_the_stamp_and_the_rollup(self):
        from zagg.hive import coverage

        for name in ("stamp_commit", "build_coverage", "build_root_coverage"):
            assert name in vars(coverage)

    def test_the_write_path_stays_on_the_facade(self):
        """Tests patch the stamp helpers on ``zagg.hive`` to assert write order."""
        import zagg.hive as hive

        assert "process_and_write_hive" in vars(hive)
        assert hive.process_and_write_hive.__module__ == "zagg.hive"
