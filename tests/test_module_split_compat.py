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

#: ``{facade module: (pre-split names, submodules it re-exports from)}``.
SPLITS = {
    "zagg.hive": (HIVE_NAMES, ("layout", "manifest", "coverage")),
    "zagg.processing.raster": (RASTER_NAMES, ("decode", "template", "write")),
}


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
        assert set(module.__all__) <= set(SPLITS[facade][0])
        assert [name for name in module.__all__ if not hasattr(module, name)] == []

    def test_facade_names_are_the_submodule_objects(self, facade):
        """Re-export, not a copy — identity, so ``is`` comparisons still hold."""
        module = importlib.import_module(facade)
        seen = 0
        for sub in SPLITS[facade][1]:
            submodule = importlib.import_module(f"{facade}.{sub}")
            for name in SPLITS[facade][0]:
                if hasattr(submodule, name) and name in vars(submodule):
                    assert getattr(module, name) is getattr(submodule, name)
                    seen += 1
        assert seen, f"{facade} re-exports nothing from {SPLITS[facade][1]}"

    def test_every_split_module_is_under_the_line_limit(self, facade):
        package = pathlib.Path(importlib.import_module(facade).__file__).parent
        oversize = {
            path.name: len(path.read_text().splitlines())
            for path in sorted(package.glob("*.py"))
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
