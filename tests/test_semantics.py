"""Semantic-core hash canonicalization (issue #299 phase 1, D19 / §8.3).

The §8.3 obligations: syntactic edits (whitespace, key order, comments) never
change the hash; packaging-knob edits (orders, chunking, worker size, read
machinery, carrier) never change it; any semantic edit does.
"""

import copy

import pytest
import yaml

from zagg.config import PipelineConfig, default_config
from zagg.semantics import (
    canonical_semantic_json,
    semantic_core,
    semantic_fingerprint,
    semantic_hash,
)


def _cfg(**overrides) -> PipelineConfig:
    cfg = default_config("atl06")
    for dotted, value in overrides.items():
        parts = dotted.split("__")
        target = getattr(cfg, parts[0])
        for p in parts[1:-1]:
            target = target.setdefault(p, {})
        target[parts[-1]] = value
    return cfg


class TestCanonicalization:
    def test_hash_shape(self):
        h = semantic_hash(_cfg())
        assert len(h) == 64 and int(h, 16) >= 0
        assert semantic_fingerprint(h) == h[:12]

    def test_deterministic(self):
        assert semantic_hash(_cfg()) == semantic_hash(_cfg())

    def test_yaml_syntax_never_changes_hash(self):
        # Same semantics, different comments/whitespace/key order.
        a = yaml.safe_load(
            "data_source:\n"
            "  reader: h5coro\n"
            "  groups: [gt1l]\n"
            "  variables: {h_li: /p}\n"
            "  coordinates: {latitude: /lat, longitude: /lon}\n"
            "aggregation:\n"
            "  variables:\n"
            "    c: {function: len, source: h_li, dtype: int32}\n"
            "output:\n"
            "  grid: {type: healpix, parent_order: 6, child_order: 12}\n"
        )
        b = yaml.safe_load(
            "# a comment\n"
            "output:\n"
            "  grid:\n"
            "    child_order:   12\n"
            "    parent_order:  6\n"
            "    type: healpix   # trailing comment\n"
            "aggregation:\n"
            "  variables:\n"
            "    c:\n"
            "      dtype: int32\n"
            "      source: h_li\n"
            "      function: len\n"
            "data_source:\n"
            "  coordinates: {longitude: /lon, latitude: /lat}\n"
            "  variables: {h_li: /p}\n"
            "  groups: [gt1l]\n"
            "  reader: h5coro\n"
        )
        ca = PipelineConfig(**a)
        cb = PipelineConfig(**b)
        assert canonical_semantic_json(ca) == canonical_semantic_json(cb)
        assert semantic_hash(ca) == semantic_hash(cb)

    def test_packaging_knobs_never_change_hash(self):
        base = semantic_hash(_cfg())
        packaging = [
            _cfg(output__grid__parent_order=9),
            _cfg(output__grid__child_order=19),
            _cfg(output__grid__chunk_inner=11),
            _cfg(output__grid__sharded=False),
            _cfg(output__store_layout="hive"),
            _cfg(output__store="s3://elsewhere/prefix"),
            _cfg(aggregation__handoff="pandas"),
            _cfg(data_source__reader="xarray"),
            _cfg(data_source__driver="https"),
        ]
        for cfg in packaging:
            assert semantic_hash(cfg) == base
        # Worker sizing (issue #235) and read knobs are packaging too.
        cfg = _cfg()
        cfg.worker = {"memory": 8192, "extra_disk": True}
        assert semantic_hash(cfg) == base
        # emit_cell_ids is the issue #304 transition hatch, not identity (F4):
        # the core reads only the grid type/scheme, so it drops out.
        cfg = _cfg()
        cfg.output["grid"]["emit_cell_ids"] = True
        assert semantic_hash(cfg) == base

    def test_credentials_provider_is_not_in_the_packaging_list(self):
        # PIN of TODAY's behavior, not an endorsement (issue #449): the
        # provider selects HOW source bytes are fetched, never WHAT is
        # computed, so it reads as packaging -- but it is not in
        # DATA_SOURCE_PACKAGING_KEYS, so declaring it moves the hash. Adding
        # `credentials_provider: lpdaac` to the GEDI template therefore moves
        # that template's identity (no GEDI store exists yet, so nothing
        # rehashes). Flagged for review; if it is ruled packaging, this test
        # flips to the equality assertion.
        from zagg.semantics import DATA_SOURCE_PACKAGING_KEYS

        assert "credentials_provider" not in DATA_SOURCE_PACKAGING_KEYS
        cfg = _cfg(data_source__credentials_provider="lpdaac")
        assert semantic_core(cfg)["data_source"]["credentials_provider"] == "lpdaac"
        assert semantic_hash(cfg) != semantic_hash(_cfg())

    def test_semantic_edits_always_change_hash(self):
        base = semantic_hash(_cfg())
        cfg = _cfg()
        cfg.aggregation["variables"]["h_min"]["function"] = "max"
        assert semantic_hash(cfg) != base
        cfg = _cfg()
        cfg.aggregation["variables"]["count"]["dtype"] = "int64"
        assert semantic_hash(cfg) != base
        cfg = _cfg()
        cfg.aggregation["variables"]["extra"] = {
            "function": "len",
            "source": "h_li",
            "dtype": "int32",
        }
        assert semantic_hash(cfg) != base
        cfg = _cfg()
        cfg.data_source["groups"] = ["gt1l"]
        assert semantic_hash(cfg) != base
        cfg = _cfg()
        cfg.data_source["quality_filter"]["value"] = 1
        assert semantic_hash(cfg) != base
        cfg = _cfg()
        cfg.data_source["variables"]["h_li"] = "/other/path"
        assert semantic_hash(cfg) != base

    def test_grid_family_is_semantic(self):
        # Grid TYPE (+ indexing scheme) is identity; the orders are not (D24).
        healpix = semantic_core(_cfg())
        assert healpix["grid"] == {"type": "healpix", "indexing_scheme": "nested"}
        rect = default_config("atl06_polar")
        rect = copy.deepcopy(rect)
        rect.output["grid"] = {
            "type": "rectilinear",
            "crs": "EPSG:3031",
            "resolution": 100,
            "bounds": [0, 0, 1000, 1000],
        }
        # F1: rect folds in the spatially-defining params (crs/resolution/
        # bounds) — D24's resolution-axis exclusion is HEALPix-only.
        assert semantic_core(rect)["grid"] == {
            "type": "rectilinear",
            "crs": "EPSG:3031",
            "resolution": 100,
            "bounds": [0, 0, 1000, 1000],
        }
        assert semantic_hash(rect) != semantic_hash(_cfg())

    def test_rect_resolution_changes_hash(self):
        # F1: two rect products differing only in resolution (or CRS) are
        # different products — they must not collide on semantic_hash.
        base = default_config("atl06_polar")
        base = copy.deepcopy(base)
        base.output["grid"] = {
            "type": "rectilinear",
            "crs": "EPSG:3031",
            "resolution": 100,
            "bounds": [0, 0, 1000, 1000],
        }
        finer = copy.deepcopy(base)
        finer.output["grid"]["resolution"] = 50
        assert semantic_hash(finer) != semantic_hash(base)
        other_crs = copy.deepcopy(base)
        other_crs.output["grid"]["crs"] = "EPSG:3413"
        assert semantic_hash(other_crs) != semantic_hash(base)

    def test_null_packaging_values_drop_out(self):
        # A present-but-null read knob (YAML `driver:`) is identical to an
        # absent one — the canonical core omits nulls.
        a, b = _cfg(), _cfg()
        b.data_source["read_plan"] = None
        assert semantic_hash(a) == semantic_hash(b)

    def test_nested_null_values_drop_out(self):
        # F2: an explicit null NESTED inside a semantic dict (YAML `value:`)
        # hashes identically to the key being absent — pruning is recursive.
        absent = _cfg()
        absent.data_source["quality_filter"] = {"dataset": "/q"}
        null_valued = _cfg()
        null_valued.data_source["quality_filter"] = {"dataset": "/q", "value": None}
        assert semantic_hash(null_valued) == semantic_hash(absent)
        # ...but a real value (0 included) is content, not an absent key.
        zero_valued = _cfg()
        zero_valued.data_source["quality_filter"] = {"dataset": "/q", "value": 0}
        assert semantic_hash(zero_valued) != semantic_hash(absent)

    def test_golden_hash_pin(self):
        # F4: a fully-inline config with fixed dicts, pinned to its exact
        # digest (re-pinned when pipeline.type joined the core, espg-ruled
        # 2026-07-21). This catches accidental canonicalization drift — any change
        # to key ordering, null pruning, the packaging-key sets, or the JSON
        # separators would move this hash, so an unintended edit fails loudly.
        cfg = PipelineConfig(
            data_source={
                "reader": "h5coro",
                "groups": ["gt1l", "gt2l"],
                "variables": {"h_li": "/land_ice_segments/h_li"},
                "coordinates": {
                    "latitude": "/land_ice_segments/latitude",
                    "longitude": "/land_ice_segments/longitude",
                },
                "quality_filter": {"dataset": "/land_ice_segments/atl06_quality_summary"},
            },
            aggregation={
                "handoff": "arrow",
                "variables": {
                    "count": {"function": "len", "source": "h_li", "dtype": "int32"},
                    "h_mean": {"function": "mean", "source": "h_li", "dtype": "float32"},
                },
            },
            output={"grid": {"type": "healpix", "parent_order": 6, "child_order": 12}},
        )
        assert semantic_hash(cfg) == (
            "98450eb2909a105f8989c55e317d27180bf92c559859ad1415bf77445f925f22"
        )

    def test_pipeline_type_is_semantic(self):
        # espg-ruled on the PR #316 review (2026-07-21): a temporal engine
        # over the same aggregation block is a DIFFERENT product. Absent
        # normalizes to the spatial default, so existing configs hash stably.
        base = _cfg()
        explicit = _cfg()
        explicit.pipeline = {"type": "spatial"}
        assert semantic_hash(base) == semantic_hash(explicit)
        temporal = _cfg()
        temporal.pipeline = {"type": "temporal"}
        assert semantic_hash(temporal) != semantic_hash(base)
        assert semantic_core(temporal)["pipeline"] == {"type": "temporal"}

    def test_fingerprint_rejects_short_input(self):
        with pytest.raises(ValueError, match="not a semantic hash"):
            semantic_fingerprint("abc")


def _digest_cfg(**meta_extra) -> PipelineConfig:
    """A minimal config carrying one ragged digest field ``d``."""
    return PipelineConfig(
        data_source={
            "reader": "h5coro",
            "groups": ["g"],
            "variables": {"h": "/p"},
            "coordinates": {"latitude": "/lat", "longitude": "/lon"},
        },
        aggregation={
            "variables": {
                "d": {
                    "kind": "ragged",
                    "function": "zagg.stats.tdigest.build_tdigest",
                    "source": "h",
                    "inner_shape": [2],
                    "dtype": "float32",
                    "params": {"delta": 8192},
                    **meta_extra,
                }
            }
        },
        output={"grid": {"type": "healpix", "parent_order": 6, "child_order": 12}},
    )


class TestWeightsAndOverviewDeltaHashing:
    """Issue #424: default weights and overview_delta never move a hash."""

    def test_weights_counts_hashes_as_absent(self):
        # The §2.0 absent-key default: explicit and absent spellings are the
        # same bytes, so they must be the same product identity.
        assert semantic_hash(_digest_cfg(weights="counts")) == semantic_hash(_digest_cfg())

    def test_weights_flux_is_semantic(self):
        gain = {"gain": {"name": "g", "version": "1"}}
        base = _digest_cfg(attrs=dict(gain))
        flux = _digest_cfg(weights="flux", attrs=dict(gain))
        assert semantic_hash(flux) != semantic_hash(base)
        assert semantic_core(flux)["aggregation"]["variables"]["d"]["weights"] == "flux"

    def test_overview_delta_is_packaging(self):
        # The pyramid-fold budget shapes overview artifacts only; the base
        # product identity must not move when it is declared (issue #424).
        assert semantic_hash(_digest_cfg(overview_delta=512)) == semantic_hash(_digest_cfg())
        core = semantic_core(_digest_cfg(overview_delta=512))
        assert "overview_delta" not in core["aggregation"]["variables"]["d"]


class TestTimeEncodingHashing:
    """Issue #443: the §8 time-coordinate encoding is output-defining."""

    def _raster_cfg(self, encoding=None) -> PipelineConfig:
        output = {"grid": {"type": "healpix", "parent_order": 10, "child_order": 16}}
        if encoding:
            output["time_encoding"] = encoding
        return PipelineConfig(
            data_source={"reader": "raster", "bands": {"red": {"asset": "red", "dtype": "uint16"}}},
            output=output,
        )

    def test_default_encoding_hashes_as_absent(self):
        # The §8 absent-key default: an explicit `microseconds` is the same
        # stored axis, so it must be the same product identity — which is
        # also what keeps every pre-#443 config's hash byte-identical.
        assert semantic_hash(self._raster_cfg("microseconds")) == semantic_hash(self._raster_cfg())
        assert "time_encoding" not in semantic_core(self._raster_cfg("microseconds"))

    def test_toc_is_semantic(self):
        toc = self._raster_cfg("toc")
        assert semantic_hash(toc) != semantic_hash(self._raster_cfg())
        assert semantic_core(toc)["time_encoding"] == "toc"

    def test_packaged_s2_config_declares_toc(self):
        cfg = default_config("sentinel2_l2a")
        assert semantic_core(cfg)["time_encoding"] == "toc"
