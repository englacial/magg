# Catalog

Catalog construction has two separable concerns:

1. **Fetch** — query a STAC endpoint (CMR-STAC) for *what / when / where* → a
   `Catalog` (a stac-geoparquet table of granule metadata, reusable across
   many grids).
2. **Shard map** — take a `Catalog` plus an output grid → a `ShardMap`: the
   work-distribution manifest mapping shard keys to granules.

The CLI chains them, building the output grid from the **same pipeline config
the aggregator uses**, so a shard map can never be built against a different
grid than the run (enforced at run time via `grid.signature()`).

## Building a shard map (CLI)

```bash
# HEALPix grid from atl06.yaml, an ICESat-2 cycle, Antarctic polygon:
python -m zagg.catalog --config atl06.yaml --short-name ATL06 --cycle 22 \
    --polygon antarctica.geojson

# Rectilinear (UTM) grid from a config, explicit dates, over a bbox:
python -m zagg.catalog --config serc_atl03.yaml --short-name ATL03 \
    --start-date 2025-01-01 --end-date 2025-12-31 \
    --bbox=-76.62107,38.84504,-76.50583,38.93512

# Persist the fetched Catalog too (reusable for other grids):
python -m zagg.catalog --config atl06.yaml --short-name ATL06 --cycle 22 \
    --polygon antarctica.geojson --catalog-out cycle22.parquet
```

`--polygon` drives both the CMR query bbox and the coverage mask; `--bbox`
gives the query box directly (coverage falls back to that rectangle). The
geometry backend (`--backend`) defaults to `auto`: exact-S2 spherely if the
spherely fork (with `SpatialIndex`) is installed separately, else mortie
(HEALPix) / shapely (rectilinear).

Endpoint selection (S3 vs HTTPS) is **not** made here — each granule record
keeps both hrefs, and the aggregator picks one at run time via
`data_source.driver`.

## The `footprint_cells` convention column

The per-granule footprint→cells cover is identical for every shard map built
against the same catalog, so a catalog can carry it
([issue #396](https://github.com/englacial/zagg/issues/396)):

```bash
# index while fetching, so the saved catalog ships pre-indexed
python -m zagg.catalog --config atl06.yaml --short-name ATL06 --cycle 22 \
    --polygon antarctica.geojson --catalog-out cycle22.parquet \
    --index-footprints 9
```

```python
cat = Catalog.from_geoparquet("cycle22.parquet").index_footprints(9)
cat.to_geoparquet("cycle22_indexed.parquet")
```

`footprint_cells` is a ragged `large_list` of morton MOC words — mortie's
`morton_index` extension type, the same typed-morton convention as
[morton arrow](../morton_arrow.md) — one entry per table row. This is a **zagg
convention column, not a stac-geoparquet upstream field**: an indexed catalog is
still an ordinary stac-geoparquet file, and any other reader ignores the extra
column. It is *not* part of the [store specification](../specification.md),
which governs the Zarr store the aggregator writes, not the catalog it reads.

**It cannot go stale.** The column rides in the same file as the `geometry` it
was covered from, so there is no second artifact to keep in sync and no version
skew to detect: subsetting the catalog (`filter_bbox`) carries the column with
the rows, and rewriting the geometry means rewriting the file.

`ShardMap.build` against an indexed catalog does no geometry at all — each
granule's stored MOC is intersected (`moc_and`) with the AOI's own shard MOC —
and records `metadata["footprint_cells"] = True` when it took that path. The
fast path engages only where the stored cover is exactly what the build asked
for: the mortie backend, `footprint="swath"`, and no caller-pinned
`mortie_order`. Anything else takes the geometry path unchanged, so an
exact-S2 spherely run is never silently swapped for a MOC one.

### Choosing the order

**Index at the grid's `parent_order`.** A MOC answers any shard order coarser
than or equal to its own; a build against a grid whose `parent_order` is *finer*
than the column is refused outright, because answering it would refine every
cell onto all its descendants and put ~every granule in ~every shard
([issue #92](https://github.com/englacial/zagg/issues/92)).

Going finer than you need is expensive in both directions — words per granule
roughly double per order:

| order | words/granule (88S) | column, 35,639 granules | index pass | build (35,639 granules) |
| --- | --- | --- | --- | --- |
| 9 | 288 | 17 MB parquet | 2.6 s | 1.6 s (vs 4.8 s from geometry) |
| 13 | 7,207 | 560 MB parquet | 53.3 s | 22.9 s (vs 59.3 s from geometry) |

At the shard order the index pays for itself on the first build. At the chunk
order it does not: the pass costs about as much as the build it replaces, and
the column is two orders of magnitude larger for resolution the shard cells
cannot see.

## Fetch

::: zagg.catalog.sources.Query

::: zagg.catalog.sources.CMRSource

::: zagg.catalog.sources.Catalog

## Shard map

::: zagg.catalog.shardmap.ShardMap

## Convenience

::: zagg.catalog.make_shardmap

## Temporal / spatial helpers

::: zagg.catalog.cycle_to_dates

::: zagg.catalog.load_polygon

::: zagg.catalog.polygon_to_bbox

::: zagg.catalog.load_antarctic_basins
