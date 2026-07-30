# zagg store specification (1.0)

This page is the **normative record** of zagg's byte-level store conventions:
the ragged vlen-bytes layout, the t-digest payload bytes, the packed
composition word, the pyramid/overview declarations, and the O11 content-hash
recipe. It exists so an independent reader
([moczarr](https://github.com/espg/moczarr)) can decode a zagg store from this
page and the committed conformance fixtures alone — no zagg import, no
reverse-engineering of `grids/base.py`
([issue #340](https://github.com/englacial/zagg/issues/340), the
reader-migration gate).

The precedent is mortie's
[`docs/specification.md`](https://github.com/espg/mortie/blob/main/docs/specification.md),
which governs everything *below* this page: the packed morton word, the
decimal path grammar, the morton-hive tree layout and leaf naming, the
coverage-MOC serializations, and the rank-space deinterleave. This page owns
the **array-level** contracts inside a leaf; it cites mortie's page for path
and word semantics and never restates them.

Design *rationale* — why each decision was made, with trade studies and
ratification records — lives in
[`design/sparse_coverage.md`](design/sparse_coverage.md) (the D/O-numbered
decisions registry) and in the narrative companions
[`ragged_layout.md`](ragged_layout.md) and
[`signal_strata.md`](signal_strata.md). Those documents *cite* this page; this
page is the spec. Byte layouts, attrs grammars, and constants are normative
**here only** — duplicated normative text drifts.

## Normative language

The key words **MUST**, **MUST NOT**, **SHOULD**, and **MAY** are used as in
RFC 2119. Text marked **Contract** is frozen for the revision that carries it;
text marked *informative* explains or motivates and binds nothing.

## Conformance

- Every versioned convention on this page is signaled in store metadata — by a
  `spec` marker (`"zagg-ragged/1"`, `"zagg-composition/1"`, …), the
  coverage-envelope discipline, **or, where a section names it, by the array's
  registered zarr data type**. A conforming reader MUST strict-check whichever
  signal the owning section names and **fail loudly on an unknown or future
  revision**, never half-parse under a guessed layout.
- Marker-*absence* is legal only where a section names the data type that
  replaces the marker. Today there is exactly one such carve-out: the typed
  `vlen-ndarray` dtype **is** the `zagg-ragged/2` signal, and the `ragged`
  attrs marker is deliberately retired (not written) on those arrays
  (§1.6/§6.1) — so a reader MUST NOT treat a missing marker there as an
  unknown revision. Everywhere else absence is a hard failure: a
  `variable_length_bytes` array with no `ragged` block is not signaled at all
  and MUST be refused (§1.2).
- A revision, once published here, is **frozen**: its text never changes
  semantics, and stores written under it remain valid indefinitely. New
  behavior is a new revision (`/2`, `/3`, …) with its own section and an
  explicit succession clause; readers add revisions, they never drop them.
- The committed conformance fixtures (§7) are part of the contract: a reader
  implementation that reproduces the fixtures' expected decoded values and
  content hashes conforms to §1–§3 and §5. zagg's own test suite asserts the
  same expectations (`tests/test_spec_conformance.py`), so the spec, the
  fixtures, and the shipping reader cannot drift apart silently.

Contents:

1. [`zagg-ragged/1` — the vlen-bytes ragged layout](#1-zagg-ragged1)
2. [Digest payload semantics](#2-digest-payload-semantics)
3. [`zagg-composition/1` — the packed composition word](#3-zagg-composition1)
4. [Pyramid / overview declarations](#4-pyramid-overview-declarations)
5. [O11 content hashes](#5-o11-content-hashes)
6. [`zagg-ragged/2` — the typed `vlen-ndarray` revision](#6-zagg-ragged2)
7. [Conformance fixtures](#7-conformance-fixtures)

---

## 1. `zagg-ragged/1`

**Status: contract — pinned as the 1.0 wire contract**
([#340 amendment](https://github.com/englacial/zagg/issues/340)). This is the
shipping format, not a placeholder: `/1` stores remain valid indefinitely,
existing stores never require rewriting, and every conforming reader MUST
support `/1` unconditionally. Rationale and history:
[`ragged_layout.md`](ragged_layout.md),
[issue #209](https://github.com/englacial/zagg/issues/209).

A `kind: ragged` field (per-cell variable-length data — e.g. a t-digest) is
stored as **one `variable_length_bytes` zarr v3 array per field** on the cells
axis. Source of truth in code: `zagg.grids.base.ragged_array_spec` /
`RAGGED_ELEMENT_ATTR` / `RAGGED_SPEC`.

### 1.1 Layout

A ragged field `{field}` under a product group is up to three sibling arrays:

```text
{group}/{field}             <- vlen payload array; populated cell i holds the raw
                               little-endian bytes of its (n, *inner_shape) payload
{group}/{field}_locations   <- LOCATED fields only (issue #87): per-row uint64
                               location words, row-aligned with {field}
{group}/morton              <- per-cell uint64 morton coordinate (zagg's standard
                               HEALPix coordinate array; the chunk-identity source)
```

- Each populated cell's value MUST be the raw **little-endian** bytes of an
  `(n, *inner_shape)` array (`n` varies per cell — e.g. `(k_centroids, 2)` for
  a t-digest whose `inner_shape` is `(2,)`), C-order, in the declared element
  dtype, independent of the producing machine.
- Empty cells keep the `b""` fill (`fill_value: ""`); an inner chunk with no
  ragged data MUST be **absent from the store** — omitted as an object in the
  per-inner-chunk geometry, and marked absent with the §1.5 sentinel inside
  the shard object when sharded (the default). Either way object size scales
  with populated chunks only — the same sub-shard sparsity the dense arrays
  get.
- A **located** field's sibling `{field}_locations` array is itself a
  `zagg-ragged/1` vlen array (element dtype `uint64`, empty `inner_shape`)
  with the same shape and chunk geometry as the payload array, and MUST be
  **row-aligned**: cell `i` of the sibling holds exactly one `uint64` word per
  payload row of cell `i`. Readers MUST bind the sibling by the payload
  array's `locations` attrs declaration (§1.2), never by reconstructing the
  `{field}_locations` naming convention.

### 1.2 The `ragged` attrs block

The element interpretation is **self-describing** in the payload array's attrs
under the `ragged` key:

```json
"ragged": {
  "spec": "zagg-ragged/1",
  "element": {"dtype": "float32", "shape": [-1, 2]},
  "locations": "h_tdigest_locations"
}
```

- **`spec`** — the convention revision. Readers MUST strict-check it: an
  unknown/future spec raises, never half-parses.
- **`element`** — `{"dtype": "<numpy dtype>", "shape": [-1, *inner_shape]}`.
  The `-1` marks the per-cell varying count; a reader reconstructs cell `i` as
  `np.frombuffer(a[i], dtype).reshape(-1, *inner_shape)`, with the dtype read
  little-endian.
- **`locations`** — present only on a located field's payload array; its value
  is the name of the sibling uint64 array carrying the per-row location words.
  An unlocated field records nothing here.

A vlen array without a well-formed `element` declaration is **not** a
`zagg-ragged/1` array; a reader MUST refuse it with a pointed error rather
than decode under a guessed layout (pre-issue-209 CSR stores are a hard
break). The `ragged` attrs key is reserved: config-declared field attrs MUST
NOT shadow it (enforced at config validation). A located field's provenance
attrs (e.g. `stratum`, `signal_threshold` — §3.3) land on the **payload array
only**; the `{field}_locations` sibling carries no user attrs.

### 1.3 Codec chain

The per-chunk codec chain MUST be `[vlen-bytes, zstd(level=3, checksum=false)]`.
The zstd deviates from the dense arrays' bytes-only/uncompressed policy
deliberately: a vlen payload has no fixed-width raw layout to preserve, and
the level is fixed so identical payloads produce identical objects across
workers. (zarr-python names the dtype `variable_length_bytes` in array
metadata while the v3 registry name is `bytes` — zarr-python#3517, accepted
both ways on read; readers MUST accept the `variable_length_bytes` spelling.)

### 1.4 Wire framing

**Contract (golden-pinned).** Within one inner chunk the `vlen-bytes` codec
frames the cells before compression as (little-endian throughout):

```text
u32  cell_count
per cell:  u32 payload_length  ||  payload_bytes
```

i.e. numcodecs' `VLenBytes`/`VLenArray` framing — a `u32` count of cells, then
for each cell a `u32` byte length followed by that many payload bytes (`0` for
an empty cell). The `payload_bytes` are
`np.ascontiguousarray(value).tobytes()` of the cell's `(n, *inner_shape)`
array in the declared dtype. This exact byte vector is frozen by a golden test
(`tests/test_processing.py::TestRaggedVlenLayout::test_golden_inner_chunk_framing`)
and exercised by the §7 fixtures; it is what makes the §6 typed-dtype
revision a metadata-only migration.

### 1.5 Storage geometries

Both geometries hold the same logical data and are self-describing in the
array's own zarr metadata (its `chunk_grid`, and whether a `sharding_indexed`
codec wraps the chain), so a single reader code path MUST read either —
deriving the stored span from the array's shard shape when sharded, else its
chunk shape:

| geometry | on disk | single-cell read |
|---|---|---|
| **sharded** (`ShardingCodec`; every hive leaf and the sharded flat path) | ONE object per shard; the shard's K inner chunks live inside it with an internal index | 2 ranged GETs (index suffix + one ranged inner chunk) |
| **per-inner-chunk** (regular array; the unsharded streaming path) | one object per inner chunk | 1 GET (the object) |

When sharded, the §1.3 chain rides INSIDE a `sharding_indexed` codec whose
outer chunk spans the shard, with index codecs
`[bytes(endian=little), crc32c]` and `index_location: end`: the shard object's
suffix is K `(offset, nbytes)` u64 pairs plus a crc32c, and an inner chunk
with no ragged data is marked absent with the `2^64 - 1` sentinel in both
fields (zarr v3 sharding spec) — object size scales with **populated** chunks
only. The 2-GET random-access recipe follows: fetch the
`16*K + 4`-byte index suffix, then the one ranged inner chunk holding the
cell.

### 1.6 Succession

The `ragged` attrs block is `/1`'s element contract. The candidate successor
is the [§6](#6-zagg-ragged2) typed-dtype revision (`zagg-ragged/2`,
[issue #210](https://github.com/englacial/zagg/issues/210)). The signaling
mechanics are normative now:

- An array whose zarr data type is the typed `vlen-ndarray` dtype **is**
  `zagg-ragged/2`; on such arrays the `ragged` attrs marker is **retired**
  (not written).
- An array with the `variable_length_bytes`/`bytes` dtype and a
  `spec: "zagg-ragged/1"` attrs block **is** `zagg-ragged/1`.
- `/1` remains valid indefinitely — existing stores never require rewriting,
  and every conforming reader supports `/1` unconditionally, whatever timing
  the `/2` implementation lands on.

## 2. Digest payload semantics

**Status: contract (payload bytes); the digest algebra is deliberately NOT
specified.**

A t-digest field is a `zagg-ragged/1` (or `/2`) array whose element
declaration is `{"dtype": "float32", "shape": [-1, 2]}`. Source of truth in
code: `zagg.stats.tdigest`.

### 2.1 Centroid array

**Contract.** A populated cell's decoded payload is a `(k, 2)` **float32**
array of weighted centroids:

- column 0 is the centroid **mean**; column 1 is the centroid **weight**
  (the number of observations merged into it, ≥ 1);
- rows MUST be sorted **ascending by mean**;
- `sum(weights)` MUST equal the cell's **exact** observation count — the
  number of finite `source` values the digest was built over (non-finite
  source rows are dropped before building) — **while that count is
  representable in float32, i.e. `<= 2^24` (16,777,216)**; above that bound
  the weights and their sum are the nearest float32 values to the true counts,
  so a reader recovering counts from weights (§3.3 tells it to, for
  `N_signal`) gets the exact integer at or below the bound and a rounded one
  above it. The bound is comfortable at leaf cell orders and is the one to
  watch at coarse overview orders (§4.4). For a stratified product (§3) each
  stratum digest's total weight is the exact stratum count, under the same
  bound;
- an absent cell decodes as the zero-length `(0, 2)` array (the `b""` fill).

### 2.2 The location channel

**Contract.** A located digest field (issue #87) carries one **uint64 morton
word per centroid row** in its `{field}_locations` sibling (§1.1), row-aligned
with the payload:

- Per-observation locations enter as **order-29 point-kind morton words**
  (mortie spec §4 — encoding-carried kind; an exact observation position).
- A centroid of weight 1 carries its single member's word unchanged — an
  exact observation position.
- A merged centroid carries the **deepest common ancestor** of its members'
  words (`mortie.common_ancestor`): the finest morton cell enclosing every
  member. Point and area words share the same path prefix, so mixed inputs
  (a fresh point word folded with an earlier merge's coarser area word)
  compose under the same rule.

Word semantics (bit layout, kind marking, coarsening) are mortie's
specification §1/§4, not restated here.

### 2.3 What is deliberately not specified (informative)

The build/merge **algebra** — Dunning's k1 scale function, the `delta`
compression budget, merge order, `merge_tdigests` / `merge_tdigests_kway` —
is zagg-owned and referenced informatively only (`zagg.stats.tdigest`
docstrings). Readers do not need it to decode: the stored bytes above are the
whole contract, and keeping the algebra out of the spec preserves zagg's
freedom to optimize it (issue #279) without a spec revision. Two consequences
a consumer should know (informative): digest merging is **order-dependent**
(approximate composability class — `np.isclose`, not byte equality, across
different fold orders), and quantile estimates from the centroids are
approximations with the usual t-digest accuracy profile (tight tails, looser
middle).

## 3. `zagg-composition/1`

**Status: contract.** Source of truth in code: `zagg.stats.composition`
(issue #321). Rationale and narrative:
[`signal_strata.md`](signal_strata.md).

A composition field is one dense **uint64** word per cell carrying eight
8-bit lanes of quantized fractions of the cell's **signal stratum**
(`N_signal` = the signal digest's total weight — magnitude lives in the
digest, composition here). An empty signal stratum packs to `0`, and a
composition array **MUST** declare `fill_value: 0` so an *unwritten* cell
decodes to the same word: readers key presence off `lane > 0` (§3.2), so a
nonzero fill would make every unwritten cell report spurious flag presence
(a fill of `1` reads as lanes `[1,0,0,0,0,0,0,0]` — "`land` occurred
exactly"). Enforced at config validation, alongside the §3.3 `of`/`threshold`
cross-checks.

### 3.1 Word layout

**Contract.** Lanes are packed **LSB-first**: lane `i` occupies bits
`8*i .. 8*i + 7` of the word. Lane order (`LANES`):

| lane (byte) | meaning |
|---|---|
| 0–4 | per-surface fractions, `signal_conf_ph` column order: `land`, `ocean`, `sea_ice`, `land_ice`, `inland_water` — the count of signal photons whose per-surface confidence clears the threshold, over `N_signal` |
| 5–7 | `low` / `med` / `high`: signal photons whose *strongest* per-surface confidence is exactly 2 / 3 / 4, over `N_signal` |

- The per-surface lanes are **overlapping marginals** (`surf_type` is
  multi-hot): they do not sum to 255 and cannot split the height distribution
  per surface.
- The level lanes are **absolute** — always `conf == 2 / 3 / 4`, never
  renumbered against the signal `threshold`. A product committing a higher
  threshold ships **empty** lower lanes rather than shifted ones
  (`threshold=3` leaves `low` structurally 0; `threshold=4` leaves `low` and
  `med` 0), so one lane layout serves every product. For ATL03 confidences
  (`-2..4`) the three level lanes partition the signal stratum exactly; a
  source with confidences above 4 is out of contract for this revision.

**Golden word.** For a single signal photon with per-surface confidences
`[4, -1, 0, 3, 1]` at `threshold=2`, the lanes are
`[255, 0, 0, 255, 0, 0, 0, 255]` (land, land_ice; strongest = 4 ⇒ high) and
the packed word is exactly

```text
0xFF000000FF0000FF
```

(an MSB-first layout would give `0xFF0000FF000000FF`). Pinned by
`tests/test_composition.py::TestPackComposition::test_golden_word_pins_lsb_first_byte_order`
and the §7 kitchen-sink fixture.

### 3.2 Quantization: the presence floor

**Contract.** Lanes quantize as `k = round(255 * c / N)` — round-half-even,
clipped to `0..255` — **except any nonzero count quantizes to at least 1**
(the presence floor). Consequences:

- `lane > 0` means "this flag occurred" **exactly, at every N**, through
  arbitrary merge chains.
- Count recovery `round(k * N / 255)` is exact whenever `N <= 254`
  (quantization error `<= N/510 < 1/2`).
- Above that, counts are within `±N/510` (plus `O(N/510)` per re-quantizing
  merge); presence stays exact.
- A cell with one signal photon has lanes in `{0, 255}` — the lanes *are*
  that photon's flags.

### 3.3 The attrs block

**Contract.** The composition array's attrs carry the versioned
`composition` block; readers MUST bind to it, never to config conventions,
and MUST strict-check `spec` per the conformance rule:

```json
"composition": {
  "spec": "zagg-composition/1",
  "lanes": ["land", "ocean", "sea_ice", "land_ice", "inland_water", "low", "med", "high"],
  "of": "h_tdigest_signal",
  "threshold": 2
}
```

- **`lanes`** — the lane names in bit order (LSB byte first). For `/1` the
  value is exactly the §3.1 order — lane order is **not** a product knob: the
  packer writes lane `i` at bits `8*i .. 8*i+7` in that fixed order, so a
  permuted, truncated, or renamed `lanes` value is **out of contract** and a
  writer MUST reject it rather than emit it.
- `spec` and `lanes` are **writer-stamped, not author-declared**: the store's
  values come from the writer's own constants
  (`zagg.stats.composition.COMPOSITION_SPEC` / `LANES`, stamped onto the array
  spec by `grids.base.apply_field_attrs`), the same posture as the §1.2
  `ragged` block, and a config declaration that disagrees with either is
  rejected at config validation. `of` and `threshold` are per-product and
  author-declared, cross-checked at config validation (`of` must name a
  declared `kind: ragged` field; `threshold` must equal the reducer's own
  `params.threshold`).
- **`of`** — the name of the sibling digest field whose total weight is the
  `N_signal` the lanes are fractions of. The composition word is
  uninterpretable without it: readers recover counts by pairing the word with
  that digest's `sum(weights)`.
- **`threshold`** — the committed signal cut (`conf >= threshold`; the ATBD
  predicate is `> 1`, i.e. `threshold=2`). Each stratum digest's payload
  array carries the companion provenance attrs `stratum`
  (`"signal"`/`"noise"`) and `signal_threshold`, which MUST agree with this
  value.

### 3.4 Merge law

**Contract** (normative here, not a zagg implementation detail: a reader
folding views — e.g. cells into a coarser cell — must reproduce it). Two
`(word, n_signal)` pairs fold as the digest-weighted mean per lane,
re-quantized with the same presence floor:

```text
lane_merged = quantize((n_a * lane_a + n_b * lane_b) / (n_a + n_b))
```

where `quantize` rounds half-even, clips to `0..255`, and floors a lane that
is nonzero on **either** input to at least 1. The identity element is
`(0, 0)`; a pair with `n <= 0` returns the other word unchanged. The
operation is symmetric and, up to the bounded re-quantization error,
associative — fold order never affects presence, and affects counts only
within `O(n/510)`. The `n` inputs come from the `of` digests' total weights
(§3.3).

## 4. Pyramid / overview declarations

**Status: ratified design; implementation in flight
([#201](https://github.com/englacial/zagg/issues/201)).** The decisions this
section records are ratified (D11/D22–D24 in
[`design/sparse_coverage.md`](design/sparse_coverage.md) and the
[#201 rulings](https://github.com/englacial/zagg/issues/201)); the grammar
below is what the #201 implementation lands and what moczarr's level-node
reader plans against (espg/moczarr#15, the 8b seam). Any divergence
discovered while landing #201 is resolved **on this section first** — the
implementation conforms to the spec, never the reverse.

### 4.1 Overview zarrs at ancestor nodes

An **overview zarr** is a sweep-built coarse materialization of a subtree's
committed leaves, written at an **ancestor digit node** of the hive tree
(tree layout and path grammar: mortie's specification). It has the same
structure as a leaf (§4.4), one basename dialect (§4.2), and is classified by
attrs alone (§4.3).

Overviews are **regenerable caches**, never load-bearing: deleting every
overview MUST leave all leaf reads intact, and a reader MUST NOT require
them. They are stale-detectable, not stale-proof — after a leaf re-run an
ancestor overview may lag until the sweep regenerates it; the generation
stamp (§4.3) is what makes that detectable.

### 4.2 Naming

Overviews inherit the leaf window-naming dialect (D23; grammar frozen on the
mortie spec page): at an ancestor node an overview for time window `{window}`
is `{window}.zarr`, and the reserved token **`all`** names the all-time fold
(`all.zarr` — the same token that names a `schedule: none` store's leaves;
excluded from the window grammar forever). Nothing about the *name*
distinguishes an overview from a leaf — classification is §4.3's job.

### 4.3 The `role` and `zagg_overview` attrs

**Contract.** Classification is carried in the zarr's **root-group attrs**,
never inferred from tree position or depth — a shallow zarr may equally be
*coarse source* in a sparse region (D24: one product tree may carry
regionally heterogeneous resolution).

- **`role`** — `"overview"` on every sweep-built overview. **Source leaves
  carry no `role` key: absence means source.** A reader MUST check `role` on
  every zarr it opens at an overview-carrying order; analysis readers reject
  or skip `role: overview` zarrs, display readers MAY stop at one.
- **`zagg_overview`** — the versioned provenance block, present exactly when
  `role` is `"overview"`:

```json
"zagg_overview": {
  "spec": "zagg-overview/1",
  "node": "-3111",
  "order": 3,
  "cell_order": 11,
  "source_shard_order": 5,
  "source_cell_order": 13,
  "window": "2019",
  "fields": {"count": {"class": "exact", "method": "sum", "nan_policy": "skip"},
             "h_tdigest": {"class": "approximate", "method": "tdigest_kway"}},
  "generation": {"n_leaves": 16, "max_leaf_timestamp": "2026-07-20T00:00:00Z"},
  "content_hash": "…",
  "generated_at": "2026-07-21T00:00:00Z"
}
```

  `spec` follows the conformance rule (strict-check, fail loudly on an
  unknown revision). `node` is the ancestor's morton decimal string and
  `order` its order — mortie's decimal grammar puts one digit per order after
  the base-cell digit, so `order` is always `len(digits) - 1` for the node
  string (`"-3111"` is order 3; the leading `-` marks a southern base cell and
  is not a digit); `cell_order` the overview's own cell order
  (`source_cell_order - (source_shard_order - order)` — constant tree depth,
  §4.4); `window` the §4.2 window key (`"all"` for the all-time fold);
  `generation` the D22 staleness stamp (merged-leaf count + max leaf commit
  timestamp); `content_hash` a sweep-internal skip-if-current digest
  (informative — not the §5 O11 recipe and not part of the reader contract).

  **`fields` enumerates the materialized fields only** — exactly the fields
  present as arrays in *this* overview, each recording the fold that was
  actually applied. A `none`-class field is absent from the zarr (§4.4) and so
  MUST be absent from this map: its recorded absence lives in the manifest's
  `pyramid.overview.fields` (§4.5), which is the map that enumerates **every**
  declared field. Consequently a reader MAY treat this map as the overview's
  variable list and MUST be able to open every array it names; cross-checking
  it against the arrays present is a valid integrity check. Each entry carries
  at least `class` and `method`, and MAY carry further fold provenance — an
  `exact` entry records the reduction's `nan_policy` (`"skip"`: nan-skipping,
  never NaN-propagating) — so readers MUST tolerate additional keys.

An overview also carries the standard D4 **commit stamp** as its final
write: an unstamped overview prefix is debris, exactly as for leaves.
Write order is pinned — template, arrays, `role`/provenance attrs, stamp
LAST — so presence of the stamp certifies the `role` attr landed; a reader
MUST ignore unstamped overview prefixes.

### 4.4 Structure

**Contract.** An overview at ancestor order `k` of a product with shard
order `s` and cell order `c` is the leaf structure "one order family up":
the same group layout as a source leaf, holding cells at order
`k_cell = c - (s - k)` (cells coarsen 4× per order of ascent — the pyramid
is the store's resolution axis, partially materialized). Concretely:

- the `morton` coordinate array holds the `4^(k_cell - k)` order-`k_cell`
  descendant words of the node, in canonical nested order;
- each **included** field is the same array kind as at the leaves: dense
  fields as dense arrays, digest fields as `zagg-ragged/1` (or `/2`) vlen
  arrays — §1–§3 of this page apply to overview arrays unchanged, **including
  §2.1's float32 exactness bound**: a coarse overview cell can pool more than
  `2^24` observations, and there `sum(weights)` is the nearest float32 to the
  true count rather than the count itself;
- field inclusion is gated by the field's **composability class** (§4.5):
  `exact` and `approximate` fields appear, `none` fields are **absent**.

An overview's variable set may therefore be a *subset* of the leaf's —
heterogeneous variable sets across level nodes are in contract, and a reader
MUST NOT assume every leaf field exists at every overview order (the
manifest declaration below is the zero-open way to know).

### 4.5 The manifest `pyramid` block

**Contract.** The product manifest (`morton_hive.json`; manifest bootstrap
semantics: mortie's specification and
[`design/sparse_coverage.md`](design/sparse_coverage.md) §3) declares the
overview family under the versioned `pyramid` block:

```json
"pyramid": {
  "spec": "zagg-pyramid/1",
  "overview": {
    "spacing": 2,
    "orders": [3, 1],
    "all_time": false,
    "fields": {
      "count":     {"class": "exact", "method": "sum", "nan_policy": "skip",
                    "dtype": "int32", "fill_value": 0},
      "h_tdigest": {"class": "approximate", "method": "tdigest_kway",
                    "dtype": "float32", "inner_shape": [2], "delta": 512},
      "photon_ids": {"class": "none"}
    }
  }
}
```

- **`orders`** — the ancestor orders that carry overviews (descending; empty
  = pyramid declared off). `spacing` records the schedule step (default 2 —
  the ratified display schedule). Schedules are per artifact family and
  deliberately decoupled from the tree's `path_grouping`.
- **The declared-off form is smaller, and `orders` is the only key a reader
  may bind unconditionally.** With the pyramid knob off the block is exactly

  ```json
  "pyramid": {"spec": "zagg-pyramid/1", "overview": {"orders": []}}
  ```

  — `spacing`, `all_time`, `fields`, and `summarize` are **absent**, not empty.
  A reader MUST branch on `orders` first: an empty `orders` (or no `pyramid`
  block at all — pre-pyramid manifests) means no overview family exists and no
  other key of the block may be assumed. When `orders` is **non-empty**,
  `spacing`, `all_time`, and `fields` MUST all be present (`summarize` stays
  optional), so the zero-open field query of §4.4 is well-defined exactly when
  there is something to query.
- **`fields`** — every aggregation field, keyed by name, with its
  **composability class**: `exact` (folds byte-equal — count/sum/min/max),
  `approximate` (t-digest merge — `np.isclose` equality class), or `none`
  (non-composable). A `"class": "none"` entry is the **recorded absence**
  (the ruled D24 default, option A): the field exists only at native
  resolution, and this declaration is how a reader knows without opening
  anything. A `none` entry carries **`class` only** — no `method`, no
  dtype/shape metadata: there is no fold to name, and stamping a default fold
  method on an excluded field would declare a t-digest array that does not
  exist. `exact`/`approximate` entries carry the fold `method`, any further
  fold provenance (an `exact` fold's `nan_policy`), and enough dtype/shape
  metadata to know the overview array's form up front. This map is the
  **all-fields** view; the per-overview `zagg_overview.fields` attrs map
  (§4.3) is the materialized subset.
- **`all_time`** — whether the `all.zarr` all-time fold is materialized at
  the declared orders (windowed stores only; a `schedule: none` store's
  single fold is already all-time).
- **`summarize`** (optional) — the opt-in **declared derived summary** for
  `none`-class fields: a mapping from a new, *different* field name to its
  derivation (e.g. an auto-digest of a roster field's raw values), living in
  the pyramid block and **never** in the semantic core — leaf truth is
  unchanged, and overview schema never silently differs from source except
  by declaration. (Ruled on the
  [#201 thread](https://github.com/englacial/zagg/issues/201); deterministic
  seeded subsampling is deferred; roster concatenation is rejected.)
- The sweep MAY additionally record materialized-actuals bookkeeping in the
  block; the template-time declaration above is never rewritten by the
  sweep, and the `pyramid` block is excluded from the manifest's frozen
  append-guard keys.

### 4.6 What §4 does not cover (informative)

The sweep's *other* derived-artifact families (MOC regeneration, stats and
sub-shardmap rollups, debris collection) are operational concerns recorded
in [`design/sparse_coverage.md`](design/sparse_coverage.md) D22 — they add
no new byte layouts (the stats rollup reuses the D20 sidecar schema; the
sub-shardmap is ShardMap JSON). The fold *algebra* for overview contents is
zagg-owned per §2.3; a reader consumes overview arrays exactly as it
consumes leaf arrays.

## 5. O11 content hashes

**Status: contract — frozen on
[#342](https://github.com/englacial/zagg/issues/342).** The recipe was
pinned by the moczarr verify reader
([espg/moczarr PR #23](https://github.com/espg/moczarr/pull/23),
`moczarr.stats.hash_arrays` / `combined_hash`); zagg's writer, when it lands
(#342), MUST adopt it verbatim. The O11 decision record (scope, compute-at-
write, exact-bytes rationale) is
[`design/sparse_coverage.md`](design/sparse_coverage.md) §8.2 O11.

The **logical content hash** of a leaf is per-array sha256 over *decoded*
values — never stored object bytes, so codec and packaging changes
(ShardingCodec inner chunks, compressor upgrades, §1.5 geometry) are
invisible by construction, while any value change flips the hash (exact
bytes, no float tolerance — interpretation pairs the hash with the recorded
zagg version).

### 5.1 Scope and keys

**Contract.** The hash set covers **every named zarr array beneath the leaf
root** — data fields, the ragged vlen payload arrays and their
`{field}_locations` siblings, `morton`, every coordinate — keyed by the
array's **path relative to the leaf root** (e.g. `"8/morton"`).

The scope is therefore **discovery-based**: both shipped implementations
enumerate (`group.members(max_depth=None)`), so the key set is whatever named
arrays exist under the leaf root, not what the template declared. That has
one normative consequence a verifier MUST honour: **a key-set difference is a
distinct outcome from a digest mismatch.** Debris inside a leaf — a foreign
array prefix, the [issue #341](https://github.com/englacial/zagg/issues/341)
Bug A class, an
[issue #327](https://github.com/englacial/zagg/issues/327) `.zarr.status/`
prefix — adds a key and so changes `combined`, which means a verifier
comparing `combined` alone reports an intact leaf as tampered. A verifier MUST
compare the per-array map first and report extra or missing keys as their own
outcome ("extra array present" / "array missing"), reserving "mismatch" for a
differing digest on a **shared** key. Hashing is the only path that
enumerates: the read and fold paths open leaf arrays **by name** (no member
enumeration, no LIST — [#344](https://github.com/englacial/zagg/pull/344)), so
debris is inert everywhere else.

### 5.2 Per-array recipe

**Contract.**

- **Fixed-width arrays**: sha256 over the array's full decoded contents as
  raw **C-order little-endian** bytes at the declared dtype (a big-endian
  dtype is byte-swapped to little-endian before hashing).
- **Vlen (ragged) arrays**: an object-dtype array has no flat buffer
  (`ndarray.tobytes()` on it would serialize per-process pointer
  addresses). It is hashed instead as, over cells in flat C order:

```text
sha256( for each cell:  u64_le(len(payload)) || payload )
```

  where `payload` is the cell's decoded bytes — exactly the §1.4
  `payload_bytes` for a `zagg-ragged/1` array (an empty or unwritten cell
  contributes its zero length; a locations sibling's payload is its raw
  little-endian `uint64` words). The 8-byte length prefix is what makes the
  digest injective (`[b"ab", b"c"]` and `[b"a", b"bc"]` must not collide),
  and it covers the cell *grid*, not just the payloads.

  The **element → bytes** normalization is itself normative, because a `/2`
  (§6) cell decodes to an ndarray rather than to bytes:

  | decoded element | payload bytes |
  |---|---|
  | `None` — an unwritten vlen cell may decode as `None`, not `b""` | zero-length |
  | `bytes` / `bytearray` / `memoryview` (a `/1` cell) | as-is |
  | `str` (a vlen-utf8 future) | UTF-8 encoded |
  | ndarray (a typed `/2` cell) | **C-contiguous, little-endian** bytes at the declared element dtype |
  | anything else | the recipe **does not apply**: a verifier MUST raise rather than hash |

  The last row is deliberate: a digest that is silently wrong is worse than no
  digest, so hashing a `repr` or a pointer buffer is forbidden. `None` and
  `b""` hash identically by construction — a `b""` fill is distinguishable
  from a missing cell only by position, which is the intent.

### 5.3 Combined hash and sidecar record

**Contract.** The combined hash is sha256 over the **sorted** per-array hex
digests joined by `"\n"`, hashed as ASCII — array names deliberately
excluded ("hash of the sorted per-array hashes").

*(Informative.)* Because it sorts the *digests*, `combined` is immune to the
order in which the §5.1 enumeration happened to yield arrays — a real
robustness property, and the reason two implementations agree without agreeing
on traversal order. The recorded `arrays` map is a different matter: a writer
SHOULD record it key-sorted so a regenerated record diffs cleanly.

The hashes are recorded in the leaf's D20 stats sidecar under
`content_hashes`, in the structured shape:

```json
"content_hashes": {
  "arrays": {"8/count": "…", "8/h_tdigest": "…", "8/morton": "…"},
  "combined": "…"
}
```

A writer MUST emit the structured shape. A reader SHOULD also accept the
flat shape (`{array_key: hash, "combined": hash}` — `combined` is reserved
and is not a legal zagg array name). A leaf with no recorded
`content_hashes` is **unverifiable, not tampered**: verification MUST
report "nothing recorded" as a distinct outcome from a mismatch (the
conservative dedup posture — an unverifiable leaf is never a hit).

*(Informative.)* The O11 hash is the verification half of the D19 identity
split — the `semantic_hash` says two leaves were *intended* identical; O11
says they *are* byte-identical — and doubles as the mismatch localizer
("only `h_tdigest` differs in this leaf") and the detection mechanism for
stamped-but-torn leaves under the concurrency contract's out-of-contract
case.

## 6. `zagg-ragged/2`

**Status: specified, implementation pending
([#210](https://github.com/englacial/zagg/issues/210); timing ratified —
the dtype package ships on its own release train, the zagg writer knob and
reader dispatch are gated on 1.0).** `/2` **adds to** `/1`, it does not
replace it: `/1` is the pinned 1.0 wire contract (§1), existing stores keep
it forever, and every conforming reader supports `/1` unconditionally.

`/2` moves the element declaration out of the §1.2 attrs block and into the
zarr **data type itself**: a parameterized typed vlen dtype, so a generic
zarr stack knows the element interpretation without zagg's attrs convention.

### 6.1 The typed dtype

**Contract.** The `/2` data type is the registered zarr v3 extension
**`vlen-ndarray`** (espg-ratified name; reference implementation: the
`zarr-vlen-ndarray` package under `github.com/espg`), parameterized by the
element dtype and trailing inner shape — exactly the pair the `/1` attrs
block declares:

- element dtype `float32`, inner shape `(2,)` for a digest payload array;
- element dtype `uint64`, inner shape `()` for a locations sibling.

A cell's logical value is the `(n, *inner_shape)` array itself rather than
its raw bytes; everything else about the array — shape, cells axis,
`fill_value` (the empty cell), located sibling alignment (§1.1), storage
geometries (§1.5) — is unchanged from §1.

### 6.2 Byte identity

**Contract.** The `/2` codec chain MUST produce chunk bytes **byte-identical
to `/1`'s**: the §1.4 wire framing and the §1.3
`[…, zstd(level=3, checksum=false)]` chain are unchanged, with the typed
array↔bytes codec serializing each cell as the same
`np.ascontiguousarray(value).tobytes()` little-endian payload. The typed
dtype changes *interpretation only*, never stored bytes. Consequences (the
point of the revision):

- migrating an existing `/1` store to `/2` is a **metadata-only** rewrite
  (`zarr.json` objects; no data object is touched);
- the §7 conformance fixtures serve both revisions — a `/1` fixture's chunk
  objects re-labeled `/2` MUST decode identically through the typed path;
- the §5 O11 vlen recipe is unaffected (it hashes decoded payload bytes,
  which are identical by construction) — this is exactly what §5.2's
  element→bytes normalization buys: a `/2` cell decodes to an ndarray, whose
  C-contiguous little-endian bytes are the `/1` cell's bytes.

### 6.3 Revision signaling

**Contract** (restating §1.6 from the `/2` side):

- An array whose zarr data type is `vlen-ndarray` **is** `zagg-ragged/2`;
  the `ragged` attrs marker is retired on such arrays (not written). The
  element declaration lives in the dtype configuration alone — a reader
  MUST NOT require the attrs block on a `/2` array.
- A located `/2` payload array still declares its sibling binding in
  **metadata, never by naming convention** (the §1.2 rule survives the
  revision), and it does so under a **new top-level attrs key** — not a
  residual `ragged` block with `spec`/`element` dropped. "Retired" is
  literal: no `ragged` key is written on a `/2` array. That is a ruling, not
  a leftover choice — a `ragged` block carrying only `locations` would be, by
  §1.2's own words, an array with no well-formed `element` declaration, which
  a `/1`-only reader MUST refuse with a pointed "not a `zagg-ragged/1` array"
  error, i.e. the misleading path instead of the actionable "install
  `zarr-vlen-ndarray`" one below. The `ragged` key stays **reserved but
  unwritten** under `/2` (config-declared attrs still MUST NOT shadow it), and
  the new key's exact name is for the `/2` implementation PR to pin *in this
  section* before any `/2` store exists. The sibling-alignment semantics of
  §1.1 are unchanged.
- An array with the `variable_length_bytes`/`bytes` dtype and a
  `spec: "zagg-ragged/1"` attrs block **is** `zagg-ragged/1`.
- A reader without the `vlen-ndarray` extension installed MUST surface an
  actionable "install `zarr-vlen-ndarray` to read this store" failure, not
  a silent mis-decode (and cannot half-parse: the dtype is unknown to its
  zarr stack by construction).

*(Informative.)* Writing `/2` will be a per-product opt-in
(`output.ragged_encoding: typed`), which shifts the product's
`semantic_hash` — a new product identity, by design. The default stays `/1`;
flipping it is a schema epoch deferred to its own ruling (public/interop
stores may deliberately stay `/1` for vanilla-zarr openability).

## 7. Conformance fixtures

**Status: contract.** The committed stores under
[`tests/data/spec/`](https://github.com/englacial/zagg/tree/main/tests/data/spec)
are part of this specification: a reader implementation that reproduces
their expected decoded values and content hashes conforms to §1–§3 and §5.
They are generated by
[`tools/generate_spec_fixtures.py`](https://github.com/englacial/zagg/blob/main/tools/generate_spec_fixtures.py)
through zagg's **production write path** (manifest, sharded leaf template,
dense + ragged writes, coverage sidecar, commit stamp), so writer↔spec
drift fails zagg's own suite (`tests/test_spec_conformance.py`) on
whichever side moved. moczarr vendors the same fixtures for its parity
gates (espg/moczarr#19/#20).

Two tiny single-shard hive stores, both on the same deliberately small
geometry — shard order 4, inner-chunk order 5, cell order 6 (16 cells,
K = 4 inner chunks of 4 cells), sharded (the hive default):

- **`minimal/`** — one *unlocated* digest field (`h_tdigest`) plus `count`.
  The smallest thing that is a conforming store.
- **`kitchen_sink/`** — the full stratified-product surface: located
  signal/noise digest strata (payload + `{field}_locations` siblings,
  `stratum`/`signal_threshold` provenance attrs), the `composition` word
  (including a single-photon cell packing the §3.1 golden word
  `0xFF000000FF0000FF` and a noise-only cell whose signal payload is the
  empty `(0, 2)` array), and `count`.

Both stores pin the layout edge cases a reader must handle: inner chunk
ordinal 2 is **empty** (absent from the shard index — the §1.5 sentinel, and
that sparsity reaches the dense arrays too: the `morton` coordinate and
`count` hold their fill across that chunk, so a reader MUST NOT assume the
coordinate is dense across a shard), populated chunks contain empty cells
(the `b""` fill), and one cell's digest carries merged centroids (weight > 1)
whose location words are common ancestors (§2.2).

Both fixtures are **sharded**, so §7's conformance claim is scoped to the
§1.5 sharded geometry. The per-inner-chunk geometry — identical §1.4 framing,
one object per inner chunk, no shard index — has no committed golden here; it
is pinned zagg-side by
`tests/test_processing.py::TestRaggedVlenLayout`. A reader that derives the
stored span from the array's own metadata, as §1.5 requires, reads both from
one code path; a reader that hard-codes the shard-index suffix passes §7 and
still fails on an unsharded store.

Each fixture ships a sibling **`{name}.expected.json`** recording the shard
key, leaf path, geometry, every populated cell's decoded values (digest
centroids, location words and composition words as decimal strings — JSON
numbers cannot carry `uint64` faithfully), the per-stratum exact counts,
and the §5 O11 `content_hashes` (per-array + combined). The expected
**decoded values** were computed from the generator's *inputs* — the arrays
handed to the writers — never read back through a zagg reader, so the reader
is pinned, not self-certified. The `content_hashes` are necessarily computed
from the **written leaf** (a content hash of nothing else would mean
anything), so they are pinned differently: the suite carries the combined
digest and one per-array digest per §5.2 element kind as **frozen hex
literals**, and recomputes the vlen digests a second time from the shard
objects alone (§1.4/§1.5 byte recipes, no zarr). A recipe change — prefix
width, joiner, key set — therefore fails a test instead of agreeing with
itself on both sides, which is also the only mechanism that catches a future
zagg↔moczarr divergence (neither side's fixture can: espg/moczarr#23).

**Conformance criteria for an external reader**: decode every ragged array
per §1–§2 and the composition array per §3, reproducing the expected
decoded values exactly (byte-exact float32/uint64 — no tolerance), and
reproduce `content_hashes` per §5. zagg's own suite additionally decodes
the shard objects with **spec-text-only** decoders (struct + zstd, no zagg
read path) to prove the byte recipes in §1.4/§1.5 are sufficient on their
own.

Regenerating the fixtures reproduces the same logical values (seeded rng);
stamp timestamps and compressed bytes may differ across zstd versions —
conformance is over *decoded* values, never stored object bytes (the same
principle as §5).
