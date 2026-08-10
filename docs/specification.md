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

**Subtree spans.** The cells axis MUST be in canonical nested order — the
per-cell `morton` coordinate ascending, every aligned power-of-four span
sharing its ancestor cell (the ordering every §1 identity derivation and the
rank-space deinterleave already presuppose; a zagg writer has never produced
anything else, this sentence makes it citable). Consequently the order-`k`
subtree below an ancestor at nested rank `r` on an order-`c` cells axis
occupies exactly the contiguous index span `[r·4^(c−k), (r+1)·4^(c−k))`, and
a reader MAY serve "everything below one morton node" as a contiguous-slice
read: on the sharded geometry the index suffix plus only the covering inner
chunks (the 2-GET recipe generalized to a span), on the per-inner-chunk
geometry only the covering chunk objects — never a whole-array sweep. The
span property is normative; a dedicated subtree reader is implementation
(zagg: [issue #351](https://github.com/englacial/zagg/issues/351)).

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

A writer-side spill-block close (`aggregation.streaming.mode: spill` crossing
its block threshold, issue #370) is an additional merge source under the same
rule: an overflow shard's centroids may carry coarser common-ancestor words,
with the stored layout and byte-level contract above unchanged.

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
section records are ratified — D11/D22–D24 in
[`design/sparse_coverage.md`](design/sparse_coverage.md), plus three rulings on
the #201 thread that this section's grammar traces to directly:
[`all.zarr` + `role: overview` confirmed, display schedule every 2 orders](https://github.com/englacial/zagg/issues/201#issuecomment-5025459421),
[the D24 `none`-class ruling (per-field exclusion the default, declared derived summary the opt-in, never the semantic core)](https://github.com/englacial/zagg/issues/201#issuecomment-5025509889),
and
[the A/B/C/D option space (C an espg-flagged opt-in phase, D rejected)](https://github.com/englacial/zagg/issues/201#issuecomment-5025519604);
the grammar
below is what the #201 implementation lands and what moczarr's level-node
reader plans against (espg/moczarr#15, the 8b seam). Any divergence
discovered while landing #201 is resolved **on this section first** — the
implementation conforms to the spec, never the reverse.

The **level grammar** is revision `zagg-pyramid/2`
([#382](https://github.com/englacial/zagg/issues/382); design record
[#381](https://github.com/englacial/zagg/issues/381), points (2)–(5), as
collapsed by the espg grammar ruling on the declaring PR): the config
declares **leaf cell resolutions only**, and everything above the shard is
the **fixed every-order ladder** of §4.4; the manifest records the fully
expanded `(node, cells)` list, of which the original constant-depth
`zagg-pyramid/1` grammar is the special case
`cells = [node + (cell_order - shard_order)]`. `/1` stores stay readable
under their own rule forever (§4.5 — we are the sole consumer; there is no
migration machinery). A `/2` declaration is materialized by the leaf
columns ([#383](https://github.com/englacial/zagg/issues/383), §4.6) and the
**staged sweep** ([#384](https://github.com/englacial/zagg/issues/384)):
declared-but-unswept remains a legal recorded state (#381 point (11):
declaring is free, sweeping is the operational decision), and since the
issue #384 default flip a **default declaration** (no schedule spelled) is
`/2` at the grid's resolved chunk order whenever that order is strictly
interior to the shard's resolution window (raster configs, explicit legacy
`orders`/`spacing` schedules, and K == 1 grids keep `/1`).

A `zagg-pyramid/2` store is therefore inherently a **multiresolution
statistical grid**: every HEALPix order from 0 through the declared base,
plus the native resolution, each level spec-guaranteed and individually
addressable — a reader picks its resolution and reads it; skipping levels
is a reader-side choice, never a store property. `output.pyramid: false`
is the degenerate single-resolution opt-out.

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
  "fold_source": "leaves",
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

  **`fold_source` names the regime that produced this level**
  ([#376](https://github.com/englacial/zagg/issues/376)) — the one piece of
  provenance a reader cannot recover from the arrays:

  - `"leaves"` — folded directly from the subtree's source leaves: **single
    quantization**, and for the `exact` class byte-equal to a direct
    aggregation at this cell order (§4.4);
  - `"cascade"` — folded from an already-materialized **finer overview**
    (fold-of-folds), whose order the entry then also names:

    ```json
    "fold_source": "cascade", "fold_from_order": 3
    ```

  The distinction is only material for the `approximate` class: the exact
  merge laws are associative, so a cascaded `sum`/`min`/`max` is the same
  value either way, while a cascaded digest is a **merge of merges** — it
  inherits the merge's documented behavior once per level and carries **no
  precision guarantee**. That is in contract: overviews are display
  artifacts, and the precision promise stops at the levels declared exact
  (`pyramid.overview.exact_levels`, §4.5). A reader that needs the exact
  regime MUST check this key rather than the level's depth, and MUST read an
  overview that carries **no** `fold_source` as `"leaves"` — the only fold
  that existed before #376.

  **`source_children` records the cascade's coverage** — present on a
  `"cascade"` overview only, and the companion a reader needs to interpret
  the level's *fill* cells:

  ```json
  "source_children": {"folded": 15, "missing": 1, "unreadable": 0}
  ```

  A cascade folds the child overviews that are **on disk**, where a
  `"leaves"` fold folds every leaf the coverage MOC knows about. `folded`
  counts the children that contributed, `missing` those with no materialized
  (D4-stamped) overview, and `unreadable` those that failed to open or did
  not classify as an overview at `fold_from_order`. When either of the latter
  two is nonzero the level **under-covers its subtree**: the spans those
  children own hold the fill value, and a fill cell there is **not** evidence
  that the subtree is empty. Such a level is repaired by a later sweep, which
  sees the parent's summed generation change once the child exists. A reader
  MUST tolerate the key's absence (a `"leaves"` level, or a pre-#376
  artifact) and MUST NOT read absence as `missing: 0`.

An overview also carries the standard D4 **commit stamp** as its final
write: an unstamped overview prefix is debris, exactly as for leaves.
Write order is pinned — template, arrays, `role`/provenance attrs, stamp
LAST — so presence of the stamp certifies the `role` attr landed; a reader
MUST ignore unstamped overview prefixes.

### 4.4 Structure

**Contract.** A pyramid is an ordered list of **level entries**
`{node, cells}` (§4.5): `node` is the hive-tree **ancestor-or-self** order
whose artifact carries the level — one artifact per `(node, window)`, named
by §4.2 — and `cells` the **reader-facing cell resolutions** stored there.
The list is not free-form; it is determined by two declarations and one
law:

- the **leaf entry** — `node == shard_order` (not an ancestor artifact at
  all: the leaf's own level column, #381 point (2)) carries every declared
  leaf resolution, each strictly between `shard_order` and `cell_order`;
- the **fixed every-order ladder** — with `d = base - shard_order` (`base`
  the coarsest leaf resolution, so `d >= 1`), every order `k` from
  `shard_order - 1` down to **0 inclusive** carries exactly one member at
  resolution `k + d`. Two numbers — `shard_order` and `d` — determine
  everything above the shard; every store roots at order 0. (Non-normative:
  the ladder is cheap by construction — cell counts shrink 4× per rung, and
  digest bytes are cap-bounded per cell — which is why it is law rather
  than a knob.)

For each member resolution `r` the artifact holds one **resolution group**:
the zarr group named `str(r)`, with the same layout as a source leaf's cell
group. Concretely, for a member `r` at an order-`k` node:

- the group's `morton` coordinate array holds the `4^(r - k)` order-`r`
  words the node covers, in canonical nested order — its **descendant**
  words where `r > k`, its **own** word where `r == k`. For a manifest
  **level member** `r > k` always, by the window and ladder rules above;
  the one recorded `r == k` group is the §4.6 column's **node-order
  member** (the whole-footprint aggregate of #381 point (2) — the leaf's
  universal partial for every coarser cell), which is a recorded group of
  that artifact and still never a manifest member. No separate partial
  grammar or `partial/` path exists anywhere;
- each **included** field is the same array kind as at the leaves: dense
  fields as dense arrays, digest fields as `zagg-ragged/1` (or `/2`) vlen
  arrays — §1–§3 of this page apply to overview arrays unchanged, **including
  §2.1's float32 exactness bound**: a coarse overview cell can pool more than
  `2^24` observations, and there `sum(weights)` is the nearest float32 to the
  true count rather than the count itself;
- field inclusion is gated by the field's **composability class** (§4.5):
  `exact` and `approximate` fields appear, `none` fields are **absent**.

Under `zagg-pyramid/1` every artifact holds exactly **one** resolution
group, at the constant depth `k_cell = c - (s - k)` for shard order `s` and
cell order `c` (cells coarsen 4× per order of ascent — the pyramid is the
store's resolution axis, partially materialized): the `/1` grammar is the
special case `cells = [node + (c - s)]`, and §4.1–§4.2 apply to both
revisions unchanged. §4.3's per-artifact `zagg_overview` attrs block — in
particular its single scalar `cell_order = c - (s - k)` — is specified for
`/1`'s single-group artifacts. Stage-written `/2` ladder artifacts (issue
#384) carry attrs revision **`zagg-overview/2`**: the same keys with
`cell_order` the entry's own `cells` member (`k + d`, not the constant-depth
formula), the `fold_source`/`fold_from_order` pair replaced by the #381
point (7) provenance — `regime` (`stage-gather` | `stage-merge`),
`merges_from_raw` (1 for a gather of gen-1 members, 2 for a merge of the
relayed gen-1 partials — never 3 for an upfront level; gen 3 belongs only
to the append-later cascade regime), and `source_children` (present in both
stage regimes: a gather that under-covers says so exactly like a merge) —
plus `run_id`, the sweep run that wrote the artifact, and a `generation`
block summing the consumed children (the stage skip gate's ratchet key —
`{n_leaves, max_leaf_timestamp, run_ids}`, composition in §4.5).
The commit stamp of a stage-written artifact carries the same `run_id` key
(additive to the stamp grammar; fleet-written stamps never carry it, and a
reader treats its absence as "not a stage artifact", never an error): it is
the residual-race backstop of the sweep-admission lease. The `/2` leaf
entry's artifact is the §4.6 column — declared-but-unmaterialized remains
legal (§4.5).

An overview's variable set may therefore be a *subset* of the leaf's —
heterogeneous variable sets across level nodes are in contract, and a reader
MUST NOT assume every leaf field exists at every overview order (the
manifest declaration below is the zero-open way to know).

The fold **regime** (§4.3's `fold_source`) does not enter this contract: a
cascaded level has exactly the layout above. It differs in the values its
`approximate` fields carry and, when it under-covers its subtree, in *which
cells are populated at all* — §4.3's `source_children` is the key that says
so, in every class.

**Accuracy doctrine (`/2`).** For `/2` levels the #376-era "display
artifacts, no precision guarantee past the exact levels" posture is
superseded (espg ruling on the declaring PR): **exact-class** fields are
exactly correct at every order — the reductions are associative, so the
pyramid is a true downsampling pyramid for them — and **approximate-class**
fields are **analysis-grade at their recorded generation**: the per-entry
merges-from-raw count (#381 point (7), recorded in the §4.5 `actuals` and
per artifact in `zagg_overview`) is
the contract a reader holds them to, not a blanket display-only caveat.
The `/1` operational caveats about the deprecated leaves/cascade regimes
(§4.3, §4.5) stay as written for `/1` artifacts.

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
    "fold_source": "cascade",
    "exact_levels": 1,
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

Under revision `/2` ([#382](https://github.com/englacial/zagg/issues/382))
the schedule is the **block-level `overviews`** list instead — the FULLY
EXPANDED form of §4.4's leaf entry + fixed ladder; `orders` and `spacing`
do not exist in a `/2` block; every other key is unchanged:

```json
"pyramid": {
  "spec": "zagg-pyramid/2",
  "overviews": [
    {"node": 3, "cells": [5, 4]},
    {"node": 2, "cells": [3]},
    {"node": 1, "cells": [2]},
    {"node": 0, "cells": [1]}
  ],
  "overview": {
    "all_time": false,
    "fold_source": "cascade",
    "exact_levels": 1,
    "fields": {"…": "…"}
  }
}
```

The split is deliberate: the block-level `overviews` list is the
**store-wide product declaration** — what fleet writers, stage sweeps,
declare-time forecasts, and external readers consume — while the singular
`overview` dict remains the **overview sweep family's execution regime**
(D22 per-family bookkeeping: `all_time`, the #376 fold keys, `fields`, the
sweep's `materialized` actuals), so the two never re-nest. Per-entry
materialization actuals (#381 point (7)) nest inside the block-level
entries themselves — the `actuals` key below, written by the issue #384
staged sweep's finisher.

- **`orders`** (`/1`) — the ancestor orders that carry overviews
  (descending; empty = pyramid declared off). `spacing` records the schedule
  step (default 2 — the ratified display schedule). Schedules are per
  artifact family and deliberately decoupled from the tree's
  `path_grouping`.
- **`overviews`** (`/2`) — the level entries of §4.4, recorded at **block
  level** (a sibling of `spec` and `overview`, never inside the family
  dict) and always **fully expanded**: the leaf entry first, then the fixed
  every-order ladder down to node 0 — a reader never re-derives the ladder,
  the recorded list IS the contract. Every entry is a `{node, cells}`
  mapping with `cells` a list. The corresponding **config grammar** is leaf
  resolutions only: `output.pyramid.overviews` is an int (sugar for one) or
  a strictly descending list of ints, each **strictly between**
  `shard_order` and `cell_order` (a member at the shard's own order is the
  writer-side aggregate, never declared; a member at the base data's own
  order would *be* the base data); omitted, the default is one resolution
  at the grid's resolved chunk order — normative since the issue #384
  default flip: a default declaration emits this `/2` block for every new
  store whose resolved chunk order is strictly interior (raster configs,
  explicit legacy `orders`/`spacing` schedules, and K == 1 grids keep
  `/1`), and the worker column gate (§4.6) derives the SAME default from
  the grid, so declaration and artifact can never disagree. There is no
  above-shard configurability: no per-node spelling, no gather declaration,
  no member promotion — the ladder makes every within-footprint member
  spec-guaranteed (espg grammar-collapse ruling on the declaring PR).
- **The declared-off form is smaller, and `orders` is the only key a reader
  may bind unconditionally.** With the pyramid knob off the block is exactly

  ```json
  "pyramid": {"spec": "zagg-pyramid/1", "overview": {"orders": []}}
  ```

  — `spacing`, `all_time`, `fold_source`, `exact_levels`, `fields`, and
  `summarize` are **absent**, not empty. Recording absence never needs the
  new grammar, so the declared-off form is always this `/1` shape — a `/2`
  block's `overviews` list is **never empty**.
  A reader MUST branch on `spec` first, then on the revision's schedule key:
  under `/1` that is `orders` — an empty `orders` (or no `pyramid` block at
  all — pre-pyramid manifests) means no overview family exists and no other
  key of the block may be assumed — and under `/2` it is the block-level
  `overviews`. When the schedule key is non-empty, `all_time` and `fields`
  MUST be present in the `overview` family dict (`spacing` too under `/1`;
  `summarize` stays optional), so the zero-open field query of §4.4 is
  well-defined exactly when there is something to query.
- **`fold_source` / `exact_levels`** — the declared fold regime
  ([#376](https://github.com/englacial/zagg/issues/376)): `"cascade"` (the
  default) folds each declared level from the next **finer** declared level's
  overviews, and `exact_levels` is how many of the finest levels are folded
  from the leaves instead — so under `{"orders": [3, 1], "fold_source":
  "cascade", "exact_levels": 1}` order 3 is exact and order 1 is a fold of
  order 3's folds. `"leaves"` is the deprecated exact-from-leaves regime,
  where every declared level folds the raw leaves and no `exact_levels` key
  is written (every level is exact there, so a boundary would name a
  distinction the store does not have). These two keys are written by
  #376-and-later writers; a reader that finds **no `fold_source`** MUST read
  the declaration as `"leaves"`, which is the only regime that existed
  before. They declare what the **next** sweep will do — what is on disk is
  `materialized.fold_sources` below and, per artifact, §4.3's
  `zagg_overview.fold_source`, which is authoritative for a given overview.
  Under `/2` the pair is declared identically; `exact_levels` counts **level
  entries** from the finest end, exactly as it counts `orders` under `/1`.
  (`exact_levels` predates the `overviews` key-naming convention and is
  deliberately untouched by it — ruled vestigial-in-waiting under the #381
  regime law, which makes the exact/approximate boundary structural; the
  `/2` accuracy contract is §4.4's doctrine, not this pair.)
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
  by declaration. ([Ruled on the #201 thread](https://github.com/englacial/zagg/issues/201#issuecomment-5025509889);
  deterministic seeded subsampling is **deferred as an opt-in phase, not
  rejected** — it shares this block's declaration grammar and stays
  declared/never-default when it lands
  ([option space](https://github.com/englacial/zagg/issues/201#issuecomment-5025519604));
  roster concatenation is rejected.)
- The sweep MAY additionally record materialized-actuals bookkeeping in the
  block; the template-time declaration above is never rewritten by the
  sweep, and the `pyramid` block is excluded from the manifest's frozen
  append-guard keys. zagg's sweep writes

  ```json
  "materialized": {"orders": [3, 1],
                   "fold_sources": {"3": "leaves", "1": "cascade"},
                   "generated_at": "2026-08-04T00:00:00Z"}
  ```

  — the orders it has written and, per level, the §4.3 regime that wrote it
  (keys are the orders as strings, JSON having no integer keys). Both are
  **actuals**, accumulated across sweeps: a level swept under an earlier
  declaration keeps the regime it was made with until it is regenerated, so
  this map can disagree with the declaration above, and that disagreement is
  informative rather than an error. It is a convenience — a reader MAY
  instead open the overviews and read §4.3 — and, like every other actual,
  it says nothing about an overview still being present (overviews are
  regenerable caches, §4.1). On a `/2` store this family-dict `materialized`
  map is the **`/1`-era inventory**, preserved verbatim across a declaration
  revision bump: a retrofit never discards actuals, and the overviews it
  names stay on disk as regenerable-cache debris; the staged sweep never
  writes this family-dict map on a `/2` store — the per-entry `actuals`
  below are the one source of truth for `/2` materialization. `/2`
  materialization actuals nest **inside the level entry that owns them**
  (#381 point (7)), written by the issue #384 finisher's manifest RMW:

  ```json
  {"node": 2, "cells": [3],
   "actuals": {"regime": "stage-merge", "merges_from_raw": 2,
               "source_children": {"folded": 3, "missing": 0, "unreadable": 0},
               "run_id": "stage-20260809T000000Z-ab12cd",
               "generated_at": "2026-08-09T00:00:00+00:00"}}
  ```

  — `regime` is `leaf-column` (the leaf entry: the fleet's own column,
  merges-from-raw 1, no `source_children` — its source is complete by
  construction), `stage-gather` (a concatenation of gen-1 members,
  merges-from-raw 1) or `stage-merge` (a k-way fold of the relayed gen-1
  node-order partials, merges-from-raw 2 — **never 3 for an upfront level**;
  gen 3 belongs only to the append-later cascade regime). `source_children`
  accumulates the run's per-artifact coverage counts; `run_id` names the
  sweep run (stage entries only). The key is **additive**: a reader MUST
  tolerate additional keys on a level entry, and `actuals` says nothing
  about artifacts still being present (overviews are regenerable caches,
  §4.1).
- **The stage skip key** (`generation`, recorded per artifact in §4.4's
  attrs, per stage column in §4.6, and in the sweep-internal envelope) is
  the triple

  ```json
  "generation": {"n_leaves": 16,
                 "max_leaf_timestamp": "2026-08-09T00:00:00+00:00",
                 "run_ids": ["stage-20260809T000000Z-ab12cd"]}
  ```

  — the summed leaf count of the consumed children, the newest child stamp,
  and the **sorted set of `run_id`s** those children's stamps carry, unioned
  with the ids their own recorded blocks relay
  ([#417](https://github.com/englacial/zagg/issues/417)). A level is folded
  again exactly when this triple moves. The third term is load-bearing:
  stamps resolve to **one second**, so the first two alone read a child
  rewritten inside its own recorded second at an unchanged leaf count as
  *current*, and the `/1` content-hash backstop cannot apply without doing
  the fold the skip exists to avoid. A run may not rewrite its own object
  mid-run (single-writer law, §4.8), so a same-second rewrite is a foreign
  run's and moves the set. `run_ids` is **additive**: absent, it MUST be
  read as the empty set (a pre-#417 entry, or children that are all
  fleet-written leaf columns — those stamps carry no `run_id`), never as a
  wildcard that matches any set, so an upgraded store folds once more
  rather than inheriting the blind spot. Fleet-written leaf columns
  contribute no run id, so at the finest tuple the term is empty and the
  gate rests on the pair alone.

### 4.6 Leaf column artifacts (`zagg-column/1`)

**Status: contract — issue [#383](https://github.com/englacial/zagg/issues/383)
(umbrella [#381](https://github.com/englacial/zagg/issues/381) points
(1)–(3)).** A **column artifact** is the leaf worker's own pyramid
contribution, written at aggregation time while the shard's cell data is
resident: one zarr per `(leaf, window)`, a **sibling of the leaf under the
leaf's own node prefix**. A column exists exactly when the **writing run's**
pyramid declaration is `zagg-pyramid/2` (§4.5), carries leaf-node levels —
the expanded `overviews` list always places the declared resolutions at the
shard node — **and declares at least one composable field** (§4.5): a
declaration whose fields are all `none`-class writes **no artifact at all**
(not a morton-only column), and clears any prior one exactly as the
no-levels arm below does. When it does exist it is written by the same
worker invocation that commits the leaf, after the leaf's own stamp. The gate is the config the unit carries, not
a store read: workers never open the manifest, so the manifest's `pyramid`
block is the **reader- and sweep-facing** declaration and MAY lag a
config-only change until the store is re-templated or retrofitted
(`ensure_manifest` deliberately excludes `pyramid` from the keys it freezes,
so a re-run into an existing store never re-PUTs it). A reader that needs to
know which leaves actually carry columns therefore reads the columns, not the
manifest. A run whose declaration carries no leaf-node levels **deletes** any
column and sidecar a previous declaration left at that `(leaf, window)`, so a
column never outlives the declaration that wrote it.
Like overviews, columns are derived artifacts a reader MUST NOT require;
unlike overviews they are **not** regenerated by a sweep — the single writer
of a column is its leaf's worker, ever (no locking anywhere), and repair is
re-invoking the idempotent leaf, never a sweep-side fold from raw cells.

- **Naming.** One column per `(node, window)`, and its basename MUST be
  `{window stem}.pyramid.zarr` — the stem derived from the D23 **window
  alone** (the §4.2 overview dialect), never from the leaf's own basename
  stem, so the rule is independent of the store's leaf-naming revision: the
  unwindowed / `schedule: none` leaf takes the reserved token (§4.2), giving
  the §7 `column/` fixture's committed pair `11213.zarr` and
  `all.pyramid.zarr` side by side, and a `/1` windowed leaf
  `11213_2019.zarr` is paired with `2019.pyramid.zarr`. The `.pyramid.zarr`
  suffix is the one name seam, and it is **normative** for name-grammar
  consumers (e.g. the root-MOC walker): a basename ending in `.pyramid.zarr`
  MUST NOT be read as a leaf or an overview. The seam is unambiguous because
  the frozen D23 window-label charset `[0-9A-Za-z-]{1,32}` (the grammar §4.2
  inherits, generative labels being digits) admits no `.`,
  so no legitimate leaf or overview basename can end that way; classification
  for everything else is the attrs below.
- **Structure.** One zarr group per **resolution group**, named by its cell
  order (the `{order}/{field}` layout of §4.4): every declared leaf
  resolution, every within-footprint rung of the fixed ladder
  (`node < cells ≤ base`), and the **node-order member** — `cells == node`,
  one cell: the leaf's whole-footprint aggregate, its **universal partial**
  for every coarser cell (there is no `partial/` grammar; a coarse level
  declared later never rewrites a leaf). Each group holds the `morton`
  coordinate (the node's order-`r` descendant words, ascending) and one
  array per **composable** field (§4.5 classes; `none` fields are absent).
  A group's arrays are **single-chunk and unsharded** — `chunk_shape` equals
  `shape` (`4^(r - node)` cells), no `sharding_indexed` codec — whatever the
  leaf's own `chunk_inner`/sharding: a column group is small by construction,
  and a reader sizes its GETs accordingly (§1.5's per-inner-chunk geometry,
  read from the array metadata as §1.5 requires).
  Under the default `[chunk_inner]` declaration on the 19/13/9 reference
  geometry a column carries groups {13, 12, 11, 10, 9}.
- **Fold laws.** Every group folds **directly from the leaf's resident cell
  data** — never group from group: exact classes by their §4.5 merge law
  (nan-skipping, §4.3), approximate classes by the order-independent k-way
  digest merge. Column bytes at a resolution MUST equal the sweep-kernel
  fold of the committed leaf's arrays at that resolution — the from-leaves
  parity contract, which reads differently per class: for **exact** fields
  it is checkable from this page (the §4.5 merge law plus §4.3's nan
  policy, so an external reader can reproduce a group by direct
  aggregation), while for **approximate** fields the merge algebra is
  zagg-owned and deliberately unspecified (§2.3), so the MUST binds
  implementations sharing those kernels and is pinned on committed bytes by
  the §7 `column/` fixture rather than derivable from spec text.
  `merges_from_raw` is 1 for every group.
- **The `role` and `zagg_column` attrs.** `role` is `"column"`;
  `zagg_column` is the versioned provenance block, present exactly when
  `role` is `"column"`:

```json
"zagg_column": {
  "spec": "zagg-column/1",
  "node": "11213",
  "order": 4,
  "source_cell_order": 6,
  "window": "all",
  "fields": {"count": {"class": "exact", "method": "sum", "nan_policy": "skip"},
             "h_tdigest": {"class": "approximate", "method": "tdigest_kway",
                            "delta": 16, "dtype": "float32", "inner_shape": [2]}},
  "groups": {"5": {"regime": "leaf-column", "merges_from_raw": 1, "n_cells": 4},
             "4": {"regime": "leaf-column", "merges_from_raw": 1, "n_cells": 1}},
  "cells_with_data_order": 5,
  "generated_at": "2026-08-05T00:00:00+00:00"
}
```

  `node` is the leaf's morton decimal and `order` its (shard) order;
  `source_cell_order` the leaf's own cell order; `window` the §4.2 window
  key (`"all"` unwindowed — the basename and this key round-trip);
  `cells_with_data_order` names the group whose populated-cell count the
  commit stamp's `cells_with_data` records (the finest group).
  `fields` follows §4.3's materialized-fields contract (approximate entries
  additionally carry `dtype`/`inner_shape`/`delta` — enough to decode
  without the manifest). `groups` carries the per-group provenance slots:
  the fold **regime** (`"leaf-column"` — folded from the leaf's own
  resident cells; `source_children` never rides this regime, its source is
  complete by construction), the `merges_from_raw` integer, and `n_cells`
  — the group's **grid** size `4^(r - order)`, i.e. its arrays' length, not
  its populated-cell count (that is the stamp's `cells_with_data`, for the
  `cells_with_data_order` group only). `n_cells` is derivable and recorded
  as a convenience.
- **Write discipline.** The leaf's own D4 order: template (wholesale — the
  column prefix, and any stale stats sidecar, are deleted first) → every
  group's arrays → `role`/`zagg_column` attrs → **one commit stamp last
  covering the whole column**. The stamp is the D15 `morton_hive_commit`
  root-attrs block, the same key and grammar a leaf carries:

```json
"morton_hive_commit": {"spec": "morton-hive/1", "complete": true,
                       "cells_with_data": 3, "granule_count": 1,
                       "written_at": "2026-08-05T00:00:00+00:00"}
```

  `cells_with_data` is the populated-cell count of the group named by
  `cells_with_data_order`; `granule_count` is the **leaf's** granule count,
  not a column quantity; and a column stamp carries **no `coverage`
  payload** (a leaf's does), so a stamp reader MUST NOT require one. On a
  **windowed** store the column's stamp is `spec: "morton-hive/2"` and
  carries the D15 half exactly as the leaf's does — `window` plus the
  observed `time_range` — so a reader that strict-checks the `spec` marker
  must accept both revisions here. An unstamped column prefix is debris and
  MUST be ignored; an idempotent re-run rewrites the whole column to the
  same array bytes (provenance timestamps move). The D20 stats sidecar —
  `{stem}.stats.json`, e.g. `all.pyramid.stats.json`, carrying the §5
  record for the column's own arrays — is a sibling object PUT **after**
  the stamp, fail-open: absence reads unverifiable, never tampered.
- **Failure identity.** A column-write failure fails the worker unit; the
  retry rewrites leaf and column wholesale. A committed leaf whose column
  is absent or unstamped therefore reads as **either** a torn worker
  **or** a leaf whose writing declaration carried no column (the gate and
  the clear above) — and the manifest cannot always separate the two,
  since its `pyramid` block MAY lag. Readers never require a column, so
  absence is never an error state; where the **writing** declaration is
  known to carry leaf-node levels, absence is the torn-worker signature
  and the repair is re-invoking the idempotent leaf.

**Stage columns (issue #384).** The staged sweep writes the SAME artifact
shape at its dispatch nodes (`{window}.pyramid.zarr` under an ancestor
node's prefix, `zagg-column/1` attrs, D4 order, one commit stamp last, D20
sidecar after): every group is a **pure gather** of the child columns'
members at the same resolution — `groups` entries record `regime:
"stage-gather"` with `merges_from_raw: 1` — and the artifact MUST carry the
**relay member** (the group at `shard_order`: the subtree's leaf node-order
partials, the merge-source tier every coarser merge consumes — the espg
merge-source ruling on the #384 thread). Stage-column attrs additionally
carry `generation` (`{n_leaves, max_leaf_timestamp, run_ids}` summed over
the consumed children — the parent's skip-gate key, §4.5), `source_children` (a
gather that under-covered says so in the artifact), and `run_id`; the
commit stamp carries `run_id` too. Cadence decides placement (columns sit
at dispatch orders), so column EXISTENCE at a given ancestor order is
orchestration, never contract — a reader binds to the ladder artifacts of
§4.4, not to stage columns. The root tuple writes no column (nothing
consumes it). Raster hive stores are column-less by construction: nothing
in this section applies to them (issue #399 owns their overview regime).

### 4.7 What §4 does not cover (informative)

The sweep's *other* derived-artifact families (MOC regeneration, stats and
sub-shardmap rollups, debris collection) are operational concerns recorded
in [`design/sparse_coverage.md`](design/sparse_coverage.md) D22 — they add
no new byte layouts (the stats rollup reuses the D20 sidecar schema; the
sub-shardmap is ShardMap JSON) — except the §4.8 sweep-admission lease,
which is control plane rather than data. The fold *algebra* for overview
contents is zagg-owned per §2.3; a reader consumes overview arrays exactly
as it consumes leaf arrays.

### 4.8 The sweep-admission lease (`zagg-sweep-lease/1`)

**Status: contract — issue #384 (espg admission ruling).** Pyramid sweeps
**serialize per store**: a column is a multi-object artifact whose D4
stamp-last discipline proves completeness only under a single writer, so
two concurrent sweeps could interleave PUTs into a *chimera* column that
validates as complete. Admission is one atomic conditional PUT
(`If-None-Match: *`) of the store-root **intent object** `sweep.lease.json`:

```json
{
  "spec": "zagg-sweep-lease/1",
  "run_id": "stage-20260809T000000Z-ab12cd",
  "scope": null,
  "acquired_at": "2026-08-09T00:00:00+00:00",
  "heartbeat_at": "2026-08-09T00:05:00+00:00",
  "ttl_s": 900,
  "claimed_from": "stage-20260808T230000Z-9f00aa"
}
```

— `scope` is the admitted run's node-prefix set (informative; the lease is
**store-granular by correctness**: scope-disjointness does not imply
write-disjointness, because disjoint-leaf sweeps converge on shared coarse
ancestors). A live intent refuses admission naming the runner; a
`heartbeat_at` older than `ttl_s` is **claimable** (crash recovery —
`claimed_from` records the takeover), and the claimant simply completes the
partial prior run under the ratchet. The finisher deletes the intent as its
final act. **Control plane, explicitly**: no data object is ever locked —
the lease is what makes "every data object has exactly one writer, ever"
true *across* runs, extending (never amending) the no-locking law. Fleets
are unaffected: fleet ∥ fleet is governed by the leaf single-writer law and
fleet ∥ sweep is allowed (the stage workers validate every column stamp
before and after reading its groups and re-read on movement; stage stamps
carry `run_id`, and a skip-if-current read that sees a foreign stamp
written after the run started aborts loudly). The same `run_id` is a **term
of the skip key** (§4.5): the abort covers a foreign stamp written *since
this run started*, and the key covers the foreign rewrite that landed
before it — inside the second the timestamp cannot resolve.

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

**Contract.** A sweep-built **overview** (§4) records its hashes in a sidecar
the same way, and that sidecar is named from the overview's own basename —
the stem plus `.stats.json` (`all.zarr` → `all.stats.json`, `2019.zarr` →
`2019.stats.json`), a sibling object at the ancestor node. Unlike a source
leaf's sidecar name, which is keyed to the store's `spec` revision, the stem
grammar applies to an overview sidecar **unconditionally, at every
revision**: one ancestor node holds every window's overview (§4.2), so a
revision-keyed bare name would resolve all of them to a single `stats.json`
at that node.

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

Three tiny single-shard hive stores plus one manifest-only declaration, all
on the same deliberately small geometry — shard order 4, inner-chunk order
5, cell order 6 (16 cells, K = 4 inner chunks of 4 cells), sharded (the
hive default):

- **`minimal/`** — one *unlocated* digest field (`h_tdigest`) plus `count`.
  The smallest thing that is a conforming store.
- **`kitchen_sink/`** — the full stratified-product surface: located
  signal/noise digest strata (payload + `{field}_locations` siblings,
  `stratum`/`signal_threshold` provenance attrs), the `composition` word
  (including a single-photon cell packing the §3.1 golden word
  `0xFF000000FF0000FF` and a noise-only cell whose signal payload is the
  empty `(0, 2)` array), and `count`.
- **`pyramid/`** — MANIFEST ONLY: the §4.5 `zagg-pyramid/2` declaration
  grammar. The committed `morton_hive.json` was produced by the production
  declaration paths end to end — templated `/1` (`hive.build_manifest`),
  given sweep actuals by the production bookkeeping writer, then retrofitted
  to `/2` with `declare_pyramid` — and carries every `/2` reading a decoder
  must tell apart: a multi-resolution leaf entry (`{"node": 3, "cells":
  [5, 4]}` — this fixture is shard order 3, not the leaf fixtures' 4, so
  the leaf window has two interior resolutions), the §4.4 fixed every-order
  ladder rooted at node 0, the #376 fold keys (`fold_source`,
  `exact_levels`), and the preserved `/1`-era `materialized.fold_sources`
  actuals — and, via the committed `pyramid.expected.json`, which records
  the raw config knob, the leaf-resolution declaration the expansion was
  derived from. It writes no store beneath it on purpose: the pyramid block
  is a template-time manifest artifact, decodable from `morton_hive.json`
  alone — the `/2` artifacts a fleet writes are `column/`'s job below
  (sweep-side levels are #384's).
- **`column/`** — the `minimal/` inputs plus an explicit
  `output.pyramid.overviews: 5` knob, so the same worker invocation that
  committed the leaf also wrote its §4.6 **column**: `all.pyramid.zarr`
  beside the leaf, groups `{5, 4}` (the declared base and the node-order
  member — this geometry has no interior ladder rung between them), `role:
  column` + the `zagg_column` attrs grammar, its own commit stamp, and the
  `all.pyramid.stats.json` D20 sidecar. `column.expected.json` records the
  decoded group values, the attrs block verbatim, the stamp's clock-free
  fields, and the column's §5 hashes; the conformance tests additionally
  re-derive the base group from the committed leaf through the §4.4 fold
  kernels — the §4.6 from-leaves parity contract, pinned on committed
  bytes. The two-group set is forced, not chosen: on this 4/5/6 geometry
  §4.4 admits only `overviews: 5`, so no committed golden here can carry an
  **interior** ladder rung (a three-or-more-group column, its group
  ordering, or a declared base distinct from an implied rung). That case is
  pinned zagg-side by `tests/test_column.py`; a committed multi-rung golden
  arrives with the sweep-side fixtures of
  [#384](https://github.com/englacial/zagg/issues/384).

`minimal/` and `kitchen_sink/` pin the layout edge cases a reader must
handle (`column/`'s leaf is `minimal/`'s, so it pins them again): inner chunk
ordinal 2 is **empty** (absent from the shard index — the §1.5 sentinel, and
that sparsity reaches the dense arrays too: the `morton` coordinate and
`count` hold their fill across that chunk, so a reader MUST NOT assume the
coordinate is dense across a shard), populated chunks contain empty cells
(the `b""` fill), and one cell's digest carries merged centroids (weight > 1)
whose location words are common ancestors (§2.2).

Every fixture **leaf** is **sharded**, so §7's leaf conformance claim is
scoped to the §1.5 sharded geometry. The per-inner-chunk geometry — identical
§1.4 framing, one object per inner chunk, no shard index — now has a
committed golden: `column/`'s resolution groups are single-chunk unsharded
arrays (§4.6), including a `zagg-ragged/1` payload array (`h_tdigest`,
`chunk_shape == shape`, no `sharding_indexed`), so a reader that hard-codes
the shard-index suffix fails a §7 fixture instead of sailing through. The
unsharded **multi**-chunk case remains pinned zagg-side by
`tests/test_processing.py::TestRaggedVlenLayout`. A reader that derives the
stored span from the array's own metadata, as §1.5 requires, reads all of
these from one code path.

Each fixture ships a sibling **`{name}.expected.json`** recording the shard
key, leaf path, geometry, every populated cell's decoded values (digest
centroids, location words and composition words as decimal strings — JSON
numbers cannot carry `uint64` faithfully), the per-stratum exact counts,
and the §5 O11 `content_hashes` (per-array + combined). The expected **leaf
decoded values** were computed from the generator's *inputs* — the arrays
handed to the writers — never read back through a zagg reader, so the reader
is pinned, not self-certified. The `column` record's group values are the
one exception: they are the writer's committed output, read back. Their
independence comes from elsewhere — the conformance suite **re-derives** the
base group from the committed leaf through the §4.4 fold kernels and asserts
byte equality (the §4.6 from-leaves parity contract), so the recorded values
are a regression pin over an independently derived result, not a
self-certification. The `content_hashes` are necessarily computed
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
principle as §5). In the committed D20 sidecar
(`column/…/all.pyramid.stats.json`) only `content_hashes` and
`cells_with_data` are pinned: `timestamp`, `zagg_version`, `run_id` and the
run counters are **informative provenance**, they churn on every
regeneration, and conformance never asserts them.
