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

- Every versioned convention on this page is signaled in store metadata by a
  `spec` marker (`"zagg-ragged/1"`, `"zagg-composition/1"`, …) — the
  coverage-envelope discipline. A conforming reader MUST strict-check the
  marker and **fail loudly on an unknown or future revision**, never
  half-parse under a guessed layout.
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
  ragged data MUST be omitted from disk entirely — the same sub-shard
  sparsity the dense arrays get.
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

**Status: contract (payload bytes); the digest algebra is informative.**

*Populated in phase 3 of the #340 PR.*

## 3. `zagg-composition/1`

**Status: contract.**

*Populated in phase 4 of the #340 PR.*

## 4. Pyramid / overview declarations

**Status: ratified design; implementation in flight
([#201](https://github.com/englacial/zagg/issues/201)).**

*Populated in phase 5 of the #340 PR.*

## 5. O11 content hashes

**Status: contract — frozen on
[#342](https://github.com/englacial/zagg/issues/342).**

*Populated in phase 5 of the #340 PR.*

## 6. `zagg-ragged/2`

**Status: specified, implementation pending
([#210](https://github.com/englacial/zagg/issues/210)).**

*Populated in phase 6 of the #340 PR.*

## 7. Conformance fixtures

*Populated in phase 7 of the #340 PR.*
