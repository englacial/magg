# Signal/noise strata and the composition field

Issue #321: ATL03 photon signal confidence cannot be reconstructed after
aggregation (the classifier needs the along-track axis, the atmospheric
background stream, and per-surface tuning), so the signal/background decision
is **committed at ingest** and carried through the store in three pieces:

1. **Two disjoint t-digest fields** per cell — `h_tdigest_signal` and
   `h_tdigest_noise` — split by the ATBD signal predicate: a photon is signal
   when **any** `signal_conf_ph` surface column clears the threshold
   (default `>= 2`, i.e. the ATBD `> 1`; selection rule A1 — the union is
   idempotent and narrowable downstream). Each stratum digest's total weight
   is the **exact** stratum photon count. The total photon-flux distribution
   is recovered at read by one deterministic two-way `merge_tdigests`.
2. **One packed `uint64` composition word** per cell
   (`zagg.stats.composition`, spec `zagg-composition/1`): eight 8-bit lanes
   of quantized fractions of the signal stratum — five per-surface lanes
   (`signal_conf_ph` column order: land, ocean, sea_ice, land_ice,
   inland_water) and three low/med/high lanes (a signal photon's *strongest*
   per-surface confidence, 2/3/4). The level lanes are **absolute** — always
   `conf == 2/3/4`, never renumbered against the threshold — so a product
   committing a higher threshold ships empty lower lanes rather than shifted
   ones, and one lane layout serves every product.
3. **Store attrs** recording the commitment: each stratum array carries
   `stratum` + `signal_threshold`, and the composition array carries the
   versioned `composition` block (`spec`, `lanes`, `of` — the digest whose
   total weight is `N_signal` — and `threshold`). Readers bind to these,
   never to config conventions.

The shipped template is `zagg/configs/atl03_tdigest_strata_healpix.yaml` —
**located strata is the default**: both digest fields carry `location:
leaf_id`, so each centroid stores its order-29 morton word (an exact photon
position at weight 1) in the `{field}_locations` sibling arrays. The
five confidence columns are read from the single 2-D `signal_conf_ph` dataset
via the per-variable `column` selector (`{path: ..., column: k}` — the
variable analogue of the structured-filter `column`); the shared path is
still read once.

## Quantization: the presence floor

Lanes quantize as `k = round(255 * c / N)` **except any nonzero count
quantizes to at least 1**. Consequences:

- `lane > 0` means "this flag occurred" **exactly, at every N**, through
  arbitrary merge chains.
- Count recovery `round(k * N / 255)` is exact whenever `N <= 254` — the
  entire below-compression-knee regime (measured 99.56% of non-empty cells
  on the live NEON store, full-mission pooling).
- Above that, counts are within `±N/510` (plus `O(N/510)` per
  re-quantizing merge); presence stays exact.
- A cell with one signal photon has lanes in `{0, 255}` — the lanes *are*
  that photon's flags.

The per-surface lanes are overlapping marginals (`surf_type` is multi-hot):
they do not sum to 255, and they cannot split the height distribution per
surface (decision D: strata stay signal/noise, never per-surface).

## Merge law

`merge_composition(word_a, n_a, word_b, n_b)` folds lanes as the
digest-weighted mean `(n_a·lane_a + n_b·lane_b)/(n_a + n_b)`, re-quantized
with the same presence floor — an order-independent monoid whose `n` inputs
come from the signal digests' total weights.

## Operational caveat

Strata and composition run on the **pooled path and the single-block spill
regime** (exact, via the pooled replay). The streaming merge surface rejects
them: its fold rebuilds digests from the raw source column and would silently
ignore the `where` mask, and the composition fold is not wired into the
streaming state. Heavy multi-block shards need the fold-law follow-up
(issue #321).

## What the mask channel gains

With the strata in place, the HHDC reader's occupancy mask (issue #265)
upgrades from 2-state to 3-state from store contents alone. State the rule
against `count`, which the template already writes:

- **unobserved** — `count == 0`: no observation fell in the cell.
- **observed with zero signal returns** — `count > 0` and the signal stratum
  is empty: the laser crossed the cell and every photon was background.
- **observed with signal** — the signal stratum is non-empty.

`count` rather than "both strata empty" because the strata are keyed to
*finite* heights: `build_tdigest_where` and `pack_composition` both drop
non-finite `source` rows (that is what makes the `N_signal` alignment exact),
so a cell whose every observation carried a non-finite height would report
`count > 0`, both strata empty, `composition == 0`. ATL03 `h_ph` is not
nullable, so the shipped template cannot produce that cell — but the `count`
form is the robust rule for any source and costs nothing to prefer.
