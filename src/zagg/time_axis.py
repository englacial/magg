"""The temporal declaration and the time axis' encoding (spec §8, #443/#410).

Spec §8 declares a word-typed array by the domain-neutral triple
``{spec, shape, grammar}``, stamped on the array that HOLDS the words. This
module owns the temporal domain: the ``temporal`` attrs key, the shape
vocabulary (:data:`TOC_SHAPES` — a time ``coordinate``, a ``per-cell``
companion, a ``per-centroid`` companion), and the coordinate shape's encode
and decode. The companion shapes' words are produced by the aggregation
kernel; what lives here is their declaration, so writer and reader agree on
one grammar.

A raster ``(time, cells)`` product's ``time`` array carries one value per
acquisition group. Two encodings are defined:

- ``"microseconds"`` (the default, and every store written before §8) —
  signed ``int64`` microseconds since the Unix epoch, self-described by the
  CF ``units``/``calendar`` attrs in :data:`LEGACY_TIME_ATTRS`;
- ``"toc"`` — mortie **toc words** (``uint64``), a tagged union of an exact
  nanosecond timestamp and an outward-rounded conservative range, declared
  by the spec-owned ``temporal`` attrs block this module stamps and reads.

The point of the second one is honesty: a Sentinel-2 datatake is a
~seconds-long acquisition whose adjacent MGRS tiles are sensed seconds
apart, which ``datetime64`` cannot represent and a toc range can. Word
semantics — bit layout, sort order, merge law — are mortie's (the flat
``mortie.time2toc`` / ``mortie.toc2time`` / ``mortie.toc_*`` kernels;
``mortie.toc`` is the ``Toc`` constructor as of mortie 0.9.10, not a
module), never restated here; this module owns zagg's half: the
declaration grammar, the acquisition-group mapping, and the decode.
"""

from __future__ import annotations

import numpy as np

#: The spec §8 attrs key on a time-carrying array. Spec-owned: writers stamp
#: it from the config's declared encoding, config authors never write it.
TEMPORAL_ATTR = "temporal"
#: The §8 convention revision, strict-checked on read.
TOC_SPEC = "zagg-toc/1"
#: §8's domain-neutral shape vocabulary, all three defined by this revision
#: (issue #410). ``coordinate`` — the declaring array IS the CF/xarray
#: coordinate variable of the time dimension (§8.1). ``per-cell`` — a dense
#: uint64 companion, one word per cell of the cell grid (§8.2).
#: ``per-centroid`` — a ragged uint64 sibling, one word per centroid of the
#: digest it accompanies (§8.3). The declaring array is always the array
#: that HOLDS the words; a payload array carries a binding, never a
#: declaration.
TOC_SHAPE_COORDINATE = "coordinate"
TOC_SHAPE_PER_CELL = "per-cell"
TOC_SHAPE_PER_CENTROID = "per-centroid"
TOC_SHAPES = (TOC_SHAPE_COORDINATE, TOC_SHAPE_PER_CELL, TOC_SHAPE_PER_CENTROID)
#: The subset a config declares per FIELD (``temporal:`` on an aggregation
#: variable). ``coordinate`` is not among them — a time axis is declared by
#: ``output.time_encoding``, not by a field.
TOC_FIELD_SHAPES = (TOC_SHAPE_PER_CELL, TOC_SHAPE_PER_CENTROID)
#: The §8.2 reserved word: a per-cell companion's ``fill_value``, marking a
#: cell the writer never observed. Never an acquisition.
TOC_UNOBSERVED = 0
#: The reducers that PRODUCE toc words, and may therefore declare a field-level
#: ``temporal:`` companion (issue #410). A ``per-centroid`` companion comes from
#: the t-digest kernels' ``temporal=`` channel (§8.3); a ``per-cell`` one from
#: :func:`zagg.stats.toc.cell_envelope`, whose whole output IS the word (§8.2).
#: The gate exists because the declaration surface landed one PR ahead of the
#: kernel: a reducer absent from this set would stamp a `zagg-toc/1`
#: declaration over words nothing produced, which is a store violating the
#: section it declares.
TOC_PRODUCING_FUNCTIONS: frozenset[str] = frozenset(
    {
        "zagg.stats.tdigest.build_tdigest",
        "zagg.stats.tdigest.build_tdigest_pairwise",
        "zagg.stats.tdigest.build_tdigest_where",
        "zagg.stats.waveform.build_waveform_digest",
        "zagg.stats.toc.cell_envelope",
    }
)
#: The subset of :data:`TOC_PRODUCING_FUNCTIONS` producing a whole-cell word —
#: the ``per-cell`` shape's reducers (§8.2). The rest carry the channel beside a
#: ragged payload, so they produce ``per-centroid`` (§8.3), and the two sets
#: partition the allowlist: a shape-vs-reducer mismatch is a config error.
TOC_PER_CELL_FUNCTIONS: frozenset[str] = frozenset({"zagg.stats.toc.cell_envelope"})
#: The derived per-observation column the read/aggregation path materializes
#: from :func:`toc_source` — the toc analogue of the HEALPix ``leaf_id`` morton
#: column, and the one place a per-observation instant becomes a toc word.
#: Reserved: a config may READ it (a ``per-cell`` companion declares
#: ``source: toc_word``) but never declare a column of that name.
TOC_WORD_COLUMN = "toc_word"
#: The continuous timescales a temporal companion may ingest from. ``utc`` is
#: excluded deliberately: §8.1/§8.3 require an instant to be encoded exact to
#: the nanosecond, and a nominal-UTC offset column is only good to the leap
#: seconds elapsed since its epoch (``zagg.windows``' documented <= 1 s
#: tolerance) — a full quantum wider than the range variant's ~2.15 s start
#: grid is meant to imply. Window ROUTING tolerates that; a word claiming
#: nanosecond exactness cannot.
TOC_SOURCE_SCALES = ("gps", "tai")
#: The §8 word-grammar citation — a grammar REVISION token in the ecosystem's
#: {name}/{major} style (``zagg-ragged/1``, ``morton-hive/2``), never a
#: documentation URL or a stamp of the writer's installed mortie: store bytes
#: must not move when a floor moves or the documentation moves. §8's prose
#: carries the documentation pointer (the mike-versioned API page today,
#: mortie's frozen spec section once espg/mortie#193 lands).
TOC_GRAMMAR = "mortie-toc/1"
#: The cited grammar's time origin. A grammar property, not a declared key
#: (#410 ruled out per-store epoch/quantization guards); kept here only to
#: refuse a time the words cannot represent.
TOC_EPOCH = "1850-01-01T00:00:00"
#: The legacy encoding's CF attrs, stamped only when ``temporal`` is absent.
LEGACY_TIME_ATTRS = {
    "units": "microseconds since 1970-01-01T00:00:00",
    "calendar": "proleptic_gregorian",
}
#: Config values for ``output.time_encoding``.
TIME_ENCODINGS = ("microseconds", "toc")
#: The absent-key default: what an undeclared axis means, both in config and
#: in stored attrs.
DEFAULT_TIME_ENCODING = "microseconds"

__all__ = [
    "DEFAULT_TIME_ENCODING",
    "LEGACY_TIME_ATTRS",
    "TEMPORAL_ATTR",
    "TIME_ENCODINGS",
    "TOC_EPOCH",
    "TOC_FIELD_SHAPES",
    "TOC_GRAMMAR",
    "TOC_PER_CELL_FUNCTIONS",
    "TOC_PRODUCING_FUNCTIONS",
    "TOC_SHAPE_COORDINATE",
    "TOC_SHAPE_PER_CELL",
    "TOC_SHAPE_PER_CENTROID",
    "TOC_SHAPES",
    "TOC_SOURCE_SCALES",
    "TOC_SPEC",
    "TOC_UNOBSERVED",
    "TOC_WORD_COLUMN",
    "decode_time_axis",
    "encode_time_axis",
    "observation_words",
    "read_time_axis",
    "toc_source",
    "temporal_attrs",
    "temporal_declaration",
    "temporal_declaration_block",
    "time_axis_attrs",
    "time_axis_dtype",
    "time_axis_overlaps",
    "time_encoding",
]


def time_encoding(config) -> str:
    """The config's declared ``output.time_encoding`` (§8), validated.

    Absent is :data:`DEFAULT_TIME_ENCODING` — the legacy encoding, so every
    existing config resolves exactly as before.
    """
    value = (getattr(config, "output", None) or {}).get("time_encoding")
    if value is None:
        return DEFAULT_TIME_ENCODING
    if value not in TIME_ENCODINGS:
        raise ValueError(
            f"output.time_encoding must be one of {TIME_ENCODINGS} (got {value!r}) — spec §8"
        )
    return str(value)


def time_axis_dtype(encoding: str) -> str:
    """The stored element type for a declared encoding."""
    return "uint64" if encoding == "toc" else "int64"


def time_axis_attrs(encoding: str) -> dict:
    """The attrs a writer stamps on the time array for ``encoding``.

    The legacy encoding keeps its CF pair and declares nothing (§8: an absent
    ``temporal`` key IS the legacy encoding); ``toc`` carries the declaration
    and no CF attrs, because ``units``/``calendar`` would describe the words
    wrongly and a CF-decoding client would silently produce garbage dates.
    """
    if encoding != "toc":
        return dict(LEGACY_TIME_ATTRS)
    return {TEMPORAL_ATTR: temporal_declaration_block(TOC_SHAPE_COORDINATE)}


def temporal_declaration_block(shape: str) -> dict:
    """The §8 declaration a writer stamps for ``shape``.

    Three keys and no more (#410): the convention revision, the shape, and
    the word-grammar revision. The epoch, the timescale and the range
    variant's quanta are properties of the cited grammar and are deliberately
    NOT echoed into store bytes.
    """
    if shape not in TOC_SHAPES:
        raise ValueError(f"temporal shape {shape!r} is not one of {TOC_SHAPES} (spec §8)")
    return {"spec": TOC_SPEC, "shape": str(shape), "grammar": TOC_GRAMMAR}


def temporal_attrs(shape: str) -> dict:
    """The attrs a writer stamps on the array that HOLDS the words (§8).

    The declaring array is the companion itself — a dense uint64 array for
    ``per-cell``, a ragged uint64 sibling for ``per-centroid`` — never the
    digest payload it accompanies, which carries only the ``times`` binding
    (§8.3). The whole attrs mapping, so a caller merges rather than guesses
    the key.
    """
    return {TEMPORAL_ATTR: temporal_declaration_block(shape)}


def temporal_declaration(attrs, *, shape: str | None = None) -> dict | None:
    """The §8 ``temporal`` block from an array's attrs, strict-checked.

    Returns ``None`` when the key is absent — never raises for a store
    written before §8, which is the schema-evolution rule (§8.4: an absent
    declaration is never a refusal). Raises on a declaration this reader
    cannot decode: an unknown ``spec``, a ``shape`` outside the vocabulary,
    or an uncited word ``grammar``. Unrecognized keys are informative by §8
    and ignored, never a refusal.

    ``shape`` narrows the check to one expected value — what a caller that
    can only consume one shape (e.g. the time-axis decode, which is
    ``coordinate``-only) passes, so a companion's block is refused rather
    than decoded as an axis.
    """
    block = dict(attrs or {}).get(TEMPORAL_ATTR)
    if block is None:
        return None
    if not isinstance(block, dict):
        raise ValueError(f"{TEMPORAL_ATTR!r} attrs must be a mapping (got {block!r}) — spec §8")
    spec = block.get("spec")
    if spec != TOC_SPEC:
        raise ValueError(
            f"unknown temporal declaration spec {spec!r} (spec §8 defines {TOC_SPEC!r}); "
            f"refusing to guess a future revision's time encoding"
        )
    expected = TOC_SHAPES if shape is None else (shape,)
    if block.get("shape") not in expected:
        raise ValueError(
            f"temporal declaration shape {block.get('shape')!r} is not implemented "
            f"(spec §8 defines {expected} here)"
        )
    grammar = block.get("grammar")
    if grammar != TOC_GRAMMAR:
        raise ValueError(
            f"temporal declaration cites word grammar {grammar!r}, not {TOC_GRAMMAR!r} "
            f"(spec §8); refusing to decode words under a grammar this reader does not "
            f"implement"
        )
    return block


def toc_source(config) -> dict | None:
    """The store's per-observation clock declaration, normalized (issue #410).

    ``output.time_source`` names the column a temporal companion's words are
    encoded from, and the conversion that reaches the grammar's continuous
    scale::

        output:
          time_source:
            field: delta_time                 # a base-rate variables column
            epoch: "2018-01-01T00:00:00"      # the column's zero, as UTC
            scale: gps                        # continuous only (TOC_SOURCE_SCALES)
            units: seconds

    Returns ``{"field", "epoch", "scale", "units"}`` with defaults resolved, or
    ``None`` when nothing is declared. **An absent block falls back to
    ``output.windowing``** when that block already carries a continuous-scale
    clock: window routing and toc ingest then derive from ONE declaration, which
    is what keeps them from disagreeing at a window boundary (an observation
    routed into window *W* whose stored word reads as outside it). When BOTH are
    declared the fallback cannot single-source anything, so
    ``config._validate_time_source`` requires the two to agree on all four keys
    instead (issue #410 review) — otherwise a store would route on one clock and
    encode words from another. A windowed store on ``scale: utc`` gets no
    fallback — see :data:`TOC_SOURCE_SCALES` — and must declare a continuous
    column explicitly; that pairing is exempt from the cross-check by design, and
    is the one path where the same column is declared twice on two scales.

    Shape and vocabulary are validated in :mod:`zagg.config`
    (``_validate_time_source``); this is the read point every consumer uses.
    """
    block = (getattr(config, "output", None) or {}).get("time_source")
    if block is None:
        from zagg.config import get_windowing

        windowing = get_windowing(config)
        if windowing is None or windowing["scale"] not in TOC_SOURCE_SCALES:
            return None
        block = {
            "field": windowing["time_field"],
            "epoch": windowing["epoch"],
            "scale": windowing["scale"],
            "units": windowing["units"],
        }
    return {
        "field": str(block["field"]),
        "epoch": str(block["epoch"]),
        "scale": str(block.get("scale") or TOC_SOURCE_SCALES[0]),
        "units": str(block.get("units") or "seconds"),
    }


def observation_words(values, *, epoch, scale: str, units: str) -> np.ndarray:
    """Per-observation offsets -> exact toc **timestamp** words (§8.3).

    ``values`` is a dataset-time column (e.g. ICESat-2 ``delta_time``): offsets
    in ``units`` on ``scale`` since ``epoch``, the same triple
    :func:`zagg.windows.utc_to_offset` takes. Every output word is a
    **timestamp**, never a range: zagg's readers deliver per-observation
    instants, and §8.3 forbids widening an instant into a range. Range ingest is
    legal under the spec, just not something any shipped reader produces.

    The conversion is offset arithmetic on the grammar's own continuous scale —
    the epoch instant's internal ns plus the column's own continuous offset — so
    no leap table enters the hot path, and no scale-vs-UTC correction belongs
    here: ``mortie.from_datetime64`` is leap-aware, so ``base`` already places
    the epoch on the grammar's continuous scale and ``values`` counts SI seconds
    forward from it. Correcting on top of that would push a pre-2017-epoch word
    18 s (``gps``) / 37 s (``tai``) off the true instant AND away from
    :func:`zagg.windows.offset_to_utc`'s routing of the same observation, which
    is exactly the drift this seam exists to avoid; with no correction, word and
    router agree for every post-2017 epoch and for a GPS-native one, leaving
    only windows.py's documented <= 1-leap-second tolerance. One residue stays:
    :func:`zagg.windows.offset_to_utc` reads a *pre-2017* epoch on a non-UTC
    scale as native to that scale, so for a ``tai`` column it routes 19 s from
    the instant ``epoch + values`` SI seconds names. That is a windows.py epoch
    convention, not something this encode can bridge.
    """
    import mortie
    from mortie import TOC_MAX_NS

    from zagg.windows import UNIT_SECONDS, parse_utc

    if scale not in TOC_SOURCE_SCALES:
        raise ValueError(
            f"toc ingest requires a continuous timescale {TOC_SOURCE_SCALES} (got {scale!r}); "
            f"a nominal-UTC column cannot carry the nanosecond-exact instant §8.3 requires"
        )
    if units not in UNIT_SECONDS:
        raise ValueError(f"toc ingest units {units!r} is not one of {sorted(UNIT_SECONDS)}")
    epoch_dt = parse_utc(epoch)
    base = int(_internal_ns(np.array([epoch_dt.replace(tzinfo=None)], dtype="datetime64[us]"))[0])
    offsets = np.asarray(values, dtype=np.float64) * UNIT_SECONDS[units]
    if offsets.size and not np.isfinite(offsets).all():
        raise ValueError(
            "toc ingest met a non-finite observation time; a NaN or inf instant has no "
            "word, and silently dropping it would misalign the companion from its payload"
        )
    # Nearest ns in two exact steps rather than one lossy multiply: at ICESat-2
    # magnitudes (``delta_time`` ~2.5e8 s) ``offsets * 1e9`` lands near 2.5e17
    # where a float64 ulp is 32 ns, so a single ``rint`` quantizes before it
    # rounds and can miss the nearest ns by 16 ns — against §8.3's "exact to the
    # nanosecond" MUST. ``whole`` is integral and the residue lies in [0, 1), so
    # each part is exact/nearest and the int64 combine below carries the nearest
    # ns of the float64 offset the reader delivered, which is the strongest claim
    # a writer can make (the column's own ~30 ns ulp is inherent to the source).
    whole = np.floor(offsets)
    frac_ns = np.rint((offsets - whole) * 1_000_000_000.0)
    # Bound-check BEFORE the int64 combine, which an out-of-domain offset would
    # overflow silently, and against the grammar's REAL domain: mortie refuses
    # at ``TOC_MAX_NS`` (2^63 - 2^32, its quantum-aligned span ceiling), not at
    # 2^63 - 1, so the last 4.29 s of the naive range would otherwise reach
    # mortie and get its message where the caller needs this one. Both extremes
    # are compared as Python ints — ``base`` exceeds 2^53, so ``float(base)``
    # floats the bound by up to a ulp (512 ns for an odd-second epoch) and this
    # check claims to BE the domain check.
    ceiling = TOC_MAX_NS - 1 - base
    if offsets.size:
        lo, hi = int(np.argmin(offsets)), int(np.argmax(offsets))
        low = int(whole[lo]) * 1_000_000_000 + int(frac_ns[lo])
        high = int(whole[hi]) * 1_000_000_000 + int(frac_ns[hi])
        if low < -base:
            raise ValueError(
                f"observation time precedes the toc epoch {TOC_EPOCH} by "
                f"{-(low + base)} ns — the word grammar cannot represent it (spec §8)"
            )
        if high > ceiling:
            raise ValueError(
                f"observation time exceeds the toc grammar's range ({TOC_EPOCH} + "
                f"2^63 - 2^32 ns, ~2142) by {high - ceiling} ns (spec §8)"
            )
    internal = base + whole.astype(np.int64) * 1_000_000_000 + frac_ns.astype(np.int64)
    return np.asarray(mortie.time2toc(internal.astype(np.uint64)), dtype=np.uint64)


def _internal_ns(when: np.ndarray) -> np.ndarray:
    """UTC ``datetime64`` -> mortie's internal ns on the continuous scale."""
    import mortie

    if when.size and when.min() < np.datetime64(TOC_EPOCH, "us"):
        raise ValueError(
            f"time {when.min()} precedes the toc epoch {TOC_EPOCH} — the word "
            f"grammar cannot represent it (spec §8)"
        )
    return np.asarray(mortie.from_datetime64(when), dtype="uint64")


def encode_time_axis(starts_us, ends_us, *, encoding: str) -> np.ndarray:
    """Encode acquisition envelopes as the stored time-axis values.

    ``starts_us``/``ends_us`` are the group's earliest and latest member
    times in microseconds since the Unix epoch (equal for a single-instant
    acquisition). Under the legacy encoding only ``starts_us`` is stored;
    under ``toc`` a degenerate envelope becomes an exact **timestamp** word
    and a real interval an outward-rounded **range** word, so the writer
    neither widens an instant nor narrows an interval (§8.1).
    """
    starts = np.asarray(starts_us, dtype="int64")
    if encoding != "toc":
        return starts
    import mortie

    ends = np.asarray(ends_us, dtype="int64")
    if starts.shape != ends.shape:
        raise ValueError(f"time-axis starts {starts.shape} and ends {ends.shape} differ in shape")
    if starts.size and (ends < starts).any():
        raise ValueError("time-axis envelope ends before it starts")
    s_ns = _internal_ns(starts.astype("datetime64[us]"))
    e_ns = _internal_ns(ends.astype("datetime64[us]"))
    words = np.asarray(mortie.span2toc(s_ns, e_ns), dtype="uint64")
    instant = s_ns == e_ns
    if instant.any():
        words[instant] = np.asarray(mortie.time2toc(s_ns[instant]), dtype="uint64")
    return words


def decode_time_axis(values, attrs) -> tuple[np.ndarray, np.ndarray]:
    """Decode a stored time axis to ``(start, end)`` ``datetime64[ns]``.

    Both bounds are the exact instant for a legacy value or a toc timestamp
    word; for a toc range word ``end`` is the envelope's **exclusive** upper
    bound (§8.1). The pair — rather than one instant — is the return shape
    because the envelope's midpoint is not an observation, and because the
    reader surface is numpy tuples by convention (D13).
    """
    if temporal_declaration(attrs, shape=TOC_SHAPE_COORDINATE) is None:
        when = np.asarray(values, dtype="int64").astype("datetime64[us]").astype("datetime64[ns]")
        return when, when
    import mortie

    words = np.asarray(values, dtype="uint64")
    start_ns, end_ns = mortie.toc2time(words)
    return mortie.to_datetime64(start_ns), mortie.to_datetime64(end_ns)


def time_axis_overlaps(values, attrs, start, end) -> np.ndarray:
    """Boolean mask of timesteps intersecting the half-open window.

    ``start``/``end`` are anything ``numpy.datetime64`` accepts (ISO strings,
    ``datetime64``, ``datetime``). A toc axis is tested by the grammar's own
    predicate on the stored words — no decode, and the answer is a
    conservative superset (over-reports by at most one quantum at an edge,
    never under-reports, §8.1). A legacy axis compares its instants directly.
    """
    q_start, q_end = np.datetime64(start, "us"), np.datetime64(end, "us")
    if q_end < q_start:
        raise ValueError(f"time window is inverted ({q_start} .. {q_end})")
    if temporal_declaration(attrs, shape=TOC_SHAPE_COORDINATE) is None:
        when = np.asarray(values, dtype="int64").astype("datetime64[us]")
        return (when >= q_start) & (when < q_end)
    import mortie

    words = np.asarray(values, dtype="uint64")
    bounds = _internal_ns(np.array([q_start, q_end], dtype="datetime64[us]"))
    return np.asarray(mortie.toc_overlaps(words, int(bounds[0]), int(bounds[1])), dtype=bool)


def read_time_axis(store, group_path: str = "") -> tuple[np.ndarray, np.ndarray]:
    """Open a product's ``time`` coordinate and decode it (§8).

    Returns the same ``(start, end)`` ``datetime64[ns]`` pair as
    :func:`decode_time_axis`, resolving the encoding from the array's own
    attrs — so a caller reads legacy and toc stores through one code path.
    """
    from zarr import open_array

    path = f"{group_path}/time" if group_path else "time"
    arr = open_array(store, path=path, zarr_format=3, consolidated=False)
    return decode_time_axis(arr[:], dict(arr.attrs))
