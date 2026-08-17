"""The time coordinate's encoding contract (spec §8, issues #443/#410).

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
semantics — bit layout, sort order, merge law — are mortie's
(``mortie.toc``), never restated here; this module owns zagg's half: the
declaration grammar, the acquisition-group mapping, and the decode.
"""

from __future__ import annotations

import numpy as np

#: The spec §8 attrs key on a time-carrying array. Spec-owned: writers stamp
#: it from the config's declared encoding, config authors never write it.
TEMPORAL_ATTR = "temporal"
#: The §8 convention revision, strict-checked on read.
TOC_SPEC = "zagg-toc/1"
#: The only §8 ``shape`` this revision defines: the declaring array IS the
#: CF/xarray coordinate variable of the time dimension. Per-cell /
#: per-centroid companions (#410) land as further values under the same
#: marker, from the same domain-neutral shape vocabulary.
TOC_SHAPE_COORDINATE = "coordinate"
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
    "TOC_GRAMMAR",
    "TOC_SHAPE_COORDINATE",
    "TOC_SPEC",
    "decode_time_axis",
    "encode_time_axis",
    "read_time_axis",
    "temporal_declaration",
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
    return {
        TEMPORAL_ATTR: {
            "spec": TOC_SPEC,
            "shape": TOC_SHAPE_COORDINATE,
            "grammar": TOC_GRAMMAR,
        }
    }


def temporal_declaration(attrs) -> dict | None:
    """The §8 ``temporal`` block from an array's attrs, strict-checked.

    Returns ``None`` for the legacy encoding (absent key) — never raises for
    a store written before §8, which is the schema-evolution rule. Raises on
    a declaration this revision cannot decode: an unknown ``spec``, an
    unimplemented ``shape``, or an uncited word ``grammar``. Unrecognized
    keys are informative by §8 and ignored, never a refusal.
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
    shape = block.get("shape")
    if shape != TOC_SHAPE_COORDINATE:
        raise ValueError(
            f"temporal declaration shape {shape!r} is not implemented "
            f"(spec §8 defines {TOC_SHAPE_COORDINATE!r} for a time coordinate)"
        )
    grammar = block.get("grammar")
    if grammar != TOC_GRAMMAR:
        raise ValueError(
            f"temporal declaration cites word grammar {grammar!r}, not {TOC_GRAMMAR!r} "
            f"(spec §8); refusing to decode words under a grammar this reader does not "
            f"implement"
        )
    return block


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
    if temporal_declaration(attrs) is None:
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
    if temporal_declaration(attrs) is None:
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
