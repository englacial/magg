"""Drift guard: no mortie version zagg's prose quotes may outrun the floor.

zagg names mortie release numbers in three different voices, and only one of
them is a live obligation:

1. the **package floor** — ``mortie>=…`` in ``[project.dependencies]``, the
   single source (``deployment/aws/build_layer.sh`` derives ``MORTIE_SPEC``
   from it, issue #322);
2. **arrival notes** in narrative docs and docstrings ("added in mortie
   0.8.4") — historically true, permanently satisfied because the floor
   already exceeds them, and enforced by nothing at runtime;
3. the one **runtime gate**, :data:`zagg.grids.aoi.MIN_MORTIE_VERSION`, which
   ``_assert_mortie_version`` turns into a ``RuntimeError``.

The failure mode worth guarding is (2) drifting *above* (1): prose promising
an API the dependency floor does not guarantee, which reaches a user as an
``AttributeError`` where the docs said otherwise. So every mortie version
quoted under ``docs/`` or ``src/`` must be at or below the floor, and the
``aoi_mask`` family — the sites that document (3) and therefore cannot be
de-versioned — must quote the enforced constant exactly.

Scope note: a citation is recognized only where the release number is
lexically attached to the word ``mortie``. A bare "shipped in 0.8.2" further
along the same sentence (``src/zagg/grids/aoi.py``) is not matched, because no
regex separates it from the other version numbers prose carries.
"""

import re
import tomllib
from pathlib import Path

from packaging.requirements import Requirement
from packaging.version import Version

from zagg.grids.aoi import MIN_MORTIE_VERSION

REPO_ROOT = Path(__file__).parent.parent

# Prose scanned for citations: narrative docs plus the shipped package
# (docstrings, comments, bundled config YAML). ``pyproject.toml`` is
# deliberately outside both roots — it is the floor these citations are
# measured against, not one of them.
SCAN_ROOTS = (REPO_ROOT / "docs", REPO_ROOT / "src")
SCAN_SUFFIXES = {".md", ".py", ".yaml", ".yml"}

# The ``aoi_mask`` family: the six sites documenting the runtime gate.
AOI_MASK_FILES = ("docs/aoi_mask.md", "src/zagg/grids/aoi.py")

# ``mortie`` followed by a release number, in every voice the tree uses:
# ``mortie >= 0.8.3``, ``mortie ≥ 0.8.4``, ``mortie>=0.8.3``, bare
# ``mortie 0.8.1``, and the backticked variants (the backticks fall outside
# the match). Whitespace spans newlines so a wrapped citation is not a blind
# spot. ``espg/mortie#89`` does not match: an issue reference is not a
# version, and neither is "frozen for mortie 1.x".
_CITATION = re.compile(r"mortie\s*(?:>=|≥)?\s*v?(\d+\.\d+(?:\.\d+)*)")


def _cited_versions(text):
    """Every mortie release ``text`` names, as ``(Version, line number)``."""
    return [
        (Version(m.group(1)), text.count("\n", 0, m.start()) + 1) for m in _CITATION.finditer(text)
    ]


def _scan():
    """Every mortie citation under the scan roots, as ``(path, line, Version)``."""
    sites = []
    for root in SCAN_ROOTS:
        for path in sorted(root.rglob("*")):
            if path.suffix not in SCAN_SUFFIXES or not path.is_file():
                continue
            rel = path.relative_to(REPO_ROOT).as_posix()
            for version, line in _cited_versions(path.read_text(encoding="utf-8")):
                sites.append((rel, line, version))
    return sites


def _above(sites, floor):
    """The comparison the guard makes — PEP 440 ordering, never string compare."""
    return [s for s in sites if s[2] > floor]


def _mortie_floor():
    """The ``>=`` floor from ``[project.dependencies]``."""
    deps = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    reqs = [Requirement(d) for d in deps["project"]["dependencies"]]
    spec = next(r.specifier for r in reqs if r.name == "mortie")
    lower = [Version(s.version) for s in spec if s.operator in (">=", "==", "~=")]
    assert len(lower) == 1, f"expected one mortie lower bound in pyproject, got {lower}"
    return lower[0]


class TestQuotedVersionsAgainstFloor:
    """Nothing zagg's prose promises may exceed what the floor guarantees."""

    def test_scan_finds_the_versioned_sites(self):
        # A regex that silently matched nothing would let every other test in
        # this module pass forever, so pin that the scan has real work to do.
        sites = _scan()
        assert len(sites) >= 12, f"mortie citation scan went blind: {sites}"
        files = {path for path, _, _ in sites}
        for expected in (
            "docs/aoi_mask.md",
            "docs/morton_arrow.md",
            "docs/ragged_layout.md",
            "src/zagg/grids/aoi.py",
            "src/zagg/grids/morton.py",
            "src/zagg/grids/healpix.py",
        ):
            assert expected in files, f"{expected} no longer reached by the scan"

    def test_no_citation_outruns_the_floor(self):
        floor = _mortie_floor()
        over = _above(_scan(), floor)
        assert not over, (
            f"prose quotes a mortie version above the pyproject floor {floor}: {over}. "
            "Either raise the floor in [project.dependencies] (an espg-authorized "
            "dependency change) or restate the prose against a shipped release."
        )

    def test_runtime_gate_is_at_or_below_the_floor(self):
        floor = _mortie_floor()
        assert Version(MIN_MORTIE_VERSION) <= floor, (
            f"zagg.grids.aoi.MIN_MORTIE_VERSION ({MIN_MORTIE_VERSION}) is above the "
            f"pyproject floor ({floor}) — aoi_mask would raise on a conforming install"
        )

    def test_aoi_mask_prose_matches_the_runtime_gate(self):
        # These six sites document a gate that raises (``aoi.py`` lines 66-80),
        # so they keep requirement voice — and must quote the constant itself.
        family = [s for s in _scan() if s[0] in AOI_MASK_FILES]
        assert len(family) >= 6, f"aoi_mask citations vanished from the scan: {family}"
        wrong = [s for s in family if s[2] != Version(MIN_MORTIE_VERSION)]
        assert not wrong, (
            f"aoi_mask prose disagrees with MIN_MORTIE_VERSION ({MIN_MORTIE_VERSION}): {wrong}"
        )


class TestGuardFires:
    """A guard that cannot fail is worse than no guard — prove each part bites."""

    def test_a_citation_above_the_floor_is_caught(self):
        floor = _mortie_floor()
        text = "the ragged reader rides `mortie >= 99.0.0`, whose contract is normative"
        sites = [("docs/synthetic.md", line, v) for v, line in _cited_versions(text)]
        assert [str(s[2]) for s in sites] == ["99.0.0"]
        assert _above(sites, floor) == sites

    def test_ordering_is_pep440_not_lexicographic(self):
        # The case a string compare gets backwards: "0.10.0" < "0.9.5" as text.
        assert "0.10.0" < "0.9.5"
        assert Version("0.10.0") > Version("0.9.5")
        sites = [("docs/synthetic.md", 1, Version("0.10.0"))]
        assert _above(sites, Version("0.9.5")) == sites
        assert _above(sites, Version("0.10.0")) == []

    def test_issue_references_are_not_versions(self):
        assert _cited_versions("(espg/mortie#89, espg/mortie#100) and mortie 1.x") == []

    def test_every_citation_spelling_is_recognized(self):
        text = (
            "`mortie >= 0.8.3` and mortie ≥ 0.8.4 and `mortie>=0.9.3` and "
            "mortie 0.8.1 and added in mortie\n0.9.0"  # wrapped by a line break
        )
        assert [str(v) for v, _ in _cited_versions(text)] == [
            "0.8.3",
            "0.8.4",
            "0.9.3",
            "0.8.1",
            "0.9.0",
        ]

    def test_the_floor_line_is_not_scanned_as_a_citation(self):
        # The regex would happily match ``mortie>=0.9.3`` in pyproject.toml, so
        # the exclusion has to come from the scan roots — not from luck.
        floor = _mortie_floor()
        assert [v for v, _ in _cited_versions(f'    "mortie>={floor}",')] == [floor]
        scanned = {path for path, _, _ in _scan()}
        assert not any(path.endswith("pyproject.toml") for path in scanned)
