"""Record-before-assert for the per-merge benchmark (issue #365).

A tripwire and a data-collection step must not be ordered so the tripwire
destroys the collection. Pinned here: the retention boundary in
``update_series.py`` -- an object-count mismatch is RETAINED (with its verdict
on the row), an empty measurement is dropped -- which is what makes it safe for
the merge workflow to append a point whatever the tripwire's verdict.
"""

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / ".github" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import bench_metrics  # noqa: E402
import update_series  # noqa: E402


def _rec(commit="c1", target="t1", *, event="merge", obs=5_000_000, mem=1200.0, mismatch=None):
    """A minimal merge record: the fields the retention filter and the dedup read."""
    return {
        "timestamp": "2026-01-01T00:00:00Z",
        "commit": commit,
        "ref": "main",
        "event": event,
        "target": target,
        "total_obs": obs,
        "runtime_s": 200.0,
        "cost_per_shard_usd": 0.005,
        "max_memory_mb": mem,
        "objects_total": 12,
        "objects_write_path": 12,
        "objects_expected": 10,
        "objects_mismatch": mismatch,
    }


def _write(tmp_path, records, name="metrics.json"):
    path = tmp_path / name
    path.write_text(json.dumps(records))
    return str(path)


# --- the anomaly rides the row ---------------------------------------------


def test_objects_mismatch_is_a_retained_column():
    # The verdict used to be JSON-only, so a run that tripped the tripwire left
    # nothing in the series to explain itself -- and (before this issue) left no
    # row at all. Now it survives the reindex.
    assert "objects_mismatch" in bench_metrics.RECORD_COLUMNS
    df = update_series.records_to_frame([_rec(mismatch="expected 10, measured 12")])
    assert df["objects_mismatch"].tolist() == ["expected 10, measured 12"]
    # Clean run -> null, so the column reads as "no anomaly", not "unknown".
    assert update_series.records_to_frame([_rec()])["objects_mismatch"].tolist() == [None]


def test_mismatched_run_is_retained_with_its_verdict(tmp_path):
    # The point this issue exists for: a count mismatch is an anomaly worth
    # having in the series NEXT to the numbers, not a reason to delete them.
    series = tmp_path / "series.parquet"
    update_series.main(
        [
            "--series",
            str(series),
            "--records",
            _write(tmp_path, [_rec(mismatch="expected 10, measured 12")]),
        ]
    )
    out = update_series.load_series(series)
    assert out["commit"].tolist() == ["c1"]
    assert out["objects_mismatch"].tolist() == ["expected 10, measured 12"]
    assert out["total_obs"].tolist() == [5_000_000]


# --- but junk is still rejected, at the retention boundary ------------------


def test_has_empty_metrics_is_the_one_definition():
    # Shared by run_benchmark's --fail-on-empty and the filter below, so the
    # appender rejects exactly the rows the tripwire names (issues #145/#365).
    assert bench_metrics.has_empty_metrics({"total_obs": 0, "max_memory_mb": 800.0})
    assert bench_metrics.has_empty_metrics({"total_obs": 999, "max_memory_mb": None})
    assert bench_metrics.has_empty_metrics({})
    assert not bench_metrics.has_empty_metrics({"total_obs": 1, "max_memory_mb": 1.0})


@pytest.mark.parametrize(
    ("record", "why"),
    [
        (_rec(obs=0, mem=None), "empty metrics"),
        (_rec(obs=999, mem=None), "empty metrics"),
        (_rec(event="pr"), "non-merge"),
    ],
)
def test_unretainable_records_are_dropped_and_reported(tmp_path, capsys, record, why):
    # A silent OOM (obs=0 / no peak memory) would plot as a real dip to zero, so
    # it must not ride the now-unconditional append. Reported, never silent.
    series = tmp_path / "series.parquet"
    update_series.main(["--series", str(series), "--records", _write(tmp_path, [record])])
    assert update_series.load_series(series).empty
    out = capsys.readouterr().out
    assert "skipping record for 't1'" in out and why in out


def test_a_dropped_record_cannot_evict_a_retained_one(tmp_path):
    # Same (commit, target) as a good row: the dedup keeps the LAST write, so an
    # unfiltered junk row would overwrite a real point on a re-dispatch.
    series = tmp_path / "series.parquet"
    update_series.save_series(update_series.records_to_frame([_rec()]), series)
    update_series.main(
        ["--series", str(series), "--records", _write(tmp_path, [_rec(obs=0, mem=None)])]
    )
    out = update_series.load_series(series)
    assert out["total_obs"].tolist() == [5_000_000]


def test_an_all_dropped_batch_leaves_the_series_untouched(tmp_path):
    # ...and leaves it untouched *in dtype*, not just in values: appending an
    # all-null frame widens every integer column to float, so an OOM'd run --
    # which now reaches the appender with nothing to append -- would silently
    # rewrite the whole retained history (issue #365).
    series = tmp_path / "series.parquet"
    update_series.save_series(update_series.records_to_frame([_rec()]), series)
    before = update_series.load_series(series)
    update_series.main(
        ["--series", str(series), "--records", _write(tmp_path, [_rec(obs=0, mem=None)])]
    )
    after = update_series.load_series(series)
    assert after["total_obs"].dtype == before["total_obs"].dtype
    assert after.equals(before)


def test_mixed_batch_keeps_the_good_records(tmp_path):
    series = tmp_path / "series.parquet"
    update_series.main(
        [
            "--series",
            str(series),
            "--records",
            _write(
                tmp_path,
                [
                    _rec(target="t1", mismatch="expected 10, measured 12"),
                    _rec(target="t2", obs=0, mem=None),
                    _rec(target="t3"),
                ],
            ),
        ]
    )
    out = update_series.load_series(series)
    assert sorted(out["target"]) == ["t1", "t3"]
