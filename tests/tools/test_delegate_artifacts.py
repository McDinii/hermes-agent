"""Subagent run artifacts: full report on disk, short receipt in parent context."""

import json
import re
from pathlib import Path
from unittest.mock import patch

from tools.delegate_tool import delegate_task
from tests.tools.test_delegate import _make_mock_parent


LONG_REPORT = "First line of the report.\nSecond line.\n" + ("body line\n" * 300)
RECEIPT_RE = re.compile(r"^Output saved to: (\S+) \((\d+\.\d) KB, (\d+) lines\)")


def _runs_dir() -> Path:
    from hermes_constants import get_hermes_home

    return get_hermes_home() / "state" / "subagent_runs"


def _entry(idx, summary, status="completed", duration=2.5):
    return {
        "task_index": idx,
        "status": status,
        "summary": summary,
        "api_calls": 1,
        "duration_seconds": duration,
    }


@patch("tools.delegate_tool._run_single_child")
def test_single_task_writes_output_and_meta(mock_run):
    mock_run.return_value = _entry(0, LONG_REPORT)

    result = json.loads(delegate_task(goal="Write report", parent_agent=_make_mock_parent()))
    entry = result["results"][0]

    match = RECEIPT_RE.match(entry["summary"])
    assert match, entry["summary"]
    output_path = Path(match.group(1))
    assert output_path.is_absolute()
    assert output_path.parent == _runs_dir().resolve()
    assert re.fullmatch(r".+_task0_output\.md", output_path.name)
    assert output_path.read_text(encoding="utf-8") == LONG_REPORT
    assert int(match.group(3)) == len(LONG_REPORT.splitlines())
    # Receipt keeps a 1-2 line summary, not the whole body.
    assert entry["summary"].splitlines()[1] == "First line of the report. Second line."
    assert len(entry["summary"]) < 500
    assert entry["artifact_path"] == str(output_path)

    meta = json.loads(Path(entry["meta_path"]).read_text(encoding="utf-8"))
    assert meta["status"] == "completed"
    assert meta["summary"] == LONG_REPORT[:200]
    assert meta["artifactPath"] == str(output_path)
    assert "sessionPath" in meta
    assert meta["duration_s"] == 2.5
    assert meta["started_at"] < meta["finished_at"]


@patch("tools.delegate_tool._run_single_child")
def test_batch_shares_one_run_id(mock_run):
    mock_run.side_effect = [_entry(0, "Result A"), _entry(1, "Result B")]

    result = json.loads(
        delegate_task(
            tasks=[{"goal": "A"}, {"goal": "B"}],
            parent_agent=_make_mock_parent(),
        )
    )

    names = [Path(e["artifact_path"]).name for e in result["results"]]
    run_ids = {re.match(r"(.+)_task\d+_output\.md", n).group(1) for n in names}
    assert len(run_ids) == 1
    assert names[0].endswith("_task0_output.md")
    assert names[1].endswith("_task1_output.md")
    assert Path(result["results"][1]["artifact_path"]).read_text() == "Result B"


@patch("tools.delegate_tool._run_single_child")
def test_separate_calls_get_distinct_run_ids(mock_run):
    mock_run.side_effect = [_entry(0, "one"), _entry(0, "two")]
    parent = _make_mock_parent()

    first = json.loads(delegate_task(goal="x", parent_agent=parent))["results"][0]
    second = json.loads(delegate_task(goal="y", parent_agent=parent))["results"][0]

    assert first["artifact_path"] != second["artifact_path"]


@patch("tools.delegate_tool._run_single_child")
def test_write_failure_falls_back_to_full_text(mock_run):
    mock_run.return_value = _entry(0, LONG_REPORT)

    with patch("pathlib.Path.write_text", side_effect=OSError("disk full")):
        result = json.loads(delegate_task(goal="x", parent_agent=_make_mock_parent()))

    entry = result["results"][0]
    assert entry["summary"] == LONG_REPORT
    assert "artifact_path" not in entry


@patch("tools.delegate_tool._run_single_child")
def test_failed_child_records_error_in_meta(mock_run):
    mock_run.return_value = {
        "task_index": 0,
        "status": "error",
        "summary": None,
        "error": "boom",
        "api_calls": 0,
        "duration_seconds": 0.1,
    }

    entry = json.loads(delegate_task(goal="x", parent_agent=_make_mock_parent()))["results"][0]

    assert entry["summary"] is None
    assert entry["error"] == "boom"
    meta = json.loads(Path(entry["meta_path"]).read_text(encoding="utf-8"))
    assert meta["status"] == "error"
    assert Path(entry["artifact_path"]).read_text() == "boom"
