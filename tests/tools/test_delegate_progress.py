"""Non-blocking child -> parent progress channel (progress_update tool)."""

import json
import os
import stat
from pathlib import Path
from unittest.mock import patch

from tools.delegate_tool import (
    PROGRESS_UPDATE_TOOL_NAME,
    delegate_task,
    progress_update_tool,
)
from tests.tools.test_delegate import _make_mock_parent


def _runs_dir() -> Path:
    from hermes_constants import get_hermes_home

    return get_hermes_home() / "state" / "subagent_runs"


def _entry(idx, summary="Done.", status="completed"):
    return {
        "task_index": idx,
        "status": status,
        "summary": summary,
        "api_calls": 1,
        "duration_seconds": 1.0,
    }


def _child_sends(*messages):
    """_run_single_child stand-in: the child posts updates, then finishes."""
    seen = {}

    def _run(task_index, goal, child, parent_agent, **_kw):
        seen[task_index] = child
        for msg in messages:
            out = json.loads(progress_update_tool(child, f"{msg} (task {task_index})"))
            assert out.get("success") is True, out
        return _entry(task_index, f"Report {task_index}")

    return _run, seen


def test_child_updates_reach_parent_result():
    run, seen = _child_sends("Read the config", "Found the bug")
    with patch("tools.delegate_tool._run_single_child", side_effect=run):
        result = json.loads(delegate_task(goal="Fix it", parent_agent=_make_mock_parent()))

    summary = result["results"][0]["summary"]
    assert "Progress updates:\n- Read the config (task 0)\n- Found the bug (task 0)" in summary
    # Section comes after the report receipt, not instead of it.
    assert summary.startswith("Output saved to: ")

    child = seen[0]
    assert PROGRESS_UPDATE_TOOL_NAME in child.valid_tool_names
    assert any(t["function"]["name"] == PROGRESS_UPDATE_TOOL_NAME for t in child.tools)

    req_dir = _runs_dir() / child._subagent_run_id / "requests"
    files = sorted(req_dir.glob("*.json"))
    assert len(files) == 2
    assert stat.S_IMODE(os.stat(req_dir).st_mode) == 0o700
    payload = json.loads(files[0].read_text(encoding="utf-8"))
    assert payload["type"] == "subagent.progress"
    assert payload["runId"] == child._subagent_run_id
    assert payload["taskIndex"] == 0
    assert set(payload) == {"type", "id", "createdAt", "runId", "taskIndex", "message"}
    assert files[0].name == f"{payload['id']}.json"
    assert not list(req_dir.glob(".tmp-*"))


def test_no_updates_means_no_section():
    with patch("tools.delegate_tool._run_single_child", return_value=_entry(0)):
        result = json.loads(delegate_task(goal="Quiet", parent_agent=_make_mock_parent()))

    assert "Progress updates:" not in result["results"][0]["summary"]


def test_parallel_children_keep_their_own_updates():
    run, seen = _child_sends("step")
    with patch("tools.delegate_tool._run_single_child", side_effect=run):
        result = json.loads(
            delegate_task(
                tasks=[{"goal": "A"}, {"goal": "B"}],
                parent_agent=_make_mock_parent(),
            )
        )

    assert seen[0]._subagent_run_id == seen[1]._subagent_run_id
    assert (seen[0]._subagent_task_index, seen[1]._subagent_task_index) == (0, 1)
    s0, s1 = (r["summary"] for r in result["results"])
    assert "- step (task 0)" in s0 and "task 1" not in s0
    assert "- step (task 1)" in s1 and "task 0" not in s1


def test_write_failure_returns_error_without_raising():
    seen = {}

    def _run(task_index, goal, child, parent_agent, **_kw):
        with patch("tempfile.mkstemp", side_effect=OSError("disk full")):
            seen["out"] = json.loads(progress_update_tool(child, "hello"))
        return _entry(task_index)

    with patch("tools.delegate_tool._run_single_child", side_effect=_run):
        result = json.loads(delegate_task(goal="x", parent_agent=_make_mock_parent()))

    assert "disk full" in seen["out"]["error"]
    entry = result["results"][0]
    assert entry["status"] == "completed"
    assert "Progress updates:" not in entry["summary"]


def test_tool_outside_subagent_is_an_error():
    class _Plain:
        pass

    out = json.loads(progress_update_tool(_Plain(), "hi"))
    assert "error" in out
