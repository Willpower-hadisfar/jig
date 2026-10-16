"""The trajectory index: written per tool call, readable by a later run, append-only, content-free, bounded.

The full transcript already lives in ``runs``/``run_steps``; what was missing was a reader. These tests
drive the real gate (no model server is needed: the tools used are READ and PRIVATE_WRITE, which the
gate never sends to the Sentinel) and then read the record back from another run.
"""

from __future__ import annotations

import json

import pytest

from jig.db import Database
from jig.trajectory import TrajectoryLog

from .memory_bank_helpers import BankStack


def _log(tmp_path) -> TrajectoryLog:
    return TrajectoryLog(Database(tmp_path / "jig.db"))


async def test_a_failure_in_one_run_informs_a_later_one(tmp_path):
    """The closed loop the design record asks for: the writing already happened; the reader is new."""
    stack = BankStack(tmp_path)
    failed = await stack.call("skill_read", {"name": "no-such-skill"}, run_id="run-early")
    assert not failed.ok and failed.error_type == "ToolError"

    later = await stack.call("history_search", {"status": "failed"}, run_id="run-later")
    assert later.ok, later.error
    rows = later.result["rows"]
    assert rows and rows[0]["run_id"] == "run-early" and rows[0]["tool"] == "skill_read"
    assert rows[0]["error_type"] == "ToolError"

    by_tool = await stack.call("history_search", {"tool": "skill_read", "kind": "step"}, run_id="run-later")
    assert [r["run_id"] for r in by_tool.result["rows"]] == ["run-early"]
    stack.close()


async def test_rows_land_for_successes_and_refusals_alike(tmp_path):
    stack = BankStack(tmp_path)
    ok = await stack.call("fact_remember", {"content": "Wren is the protagonist", "kind": "world"},
                          run_id="run-a")
    assert ok.ok
    refused = await stack.call("memory_update", {"memory_id": 1, "content": "x"}, run_id="run-b")
    assert not refused.ok, "memory_update is a side effect: refused in research mode by the mode rule"
    rows = stack.trajectory.query(run_id="run-b")
    assert [r["status"] for r in rows] == ["failed"] and rows[0]["error_type"] == "ModeViolation"
    assert stack.trajectory.query(run_id="run-a")[0]["status"] == "ok"
    stack.close()


def test_rows_are_never_edited(tmp_path):
    log = _log(tmp_path)
    log.record("step", tool="write_file", status="failed", error_type="ToolError", digest="write_file failed")
    with pytest.raises(Exception, match="append-only"):
        log.db.execute("UPDATE trajectory SET status = 'ok', error_type = NULL WHERE id = 1")
    assert log.query()[0]["status"] == "failed"


def test_the_index_is_bounded_and_wipes_whole(tmp_path):
    log = _log(tmp_path)
    for i in range(30):
        log.record("step", tool=f"tool_{i}", status="ok", digest=f"tool_{i} ok")
    assert log.count() == 30
    assert log.prune(max_rows=500) == 0, "under the bound nothing goes"
    assert log.prune(max_rows=10) == 20
    rows = log.query(limit=50)
    assert len(rows) == 10 and rows[0]["tool"] == "tool_29", "the newest rows survive, the oldest whole rows go"
    out = log.wipe()
    assert out["deleted"] == 10 and log.count() == 0


def test_a_digest_is_short_built_from_metadata_and_redacted(tmp_path):
    seen = []
    log = TrajectoryLog(Database(tmp_path / "jig.db"), redact=lambda text: seen.append(text) or text.replace("sk-1", "[redacted]"))
    log.record("step", digest="run_python failed error_type=ToolError sk-1234567890 " + "x" * 400)
    row = log.query()[0]
    assert len(row["digest"]) <= 160
    assert "sk-1234567890" not in row["digest"], "the redaction runs before the row is stored"


async def test_the_index_holds_no_payload_and_nothing_reaches_the_audit_log(tmp_path):
    """The privacy line: a fact, a proposed skill and a failing call all carry secret words in their
    arguments; neither the trajectory digests nor a single audit row may hold them."""
    secret = "quokkaxylophone"
    stack = BankStack(tmp_path)
    await stack.call("fact_remember", {"content": f"The locker code is {secret}"}, run_id="run-p")
    await stack.call("skill_propose", {"name": "locker-note", "title": "T", "summary": f"About {secret}.",
                                       "body": f"never write {secret}"}, run_id="run-p")
    await stack.call("skill_read", {"name": f"missing-{secret}"}, run_id="run-p")

    for row in stack.trajectory.query(limit=100):
        blob = json.dumps(row)
        assert secret not in blob, f"payload text reached the trajectory: {row}"
        assert len(row["digest"]) <= 160
    for row in stack.audit.query(limit=5000):
        assert secret not in row["summary"] and secret not in row["data_json"]
    # ... and the calls still did their jobs.
    assert stack.facts.search("locker") and stack.skills.list(status="candidate")
    stack.close()
