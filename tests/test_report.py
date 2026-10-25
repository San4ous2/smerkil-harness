"""Tests for smerkil report — trace aggregation for the dashboard."""
import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from smerkil.cli import (
    _aggregate_agent,
    _load_trace_records,
    _sum_totals,
    cmd_report,
)


class _Args:
    def __init__(self, **kw):
        self.name = None
        self.json = False
        self.all = True
        self.limit = 20
        self.__dict__.update(kw)


def _write_run(traces: Path, run_id: str, **over):
    rec = {
        "run_id": run_id,
        "agent": "demo",
        "model": "gpt-4o-mini",
        "stop_reason": "completed",
        "started_at": f"2026-10-25T10:00:{run_id}.000Z",
        "finished_at": f"2026-10-25T10:01:{run_id}.000Z",
        "budget": {"prompt_tokens": 100, "completion_tokens": 20, "usd": 0.001},
        "totals": {"turns": 2, "tool_calls": 3},
        "tool_calls": [{"name": "read_file"}, {"name": "write_file"},
                       {"name": "read_file"}],
    }
    rec.update(over)
    (traces / f"run-{run_id}.json").write_text(json.dumps(rec))
    with (traces / "runs.jsonl").open("a") as f:
        f.write(json.dumps(rec) + "\n")


class TestReport(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        # cmd_report scans $SMERKIL_HOME/agents/* — mirror the real layout
        self.agents = self.root / "agents"
        (self.agents / "demo").mkdir(parents=True)
        self.traces = self.agents / "demo" / "workspace" / "traces"
        self.traces.mkdir(parents=True)

    def test_load_merges_json_and_jsonl_without_duplicates(self):
        _write_run(self.traces, "01")
        recs = _load_trace_records(self.agents / "demo")
        self.assertEqual(len(recs), 1, "same run must not appear twice")

    def test_aggregate_sums_and_counts_tools(self):
        _write_run(self.traces, "01")
        _write_run(self.traces, "02", stop_reason="max_turns_reached")
        recs = _load_trace_records(self.agents / "demo")
        agg = _aggregate_agent("demo", recs)
        self.assertEqual(agg["runs"], 2)
        self.assertEqual(agg["turns"], 4)
        self.assertEqual(agg["tool_calls"], 6)
        self.assertEqual(agg["budget"]["prompt_tokens"], 200)
        self.assertEqual(agg["budget"]["completion_tokens"], 40)
        self.assertAlmostEqual(agg["budget"]["usd"], 0.002, places=6)
        self.assertEqual(agg["stop_reasons"], {"completed": 1, "max_turns_reached": 1})
        self.assertEqual(agg["tools"]["read_file"], 4)
        self.assertEqual(agg["tools"]["write_file"], 2)
        # most recent run first
        self.assertEqual(agg["recent"][0]["run_id"], "02")

    def test_totals_across_agents(self):
        a = _aggregate_agent("a", [
            {"budget": {"prompt_tokens": 10, "completion_tokens": 5, "usd": 0.1},
             "totals": {"turns": 1, "tool_calls": 1}},
        ])
        b = _aggregate_agent("b", [
            {"budget": {"prompt_tokens": 20, "completion_tokens": 7, "usd": 0.2},
             "totals": {"turns": 2, "tool_calls": 3}},
        ])
        t = _sum_totals([a, b])
        self.assertEqual(t["agents"], 2)
        self.assertEqual(t["runs"], 2)
        self.assertEqual(t["prompt_tokens"], 30)
        self.assertAlmostEqual(t["usd"], 0.3, places=6)

    def test_cmd_report_json(self):
        _write_run(self.traces, "01")
        import io
        import contextlib
        buf = io.StringIO()
        os.environ["SMERKIL_HOME"] = str(self.root)
        try:
            with contextlib.redirect_stdout(buf):
                rc = cmd_report(_Args(json=True))
        finally:
            os.environ.pop("SMERKIL_HOME", None)
        self.assertEqual(rc, 0)
        payload = json.loads(buf.getvalue())
        self.assertEqual(payload["totals"]["runs"], 1)
        self.assertEqual(payload["agents"][0]["agent"], "demo")
        self.assertIn("version", payload)
        self.assertIn("generated_at", payload)

    def test_missing_traces_is_empty_not_crash(self):
        empty = self.root / "empty"
        (empty / "workspace").mkdir(parents=True)
        recs = _load_trace_records(empty)
        self.assertEqual(recs, [])
        agg = _aggregate_agent("empty", recs)
        self.assertEqual(agg["runs"], 0)
        self.assertEqual(agg["tools"], {})


if __name__ == "__main__":
    unittest.main()
