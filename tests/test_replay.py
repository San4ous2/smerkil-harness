"""Tests for `smerkil replay`."""
import builtins
import io
import json
import os
import unittest
from argparse import Namespace
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

from smerkil.cli import cmd_replay

TRACE = {
    "run_id": "20261025-120000-1234",
    "agent": "tester",
    "model": "m1",
    "started_at": "2026-10-25T12:00:00+00:00",
    "steps": [
        {"turn": 1, "model": "m1", "content": "I will search.",
         "tool_calls": [{"name": "search", "id": "c1"}],
         "tokens": {"prompt": 100, "completion": 20},
         "finish_reason": "tool_calls"},
        {"turn": 2, "model": "m1", "content": "Done now.",
         "tool_calls": [], "tokens": {"prompt": 200, "completion": 10},
         "finish_reason": "stop"},
    ],
    "tool_calls": [
        {"turn": 1, "name": "search", "args": '{"q":"latvia"}',
         "result": "found 3 results"},
        {"turn": 1, "name": "fetch", "args": '{"url":"http://x"}',
         "result": "page body"},
    ],
    "final": "Latvia is in the Baltics.",
    "stop_reason": "completed",
    "compactions": 1,
    "budget": {"prompt_tokens": 300, "completion_tokens": 30, "usd": 0.0041},
    "totals": {"turns": 2, "tool_calls": 2},
}


def _ns(**kw):
    n = Namespace(name="tester", run=None, tools_only=False, json=False,
                  quiet=False, verbose=False)
    n.__dict__.update(kw)
    return n


class TestReplay(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        # agents live in the CWD, one directory per agent (_agent_dir = cwd/name)
        self.root = Path(self.tmp.name)
        self._old_cwd = os.getcwd()
        os.chdir(self.root)
        self.addCleanup(os.chdir, self._old_cwd)
        self.agent = self.root / "tester"
        self.traces = self.agent / "workspace" / "traces"
        self.traces.mkdir(parents=True)
        (self.traces / "run-20261025-120000.json").write_text(
            json.dumps(TRACE), encoding="utf-8")
        # an older run, to prove "most recent" picks the right one
        older = dict(TRACE, run_id="20261024-110000-999", final="older answer")
        (self.traces / "run-20261024-110000.json").write_text(
            json.dumps(older), encoding="utf-8")

    def _run(self, **kw):
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = cmd_replay(_ns(**kw))
        return rc, buf.getvalue()

    def test_replays_most_recent(self):
        rc, out = self._run()
        self.assertEqual(rc, 0)
        self.assertIn("20261025-120000-1234", out)
        self.assertIn("Latvia is in the Baltics.", out)
        # both tool calls from turn 1 present, in order
        self.assertIn("→ search(", out)
        self.assertIn("→ fetch(", out)
        self.assertIn("stop completed", out)
        self.assertIn("300p/30c", out)

    def test_picks_explicit_run(self):
        rc, out = self._run(run="20261024-110000")
        self.assertEqual(rc, 0)
        self.assertIn("older answer", out)
        self.assertNotIn("20261025", out)

    def test_json_output_is_parseable(self):
        rc, out = self._run(json=True)
        self.assertEqual(rc, 0)
        parsed = json.loads(out)
        self.assertEqual(parsed["stop_reason"], "completed")

    def test_quiet_prints_only_final(self):
        rc, out = self._run(quiet=True)
        self.assertEqual(rc, 0)
        self.assertEqual(out.strip(), "Latvia is in the Baltics.")

    def test_tools_only_hides_prose(self):
        rc, out = self._run(tools_only=True)
        self.assertEqual(rc, 0)
        self.assertIn("→ search(", out)
        self.assertNotIn("I will search.", out)
        self.assertNotIn("Latvia is in the Baltics.", out)

    def test_verbose_shows_results(self):
        rc, out = self._run(verbose=True)
        self.assertIn("found 3 results", out)

    def test_missing_run_id_errors(self):
        rc, out = self._run(run="does-not-exist")
        self.assertEqual(rc, 1)
        self.assertIn("no trace matching", out)

    def test_no_traces_errors(self):
        (self.root / "other").mkdir(parents=True)
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = cmd_replay(_ns(name="other"))
        self.assertEqual(rc, 1)

    def test_reads_from_runs_jsonl(self):
        # a run that only exists in runs.jsonl (json-per-run was pruned)
        for p in self.traces.glob("run-*.json"):
            p.unlink()
        (self.traces / "runs.jsonl").write_text(
            json.dumps(TRACE) + "\n", encoding="utf-8")
        rc, out = self._run(run="20261025-120000-1234")
        self.assertEqual(rc, 0)
        self.assertIn("Latvia is in the Baltics.", out)

    def test_empty_trace_is_handled(self):
        (self.traces / "run-20261026-000000.json").write_text(
            json.dumps({"run_id": "x", "agent": "tester", "model": "m",
                        "steps": [], "tool_calls": [], "budget": {},
                        "totals": {}}), encoding="utf-8")
        rc, out = self._run(run="20261026")
        self.assertEqual(rc, 0)
        self.assertIn("empty trace", out)


if __name__ == "__main__":
    unittest.main()
