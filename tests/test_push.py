"""Tests for smerkil push — opt-in upload of the usage report."""
import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from smerkil.cli import cmd_push


class _Args:
    def __init__(self, **kw):
        self.name = None
        self.url = None
        self.token_file = None
        self.limit = 20
        self.timeout = 5.0
        self.__dict__.update(kw)


class _Resp:
    def __init__(self, status=200, body=b'{"ok":true,"agents":1,"history":2}'):
        self.status = status
        self._body = body

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class TestPush(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "agents").mkdir()
        # silence stdout for every case
        import contextlib
        import io
        self._buf = io.StringIO()
        self._ctx = contextlib.redirect_stdout(self._buf)
        self._ctx.__enter__()
        self.addCleanup(self._ctx.__exit__, None, None, None)

    def test_requires_url(self):
        os.environ.pop("SMERKIL_DASHBOARD_URL", None)
        self.assertEqual(cmd_push(_Args()), 2)

    def test_rejects_non_http_url(self):
        self.assertEqual(cmd_push(_Args(url="ftp://x", token_file="/dev/null")), 2)

    def test_requires_token(self):
        os.environ.pop("SMERKIL_DASHBOARD_TOKEN", None)
        self.assertEqual(cmd_push(_Args(url="https://example.test")), 2)

    def test_url_from_env(self):
        os.environ["SMERKIL_DASHBOARD_URL"] = "https://example.test"
        self.addCleanup(os.environ.pop, "SMERKIL_DASHBOARD_URL", None)
        tk = self.root / "tk"
        tk.write_text("tok-abc")
        with patch("urllib.request.urlopen", return_value=_Resp()) as m:
            rc = cmd_push(_Args(token_file=str(tk)))
        self.assertEqual(rc, 0)
        req = m.call_args[0][0]
        self.assertEqual(req.full_url, "https://example.test/api/usage")
        self.assertEqual(req.get_header("Authorization"), "Bearer tok-abc")
        sent = json.loads(req.data.decode())
        self.assertIn("totals", sent)
        self.assertIn("agents", sent)

    def test_token_file_missing_is_error(self):
        self.assertEqual(
            cmd_push(_Args(url="https://example.test", token_file="/no/such/file")), 2
        )

    def test_http_error_surfaces_status(self):
        import urllib.error
        tk = self.root / "tk"
        tk.write_text("t")
        err = urllib.error.HTTPError(
            "https://example.test/api/usage", 401, "Unauthorized", {}, None
        )
        err.read = lambda: b'{"error":"sign in first"}'
        with patch("urllib.request.urlopen", side_effect=err):
            rc = cmd_push(_Args(url="https://example.test", token_file=str(tk)))
        self.assertEqual(rc, 1)

    def test_payload_matches_report_shape(self):
        # one agent with two runs -> payload aggregated, not raw traces
        d = self.root / "agents" / "demo"
        (d / "traces").mkdir(parents=True)
        rec = {
            "run_id": "01", "agent": "demo", "model": "m", "stop_reason": "completed",
            "finished_at": "2026-10-25T10:00:00Z",
            "budget": {"prompt_tokens": 100, "completion_tokens": 20, "usd": 0.001},
            "totals": {"turns": 2, "tool_calls": 3},
            "tool_calls": [{"name": "read_file"}],
        }
        (d / "traces" / "run-01.json").write_text(json.dumps(rec))
        os.environ["SMERKIL_HOME"] = str(self.root)
        self.addCleanup(os.environ.pop, "SMERKIL_HOME", None)
        tk = self.root / "tk"
        tk.write_text("t")
        with patch("urllib.request.urlopen", return_value=_Resp()) as m:
            rc = cmd_push(_Args(url="https://example.test", token_file=str(tk)))
        self.assertEqual(rc, 0)
        sent = json.loads(m.call_args[0][0].data.decode())
        self.assertEqual(sent["totals"]["runs"], 1)
        self.assertEqual(sent["agents"][0]["agent"], "demo")


if __name__ == "__main__":
    unittest.main()
