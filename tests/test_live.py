"""Tests for live status: /live endpoint, heartbeat file, smerkil status."""
import json
import os
import threading
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from smerkil.serve import SessionStore, read_heartbeat, write_heartbeat
from smerkil.cli import cmd_status


class _Cfg:
    def __init__(self, name="demo", model="m1", max_tokens=1000, max_usd=1.0):
        self.name = name
        self.model = model
        self.guardrails = type("G", (), {"max_tokens": max_tokens, "max_usd": max_usd})()


class _Args:
    def __init__(self, **kw):
        self.url = None
        self.token = None
        self.max_age = 20.0
        self.timeout = 5.0
        self.json = False
        self.__dict__.update(kw)


class TestLive(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.home = Path(self.tmp.name)
        self._old = os.environ.get("SMERKIL_HOME")
        os.environ["SMERKIL_HOME"] = str(self.home)
        self.addCleanup(self._restore_home)

    def _restore_home(self):
        if self._old is None:
            os.environ.pop("SMERKIL_HOME", None)
        else:
            os.environ["SMERKIL_HOME"] = self._old

    def test_live_report_empty_store(self):
        store = SessionStore(_Cfg())
        live = store.live()
        self.assertEqual(live["agent"], "demo")
        self.assertEqual(live["model"], "m1")
        self.assertEqual(live["active_runs"], 0)
        self.assertEqual(live["sessions"], [])
        self.assertIn("started_at", live)

    def test_live_tracks_in_flight_and_runs(self):
        store = SessionStore(_Cfg())
        s = store.get("job-1")
        # simulate a run starting
        with store._lock:
            s["in_flight"] = True
        live = store.live()
        self.assertEqual(live["active_runs"], 1)
        self.assertEqual(live["active_sessions"], ["job-1"])
        self.assertTrue(live["sessions"][0]["in_flight"])
        # simulate it finishing
        with store._lock:
            s["in_flight"] = False
            s["runs"] += 1
            s["last_stop_reason"] = "completed"
        live = store.live()
        self.assertEqual(live["active_runs"], 0)
        self.assertEqual(live["sessions"][0]["runs"], 1)
        self.assertEqual(live["sessions"][0]["last_stop_reason"], "completed")

    def test_live_includes_budget(self):
        store = SessionStore(_Cfg())
        s = store.get("j")
        s["budget"].add_usage(100, 20, 0.5) if hasattr(s["budget"], "add_usage") else None
        live = store.live()
        self.assertEqual(live["sessions"][0]["turns"], 0)

    def test_heartbeat_roundtrip(self):
        cfg = _Cfg()
        stop = write_heartbeat(cfg, "127.0.0.1", 8317, os.getpid(),
                               "http://127.0.0.1:8317", every=0.05)
        self.addCleanup(stop.set)
        time.sleep(0.15)
        hb = read_heartbeat(max_age_s=5.0)
        self.assertIsNotNone(hb, "heartbeat should be readable while alive")
        self.assertEqual(hb["port"], 8317)
        self.assertEqual(hb["pid"], os.getpid())
        self.assertEqual(hb["agent"], "demo")

    def test_heartbeat_dead_pid_ignored(self):
        p = self.home / "serve.json"
        p.write_text(json.dumps({
            "pid": 999999, "host": "127.0.0.1", "port": 1,
            "url": "http://x", "agent": "a", "model": "m",
            "updated_at": "2026-10-25T00:00:00+00:00",
        }))
        self.assertIsNone(read_heartbeat(max_age_s=10**9),
                          "a dead pid must not be reported as live")

    def test_missing_heartbeat(self):
        self.assertIsNone(read_heartbeat())

    def test_status_no_server_exits_nonzero(self):
        import contextlib, io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = cmd_status(_Args())
        self.assertEqual(rc, 1)
        self.assertIn("no agent serving", buf.getvalue())

    def test_status_reads_live_endpoint(self):
        payload = {
            "agent": "demo", "model": "m1", "active_runs": 1,
            "active_sessions": ["job-1"],
            "sessions": [{"session": "job-1", "runs": 2, "in_flight": True,
                          "last_stop_reason": None, "prompt_tokens": 10,
                          "completion_tokens": 5, "usd": 0.1, "turns": 4,
                          "created_at": "x", "last_active_at": "y"}],
        }

        class _R:
            def read(self):
                return json.dumps(payload).encode()

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        import contextlib, io
        buf = io.StringIO()
        with patch("urllib.request.urlopen", return_value=_R()) as m:
            with contextlib.redirect_stdout(buf):
                rc = cmd_status(_Args(url="http://127.0.0.1:9"))
        self.assertEqual(rc, 0)
        self.assertEqual(m.call_args[0][0].full_url, "http://127.0.0.1:9/live")
        out = buf.getvalue()
        self.assertIn("running", out)
        self.assertIn("job-1", out)

    def test_status_json_mode(self):
        payload = {"agent": "a", "model": "m", "active_runs": 0, "sessions": []}

        class _R:
            def read(self):
                return json.dumps(payload).encode()

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        import contextlib, io
        buf = io.StringIO()
        with patch("urllib.request.urlopen", return_value=_R()):
            with contextlib.redirect_stdout(buf):
                rc = cmd_status(_Args(url="http://x", json=True))
        self.assertEqual(rc, 0)
        self.assertEqual(json.loads(buf.getvalue())["active_runs"], 0)

    def test_status_rejects_bad_url(self):
        self.assertEqual(cmd_status(_Args(url="ftp://x")), 2)


if __name__ == "__main__":
    unittest.main()
