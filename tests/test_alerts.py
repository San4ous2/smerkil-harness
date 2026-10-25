"""Tests for budget alerts."""
import json
import os
import subprocess
import threading
import unittest
import urllib.error
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from smerkil.alerts import (
    DEFAULT_THRESHOLDS,
    BudgetAlerts,
    alerts_from_config,
    load_thresholds,
)
from smerkil.budget import Budget


class Spy:
    """Stub Budget-like object with the fields BudgetAlerts reads."""

    def __init__(self, tokens=0, maxt=1000, usd=0.0, maxu=1.0):
        self.total_tokens = tokens
        self.max_tokens = maxt
        self.usd_spent = usd
        self.max_usd = maxu


class TestThresholds(unittest.TestCase):
    def test_defaults(self):
        self.assertEqual(list(DEFAULT_THRESHOLDS), [0.5, 0.8, 0.95])

    def test_load_defaults_without_workspace(self):
        self.assertEqual(load_thresholds(None), list(DEFAULT_THRESHOLDS))

    def test_load_from_yaml(self):
        try:
            import yaml  # noqa: F401
        except ImportError:
            self.skipTest("pyyaml not installed")
        with TemporaryDirectory() as t:
            ws = Path(t)
            (ws / "agent.yaml").write_text(
                "alerts:\n  thresholds: [0.25, 0.9]\n", encoding="utf-8")
            self.assertEqual(load_thresholds(ws), [0.25, 0.9])

    def test_ignores_bad_values(self):
        try:
            import yaml  # noqa: F401
        except ImportError:
            self.skipTest("pyyaml not installed")
        with TemporaryDirectory() as t:
            ws = Path(t)
            (ws / "agent.yaml").write_text(
                "alerts:\n  thresholds: [0.5, 99, -1, 'x']\n", encoding="utf-8")
            self.assertEqual(load_thresholds(ws), [0.5])


class TestFraction(unittest.TestCase):
    def test_token_fraction(self):
        a = BudgetAlerts(Spy(tokens=500, maxt=1000, usd=0.0, maxu=1.0))
        self.assertAlmostEqual(a._fraction(), 0.5)

    def test_worst_of_token_and_usd(self):
        # 90% of tokens but 50% of money -> 0.9
        a = BudgetAlerts(Spy(tokens=900, maxt=1000, usd=0.5, maxu=1.0))
        self.assertAlmostEqual(a._fraction(), 0.9)

    def test_no_limits_is_zero(self):
        a = BudgetAlerts(Spy(tokens=10, maxt=0, usd=5.0, maxu=0))
        self.assertEqual(a._fraction(), 0.0)

    def test_capped_at_one(self):
        a = BudgetAlerts(Spy(tokens=5000, maxt=1000, usd=9.0, maxu=1.0))
        self.assertEqual(a._fraction(), 1.0)


class TestFiring(unittest.TestCase):
    def setUp(self):
        self.emitted = []
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.ws = Path(self.tmp.name)

    def _alerts(self, spy, **kw):
        a = BudgetAlerts(spy, agent="tester", run_id="r1",
                         workspace=self.ws, **kw)
        return a

    def test_fires_each_threshold_once(self):
        spy = Spy()
        a = self._alerts(spy)
        self.assertEqual(a.check(), [])          # 0%
        spy.total_tokens = 600                   # 60% -> crosses 0.5
        self.assertEqual(a.check(), ["budget_warning"])
        self.assertEqual(a.check(), [])          # already fired
        spy.total_tokens = 900                   # 90% -> crosses 0.8
        self.assertEqual(a.check(), ["budget_warning"])
        spy.total_tokens = 1000                  # 100% -> 0.95 + exceeded
        evs = a.check()
        self.assertIn("budget_exceeded", evs)

    def test_no_pyre_above_budget(self):
        spy = Spy(tokens=100, maxt=1000)
        a = self._alerts(spy)
        self.assertEqual(a.check(), [])

    def test_real_budget_object(self):
        b = Budget(1000, 1.0).model("gpt-4o-mini")
        b.record(600, 0, "gpt-4o-mini")
        a = BudgetAlerts(b, thresholds=[0.5])
        # 600/1000 tokens = 60%
        self.assertEqual(a.check(), ["budget_warning"])

    def test_payload_shape(self):
        a = self._alerts(Spy(tokens=600, maxt=1000))
        p = a._payload("budget_warning", 0.6, 0.5)
        self.assertEqual(p["event"], "budget_warning")
        self.assertEqual(p["agent"], "tester")
        self.assertEqual(p["run_id"], "r1")
        self.assertEqual(p["threshold"], 0.5)
        self.assertEqual(p["tokens"], 600)


class TestDelivery(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.ws = Path(self.tmp.name)
        self.received = []

    def test_posts_to_url(self):
        class H(BaseHTTPRequestHandler):
            def do_POST(inner):
                n = int(inner.headers.get("Content-Length", 0))
                data.append(json.loads(inner.rfile.read(n)))
                inner.send_response(200)
                inner.end_headers()

            def log_message(self, *a):
                pass

        data = []
        srv = HTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=srv.handle_request, daemon=True).start()
        url = f"http://127.0.0.1:{srv.server_address[1]}/hook"
        os.environ["SMERKIL_ALERT_URL"] = url
        self.addCleanup(os.environ.pop, "SMERKIL_ALERT_URL", None)

        a = BudgetAlerts(Spy(tokens=600, maxt=1000), agent="a", run_id="r",
                         workspace=self.ws)
        a.check()
        srv.server_close()
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["event"], "budget_warning")
        self.assertEqual(data[0]["tokens"], 600)

    def test_broken_url_does_not_raise(self):
        os.environ["SMERKIL_ALERT_URL"] = "http://127.0.0.1:1/nope"
        self.addCleanup(os.environ.pop, "SMERKIL_ALERT_URL", None)
        a = BudgetAlerts(Spy(tokens=600, maxt=1000), workspace=self.ws)
        # must not raise
        self.assertEqual(a.check(), ["budget_warning"])

    def test_shell_hook_runs(self):
        try:
            import yaml  # noqa: F401
        except ImportError:
            self.skipTest("pyyaml not installed")
        out = self.ws / "fired.txt"
        (self.ws / "alerts.yaml").write_text(
            f"alerts:\n  budget_warning: 'echo $SMERKIL_ALERT_TOKENS > {out}'\n",
            encoding="utf-8")
        a = BudgetAlerts(Spy(tokens=600, maxt=1000), workspace=self.ws)
        a.check()
        self.assertTrue(out.exists())
        self.assertEqual(out.read_text().strip(), "600")

    def test_missing_hook_file_is_silent(self):
        a = BudgetAlerts(Spy(tokens=600, maxt=1000), workspace=self.ws)
        self.assertEqual(a.check(), ["budget_warning"])  # no raise

    def test_alerts_from_config_helper(self):
        a = alerts_from_config(self.ws, Spy(tokens=600, maxt=1000),
                               agent="x", run_id="y")
        self.assertEqual(a.agent, "x")
        self.assertIsNotNone(a.workspace)


if __name__ == "__main__":
    unittest.main()
