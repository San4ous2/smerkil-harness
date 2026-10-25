"""Tests for smerkil login and the live-status uplink."""
import json
import os
import threading
import time
import unittest
import urllib.error
import urllib.parse
from email.message import Message
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from smerkil.cli import (
    cmd_login,
    dashboard_token_path,
    load_dashboard_session,
    publish_live_loop,
)


class _Args:
    def __init__(self, **kw):
        self.url = None
        self.email = None
        self.password = None
        self.timeout = 5.0
        self.__dict__.update(kw)


class _Resp:
    """Minimal urllib response: headers + body."""

    def __init__(self, body= b'{"ok":true}', cookies=None, status=200):
        self._body = body
        self.status = status
        self.headers = Message()
        for c in cookies or []:
            self.headers.add_header("Set-Cookie", c)

    def read(self):
        return self._body

    def get_all(self, name):
        return self.headers.get_all(name)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class TestLogin(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self._old = os.environ.get("SMERKIL_HOME")
        os.environ["SMERKIL_HOME"] = self.tmp.name
        self.addCleanup(self._restore)

    def _restore(self):
        if self._old is None:
            os.environ.pop("SMERKIL_HOME", None)
        else:
            os.environ["SMERKIL_HOME"] = self._old

    def test_requires_url(self):
        os.environ.pop("SMERKIL_DASHBOARD_URL", None)
        self.assertEqual(cmd_login(_Args()), 2)

    def test_rejects_non_http(self):
        self.assertEqual(cmd_login(_Args(url="ftp://x")), 2)

    def test_saves_token_from_cookie(self):
        tok = "1.abc%40example.com.sig"
        r = _Resp(cookies=[f"smerkil_session={tok}; Path=/"])
        with patch("urllib.request.urlopen", return_value=r):
            rc = cmd_login(_Args(url="https://dash.test",
                                 email="a@example.com", password="pw"))
        self.assertEqual(rc, 0)
        saved = json.loads(dashboard_token_path().read_text())
        # must be URL-decoded so Authorization headers work verbatim
        self.assertEqual(saved["token"], urllib.parse.unquote(tok))
        self.assertIn("@", saved["token"])
        self.assertEqual(saved["email"], "a@example.com")
        # and re-readable by every other command
        self.assertEqual(load_dashboard_session()["token"], saved["token"])

    def test_token_file_is_private(self):
        with patch("urllib.request.urlopen",
                   return_value=_Resp(cookies=["smerkil_session=1.a%40b.c.s"])):
            cmd_login(_Args(url="https://d.test", email="a@b.c", password="p"))
        mode = dashboard_token_path().stat().st_mode & 0o777
        self.assertEqual(mode, 0o600, "token file must not be world-readable")

    def test_http_error_reported(self):
        err = urllib.error.HTTPError("u", 401, "Unauthorized", {}, None)
        err.read = lambda: b'{"error":"bad creds"}'
        with patch("urllib.request.urlopen", side_effect=err):
            rc = cmd_login(_Args(url="https://d.test", email="a@b.c", password="p"))
        self.assertEqual(rc, 1)

    def test_no_token_returned_is_error(self):
        with patch("urllib.request.urlopen", return_value=_Resp(body=b'{"ok":true}')):
            rc = cmd_login(_Args(url="https://d.test", email="a@b.c", password="p"))
        self.assertEqual(rc, 1)

    def test_load_session_empty_when_absent(self):
        self.assertEqual(load_dashboard_session(), {})


class TestPublishLive(unittest.TestCase):
    def test_loop_posts_and_stops(self):
        calls = []

        class _R:
            status = 200

            def __enter__(self):
                calls.append(time.time())
                return self

            def __exit__(self, *a):
                return False

        snap = {"agent": "a", "sessions": [], "active_runs": 0}
        with patch("urllib.request.urlopen", return_value=_R()):
            stop = publish_live_loop("https://d.test", "tok", 0.05,
                                     lambda: snap, verbose=False)
            time.sleep(0.3)
            stop.set()
            time.sleep(0.1)
        self.assertGreater(len(calls), 1, "should have published repeatedly")

    def test_loop_survives_errors(self):
        import urllib.error

        err = urllib.error.HTTPError("u", 500, "Server Error", {}, None)
        err.read = lambda: b"boom"
        with patch("urllib.request.urlopen", side_effect=err):
            # must not raise, must not hang
            stop = publish_live_loop("https://d.test", "tok", 0.05,
                                     lambda: {"agent": "a"}, verbose=False)
            time.sleep(0.2)
            stop.set()
        self.assertTrue(stop.is_set())

    def test_stop_ends_loop(self):
        with patch("urllib.request.urlopen", return_value=_Resp()):
            stop = publish_live_loop("https://d.test", "t", 0.05,
                                     lambda: {"agent": "a"}, verbose=False)
            time.sleep(0.1)
            stop.set()
            time.sleep(0.2)
        self.assertTrue(stop.is_set())


if __name__ == "__main__":
    unittest.main()
