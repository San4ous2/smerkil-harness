"""Tests for smerkil.serve — HTTP API over an agent (offline, no network)."""
import json
import threading
import unittest
import urllib.request
from pathlib import Path
from tempfile import TemporaryDirectory

from smerkil.agent import Agent
from smerkil.config import AgentConfig
from smerkil.serve import make_server


def _fake_agent(workspace: Path) -> Agent:
        cfg = AgentConfig.from_dict({
            "model": "gpt-4o-mini",
            "tools": [],
            "api_key_env": "SMERKIL_TEST_KEY",
            "guardrails": {"max_turns": 3, "max_tokens": 100000, "usd": 1.0},
        }, "webby")
        import os
        os.environ.setdefault("SMERKIL_TEST_KEY", "test-key")
        agent = Agent(cfg, workspace, stream=False)
        # stub the LLM call: echoes the last user message + one fake usage
        def fake_chat(messages, on_delta=None, on_tool=None, model_chain=None):
            user = [m for m in messages if m["role"] == "user"][-1]["content"]
            return {"model": "gpt-4o-mini",
                    "choices": [{"message": {"content": "echo: " + user},
                                 "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5}}
        agent._chat = fake_chat
        return agent


class TestServe(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.ws = Path(self.tmp.name) / "ws"
        self.ws.mkdir()
        self.addCleanup(self.tmp.cleanup)

    def _server(self, **kw):
        httpd = make_server(AgentConfig.from_dict({"tools": []}, "webby"),
                            self.ws, host="127.0.0.1", port=0,
                            agent_factory=lambda: _fake_agent(self.ws), **kw)
        host, port = httpd.server_address
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(httpd.server_close)
        return f"http://{host}:{port}"

    def _post(self, base, path, body, headers=None):
        req = urllib.request.Request(
            base + path, data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", **(headers or {})})
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.status, json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read().decode())

    def _get(self, base, path, headers=None):
        req = urllib.request.Request(base + path, headers=headers or {})
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.status, json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read().decode())

    def test_health_and_unknown_route(self):
        base = self._server()
        code, body = self._get(base, "/health")
        self.assertEqual(code, 200)
        self.assertTrue(body["ok"])
        self.assertEqual(body["model"], "gpt-4o-mini")
        code, body = self._get(base, "/nope")
        self.assertEqual(code, 404)

    def test_chat_multi_turn_session(self):
        base = self._server()
        code, body = self._post(base, "/v1/chat",
                                {"message": "hello", "session": "s1"})
        self.assertEqual(code, 200)
        self.assertEqual(body["reply"], "echo: hello")
        self.assertEqual(body["stop_reason"], "completed")
        self.assertEqual(body["budget"]["prompt_tokens"], 10)
        # same session accumulates tokens (session budget shared)
        code, body = self._post(base, "/v1/chat",
                                {"message": "again", "session": "s1"})
        self.assertEqual(body["budget"]["prompt_tokens"], 20)
        # different session starts fresh
        code, body = self._post(base, "/v1/chat",
                                {"message": "new one", "session": "s2"})
        self.assertEqual(body["budget"]["prompt_tokens"], 10)

    def test_sessions_listing(self):
        base = self._server()
        self._post(base, "/v1/chat", {"message": "a", "session": "x"})
        self._post(base, "/v1/chat", {"message": "b", "session": "y"})
        code, body = self._get(base, "/sessions")
        self.assertEqual(code, 200)
        ids = [s["session"] for s in body["sessions"]]
        self.assertEqual(ids, ["x", "y"])
        for s in body["sessions"]:
            self.assertEqual(s["prompt_tokens"], 10)

    def test_bad_requests(self):
        base = self._server()
        code, body = self._post(base, "/v1/chat", {})
        self.assertEqual(code, 400)
        code, body = self._post(base, "/v1/chat", {"message": ""})
        self.assertEqual(code, 400)

    def test_agent_error_returns_502(self):
        def broken_factory():
            a = _fake_agent(self.ws)
            def boom(messages, **kw):
                raise RuntimeError("model down")
            a._chat = boom
            return a
        httpd = make_server(AgentConfig.from_dict({"tools": []}, "webby"),
                            self.ws, host="127.0.0.1", port=0,
                            agent_factory=broken_factory)
        host, port = httpd.server_address
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        self.addCleanup(httpd.server_close)
        base = f"http://{host}:{port}"
        code, body = self._post(base, "/v1/chat", {"message": "hi"})
        self.assertEqual(code, 502)
        self.assertIn("model down", body["error"])

    def test_bearer_token_auth(self):
        base = self._server(token="sekrit")
        code, _ = self._get(base, "/health")
        self.assertEqual(code, 401)
        code, _ = self._get(base, "/health",
                            headers={"Authorization": "Bearer wrong"})
        self.assertEqual(code, 401)
        code, body = self._get(base, "/health",
                               headers={"Authorization": "Bearer sekrit"})
        self.assertEqual(code, 200)
        code, body = self._post(base, "/v1/chat", {"message": "hi"},
                                headers={"Authorization": "Bearer sekrit"})
        self.assertEqual(code, 200)


if __name__ == "__main__":
    unittest.main()
