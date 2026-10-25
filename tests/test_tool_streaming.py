"""Tool-call streaming: on_tool fires incrementally with (name, args_fragment)
as the model writes arguments; assembly is unchanged."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from smerkil.llm import LLMClient


def _sse(chunks: bytes) -> "FakeResp":
    class FakeResp:
        headers = {"Content-Type": "text/event-stream"}
        def __init__(self, data):
            self._data = data
            self._pos = 0
        def read(self, n):
            piece = self._data[self._pos:self._pos + n]
            self._pos += n
            return piece
    return FakeResp(chunks)


SSE_BODY = b"""data: {"choices":[{"delta":{"role":"assistant"}}]}

data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","function":{"name":"write_file","arguments":""}}]}}]}

data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\"path\\":"}}]}}]}

data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\\"a.txt\\",\\"content\\":\\"hi\\"}"}}]}}]}

data: {"choices":[{"delta":{"content":"done"},"finish_reason":null}]}

data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}

data: [DONE]

"""


class TestToolStreaming(unittest.TestCase):
    def _client(self):
        c = LLMClient("http://127.0.0.1:9", "k", timeout=1)
        fake_conn = type("C", (), {"request": lambda *a, **k: None,
                                   "close": lambda self: None})()
        c._conn = lambda url: fake_conn
        return c

    def _wire(self, c):
        def fake_stream(url, body, headers, on_delta, on_tool=None):
            return c._read_sse(_sse(SSE_BODY), on_delta, on_tool)
        def fake_plain(url, body, headers):
            return c._read_sse(_sse(SSE_BODY), None, None)
        c._request_stream = fake_stream
        c._request_plain = fake_plain

    def test_on_tool_fires_per_fragment(self):
        c = self._client()
        self._wire(c)
        events = []
        out = c.chat({"model": "m", "messages": []}, None,
                     on_tool=lambda name, frag: events.append((name, frag)),
                     model_chain=["m"])
        self.assertEqual(events[0][0], "write_file")
        # fragments concatenate to the full argument string
        joined = "".join(f for _, f in events)
        self.assertEqual(joined, out["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"])
        # two argument fragments (name-only chunk emits nothing)
        self.assertEqual(len(events), 2)

    def test_assembly_unchanged(self):
        c = self._client()
        self._wire(c)
        out = c.chat({"model": "m", "messages": []}, None, model_chain=["m"])
        msg = out["choices"][0]["message"]
        self.assertEqual(msg["tool_calls"][0]["function"]["name"], "write_file")
        self.assertEqual(msg["content"], "done")
        self.assertEqual(out["choices"][0]["finish_reason"], "tool_calls")

    def test_on_tool_never_breaks_stream(self):
        c = self._client()
        self._wire(c)
        def bad(name, frag):
            raise RuntimeError("callback bug")
        out = c.chat({"model": "m", "messages": []}, None,
                     on_tool=bad, model_chain=["m"])
        self.assertEqual(out["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
                         "write_file")

    def test_agent_forwards_on_tool_delta(self):
        from smerkil.agent import Agent
        from unittest.mock import MagicMock
        from tempfile import TemporaryDirectory
        cfg = MagicMock()
        cfg.name = "t"; cfg.model = "m"; cfg.fallback_models = []
        cfg.temperature = 0.2
        cfg.guardrails.max_turns = 2; cfg.guardrails.max_tokens = 100000
        cfg.guardrails.max_usd = 1.0; cfg.guardrails.sandbox_strict = False
        cfg.guardrails.allow_shell = False
        cfg.tools = []; cfg.system = "s"; cfg.base_url = ""; cfg.api_key_env = "X"
        td = TemporaryDirectory()
        seen = []
        a = Agent(cfg, Path(td.name), on_token=lambda s: None,
                  on_tool_delta=lambda n, f: seen.append((n, f)))
        a._td = td
        # stream mode is on → chat must receive on_tool (spy on client)
        calls = {}
        def spy_chat(payload, on_delta=None, on_tool=None, **kw):
            calls["on_tool"] = on_tool
            return {"model": "m",
                    "choices": [{"message": {"role": "assistant", "content": "ok"},
                                 "finish_reason": "stop"}], "usage": {}}
        a.client.chat = spy_chat
        a._chat([{"role": "user", "content": "q"}])
        self.assertTrue(calls.get("on_tool") is not None)


if __name__ == "__main__":
    unittest.main()
