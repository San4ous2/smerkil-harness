"""Sticky model failover: once a fallback serves, later turns skip the dead
primary; last_good_model is recorded; trace steps carry the serving model."""
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from smerkil.llm import LLMClient, LLMError


def _client():
    return LLMClient("http://127.0.0.1:9", "k", timeout=1)


class TestStickyFailover(unittest.TestCase):
    def test_last_good_model_set_on_success(self):
        c = _client()
        ok = {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
              "usage": {}}
        c._request = lambda payload, on_delta=None, max_attempts=4: ok
        c.chat({"model": "a", "messages": []}, None, model_chain=["a"])
        self.assertEqual(c.last_good_model, "a")

    def test_last_good_not_set_on_failure(self):
        c = _client()
        def boom(payload, on_delta=None, max_attempts=4):
            raise LLMError("dead")
        c._request = boom
        with self.assertRaises(LLMError):
            c.chat({"model": "a", "messages": []}, None, max_attempts=1,
                   model_chain=["a"])
        self.assertEqual(c.last_good_model, "")

    def test_sticky_reorder_skips_dead_primary(self):
        """After fallback 'b' serves once, a later chat() must not re-try 'a'."""
        c = _client()
        tried = []

        def fake_request(payload, on_delta=None, max_attempts=4):
            tried.append(payload["model"])
            if payload["model"] == "a":
                raise LLMError("HTTP 400: bad model")
            return {"choices": [{"message": {"content": "ok"},
                                 "finish_reason": "stop"}], "usage": {}}

        c._request = fake_request
        payload = {"model": "a", "messages": [{"role": "user", "content": "x"}]}
        c.chat(payload, None, max_attempts=1, model_chain=["a", "b"])
        self.assertEqual(c.last_good_model, "b")
        c.chat(payload, None, max_attempts=1, model_chain=["a", "b"])
        self.assertEqual(tried, ["a", "b", "b"])  # third call: sticky → 'b' first

    def test_sticky_respects_primary_recovery(self):
        """If the (sticky) fallback dies, chain still walks to the primary."""
        c = _client()
        state = {"b_down": False}

        def fake_request(payload, on_delta=None, max_attempts=4):
            m = payload["model"]
            if m == "a":  # primary always down in this test
                raise LLMError("HTTP 400: a down")
            if m == "b" and state["b_down"]:
                raise LLMError("HTTP 400: b down")
            return {"choices": [{"message": {"content": "ok"},
                                 "finish_reason": "stop"}], "usage": {}}

        c._request = fake_request
        payload = {"model": "a", "messages": []}
        # turn 1: a down, b serves → sticky b
        c.chat(payload, None, max_attempts=1, model_chain=["a", "b"])
        self.assertEqual(c.last_good_model, "b")
        # turn 2: sticky b tried first, now b is down too → walks back to a? no,
        # a is still down — must raise exhausted, but attempts must cover both
        state["b_down"] = True
        with self.assertRaises(LLMError) as ctx:
            c.chat(payload, None, max_attempts=1, model_chain=["a", "b"])
        self.assertIn("a down", str(ctx.exception))
        # turn 3: primary recovers, sticky b tried first, fails, walks to a
        calls = {"n": 0}
        def fake2(payload, on_delta=None, max_attempts=4):
            m = payload["model"]
            calls["n"] += 1
            if m == "b":
                raise LLMError("HTTP 400: b down")  # b dies after recovery
            return {"choices": [{"message": {"content": "ok"},
                                 "finish_reason": "stop"}], "usage": {}}
        c._request = fake2
        out = c.chat(payload, None, max_attempts=1, model_chain=["a", "b"])
        self.assertEqual(out["choices"][0]["message"]["content"], "ok")
        self.assertEqual(calls["n"], 2)  # sticky b first (fails), then a serves
        self.assertEqual(c.last_good_model, "a")


class TestTraceServingModel(unittest.TestCase):
    def test_step_records_serving_model(self):
        """Trace steps must record resp['model'] (what actually served)."""
        from smerkil.agent import Agent

        cfg = MagicMock()
        cfg.name = "t"
        cfg.model = "primary-m"
        cfg.fallback_models = []
        cfg.temperature = 0.2
        cfg.guardrails.max_turns = 5
        cfg.guardrails.max_tokens = 100000
        cfg.guardrails.max_usd = 1.0
        cfg.guardrails.sandbox_strict = False
        cfg.guardrails.allow_shell = False
        cfg.tools = []
        cfg.system = "sys"
        cfg.base_url = ""
        cfg.api_key_env = "NOPE"

        td = TemporaryDirectory()
        a = Agent(cfg, Path(td.name), stream=False)
        a._td = td  # keep alive
        resp = {"model": "fallback-m",
                "choices": [{"message": {"role": "assistant", "content": "hi"},
                             "finish_reason": "stop"}], "usage": {}}
        a._chat = lambda msgs: resp
        a.run("q")
        self.assertEqual(a.trace["steps"][0]["model"], "fallback-m")


if __name__ == "__main__":
    unittest.main()
