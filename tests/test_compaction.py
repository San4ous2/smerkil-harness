"""Unit tests: compaction unit-atomicity, shell sandbox, model fallback chain.
Run: python3 -m unittest discover tests -v
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smerkil.compaction import Compactor, ModelSummarizer, TruncatingSummarizer, estimate_tokens
from smerkil.tools import RunShell, Sandbox


def _assistant_toolcall(call_id: str = "call_1", name: str = "read_file") -> dict:
    return {"role": "assistant", "content": "", "tool_calls": [
        {"id": call_id, "type": "function",
         "function": {"name": name, "arguments": "{}"}}]}


class TestCompactionUnits(unittest.TestCase):
    def test_tool_pair_stays_together(self):
        msgs = [
            {"role": "system", "content": "sys"},
            _assistant_toolcall("c1"),
            {"role": "tool", "tool_call_id": "c1", "content": "result"},
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"},
        ]
        units = Compactor._units(msgs[1:])
        # assistant+tool = one unit; user, assistant = own units
        self.assertEqual(len(units), 3)
        self.assertEqual(len(units[0]), 2)

    def test_compaction_never_splits_pair(self):
        """Even with keep_recent=1, tool result must stay with its tool_call."""
        big = "x" * 4000
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": big},
            {"role": "assistant", "content": big},
            _assistant_toolcall("c1"),
            {"role": "tool", "tool_call_id": "c1", "content": "r"},
            {"role": "user", "content": "final question"},
            {"role": "assistant", "content": "final answer"},
        ]
        c = Compactor(TruncatingSummarizer(), max_tokens=100, keep_recent=1)
        out = c.maybe_compact([dict(m) for m in msgs])
        # any assistant tool_calls in output must have their tool result after it
        for i, m in enumerate(out):
            if m.get("role") == "assistant" and m.get("tool_calls"):
                ids = {tc["id"] for tc in m["tool_calls"]}
                following = [o for o in out[i + 1:] if o.get("role") == "tool"]
                self.assertTrue(following, f"tool_calls {ids} dangling without tool result")
                out_ids = {o.get("tool_call_id") for o in following}
                self.assertTrue(ids <= out_ids,
                                f"tool results missing for {ids - out_ids}")

    def test_system_prompt_preserved(self):
        msgs = [
            {"role": "system", "content": "ORIGINAL SYSTEM"},
            {"role": "user", "content": "x" * 9000},
            {"role": "assistant", "content": "y" * 9000},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "there"},
        ]
        c = Compactor(TruncatingSummarizer(), max_tokens=100, keep_recent=4)
        out = c.maybe_compact(msgs)
        self.assertEqual(out[0]["role"], "system")
        self.assertEqual(out[0]["content"], "ORIGINAL SYSTEM")

    def test_no_compaction_under_budget(self):
        msgs = [{"role": "user", "content": "hello"}]
        c = Compactor(TruncatingSummarizer(), max_tokens=1000)
        self.assertIs(c.maybe_compact(msgs), msgs)

    def test_summary_cap(self):
        class BigSum:
            def summarize(self, messages):
                return "z" * 100_000
        msgs = [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "x" * 40_000},
            {"role": "user", "content": "recent"},
            {"role": "assistant", "content": "ok"},
        ]
        c = Compactor(BigSum(), max_tokens=1000, keep_recent=2)
        out = c.maybe_compact(msgs)
        joined = json.dumps(out)
        # summary capped at max_tokens*2 chars = 2000
        self.assertLessEqual(joined.count("z"), 2100)


class TestModelSummarizer(unittest.TestCase):
    def test_empty_content_falls_back(self):
        client = MagicMock()
        client.chat.return_value = {"choices": [{"message": {"content": ""}}]}
        ms = ModelSummarizer(client, "m")
        msgs = [{"role": "user", "content": "hello world"}]
        out = ms.summarize(msgs)
        self.assertIn("user: hello world", out)  # truncating fallback output

    def test_exception_falls_back(self):
        client = MagicMock()
        client.chat.side_effect = RuntimeError("boom")
        ms = ModelSummarizer(client, "m")
        out = ms.summarize([{"role": "user", "content": "abc"}])
        self.assertIn("abc", out)

    def test_model_chain_passed(self):
        client = MagicMock()
        client.chat.return_value = {"choices": [{"message": {"content": "sum"}}]}
        ms = ModelSummarizer(client, "primary", model_chain=["primary", "backup"])
        ms.summarize([{"role": "user", "content": "abc"}])
        _, kwargs = client.chat.call_args
        self.assertEqual(kwargs.get("model_chain"), ["primary", "backup"])

    def test_transcript_cap(self):
        client = MagicMock()
        client.chat.return_value = {"choices": [{"message": {"content": "s"}}]}
        ms = ModelSummarizer(client, "m", max_transcript_chars=100)
        ms.summarize([{"role": "user", "content": "a" * 500}])
        sent = client.chat.call_args[0][0]["messages"][1]["content"]
        self.assertLessEqual(len(sent), 100)


class TestShellSandbox(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.shell = RunShell(Sandbox(Path(self.tmp.name), strict=True), allow_shell=True)

    def tearDown(self):
        self.tmp.cleanup()

    def test_injection_blocked(self):
        for cmd in ["ls; cat /etc/passwd", "cat x && rm -rf /", "ls $(pwd)",
                    "cat `id`", "ls | grep x", "echo hi > /tmp/pwn"]:
            with self.assertRaises(Exception, msg=cmd):
                self.shell.run(command=cmd)

    def test_whitelist_blocked(self):
        with self.assertRaises(Exception):
            self.shell.run(command="curl http://evil.com")

    def test_clean_command_ok(self):
        out = self.shell.run(command="pwd")
        self.assertIn(str(self.tmp.name), out)


class TestFallbackChain(unittest.TestCase):
    def test_chain_exhaustion_raises_special_error(self):
        from smerkil.llm import LLMClient, ModelFallbackExhausted
        c = LLMClient("http://127.0.0.1:9", "k", timeout=1)
        payload = {"model": "a", "messages": [{"role": "user", "content": "x"}]}
        with self.assertRaises(ModelFallbackExhausted) as cm:
            c.chat(payload, None, max_attempts=1, model_chain=["a", "b"])
        self.assertIn("a:", str(cm.exception))
        self.assertIn("b:", str(cm.exception))

    def test_chain_skips_dead_model(self):
        """First model raises definitive error → second is tried."""
        from smerkil.llm import LLMClient, LLMError
        c = LLMClient("http://127.0.0.1:9", "k", timeout=1)
        calls = []
        real = c._request

        def fake_request(payload, on_delta, max_attempts):
            calls.append(payload["model"])
            if payload["model"] == "a":
                raise LLMError("HTTP 400: bad model")
            return {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1}}

        c._request = fake_request
        out = c.chat({"model": "a", "messages": []}, None, max_attempts=1,
                     model_chain=["a", "b"])
        self.assertEqual(out["choices"][0]["message"]["content"], "ok")
        self.assertEqual(calls, ["a", "b"])


class TestLengthFinish(unittest.TestCase):
    """finish_reason=length: truncated output must not be treated as final."""

    def _agent(self, responses):
        from smerkil.agent import Agent
        cfg = MagicMock()
        cfg.name = "t"
        cfg.model = "m"
        cfg.fallback_models = []
        cfg.temperature = 0.2
        cfg.guardrails.max_turns = 10
        cfg.guardrails.max_tokens = 100000
        cfg.guardrails.max_usd = 1.0
        cfg.guardrails.sandbox_strict = False
        cfg.guardrails.allow_shell = False
        cfg.tools = []
        cfg.system = "sys"
        cfg.base_url = ""
        cfg.api_key_env = "NOPE"
        self._tmp = TemporaryDirectory()
        a = Agent(cfg, Path(self._tmp.name), stream=False)
        seq = iter(responses)
        a._chat = lambda msgs: next(seq)
        return a

    def tearDown(self):
        if getattr(self, "_tmp", None):
            self._tmp.cleanup()

    def test_length_gets_nudge_then_continues(self):
        resp = {"choices": [{"message": {"role": "assistant", "content": "half a"},
                             "finish_reason": "length"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1}}
        done = {"choices": [{"message": {"role": "assistant", "content": "full answer"},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1}}
        a = self._agent([dict(resp), dict(done)])
        out = a.run("goal")
        self.assertEqual(a.trace["stop_reason"], "completed")
        self.assertEqual(out, "full answer")
        # a nudge was injected: check trace kept the length step
        self.assertEqual(a.trace["steps"][0]["finish_reason"], "length")

    def test_three_lengths_bail_out(self):
        resp = {"choices": [{"message": {"role": "assistant", "content": "cut"},
                             "finish_reason": "length"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1}}
        a = self._agent([dict(resp) for _ in range(3)])
        out = a.run("goal")
        self.assertEqual(a.trace["stop_reason"], "max_output_tokens")


class TestEvalsNewRules(unittest.TestCase):
    def test_contains_none(self):
        from smerkil.evals import check_expectations
        v = check_expectations("all good here", {"expect": {"contains_none": ["error", "fail"]}})
        self.assertTrue(v["passed"])
        v2 = check_expectations("it failed badly", {"expect": {"contains_none": ["fail"]}})
        self.assertFalse(v2["passed"])

    def test_regex_match(self):
        from smerkil.evals import check_expectations
        v = check_expectations("Found 42 items", {"expect": {"regex_match": r"Found \d+ items"}})
        self.assertTrue(v["passed"])
        v2 = check_expectations("nothing", {"expect": {"regex_match": r"Found \d+"}})
        self.assertFalse(v2["passed"])

    def test_tool_used(self):
        from smerkil.evals import check_expectations
        agent = MagicMock()
        agent.trace = {"tool_calls": [{"name": "grep"}, {"name": "todo_write"}]}
        v = check_expectations("done", {"expect": {"tool_used": ["grep"]}}, agent=agent)
        self.assertTrue(v["passed"])
        v2 = check_expectations("done", {"expect": {"tool_used": ["write_file"]}}, agent=agent)
        self.assertFalse(v2["passed"])
        self.assertEqual(v2["checks"][0]["missing"], ["write_file"])

    def test_combined_rules(self):
        from smerkil.evals import check_expectations
        agent = MagicMock()
        agent.trace = {"tool_calls": [{"name": "grep"}]}
        v = check_expectations(
            "result has TOKEN in src/util.py",
            {"expect": {"contains_all": ["token"], "regex_match": r"src/\w+\.py",
                        "tool_used": ["grep"], "min_chars": 10}},
            agent=agent)
        self.assertTrue(v["passed"])


if __name__ == "__main__":
    unittest.main()
