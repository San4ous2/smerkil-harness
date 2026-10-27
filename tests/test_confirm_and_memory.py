"""Tests for human-in-the-loop confirm gate + memory tools wiring in Agent."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from smerkil.config import AgentConfig, Guardrails
from smerkil.agent import Agent
from smerkil.confirm import ConfirmGate, ConfirmationPolicy
from smerkil.memory import MemoryStore
from smerkil.memory_tools import MemorySave, MemorySearch, MemoryForget
from smerkil.tools import Sandbox, ToolError


class PolicyTest(unittest.TestCase):
    def test_tool_list_match(self):
        p = ConfirmationPolicy(tools=["run_shell"])
        self.assertTrue(p.requires("run_shell", "{}"))
        self.assertFalse(p.requires("read_file", "{}"))

    def test_pattern_match(self):
        p = ConfirmationPolicy(patterns=[{"tool": "git_commit", "match": "push|force"}])
        self.assertTrue(p.requires("git_commit", '{"message": "force push"}'))
        self.assertFalse(p.requires("git_commit", '{"message": "docs"}'))
        self.assertFalse(p.requires("read_file", "push"))

    def test_wildcard_pattern(self):
        p = ConfirmationPolicy(patterns=[{"tool": "*", "match": "danger"}])
        self.assertTrue(p.requires("write_file", '{"content": "danger zone"}'))

    def test_ask_once_remember_and_reset(self):
        p = ConfirmationPolicy(tools=["run_shell"], ask_once_per_run=True)
        p.remember("run_shell", '{"cmd": "ls"}')
        self.assertTrue(p.already_approved("run_shell", '{"cmd": "ls"}'))
        self.assertFalse(p.already_approved("run_shell", '{"cmd": "rm"}'))
        p.reset_run()
        self.assertFalse(p.already_approved("run_shell", '{"cmd": "ls"}'))


class GateTest(unittest.TestCase):
    def test_approved_call_passes(self):
        g = ConfirmGate(ConfirmationPolicy(tools=["run_shell"]),
                        asker=lambda t, a: True)
        self.assertIsNone(g.check("run_shell", "{}"))

    def test_denied_call_returns_reason(self):
        g = ConfirmGate(ConfirmationPolicy(tools=["run_shell"]),
                        asker=lambda t, a: False)
        out = g.check("run_shell", "{}")
        self.assertIn("denied by human", out)

    def test_no_asker_denies(self):
        # explicit asker=None AND the default terminal asker under a non-TTY
        # must both deny (fail closed)
        g = ConfirmGate(ConfirmationPolicy(tools=["run_shell"]), asker=None)
        out = g.check("run_shell", "{}")
        self.assertIn("denied", out)
        self.assertIn("no human is attached", out)
        g2 = ConfirmGate(ConfirmationPolicy(tools=["run_shell"]))  # default asker
        out2 = g2.check("run_shell", "{}")
        self.assertIn("denied", out2)

    def test_auto_approve_bypasses(self):
        g = ConfirmGate(ConfirmationPolicy(tools=["run_shell"]),
                        asker=lambda t, a: (_ for _ in ()).throw(AssertionError("must not ask")),
                        auto_approve=True)
        self.assertIsNone(g.check("run_shell", "{}"))

    def test_unflagged_tool_not_asked(self):
        g = ConfirmGate(ConfirmationPolicy(tools=["run_shell"]),
                        asker=lambda t, a: (_ for _ in ()).throw(AssertionError("must not ask")))
        self.assertIsNone(g.check("read_file", "{}"))

    def test_asker_exception_denies(self):
        def boom(t, a):
            raise KeyboardInterrupt
        import smerkil.confirm as confirm_mod
        g = ConfirmGate(ConfirmationPolicy(tools=["run_shell"]), asker=boom)
        # KeyboardInterrupt inside the asker must deny, not crash the run
        try:
            out = g.check("run_shell", "{}")
            self.assertIsNotNone(out)
        except KeyboardInterrupt:
            self.fail("KeyboardInterrupt leaked from gate")

    def test_repeat_identical_call_not_reattended(self):
        calls = []
        def asker(t, a):
            calls.append(a)
            return True
        g = ConfirmGate(ConfirmationPolicy(tools=["run_shell"]), asker=asker)
        self.assertIsNone(g.check("run_shell", '{"cmd":"make test"}'))
        self.assertIsNone(g.check("run_shell", '{"cmd":"make test"}'))
        self.assertEqual(len(calls), 1)


def _agent(tmp: Path, **kw) -> Agent:
    import os
    os.environ.setdefault("SMERKIL_TEST_KEY", "test-key-not-real")
    depth = kw.pop("depth", 0)
    cfg = AgentConfig(
        name="t", model="m", system="s",
        tools=["read_file"],
        guardrails=Guardrails(max_turns=2),
        base_url="http://localhost:1", api_key_env="SMERKIL_TEST_KEY",
        **kw)
    return Agent(cfg, tmp / "workspace", depth=depth)


class AgentWiringTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def test_memory_tools_auto_added(self):
        a = _agent(self.tmp)
        for t in ("memory_save", "memory_search", "memory_forget"):
            self.assertIn(t, a.tools)

    def test_memory_saved_via_tool_then_injected(self):
        a = _agent(self.tmp)
        out = a.tools["memory_save"].run(text="build with gradle", tags=["build"])
        self.assertIn("saved", out)
        # new agent over same workspace injects the note into system prompt
        a2 = _agent(self.tmp)
        captured = {}
        def fake_chat(messages):
            captured["messages"] = messages
            return {"choices": [{"message": {"content": "done", "role": "assistant"},
                                 "finish_reason": "stop"}],
                    "usage": {}}
        a2._chat = fake_chat
        a2.run("hello")
        sys_chunks = [m["content"] for m in captured["messages"] if m["role"] == "system"]
        self.assertTrue(any("build with gradle" in c for c in sys_chunks))

    def test_memory_search_tool(self):
        a = _agent(self.tmp)
        a.tools["memory_save"].run(text="deploy checklist item", tags=["deploy"])
        out = a.tools["memory_search"].run(query="deploy")
        self.assertIn("deploy checklist item", out)

    def test_memory_forget_tool(self):
        a = _agent(self.tmp)
        a.tools["memory_save"].run(text="stale note")
        out = a.tools["memory_forget"].run(query="stale")
        self.assertIn("forgot 1", out)

    def test_confirm_gate_blocks_tool(self):
        ws = self.tmp / "workspace"
        ws.mkdir(parents=True, exist_ok=True)
        (ws / "confirm.yaml").write_text(
            "confirm:\n  tools: [read_file]\n", encoding="utf-8")
        a = _agent(self.tmp)
        self.assertIsNotNone(a.confirm)
        # simulate a denied call through the gate directly
        denial = a.confirm.check("read_file", '{"path": "x"}')
        self.assertIn("denied by human", denial)
        a2 = _agent(self.tmp)  # no confirm.yaml in fresh tmp? same tmp -> present
        # subagent: depth 1 → no gate
        a3 = _agent(self.tmp, depth=1)
        self.assertIsNone(a3.confirm)

    def test_no_confirm_yaml_no_gate(self):
        tmp = Path(tempfile.mkdtemp())
        a = _agent(tmp)
        self.assertIsNone(a.confirm)

    def test_env_confirm(self):
        with mock.patch.dict("os.environ", {"SMERKIL_CONFIRM": "1"}):
            a = _agent(Path(tempfile.mkdtemp()))
        self.assertIsNotNone(a.confirm)


if __name__ == "__main__":
    unittest.main()
