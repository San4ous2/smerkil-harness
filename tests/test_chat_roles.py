"""Tests for chat history (multi-turn context) and `smerkil new --role` presets."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smerkil.agent import Agent


def _cfg():
    cfg = MagicMock()
    cfg.name = "t"
    cfg.model = "m"
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
    return cfg


class TestChatHistory(unittest.TestCase):
    def _agent(self):
        self._tmp = TemporaryDirectory()
        a = Agent(_cfg(), Path(self._tmp.name), stream=False)
        return a

    def test_history_prepended_and_extended(self):
        """Second run() call with history must see prior turns and append new ones."""
        a = self._agent()
        seq = iter([
            {"choices": [{"message": {"role": "assistant", "content": "first answer"},
                          "finish_reason": "stop"}],
             "usage": {}},
            {"choices": [{"message": {"role": "assistant", "content": "second answer"},
                          "finish_reason": "stop"}],
             "usage": {}},
        ])
        seen: list = []
        a._chat = lambda msgs: (seen.append([dict(m) for m in msgs]), next(seq))[1]

        history: list = []
        r1 = a.run("question one", history=history)
        self.assertEqual(r1, "first answer")
        n_after_first = len(history)
        self.assertGreaterEqual(n_after_first, 2)  # user + assistant

        r2 = a.run("question two", history=history)
        self.assertEqual(r2, "second answer")
        # second call's transcript must include the first exchange
        second_msgs = seen[1]
        contents = [m.get("content", "") for m in second_msgs]
        self.assertIn("question one", contents)
        self.assertIn("first answer", contents)
        self.assertIn("question two", contents)
        self.assertEqual(len(history), n_after_first + 2)

    def test_history_not_mutated_by_internal_system_msgs(self):
        """Hook/tool system messages from run 1 must not leak into run 2's start."""
        a = self._agent()
        seq = iter([
            {"choices": [{"message": {"role": "assistant", "content": "ok"},
                          "finish_reason": "stop"}], "usage": {}},
        ] * 3)
        a._chat = lambda msgs: next(seq)
        history: list = []
        a.run("q1", history=history)
        # no dangling tool/system junk in history
        for m in history:
            self.assertIn(m.get("role"), ("user", "assistant"))

    def test_no_history_single_turn_unchanged(self):
        a = self._agent()
        seq = iter([
            {"choices": [{"message": {"role": "assistant", "content": "done"},
                          "finish_reason": "stop"}], "usage": {}},
        ])
        a._chat = lambda msgs: next(seq)
        out = a.run("solo")
        self.assertEqual(out, "done")


class TestRolePresets(unittest.TestCase):
    def test_presets_exist(self):
        from smerkil.cli import ROLE_PRESETS
        for role in ("general", "coder", "researcher", "reviewer", "writer"):
            self.assertIn(role, ROLE_PRESETS)

    def test_coder_scaffold(self):
        import tempfile
        from smerkil.cli import cmd_new
        from smerkil.config import AgentConfig
        with tempfile.TemporaryDirectory() as td:
            import smerkil.cli as cli
            old = cli._agent_dir
            cli._agent_dir = lambda name: Path(td) / "agents" / name
            try:
                args = MagicMock()
                args.name = "bob"
                args.role = "coder"
                rc = cmd_new(args)
                self.assertEqual(rc, 0)
                text = (Path(td) / "agents" / "bob" / "agent.yaml").read_text()
                cfg = AgentConfig.from_dict(
                    cli.load_config_text(text), "bob")
                self.assertIn("run_shell", cfg.tools)
                self.assertIn("grep", cfg.tools)
                self.assertIn("senior software engineer", cfg.system)
            finally:
                cli._agent_dir = old

    def test_unknown_role_rejected(self):
        import tempfile
        from smerkil.cli import cmd_new
        with tempfile.TemporaryDirectory() as td:
            import smerkil.cli as cli
            old = cli._agent_dir
            cli._agent_dir = lambda name: Path(td) / "agents" / name
            try:
                args = MagicMock()
                args.name = "x"
                args.role = "ninja"
                self.assertEqual(cmd_new(args), 1)
            finally:
                cli._agent_dir = old


if __name__ == "__main__":
    unittest.main()
