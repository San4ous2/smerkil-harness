"""Tests for v0.5: grep/glob/checkpoint tools, hooks, skills, tool-result clearing."""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smerkil.compaction import Compactor, TruncatingSummarizer
from smerkil.hooks import Hook, HookRunner
from smerkil.skills import load_skills, skills_index
from smerkil.tools import (CheckpointTool, GrepTool, GlobTool, Sandbox,
                           build_tools)


class Base(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.sb = Sandbox(self.root, strict=True)

    def tearDown(self):
        self._tmp.cleanup()


class TestGrepGlob(Base):
    def _mk(self, rel, text):
        p = self.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")

    def test_grep_finds_lines(self):
        self._mk("a.py", "def foo():\n    return SECRET_TOKEN\n")
        self._mk("sub/b.py", "x = 'plain'\n")
        g = GrepTool(self.sb)
        out = g.run(pattern="SECRET", glob="*.py")
        self.assertIn("a.py:2", out)
        self.assertNotIn("b.py", out)

    def test_grep_max_results(self):
        self._mk("big.txt", "\n".join(f"match {i}" for i in range(100)))
        g = GrepTool(self.sb)
        out = g.run(pattern="match", max_results=5)
        self.assertIn("truncated at 5", out)

    def test_grep_bad_regex(self):
        g = GrepTool(self.sb)
        with self.assertRaises(Exception):
            g.run(pattern="([unclosed")

    def test_glob(self):
        self._mk("x/a.py", "")
        self._mk("y/b.py", "")
        g = GlobTool(self.sb)
        out = g.run(pattern="**/*.py")
        self.assertIn("x/a.py", out)
        self.assertIn("y/b.py", out)


class TestCheckpoint(Base):
    def test_create_restore_roundtrip(self):
        (self.root / "data.txt").write_text("v1")
        cp = CheckpointTool(self.sb)
        out = cp.run(action="create", name="snap1")
        self.assertIn("created", out)
        (self.root / "data.txt").write_text("v2-broken")
        (self.root / "extra.txt").write_text("junk")
        out = cp.run(action="restore", name="snap1")
        self.assertIn("restored", out)
        self.assertEqual((self.root / "data.txt").read_text(), "v1")
        # restore does not delete files created after the snapshot
        self.assertTrue((self.root / "extra.txt").exists())

    def test_list(self):
        cp = CheckpointTool(self.sb)
        self.assertEqual(cp.run(action="list"), "(no checkpoints)")
        (self.root / "f.txt").write_text("x")
        cp.run(action="create", name="s")
        self.assertIn("s", cp.run(action="list"))

    def test_restore_missing(self):
        cp = CheckpointTool(self.sb)
        with self.assertRaises(Exception):
            cp.run(action="restore", name="ghost")


class TestHooks(Base):
    def test_event_and_tool_filter(self):
        h = Hook(event="tool_before", command="echo hi", tool="run_shell")
        self.assertTrue(h.matches("tool_before", "run_shell"))
        self.assertFalse(h.matches("tool_before", "read_file"))
        self.assertFalse(h.matches("run_end", "run_shell"))

    def test_fire_returns_output(self):
        r = HookRunner([Hook(event="run_start", command="echo hello-hook")])
        self.assertIn("hello-hook", r.fire("run_start") or "")

    def test_gate_blocks(self):
        # exit 1 with BLOCK output → gate blocks
        r = HookRunner([Hook(event="tool_before",
                             command="echo 'BLOCK: no rm allowed'")])
        out = r.fire("tool_before", tool_name="run_shell")
        self.assertTrue(out.strip().upper().startswith("BLOCK"))

    def test_silent_hook_does_not_block(self):
        # exit 0 no output → no block
        r = HookRunner([Hook(event="tool_before", command="true")])
        self.assertIsNone(r.fire("tool_before", tool_name="run_shell"))

    def test_runner_load_from_yaml(self):
        (self.root / "hooks.yaml").write_text(
            "hooks:\n  - event: run_end\n    command: echo done-hook\n",
            encoding="utf-8")
        r = HookRunner.load(self.root)
        self.assertEqual(len(r.hooks), 1)
        self.assertIn("done-hook", r.fire("run_end") or "")


class TestSkills(Base):
    def test_load_and_index(self):
        sdir = self.root / "skills"
        sdir.mkdir()
        (sdir / "deploy.md").write_text(
            "---\ndescription: Deploy the site safely.\n---\n\n1. build\n2. verify\n",
            encoding="utf-8")
        (sdir / "notes.md").write_text("no frontmatter here", encoding="utf-8")
        sk = load_skills(self.root)
        self.assertEqual(sk["deploy"].description, "Deploy the site safely.")
        idx = skills_index(sk)
        self.assertIn("- deploy: Deploy the site safely.", idx)
        self.assertIn("- notes:", idx)

    def test_skill_tool_loads_body(self):
        sdir = self.root / "skills"
        sdir.mkdir()
        (sdir / "deploy.md").write_text("FULL BODY 123", encoding="utf-8")
        from smerkil.skills import SkillTool
        st = SkillTool(load_skills(self.root))
        self.assertEqual(st.run(name="deploy"), "FULL BODY 123")
        with self.assertRaises(Exception):
            st.run(name="ghost")


class TestToolResultClearing(Base):
    def _big(self, n=2000):
        return "y" * n

    def test_clearing_stubs_old_results(self):
        msgs = [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "c1", "type": "function",
                 "function": {"name": "read_file", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": self._big()},
            {"role": "user", "content": "recent q"},
            {"role": "assistant", "content": "recent a"},
            {"role": "user", "content": "recent q2"},
            {"role": "assistant", "content": "recent a2"},
        ]
        c = Compactor(TruncatingSummarizer(), max_tokens=500, keep_recent=4)
        out = c.maybe_compact([dict(m) for m in msgs])
        stubbed = [m for m in out if m.get("role") == "tool"
                   and Compactor.STUB_MARKER in str(m.get("content"))]
        self.assertEqual(len(stubbed), 1)
        # pair still intact: tool_call c1 still has its (stubbed) result
        self.assertTrue(any(m.get("tool_call_id") == "c1" for m in out))

    def test_no_clearing_needed_under_budget(self):
        msgs = [{"role": "system", "content": "s"},
                {"role": "tool", "tool_call_id": "c", "content": "x" * 500}]
        c = Compactor(TruncatingSummarizer(), max_tokens=5000)
        self.assertIs(c.maybe_compact(msgs), msgs)


class TestRegistryIncludesNew(Base):
    def test_build_new_tools(self):
        tools = build_tools(["grep", "glob", "checkpoint"], self.sb, False)
        self.assertIn("grep", tools)
        self.assertIn("glob", tools)
        self.assertIn("checkpoint", tools)


if __name__ == "__main__":
    unittest.main()
