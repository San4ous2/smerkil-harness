"""Tests for todos and MCP client."""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smerkil.todos import TodoList, TodoTool, TodoError
from smerkil.mcp import MCPManager, MCPServer, MCPError, load_mcp_config
from smerkil.tools import SubagentTool


class TestTodos(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.tl = TodoList(Path(self._tmp.name) / ".smerkil" / "todos.json")

    def tearDown(self):
        self._tmp.cleanup()

    def test_replace_and_render(self):
        out = self.tl.replace([
            {"content": "search code", "status": "done"},
            {"content": "fix bug", "status": "in_progress"},
            {"content": "run tests", "status": "pending"},
        ])
        self.assertIn("[x] search code", out)
        self.assertIn("[~] fix bug", out)
        self.assertIn("[ ] run tests", out)
        # persisted
        tl2 = TodoList(self.tl.path)
        tl2.load()
        self.assertEqual(len(tl2.items), 3)

    def test_two_in_progress_rejected(self):
        with self.assertRaises(TodoError):
            self.tl.replace([
                {"content": "a", "status": "in_progress"},
                {"content": "b", "status": "in_progress"},
            ])

    def test_done_after_in_progress_rejected(self):
        with self.assertRaises(TodoError):
            self.tl.replace([
                {"content": "a", "status": "in_progress"},
                {"content": "b", "status": "done"},
            ])

    def test_bad_status_rejected(self):
        with self.assertRaises(TodoError):
            self.tl.replace([{"content": "a", "status": "zoom"}])

    def test_empty_list_rejected(self):
        with self.assertRaises(TodoError):
            self.tl.replace([])

    def test_tool_roundtrip(self):
        t = TodoTool(self.tl)
        t.run(todos=[{"content": "x", "status": "pending"}])
        self.assertEqual(self.tl.items[0]["content"], "x")

    def test_prompt_block_empty(self):
        self.assertEqual(self.tl.prompt_block(), "")


class TestMCP(unittest.TestCase):
    def setUp(self):
        # a trivial but REAL MCP server over stdio, in-process via python
        self._tmp = TemporaryDirectory()
        self.server_py = Path(self._tmp.name) / "mini_mcp.py"
        self.server_py.write_text(
            "import json,sys\n"
            "for line in sys.stdin:\n"
            "    req = json.loads(line)\n"
            "    m = req.get('method','')\n"
            "    if m == 'initialize':\n"
            "        res = {'protocolVersion':'2024-11-05','capabilities':{},'serverInfo':{'name':'mini','version':'1'}}\n"
            "    elif m == 'tools/list':\n"
            "        res = {'tools':[{'name':'echo','description':'Echo text back','inputSchema':{'type':'object','properties':{'text':{'type':'string'}},'required':['text']}}]}\n"
            "    elif m == 'tools/call':\n"
            "        t = req['params']['arguments']['text']\n"
            "        res = {'content':[{'type':'text','text':'echo:'+t}]}\n"
            "    elif m.startswith('notifications/'):\n"
            "        continue\n"
            "    else:\n"
            "        res = {}\n"
            "    if req.get('id') is not None:\n"
            "        sys.stdout.write(json.dumps({'jsonrpc':'2.0','id':req['id'],'result':res})+'\\n'); sys.stdout.flush()\n",
            encoding="utf-8")
        self.mgr = MCPManager([{"name": "mini",
                                "command": [sys.executable, str(self.server_py)]}])

    def tearDown(self):
        self.mgr.close()
        self._tmp.cleanup()

    def test_full_roundtrip(self):
        errs = self.mgr.connect()
        self.assertEqual(errs, {})
        specs = self.mgr.tool_specs()
        self.assertEqual(len(specs), 1)
        self.assertEqual(specs[0]["name"], "mcp__mini__echo")
        out = self.mgr.call("mcp__mini__echo", {"text": "hello"})
        self.assertEqual(out, "echo:hello")

    def test_broken_server_reported(self):
        bad = MCPManager([{"name": "ghost",
                           "command": [sys.executable, "-c", "import sys; sys.exit(1)"]}])
        errs = bad.connect()
        self.assertIn("ghost", errs)
        bad.close()

    def test_load_mcp_config(self):
        ws = Path(self._tmp.name)
        (ws / "mcp.json").write_text(json.dumps(
            {"mcp_servers": [{"name": "mini",
                              "command": [sys.executable, str(self.server_py)]}]}))
        cfg = load_mcp_config(ws)
        self.assertEqual(len(cfg), 1)
        self.assertEqual(cfg[0]["name"], "mini")


class TestSubagentFanout(unittest.TestCase):
    def _factory_and_parent(self):
        from smerkil.tools import ReadFile, ListDir
        from smerkil.tools import Sandbox
        from tempfile import TemporaryDirectory
        self._tmp = TemporaryDirectory()
        self.sb = Sandbox(Path(self._tmp.name), strict=True)
        self.addCleanup(self._tmp.cleanup)
        calls = []
        def factory(tools=None, max_turns=12):
            calls.append((tuple(tools or []), max_turns))
            m = MagicMock()
            m.run.return_value = f"done-{len(calls)}"
            m.trace = {"stop_reason": "completed", "steps": [1], "tool_calls": [1]}
            return m
        parent = {"read_file": ReadFile(self.sb), "list_dir": ListDir(self.sb)}
        return factory, parent, calls

    def test_single_goal_backward_compat(self):
        factory, parent, calls = self._factory_and_parent()
        sa = SubagentTool(factory, parent)
        out = json.loads(sa.run(goal="do x"))
        self.assertEqual(out["result"], "done-1")
        self.assertEqual(out["stop_reason"], "completed")

    def test_parallel_fanout(self):
        import time
        factory, parent, calls = self._factory_and_parent()
        def slow_factory(tools=None, max_turns=12):
            m = MagicMock()
            m.trace = {"stop_reason": "completed", "steps": [1], "tool_calls": []}
            def slow(goal):
                time.sleep(1.0)
                return f"done:{goal}"
            m.run = slow
            return m
        sa = SubagentTool(slow_factory, parent)
        t0 = time.time()
        out = json.loads(sa.run(goals=["a", "b", "c", "d"]))
        dt = time.time() - t0
        self.assertEqual(out["count"], 4)
        self.assertEqual([r["result"] for r in out["results"]],
                         ["done:a", "done:b", "done:c", "done:d"])
        # 4x1s sleeps ran in parallel → well under 4s
        self.assertLess(dt, 3.0, f"fan-out not parallel: {dt:.1f}s")

    def test_fanout_order_preserved(self):
        factory, parent, calls = self._factory_and_parent()
        def var_factory(tools=None, max_turns=12):
            m = MagicMock()
            m.trace = {"stop_reason": "completed", "steps": [], "tool_calls": []}
            m.run.return_value = "ok"
            return m
        sa = SubagentTool(var_factory, parent)
        out = json.loads(sa.run(goals=["g1", "g2", "g3"]))
        self.assertEqual([r["index"] for r in out["results"]], [0, 1, 2])
        self.assertEqual([r["goal"] for r in out["results"]], ["g1", "g2", "g3"])

    def test_too_many_goals_rejected(self):
        factory, parent, calls = self._factory_and_parent()
        sa = SubagentTool(factory, parent)
        out = json.loads(sa.run(goals=[f"g{i}" for i in range(9)]))
        self.assertIn("error", out)

    def test_child_tools_filtered(self):
        factory, parent, calls = self._factory_and_parent()
        sa = SubagentTool(factory, parent)
        sa.run(goal="x", tools=["read_file", "run_shell"])  # run_shell not in parent
        self.assertEqual(calls[0][0], ("read_file",))


if __name__ == "__main__":
    unittest.main()
