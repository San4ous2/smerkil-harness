"""MCP client over stdio: JSON-RPC 2.0 via subprocess, zero non-stdlib deps.

Connects to MCP servers declared in agent.yaml:

    mcp_servers:
      - name: files
        command: ["npx", "-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
      - name: fetch
        command: ["uvx", "mcp-server-fetch"]

Protocol: initialize → tools/list → tools/call. Each tool becomes a regular
smerkil tool named mcp__<server>__<tool>. Server process is spawned once per
Agent and shut down on close().
"""
from __future__ import annotations

import json
import os
import subprocess
import threading
from typing import Any, Dict, List, Optional

from .config import load_config_text


class MCPError(RuntimeError):
    pass


class MCPServer:
    """One stdio MCP server process."""

    def __init__(self, name: str, command: List[str], timeout: float = 30.0,
                 env: Optional[Dict[str, str]] = None):
        self.name = name
        self.command = [str(c) for c in command]
        self.timeout = timeout
        self._proc: Optional[subprocess.Popen] = None
        self._next_id = 1
        self._lock = threading.Lock()
        self._env = dict(os.environ, **(env or {}))
        self.tools: Dict[str, Dict[str, Any]] = {}

    # ---- low-level JSON-RPC over stdio ----
    def _start(self) -> None:
        if self._proc and self._proc.poll() is None:
            return
        try:
            self._proc = subprocess.Popen(
                self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, env=self._env,
                text=True, bufsize=1)
        except OSError as e:
            raise MCPError(f"cannot spawn {self.command}: {e}") from e

    def _send(self, method: str, params: Optional[Dict[str, Any]] = None,
              notify: bool = False) -> Optional[Dict[str, Any]]:
        self._start()
        assert self._proc and self._proc.stdin and self._proc.stdout
        with self._lock:
            msg: Dict[str, Any] = {"jsonrpc": "2.0", "method": method}
            if params is not None:
                msg["params"] = params
            if not notify:
                msg["id"] = self._next_id
                self._next_id += 1
            try:
                self._proc.stdin.write(json.dumps(msg) + "\n")
                self._proc.stdin.flush()
            except (BrokenPipeError, OSError) as e:
                raise MCPError(f"MCP '{self.name}' pipe broken: {e}") from e
            if notify:
                return None
            deadline = self.timeout
            while True:
                line = self._proc.stdout.readline()
                if not line:
                    raise MCPError(f"MCP '{self.name}' closed stdout (crashed?)")
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if obj.get("id") == msg["id"]:
                    if "error" in obj:
                        err = obj["error"]
                        raise MCPError(f"MCP '{self.name}' {method}: "
                                       f"{err.get('message', err)}")
                    return obj.get("result")

    # ---- MCP lifecycle ----
    def initialize(self) -> Dict[str, Any]:
        res = self._send("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "smerkil", "version": "0.5.0"},
        })
        self._send("notifications/initialized", notify=True)
        return res or {}

    def list_tools(self) -> List[Dict[str, Any]]:
        res = self._send("tools/list", {})
        tools = (res or {}).get("tools") or []
        self.tools = {t.get("name", "?"): t for t in tools if isinstance(t, dict)}
        return tools

    def call(self, tool: str, arguments: Dict[str, Any]) -> str:
        res = self._send("tools/call", {"name": tool, "arguments": arguments})
        if not res:
            return ""
        if res.get("isError"):
            content = res.get("content") or []
            text = " ".join(c.get("text", "") for c in content if isinstance(c, dict))
            raise ValueError(f"MCP tool error: {text or res}")
        parts: List[str] = []
        for c in res.get("content") or []:
            if not isinstance(c, dict):
                continue
            if c.get("type") == "text":
                parts.append(str(c.get("text", "")))
            elif c.get("type") == "resource":
                parts.append(json.dumps(c.get("resource", {}), ensure_ascii=False))
        return "\n".join(parts) or "(empty result)"

    def close(self) -> None:
        if self._proc and self._proc.poll() is None:
            try:
                self._proc.stdin.close()
                self._proc.wait(timeout=5)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
        self._proc = None


class MCPManager:
    """Owns all MCP servers for an Agent; exposes their tools."""

    def __init__(self, servers: List[Dict[str, Any]]):
        self.servers: Dict[str, MCPServer] = {}
        for s in servers:
            name = str(s.get("name", "")).strip()
            cmd = s.get("command") or []
            if not name or not cmd:
                continue
            self.servers[name] = MCPServer(name, cmd,
                                           env=s.get("env") or {},
                                           timeout=float(s.get("timeout", 30)))
        self.connected = False

    def connect(self) -> Dict[str, str]:
        """initialize + tools/list on every server. Returns {server: error}."""
        errors: Dict[str, str] = {}
        for name, srv in self.servers.items():
            try:
                srv.initialize()
                srv.list_tools()
            except Exception as e:
                errors[name] = str(e)
        self.connected = True
        return errors

    def tool_specs(self) -> List[Dict[str, Any]]:
        """smerkil tool specs for all MCP tools, namespaced mcp__<srv>__<tool>."""
        specs: List[Dict[str, Any]] = []
        for sname, srv in self.servers.items():
            for tname, t in srv.tools.items():
                specs.append({
                    "name": f"mcp__{sname}__{tname}",
                    "description": f"[MCP:{sname}] {t.get('description', tname)}",
                    "parameters": t.get("inputSchema") or {"type": "object",
                                                           "properties": {}},
                    "_mcp": (sname, tname),
                })
        return specs

    def call(self, namespaced: str, arguments: Dict[str, Any]) -> str:
        if not namespaced.startswith("mcp__"):
            raise MCPError(f"not an MCP tool: {namespaced}")
        _, rest = namespaced.split("__", 1)
        sname, tname = rest.split("__", 1)
        srv = self.servers.get(sname)
        if not srv:
            raise MCPError(f"no MCP server '{sname}'")
        return srv.call(tname, arguments)

    def close(self) -> None:
        for srv in self.servers.values():
            srv.close()
        self.servers.clear()


def load_mcp_config(workspace) -> List[Dict[str, Any]]:
    """mcp_servers: section from the workspace mcp.json (Claude Code layout)."""
    path = Path(workspace) / "mcp.json" if not isinstance(workspace, Path) else workspace / "mcp.json"
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    return list(data.get("mcp_servers") or [])


# Path import lives here to keep module import cost minimal
from pathlib import Path  # noqa: E402
