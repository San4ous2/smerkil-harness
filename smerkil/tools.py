"""Sandboxed, stdlib-only tool set. File tools are rooted to the agent dir;
shell is opt-in via guardrails.allow_shell. Includes web fetch, python exec,
read-only git tools, plan/todo tools and subagent spawning."""
from __future__ import annotations

import json
import subprocess
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .planning import Plan, STATUSES


class ToolError(Exception):
    pass


class Sandbox:
    """Roots all path operations to an agent directory. strict=True forbids
    symlinks escaping the root and absolute paths outside it."""

    def __init__(self, root: Path, strict: bool = True):
        self.root = root.resolve()
        self.strict = strict

    def resolve(self, path: str) -> Path:
        p = Path(path)
        if not p.is_absolute():
            p = self.root / p
        p = p.resolve()
        if self.strict:
            if not str(p).startswith(str(self.root)):
                raise ToolError(f"path '{path}' escapes sandbox root {self.root}")
            if p.is_symlink():
                raise ToolError(f"symlink '{path}' is not allowed in strict sandbox")
        return p


class Tool:
    name: str = "tool"
    description: str = ""
    params: Dict[str, Any] = {}

    def __init__(self, sandbox: Optional[Sandbox] = None, allow_shell: bool = False):
        self.sandbox = sandbox
        self.allow_shell = allow_shell

    def schema(self) -> Dict[str, Any]:
        return {"name": self.name, "description": self.description}

    def run(self, **kwargs: Any) -> str:
        raise NotImplementedError


# ---------------- file tools (unchanged behavior) ----------------
class ReadFile(Tool):
    name = "read_file"
    description = "Read a text file inside the agent workspace. args: {path}"
    params = {"type": "object", "properties": {"path": {"type": "string", "description": "relative or sandbox-rooted path"}}, "required": ["path"]}

    def run(self, path: str = "", **_: Any) -> str:
        p = self.sandbox.resolve(path)
        if not p.exists():
            raise ToolError(f"no such file: {path}")
        text = p.read_text(encoding="utf-8", errors="replace")
        return text[:20_000]


class WriteFile(Tool):
    name = "write_file"
    description = "Write text to a file inside the agent workspace. args: {path, content}"
    params = {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}

    def run(self, path: str = "", content: str = "", **_: Any) -> str:
        p = self.sandbox.resolve(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"wrote {len(content)} chars to {p.name}"


class ListDir(Tool):
    name = "list_dir"
    description = "List files in a directory inside the workspace. args: {path}"
    params = {"type": "object", "properties": {"path": {"type": "string", "description": "defaults to workspace root"}}, "required": []}

    def run(self, path: str = ".", **_: Any) -> str:
        p = self.sandbox.resolve(path)
        if not p.is_dir():
            raise ToolError(f"not a directory: {path}")
        entries = sorted(e.name + ("/" if e.is_dir() else "") for e in p.iterdir())
        return "\n".join(entries[:200]) or "(empty)"


class RunShell(Tool):
    name = "run_shell"
    description = ("Run a whitelisted shell command inside the workspace, "
                   "streaming output line by line. args: {command}")
    params = {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}

    # metacharacters that enable command chaining / injection beyond the
    # whitelist; a whitelisted command must be a single simple command
    _FORBIDDEN = set(";|&`$><\n\\'\"(){}[]*?~!#")

    # optional live-output callback: fn(line) — set by the CLI/chat runner
    on_line = None

    def run(self, command: str = "", timeout: int = 60, **_: Any) -> str:
        if not self.allow_shell:
            raise ToolError("shell is disabled by guardrails (allow_shell: false)")
        allowed = ("ls", "cat", "head", "tail", "wc", "grep", "find", "pwd", "echo", "date")
        stripped = command.strip()
        first = stripped.split()[0] if stripped else ""
        if first not in allowed:
            raise ToolError(f"command '{first}' not in sandbox whitelist {allowed}")
        bad = self._FORBIDDEN & set(stripped)
        if bad:
            raise ToolError(
                f"forbidden characters in command: {' '.join(sorted(bad))} "
                f"(single simple whitelisted commands only)")
        proc = subprocess.Popen(stripped.split(), shell=False, cwd=str(self.sandbox.root),
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1)
        lines: list = []
        import time as _time
        deadline = _time.monotonic() + max(5, min(int(timeout), 300))
        assert proc.stdout is not None
        while True:
            line = proc.stdout.readline()
            if line == "" and proc.poll() is not None:
                break
            if line:
                lines.append(line.rstrip("\n"))
                if self.on_line:
                    try:
                        self.on_line(line.rstrip("\n"))
                    except Exception:
                        pass
            if _time.monotonic() > deadline:
                proc.kill()
                lines.append(f"[timeout after {timeout}s — process killed]")
                break
        out = "\n".join(lines).strip()
        if len(out) > 10_000:
            # keep head AND tail — the tail carries status like "[timeout …]"
            out = out[:7000] + "\n… [output truncated, middle omitted] …\n" + out[-2900:]
        return out if out else "(no output)"


class MemoryNote(Tool):
    name = "memory_note"
    description = "Persist a durable note for this agent (survives runs). args: {note}"
    params = {"type": "object", "properties": {"note": {"type": "string"}}, "required": ["note"]}

    def run(self, note: str = "", **_: Any) -> str:
        path = self.sandbox.root / "memory.md"
        with path.open("a", encoding="utf-8") as f:
            f.write(f"- {note}\n")
        return "noted"


# ---------------- new: web_fetch ----------------
class WebFetch(Tool):
    name = "web_fetch"
    description = ("Fetch a URL over HTTP(S) (stdlib), extract readable text, "
                   "size/time limited. args: {url, max_chars}")
    params = {"type": "object",
              "properties": {"url": {"type": "string"},
                             "max_chars": {"type": "integer", "description": "default 8000"}},
              "required": ["url"]}

    MAX_BYTES = 512 * 1024
    TIMEOUT = 20

    def run(self, url: str = "", max_chars: int = 8000, **_: Any) -> str:
        if not url.startswith(("http://", "https://")):
            raise ToolError("url must start with http:// or https://")
        req = urllib.request.Request(url, headers={"User-Agent": "smerkil-agent/0.3"})
        t0 = time.monotonic()
        with urllib.request.urlopen(req, timeout=self.TIMEOUT) as resp:
            raw = resp.read(self.MAX_BYTES)
            ctype = resp.headers.get("Content-Type", "")
        text: str
        if "html" in ctype.lower():
            text = _html_to_text(raw.decode("utf-8", errors="replace"))
        else:
            text = raw.decode("utf-8", errors="replace")
        elapsed = time.monotonic() - t0
        header = f"[{url} | {ctype or 'unknown'} | {len(raw)} bytes | {elapsed:.1f}s]"
        return header + "\n" + text[: int(max_chars)]


def _html_to_text(html: str) -> str:
    import re
    import html as html_mod
    # drop scripts/styles, strip tags, collapse whitespace
    html = re.sub(r"(?is)<(script|style|noscript)[^>]*>.*?</\1>", " ", html)
    html = re.sub(r"(?is)<br\s*/?>", "\n", html)
    html = re.sub(r"(?is)</(p|div|h[1-6]|li|tr|section|article)>", "\n", html)
    text = re.sub(r"<[^>]+>", " ", html)
    text = html_mod.unescape(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


# ---------------- new: python_exec ----------------
class PythonExec(Tool):
    name = "python_exec"
    description = ("Run a Python code snippet in a subprocess with a timeout, "
                   "isolated from the agent. args: {code, timeout}")
    params = {"type": "object",
              "properties": {"code": {"type": "string"},
                             "timeout": {"type": "integer", "description": "seconds, default 10, max 60"}},
              "required": ["code"]}

    def run(self, code: str = "", timeout: int = 10, **_: Any) -> str:
        timeout = max(1, min(int(timeout), 60))
        try:
            r = subprocess.run(
                ["python3", "-I", "-c", code],
                capture_output=True, text=True, timeout=timeout,
                cwd=str(self.sandbox.root) if self.sandbox else None)
        except subprocess.TimeoutExpired:
            raise ToolError(f"python_exec timed out after {timeout}s")
        out = (r.stdout or "") + (("\n[stderr] " + r.stderr) if r.stderr else "")
        if r.returncode != 0:
            out += f"\n[exit {r.returncode}]"
        return out.strip()[:10_000] or "(no output)"


# ---------------- new: read-only git tools ----------------
class _GitTool(Tool):
    args_spec: Tuple[Tuple[str, ...], ...] = ()

    def _git(self, sub: List[str], cwd: Path) -> str:
        try:
            r = subprocess.run(["git"] + sub, capture_output=True, text=True,
                               timeout=15, cwd=str(cwd))
        except subprocess.TimeoutExpired:
            raise ToolError("git command timed out")
        if r.returncode != 0:
            raise ToolError(f"git failed: {r.stderr.strip()[:300]}")
        return (r.stdout + (("\n[stderr] " + r.stderr) if r.stderr else "")).strip()

    def _repo(self) -> Path:
        p = self.sandbox.root
        for cand in (p, *p.parents):
            if (cand / ".git").exists():
                return cand
        raise ToolError(f"no git repository found at or above {p}")


class GitStatus(_GitTool):
    name = "git_status"
    description = "Read-only git status for the repository containing the workspace. args: {}"
    params = {"type": "object", "properties": {}, "required": []}

    def run(self, **_: Any) -> str:
        repo = self._repo()
        return self._git(["status", "--porcelain=v1", "-b"], repo)[:5000] or "(clean)"


class GitLog(_GitTool):
    name = "git_log"
    description = "Read-only git log. args: {max_count}"
    params = {"type": "object", "properties": {"max_count": {"type": "integer", "description": "default 10"}}, "required": []}

    def run(self, max_count: int = 10, **_: Any) -> str:
        repo = self._repo()
        n = max(1, min(int(max_count), 50))
        return self._git(["log", "--oneline", f"-n{n}"], repo)[:5000]


# ---------------- new: plan / todo tools ----------------
class PlanTool(Tool):
    """Backed by a Plan object injected at build time (not the sandbox)."""

    def __init__(self, plan: Plan, **kw: Any):
        super().__init__(**kw)
        self.plan = plan


class PlanAdd(PlanTool):
    name = "plan_add"
    description = "Add one or more tasks to the plan (workspace/plan.md). args: {tasks: [..]} or {task}"
    params = {"type": "object",
              "properties": {"tasks": {"type": "array", "items": {"type": "string"}},
                             "task": {"type": "string"}},
              "required": []}

    def run(self, tasks: Optional[List[str]] = None, task: str = "", **_: Any) -> str:
        items = [t for t in (tasks or []) if str(t).strip()]
        if task.strip():
            items.append(task.strip())
        if not items:
            raise ToolError("provide 'tasks' (list) or 'task'")
        self.plan.add_many(items)
        return "added: " + "; ".join(items)


class PlanList(PlanTool):
    name = "plan_list"
    description = "Show the current plan with statuses. args: {}"
    params = {"type": "object", "properties": {}, "required": []}

    def run(self, **_: Any) -> str:
        self.plan.load()
        return self.plan.render()


class PlanUpdate(PlanTool):
    name = "plan_update"
    description = ("Update a task's status: todo / in_progress / done. args: {index, status} "
                   "(index is 1-based, from plan_list)")
    params = {"type": "object",
              "properties": {"index": {"type": "integer"},
                             "status": {"type": "string", "enum": list(STATUSES)}},
              "required": ["index", "status"]}

    def run(self, index: int = 0, status: str = "", **_: Any) -> str:
        try:
            idx = int(index) - 1  # model-facing 1-based
        except (TypeError, ValueError):
            raise ToolError(f"invalid index {index!r}")
        try:
            self.plan.set_status(idx, status)
        except ValueError as e:
            raise ToolError(str(e))
        return f"task {index} set to {status}"


# ---------------- new: subagent tool ----------------
class SubagentTool(Tool):
    """Spawn isolated child agents with fresh contexts; children run the same
    loop in parallel threads and return their final texts. Single goal → one
    child. goals array → parallel fan-out (Claude Code / Hermes style)."""

    name = "subagent"
    description = ("Spawn isolated sub-agent(s) with fresh contexts. A single goal "
                   "spawns one child; a goals array runs children in PARALLEL and "
                   "returns results in order. args: {goal | goals: [...], tools, max_turns}")
    params = {"type": "object",
              "properties": {
                  "goal": {"type": "string"},
                  "goals": {"type": "array", "items": {"type": "string"},
                            "description": "multiple independent subtasks — run in parallel"},
                  "tools": {"type": "array", "items": {"type": "string"},
                            "description": "subset of tools for the child; defaults to safe file tools"},
                  "max_turns": {"type": "integer"}},
              "required": []}

    DEFAULT_CHILD_TOOLS = ["read_file", "write_file", "list_dir", "grep", "glob"]

    def __init__(self, agent_factory: Callable[[], Any], parent_tools: Dict[str, Tool], **kw: Any):
        super().__init__(**kw)
        self._agent_factory = agent_factory
        self._parent_tools = parent_tools
        self.depth = 0

    def _run_one(self, goal: str, tools: Optional[List[str]],
                 max_turns: int, idx: int) -> Dict[str, Any]:
        from .agent import Agent, AgentRunError
        allowed = [t for t in (tools or self.DEFAULT_CHILD_TOOLS) if t in self._parent_tools]
        if not allowed:
            allowed = [t for t in self.DEFAULT_CHILD_TOOLS if t in self._parent_tools]
        if not allowed:
            return {"index": idx, "error": "no valid tools available for subagent"}
        child = self._agent_factory(tools=allowed, max_turns=max(1, min(int(max_turns), 25)))
        try:
            result = child.run(goal)
            return {"index": idx, "goal": goal[:200],
                    "result": str(result)[:4000],
                    "stop_reason": child.trace.get("stop_reason"),
                    "turns": len(child.trace.get("steps", [])),
                    "tool_calls": len(child.trace.get("tool_calls", []))}
        except AgentRunError as e:
            return {"index": idx, "goal": goal[:200], "error": f"subagent failed: {e}"}
        except Exception as e:  # child crash → structured error
            return {"index": idx, "goal": goal[:200],
                    "error": f"subagent crashed: {type(e).__name__}: {e}"}

    def run(self, goal: str = "", goals: Optional[List[str]] = None,
            tools: Optional[List[str]] = None, max_turns: int = 12,
            **_: Any) -> str:
        task_list = [g for g in (goals or ([goal] if goal else [])) if str(g).strip()]
        if not task_list:
            return json.dumps({"error": "provide 'goal' or a non-empty 'goals' array"})
        if len(task_list) > 8:
            return json.dumps({"error": "max 8 subagents per fan-out"})
        if len(task_list) == 1:
            out = self._run_one(task_list[0], tools, max_turns, 0)
        else:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=min(len(task_list), 8)) as ex:
                outs = list(ex.map(lambda p: self._run_one(p[1], tools, max_turns, p[0]),
                                   enumerate(task_list)))
            out = {"count": len(outs), "results": outs}
        return json.dumps(out, ensure_ascii=False)


# ---------------- registry ----------------
def build_tools(names: List[str], sandbox: Sandbox, allow_shell: bool,
                plan: Optional[Plan] = None,
                agent_factory: Optional[Callable[[], Any]] = None,
                parent_tools: Optional[Dict[str, Tool]] = None) -> Dict[str, Tool]:
    tools: Dict[str, Tool] = {}
    for name in names:
        if name in ("plan_add", "plan_list", "plan_update", "todo"):
            if plan is None:
                plan = Plan(sandbox.root / "plan.md")
            tool: Tool = {
                "plan_add": lambda: PlanAdd(plan=plan, sandbox=sandbox, allow_shell=allow_shell),
                "plan_list": lambda: PlanList(plan=plan, sandbox=sandbox, allow_shell=allow_shell),
                "plan_update": lambda: PlanUpdate(plan=plan, sandbox=sandbox, allow_shell=allow_shell),
            }[{"plan_add": "plan_add", "plan_list": "plan_list", "plan_update": "plan_update",
               "todo": "plan_list"}[name]]()
            tools[name] = tool
        elif name == "subagent":
            if agent_factory is None:
                raise ToolError("subagent tool requires agent_factory (internal)")
            tools[name] = SubagentTool(agent_factory=agent_factory,
                                       parent_tools=parent_tools or {},
                                       sandbox=sandbox, allow_shell=allow_shell)
        else:
            cls = BUILTIN_TOOLS.get(name)
            if cls is None:
                # tolerate unknown names (todo_write and friends are injected
                # later by Agent.__init__; a stale yaml must not crash the run)
                continue
            tools[name] = cls(sandbox, allow_shell)
    return tools


# ---------------- retrieval: grep / glob (just-in-time, Anthropic-style) ----------------
class GrepTool(Tool):
    name = "grep"
    description = ("Search inside workspace files by regex. Returns file:line matches. "
                   "args: {pattern, path, glob, max_results}")
    params = {"type": "object",
              "properties": {"pattern": {"type": "string"},
                             "path": {"type": "string", "description": "defaults to workspace root"},
                             "glob": {"type": "string", "description": "filename filter, e.g. '*.py'"},
                             "max_results": {"type": "integer", "description": "default 50"}},
              "required": ["pattern"]}

    def run(self, pattern: str = "", path: str = ".", glob: str = "",
            max_results: int = 50, **_: Any) -> str:
        import re as _re
        try:
            rx = _re.compile(pattern)
        except _re.error as e:
            raise ToolError(f"bad regex: {e}")
        root = self.sandbox.resolve(path)
        if not root.exists():
            raise ToolError(f"no such path: {path}")
        max_results = max(1, min(int(max_results), 200))
        out, skipped = [], 0
        files = sorted(root.rglob(glob or "*")) if root.is_dir() else [root]
        for f in files:
            if len(out) >= max_results:
                break
            if not f.is_file() or f.stat().st_size > 1_000_000:
                continue
            try:
                text = f.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for n, line in enumerate(text.splitlines(), 1):
                if rx.search(line):
                    rel = f.relative_to(root) if root.is_dir() else f.name
                    out.append(f"{rel}:{n}: {line.strip()[:200]}")
                    if len(out) >= max_results:
                        break
                else:
                    continue
        if not out:
            return "(no matches)"
        return "\n".join(out) + (f"\n… [truncated at {max_results} results]" if len(out) >= max_results else "")


class GlobTool(Tool):
    name = "glob"
    description = ("Find files in the workspace by glob pattern. args: {pattern, path}")
    params = {"type": "object",
              "properties": {"pattern": {"type": "string", "description": "e.g. '**/*.py'"},
                             "path": {"type": "string"}},
              "required": ["pattern"]}

    def run(self, pattern: str = "**/*", path: str = ".", **_: Any) -> str:
        root = self.sandbox.resolve(path)
        if not root.is_dir():
            raise ToolError(f"not a directory: {path}")
        matches = sorted(root.glob(pattern))[:200]
        if not matches:
            return "(no matches)"
        return "\n".join(str(m.relative_to(root)) for m in matches)


# ---------------- checkpoint / restore (file snapshots) ----------------
class CheckpointTool(Tool):
    """Workspace snapshots stored under .smerkil/checkpoints/. Restore reverts
    tracked files to a snapshot state. stdlib zip, no network."""

    name = "checkpoint"
    description = ("Create or restore a snapshot of workspace files. "
                   "args: {action: 'create'|'restore'|'list', name} (restore needs name from list)")
    params = {"type": "object",
              "properties": {"action": {"type": "string", "enum": ["create", "restore", "list"]},
                             "name": {"type": "string"}},
              "required": ["action"]}

    def run(self, action: str = "", name: str = "", **_: Any) -> str:
        import zipfile, time as _time
        cp_dir = self.sandbox.root / ".smerkil" / "checkpoints"
        if action == "list":
            if not cp_dir.exists():
                return "(no checkpoints)"
            return "\n".join(f"{z.stem}  ({z.stat().st_size} bytes)"
                             for z in sorted(cp_dir.glob("*.zip")))
        if action == "create":
            label = name or _time.strftime("%Y%m%d-%H%M%S")
            cp_dir.mkdir(parents=True, exist_ok=True)
            zpath = cp_dir / f"{label}.zip"
            count = 0
            with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
                for f in sorted(self.sandbox.root.rglob("*")):
                    if (not f.is_file() or ".smerkil" in f.parts
                            or "traces" in f.parts or f.name == "plan.md"):
                        continue
                    if f.stat().st_size > 1_000_000:
                        continue
                    z.write(f, f.relative_to(self.sandbox.root))
                    count += 1
            return f"checkpoint '{label}' created ({count} files)"
        if action == "restore":
            if not name:
                raise ToolError("restore needs a checkpoint name (use action=list)")
            zpath = cp_dir / f"{name}.zip"
            if not zpath.exists():
                raise ToolError(f"no checkpoint '{name}'")
            restored = 0
            with zipfile.ZipFile(zpath) as z:
                for info in z.infolist():
                    target = self.sandbox.resolve(str(info.filename))
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(z.read(info))
                    restored += 1
            return f"restored {restored} files from '{name}'"
        raise ToolError("action must be create | restore | list")


BUILTIN_TOOLS: Dict[str, Callable[[Sandbox, bool], Tool]] = {
    t.name: t
    for t in (ReadFile, WriteFile, ListDir, RunShell, MemoryNote,
              WebFetch, PythonExec, GitStatus, GitLog, GrepTool, GlobTool,
              CheckpointTool)
}


def tools_to_openai_schema(tools: Dict[str, Tool]) -> List[Dict[str, Any]]:
    """Adapt to OpenAI tool-call format (works on every compatible router)."""
    out = []
    for t in tools.values():
        out.append({
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.params or {"type": "object", "properties": {}, "required": []},
            },
        })
    return out


def execute_tool(tools: Dict[str, Tool], name: str, arguments: str) -> str:
    tool = tools.get(name)
    if tool is None:
        return json.dumps({"error": f"unknown tool '{name}'"})
    try:
        kwargs = json.loads(arguments) if arguments.strip() else {}
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"bad arguments JSON: {e}"})
    try:
        return tool.run(**kwargs)
    except ToolError as e:
        return json.dumps({"error": str(e)})
    except Exception as e:  # tool crash → structured error, loop continues
        return json.dumps({"error": f"{type(e).__name__}: {e}"})


def execute_tools_parallel(tools: Dict[str, Tool],
                           calls: List[Tuple[str, str, str]],
                           max_workers: int = 8) -> List[Tuple[str, str]]:
    """Run multiple tool calls concurrently. calls = [(call_id, name, args)].
    Returns [(call_id, result)] in the SAME ORDER as input (deterministic),
    regardless of completion order."""
    if len(calls) <= 1:
        return [(cid, execute_tool(tools, n, a)) for cid, n, a in calls]

    def work(cid: str, name: str, args: str) -> str:
        return execute_tool(tools, name, args)

    with ThreadPoolExecutor(max_workers=min(max_workers, len(calls))) as ex:
        futures = [ex.submit(work, cid, n, a) for cid, n, a in calls]
        results = [f.result() for f in futures]  # ordered; exceptions already handled
    return list(zip([c for c, _, _ in calls], results))
