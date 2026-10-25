"""TodoWrite: plan tracking the model must maintain (Claude Code style).

The todo list lives in <workspace>/.smerkil/todos.json. Every turn the agent
sees the current list in its system prompt; the tool enforces consistency:
 - exactly one in_progress item at a time
 - cannot mark done while an in_progress item exists (finish what you started)
 - status must be pending | in_progress | done
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from .config import load_config_text


class TodoError(ValueError):
    pass


class TodoList:
    def __init__(self, path: Path):
        self.path = path
        self.items: List[Dict[str, str]] = []

    def load(self) -> None:
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    self.items = [i for i in data if isinstance(i, dict)]
            except (json.JSONDecodeError, OSError):
                self.items = []

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.items, indent=2, ensure_ascii=False),
                             encoding="utf-8")

    def replace(self, items: List[Dict[str, str]]) -> str:
        """Validate a full replacement of the list and persist it."""
        if not isinstance(items, list) or not items:
            raise TodoError("todos must be a non-empty array of {content, status, activeForm?}")
        cleaned: List[Dict[str, str]] = []
        in_progress = 0
        for i, it in enumerate(items):
            if not isinstance(it, dict):
                raise TodoError(f"item {i}: must be an object")
            content = str(it.get("content", "")).strip()
            status = str(it.get("status", "pending")).strip()
            if not content:
                raise TodoError(f"item {i}: empty content")
            if status not in ("pending", "in_progress", "done"):
                raise TodoError(f"item {i}: status must be pending|in_progress|done, got '{status}'")
            if status == "in_progress":
                in_progress += 1
                if in_progress > 1:
                    raise TodoError("only ONE task may be in_progress at a time")
            cleaned.append({"content": content, "status": status,
                            "activeForm": str(it.get("activeForm", content))})
        # don't allow marking things done while another task is in progress
        has_ip = any(c["status"] == "in_progress" for c in cleaned)
        if has_ip and any(c["status"] == "done" for c in cleaned):
            # allowed only if the done ones precede the in_progress one in order
            order = [c["status"] for c in cleaned]
            first_done_after_ip = False
            seen_ip = False
            for s in order:
                if s == "in_progress":
                    seen_ip = True
                elif s == "done" and seen_ip:
                    first_done_after_ip = True
                    break
            if first_done_after_ip:
                raise TodoError("cannot complete a task that comes after the in_progress task; "
                                "finish the in_progress task first")
        self.items = cleaned
        self.save()
        return self.render()

    def render(self) -> str:
        if not self.items:
            return "(no todos)"
        icons = {"pending": "[ ]", "in_progress": "[~]", "done": "[x]"}
        return "\n".join(f"{icons[i['status']]} {i['content']}" for i in self.items)

    def prompt_block(self) -> str:
        if not self.items:
            return ""
        return ("\n\nCurrent task list (maintain it with the todo_write tool; "
                "keep exactly one task in_progress):\n" + self.render())


class TodoTool:
    """Duck-typed tool; workspace-injected."""

    name = "todo_write"
    description = ("Replace the whole task list. Each item: {content, status, activeForm?}. "
                   "Rules: exactly one in_progress; finish tasks in order; keep the list "
                   "updated as you work. args: {todos: [...]}")
    params = {"type": "object",
              "properties": {"todos": {"type": "array",
                                       "items": {"type": "object"}}},
              "required": ["todos"]}

    def __init__(self, todos: TodoList, sandbox: Any = None, allow_shell: bool = False):
        self._todos = todos
        self.sandbox = sandbox
        self.allow_shell = allow_shell

    def run(self, todos: List[Dict[str, str]] = None, **_: Any) -> str:
        return self._todos.replace(todos or [])
