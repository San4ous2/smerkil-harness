"""Structured planning: a plan.md in the workspace with checkbox tasks that
carry statuses (todo / in_progress / done). Parsed and written deterministically;
exposed to the agent via plan tools."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

STATUSES = ("todo", "in_progress", "done")
_CHECK_RE = re.compile(r"^- \[([ x~])\] (.+)$")
_META_RE = re.compile(r"\[([a-z_]+)\]$")
FILE_HEADER = "# Plan\n"


def _status_from_mark(box: str, text: str) -> str:
    if box == "x":
        return "done"
    if box == "~":
        return "in_progress"
    m = _META_RE.search(text)
    if m and m.group(1) in STATUSES:
        return m.group(1)
    return "todo"


def _mark(status: str) -> str:
    return {"done": "x", "in_progress": "~"}.get(status, " ")


class Plan:
    """plan.md-backed task list. Format per line:
        - [x] task text [status-tag-optional]
    status is derived from the checkbox; the tag in brackets at end of line is
    preserved for human readability."""

    def __init__(self, path: Path):
        self.path = path
        self._tasks: List[Dict[str, str]] = []
        self.load()

    def load(self) -> "Plan":
        self._tasks = []
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                m = _CHECK_RE.match(line.strip())
                if m:
                    text = m.group(2).strip()
                    status = _status_from_mark(m.group(1), text)
                    self._tasks.append({"task": text, "status": status})
        return self

    def save(self) -> None:
        lines = [FILE_HEADER.rstrip("\n"), ""]
        for t in self._tasks:
            lines.append(f"- [{_mark(t['status'])}] {t['task']}")
        self.path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---------------- operations ----------------
    def list_tasks(self) -> List[Dict[str, str]]:
        return [dict(t) for t in self._tasks]

    def render(self) -> str:
        if not self._tasks:
            return "(plan is empty)"
        out = []
        for i, t in enumerate(self._tasks, 1):
            out.append(f"{i}. [{t['status']}] {t['task']}")
        counts = {s: sum(1 for t in self._tasks if t["status"] == s)
                  for s in STATUSES}
        out.append(f"-- todo: {counts['todo']} | in_progress: {counts['in_progress']} | done: {counts['done']}")
        return "\n".join(out)

    def add(self, task: str, status: str = "todo") -> int:
        if status not in STATUSES:
            status = "todo"
        self._tasks.append({"task": task.strip(), "status": status})
        self.save()
        return len(self._tasks) - 1

    def set_status(self, index: int, status: str) -> str:
        if status not in STATUSES:
            raise ValueError(f"invalid status '{status}'; use {STATUSES}")
        self.load()  # pick up external edits
        if not (0 <= index < len(self._tasks)):
            raise ValueError(f"task index {index} out of range (0..{len(self._tasks) - 1})")
        self._tasks[index]["status"] = status
        self.save()
        return f"task {index} -> {status}"

    def add_many(self, tasks: List[str]) -> str:
        for t in tasks:
            if t.strip():
                self._tasks.append({"task": t.strip(), "status": "todo"})
        self.save()
        return f"added {sum(1 for t in tasks if t.strip())} tasks"
