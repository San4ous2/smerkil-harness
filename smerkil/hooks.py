"""Lifecycle hooks: deterministic shell/python actions on agent events.

Fires OUTSIDE the LLM — the model cannot skip or disable them. Events:
  run_start, turn_end, tool_before, tool_after, run_end

Hooks live in <workspace>/hooks.yaml:

  hooks:
    - event: tool_before
      tool: run_shell            # optional filter; omit = all tools
      command: echo "shell used at $(date)"

Hook command runs via shell with env:
  SMERKIL_EVENT, SMERKIL_TOOL, SMERKIL_EXIT_CODE, SMERKIL_WORKSPACE
Non-zero hook exit on tool_before BLOCKS the tool call (a gate, like
Claude Code PreToolUse). Output is returned to the model as an error.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import load_config_text

EVENTS = ("run_start", "turn_end", "tool_before", "tool_after", "run_end")


class Hook:
    def __init__(self, event: str, command: str,
                 tool: Optional[str] = None, timeout: int = 15):
        if event not in EVENTS:
            raise ValueError(f"unknown hook event '{event}' (valid: {EVENTS})")
        self.event = event
        self.command = command
        self.tool = tool
        self.timeout = max(1, min(int(timeout), 60))

    def matches(self, event: str, tool_name: str = "") -> bool:
        if event != self.event:
            return False
        if self.tool is not None and self.tool != tool_name:
            return False
        return True

    def execute(self, env_extra: Dict[str, str]) -> Optional[str]:
        """Run the hook; returns output text or None. Never raises."""
        env = dict(os.environ, **env_extra)
        try:
            r = subprocess.run(self.command, shell=True, capture_output=True,
                               text=True, timeout=self.timeout, env=env)
        except subprocess.TimeoutExpired:
            return f"[hook timeout after {self.timeout}s]"
        except Exception as e:  # hooks must never crash the run
            return f"[hook error: {type(e).__name__}: {e}]"
        out = (r.stdout or "").strip()
        err = (r.stderr or "").strip()
        return out + (f"\n[stderr] {err}" if err else "")


class HookRunner:
    def __init__(self, hooks: List[Hook]):
        self.hooks = hooks

    @classmethod
    def load(cls, workspace: Path) -> "HookRunner":
        path = workspace / "hooks.yaml"
        if not path.exists():
            return cls([])
        try:
            data = load_config_text(path.read_text(encoding="utf-8"))
        except Exception as e:
            # A malformed hooks.yaml silently disabling your gate hooks is a
            # foot-gun — say so loudly instead of pretending there are none.
            print(f"[smerkil] hooks.yaml not parseable, hooks skipped: "
                  f"{type(e).__name__}: {str(e)[:200]}")
            return cls([])
        raw = data.get("hooks") or []
        hooks: List[Hook] = []
        for h in raw:
            if not isinstance(h, dict):
                continue
            try:
                hooks.append(Hook(event=str(h.get("event", "")),
                                  command=str(h.get("command", "")),
                                  tool=h.get("tool") if h.get("tool") else None,
                                  timeout=int(h.get("timeout", 15))))
            except (ValueError, TypeError):
                continue
        return cls(hooks)

    def fire(self, event: str, tool_name: str = "",
             env_extra: Optional[Dict[str, str]] = None) -> Optional[str]:
        """Fire matching hooks. Returns concatenated output, or None.
        For tool_before: a non-zero exit of ANY matching hook blocks —
        the return value is then the blocking error text."""
        env_base = dict(env_extra or {},
                        SMERKIL_EVENT=event,
                        SMERKIL_TOOL=tool_name)
        outputs: List[str] = []
        for h in self.hooks:
            if not h.matches(event, tool_name):
                continue
            out = h.execute(env_base)
            if out:
                outputs.append(out)
        return "\n".join(outputs) if outputs else None
