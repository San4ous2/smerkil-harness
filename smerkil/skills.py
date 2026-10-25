"""Skills: markdown instruction files loaded just-in-time.

A skill lives in <workspace>/skills/<name>.md with optional YAML frontmatter:

    ---
    description: Deploy the site: build, verify, upload.
    ---

    Step-by-step instructions here...

The agent sees ONLY name+description in its system prompt (a few tokens).
Full body is loaded on demand via the skill tool — same economics as
Claude Code skills (progressive disclosure).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import load_config_text


class Skill:
    def __init__(self, name: str, path: Path, description: str = ""):
        self.name = name
        self.path = path
        self.description = description

    def index_line(self) -> str:
        d = self.description or "(no description)"
        return f"- {self.name}: {d}"


def load_skills(workspace: Path) -> Dict[str, Skill]:
    """Scan <workspace>/skills/*.md. Frontmatter description parsed without
    assuming PyYAML (mini-parser fallback)."""
    skills: Dict[str, Skill] = {}
    sdir = workspace / "skills"
    if not sdir.is_dir():
        return skills
    for f in sorted(sdir.glob("*.md")):
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        desc = ""
        if text.startswith("---"):
            parts = text.split("---", 2)
            if len(parts) >= 3:
                fm_text, body = parts[1], parts[2]
                fm = load_config_text(fm_text)
                desc = str(fm.get("description", "") or "")
        skills[f.stem] = Skill(f.stem, f, desc)
    return skills


def skills_index(skills: Dict[str, Skill]) -> str:
    if not skills:
        return ""
    lines = "\n".join(s.index_line() for s in skills.values())
    return (
        "\n\nAvailable skills (load the full instructions with the "
        f"`skill` tool before using one):\n{lines}"
    )


class SkillTool:
    """Duck-typed as a Tool; injected with the workspace skill map."""

    name = "skill"
    description = ("Load full instructions for a named skill (see the skill index "
                   "in your context). args: {name}")
    params = {"type": "object",
              "properties": {"name": {"type": "string"}},
              "required": ["name"]}

    def __init__(self, skills: Dict[str, Skill], sandbox: Any = None,
                 allow_shell: bool = False):
        self._skills = skills
        self.sandbox = sandbox
        self.allow_shell = allow_shell

    def run(self, name: str = "", **_: Any) -> str:
        sk = self._skills.get(name)
        if sk is None:
            avail = ", ".join(sorted(self._skills)) or "(none)"
            raise ValueError(f"unknown skill '{name}'. available: {avail}")
        return sk.path.read_text(encoding="utf-8", errors="replace")
