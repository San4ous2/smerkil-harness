"""Durable agent memory: append-only JSONL store with keyword recall.

Lives in <workspace>/.smerkil/memory.jsonl so it survives sessions, restarts
and reinstalls (everything in .smerkil is agent state, like Claude Code's
.claude/). Zero dependencies: retrieval is token-overlap scoring with a
recency boost — good enough for personal-agent scale (hundreds of notes)
without an embedding model.

Entry shape (one JSON object per line):
  {"id": "20261027-101500-1234-a1b2", "ts": "...ISO...", "text": "...",
   "tags": ["python", "deploy"]}

Injection model (same economics as Claude Code):
  - the TOP few recent notes are injected into the system prompt at run
    start (small, always visible);
  - everything else is pulled on demand via memory_search when the agent
    needs it.
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

_TOKEN_RE = re.compile(r"[a-zа-я0-9_]{2,}", re.IGNORECASE)

# cap on the always-injected recency block, in characters
RECENT_BLOCK_MAX_CHARS = 1800
DEFAULT_RECENT = 8
MAX_NOTE_CHARS = 2000


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class MemoryStore:
    def __init__(self, workspace: Path):
        self.path = Path(workspace) / ".smerkil" / "memory.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._cache: Optional[List[Dict[str, Any]]] = None

    # ---------------- io ----------------
    def _load(self) -> List[Dict[str, Any]]:
        if self._cache is not None:
            return self._cache
        entries: List[Dict[str, Any]] = []
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(rec, dict) and rec.get("text"):
                    entries.append(rec)
        self._cache = entries
        return entries

    def _save(self, entries: List[Dict[str, Any]]) -> None:
        tmp = self.path.with_suffix(".jsonl.tmp")
        tmp.write_text(
            "\n".join(json.dumps(e, ensure_ascii=False) for e in entries) + ("\n" if entries else ""),
            encoding="utf-8")
        os.replace(tmp, self.path)
        self._cache = entries

    # ---------------- api ----------------
    def add(self, text: str, tags: Optional[List[str]] = None) -> Dict[str, Any]:
        text = (text or "").strip()
        if not text:
            raise ValueError("memory note must not be empty")
        if len(text) > MAX_NOTE_CHARS:
            text = text[:MAX_NOTE_CHARS] + "…(truncated)"
        ts = _now_iso()
        entries = self._load()
        # unique id even within the same second/pid (multiple notes per run)
        taken = {e.get("id") for e in entries}
        base = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-") + f"{os.getpid() % 10000:04d}"
        entry_id = base
        n = 0
        while entry_id in taken:
            n += 1
            entry_id = f"{base}-{n}"
        entry = {
            "id": entry_id,
            "ts": ts,
            "text": text,
            "tags": [str(t).strip().lower() for t in (tags or []) if str(t).strip()],
        }
        entries.append(entry)
        self._save(entries)
        return entry

    def forget(self, entry_id: str = "", query: str = "") -> int:
        """Delete by exact id, or all entries matching a query. Returns count."""
        entries = self._load()
        if entry_id:
            kept = [e for e in entries if e.get("id") != entry_id]
        elif query:
            qtokens = set(_tokenize(query))
            kept = [e for e in entries
                    if not (qtokens & set(_tokenize(e["text"] + " " + " ".join(e.get("tags", [])))))]
        else:
            raise ValueError("forget requires entry_id or query")
        removed = len(entries) - len(kept)
        if removed:
            self._save(kept)
        return removed

    def recent(self, n: int = DEFAULT_RECENT) -> List[Dict[str, Any]]:
        entries = self._load()
        return entries[-max(0, int(n)):]

    def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Keyword scoring: query-token overlap with text (1x) and tags (2x),
        plus a mild recency boost. Deterministic: score desc, then ts asc."""
        qtokens = _tokenize(query or "")
        if not qtokens:
            return []
        entries = self._load()
        now = datetime.now(timezone.utc)
        scored = []
        for e in entries:
            hay_text = set(_tokenize(e["text"]))
            hay_tags = set(e.get("tags", []))
            score = 0.0
            for t in set(qtokens):
                if t in hay_tags:
                    score += 2.0
                if t in hay_text:
                    score += 1.0
            if score <= 0:
                continue
            # recency boost: up to +1.0, decays over ~4 weeks
            try:
                age_days = (now - datetime.fromisoformat(e["ts"])).total_seconds() / 86400
                score += max(0.0, 1.0 - age_days / 28.0)
            except (ValueError, TypeError):
                pass
            scored.append((score, e["ts"], e))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [e for _, _, e in scored[:max(1, min(int(limit), 20))]]

    def prompt_block(self, n: int = DEFAULT_RECENT) -> str:
        """System-prompt block with the most recent notes (auto-injection)."""
        entries = self.recent(n)
        if not entries:
            return ""
        lines = []
        budget = RECENT_BLOCK_MAX_CHARS
        for e in reversed(entries):  # newest first
            line = f"- [{e['ts'][:10]}] {e['text']}"
            if e.get("tags"):
                line += f"  #{' #'.join(e['tags'])}"
            if budget - len(line) < 0:
                break
            budget -= len(line) + 1
            lines.append(line)
        if not lines:
            return ""
        return ("\n\nDurable memory (most recent notes; use memory_search for "
                "more):\n" + "\n".join(lines))
