"""Memory tools: memory_save / memory_search / memory_forget.

Backed by MemoryStore (smerkil/memory.py) — append-only JSONL in
<workspace>/.smerkil/memory.jsonl. The most recent notes are auto-injected
into the system prompt by the agent loop; memory_search pulls older notes
on demand.
"""
from __future__ import annotations

from typing import Any, List, Optional

from .memory import MemoryStore
from .tools import Sandbox, Tool, ToolError

MAX_NOTE_CHARS = 2000


class MemorySave(Tool):
    name = "memory_save"
    description = ("Save a durable note for future sessions (survives restarts; "
                   "recent notes are auto-shown at run start). args: {text, tags?}")
    params = {"type": "object",
              "properties": {"text": {"type": "string",
                                      "description": "the fact to remember, self-contained"},
                             "tags": {"type": "array", "items": {"type": "string"},
                                      "description": "optional keywords for retrieval"}},
              "required": ["text"]}

    def __init__(self, store: MemoryStore, sandbox: Optional[Sandbox] = None,
                 allow_shell: bool = False):
        super().__init__(sandbox, allow_shell)
        self.store = store

    def run(self, text: str = "", tags: Optional[List[str]] = None, **_: Any) -> str:
        text = (text or "").strip()
        if not text:
            raise ToolError("text is required")
        if len(text) > MAX_NOTE_CHARS:
            text = text[:MAX_NOTE_CHARS] + "…(truncated)"
        try:
            entry = self.store.add(text, tags)
        except ValueError as e:
            raise ToolError(str(e))
        return f"saved {entry['id']}"


class MemorySearch(Tool):
    name = "memory_search"
    description = ("Search durable memory from past sessions by keywords. "
                   "args: {query, limit?}")
    params = {"type": "object",
              "properties": {"query": {"type": "string"},
                             "limit": {"type": "integer", "description": "default 5"}},
              "required": ["query"]}

    def __init__(self, store: MemoryStore, sandbox: Optional[Sandbox] = None,
                 allow_shell: bool = False):
        super().__init__(sandbox, allow_shell)
        self.store = store

    def run(self, query: str = "", limit: int = 5, **_: Any) -> str:
        hits = self.store.search(query, limit=limit)
        if not hits:
            return "(no matching memories)"
        out = []
        for e in hits:
            tags = f"  #{' #'.join(e['tags'])}" if e.get("tags") else ""
            out.append(f"[{e['id']}] {e['ts'][:10]}: {e['text']}{tags}")
        return "\n".join(out)


class MemoryForget(Tool):
    name = "memory_forget"
    description = ("Delete durable memories: by exact id, or everything matching "
                   "a keyword query. args: {id?} or {query?} (one required)")
    params = {"type": "object",
              "properties": {"id": {"type": "string"},
                             "query": {"type": "string"}},
              "required": []}

    def __init__(self, store: MemoryStore, sandbox: Optional[Sandbox] = None,
                 allow_shell: bool = False):
        super().__init__(sandbox, allow_shell)
        self.store = store

    def run(self, id: str = "", query: str = "", **_: Any) -> str:
        try:
            n = self.store.forget(entry_id=id or "", query=query or "")
        except ValueError as e:
            raise ToolError(str(e))
        return f"forgot {n} note(s)"
