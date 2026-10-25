"""Context compaction: keep the transcript under a budget by summarizing older
messages. Summarizer is pluggable (protocol); default truncates when no model
is available.

Loop-safety invariants:
- System messages are always preserved verbatim.
- An assistant message carrying tool_calls is never separated from its tool
  result messages: they compact as one atomic unit.
- Compaction never runs mid-pair.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol


def estimate_tokens(messages: List[Dict[str, Any]]) -> int:
    """Rough token estimate: ~4 chars/token over all string content."""
    total = 0
    for m in messages:
        c = m.get("content")
        if isinstance(c, str):
            total += len(c)
        elif c is not None:
            total += len(str(c))
        for tc in m.get("tool_calls") or []:
            total += len(str(tc.get("function", {}).get("arguments", "")))
    return total // 4


class Summarizer(Protocol):
    def summarize(self, messages: List[Dict[str, Any]]) -> str:  # pragma: no cover
        ...


class TruncatingSummarizer:
    """Fallback summarizer when no model is available: keeps the head of each
    message and a count of what was dropped. Never calls the network."""

    def __init__(self, max_chars_per_message: int = 400):
        self.max_chars_per_message = max_chars_per_message

    def summarize(self, messages: List[Dict[str, Any]]) -> str:
        lines = []
        for m in messages:
            role = m.get("role", "?")
            c = m.get("content")
            if c is None:
                c = ""
            text = c if isinstance(c, str) else str(c)
            head = text[: self.max_chars_per_message]
            more = "" if len(text) <= self.max_chars_per_message else f" …[+{len(text) - self.max_chars_per_message} chars]"
            tcs = m.get("tool_calls") or []
            tc_desc = ""
            if tcs:
                names = []
                for tc in tcs:
                    fn = tc.get("function", {})
                    names.append(fn.get("name", "?"))
                tc_desc = f" [tool calls: {', '.join(names)}]"
            lines.append(f"- {role}: {head}{more}{tc_desc}")
        return "\n".join(lines)


class Compactor:
    """Two-stage compaction, cheapest lever first:

    1. Tool result clearing (Anthropic-style): older tool results are replaced
       by short stubs — they are almost never needed again once seen. The
       tool_call and its (stubbed) result stay API-valid.
    2. Summarization: when clearing is not enough, older messages are
       summarized. Units are atomic: an assistant message with tool_calls plus
       all following tool messages compact together.
    """

    STUB_MARKER = "[cleared: tool result seen earlier]"

    def __init__(self, summarizer: Summarizer,
                 max_tokens: int = 24_000,
                 keep_recent: int = 8,
                 clear_tool_results: bool = True):
        self.summarizer = summarizer
        self.max_tokens = max_tokens
        self.keep_recent = max(4, keep_recent)
        self.clear_tool_results = clear_tool_results
        self.compactions = 0
        self.cleared_results = 0

    def _clear_old_tool_results(self, messages: List[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
        """Replace content of tool results that are older than the recent
        window with a short stub. Returns a NEW list, or None if the budget
        is already met after clearing."""
        if not self.clear_tool_results:
            return None
        # find index where the recent window starts (count from the end)
        kept = 0
        cut = len(messages)
        while cut > 0 and kept < self.keep_recent:
            cut -= 1
            kept += 1
        changed = False
        out: List[Dict[str, Any]] = []
        for i, m in enumerate(messages):
            if i < cut and m.get("role") == "tool" and m.get("content") != self.STUB_MARKER:
                c = str(m.get("content") or "")
                if len(c) > 64:  # tiny results stay verbatim
                    m = dict(m, content=f"{self.STUB_MARKER} (was {len(c)} chars)")
                    changed = True
                    self.cleared_results += 1
            out.append(m)
        if changed and estimate_tokens(out) <= self.max_tokens:
            self.compactions += 1
            return out
        return None

    @staticmethod
    def _units(rest: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Group non-system messages into atomic units. A unit is an assistant
        message with tool_calls plus every subsequent tool message; any other
        message is its own unit."""
        units: List[List[Dict[str, Any]]] = []
        i = 0
        while i < len(rest):
            m = rest[i]
            if m.get("role") == "assistant" and (m.get("tool_calls") or []):
                group = [m]
                i += 1
                while i < len(rest) and rest[i].get("role") == "tool":
                    group.append(rest[i])
                    i += 1
                units.append(group)
            else:
                units.append([m])
                i += 1
        return units

    def maybe_compact(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if estimate_tokens(messages) <= self.max_tokens:
            return messages
        # stage 1: clear old tool results (cheapest, lossless-in-practice)
        cleared = self._clear_old_tool_results(messages)
        if cleared is not None:
            return cleared
        # stage 2: summarize older units
        system: List[Dict[str, Any]] = []
        rest: List[Dict[str, Any]] = []
        for m in messages:
            if m.get("role") == "system" and not rest:
                system.append(m)
            else:
                rest.append(m)
        units = self._units(rest)
        # pick how many whole units from the front fit keep_recent messages;
        # never compact a unit that reaches into the recent window
        keep_units: List[List[Dict[str, Any]]] = []
        kept = 0
        while units and kept < self.keep_recent:
            keep_units.insert(0, units.pop())
            kept += len(keep_units[0])
        if not units:
            return messages  # nothing to safely drop
        older: List[Dict[str, Any]] = [m for u in units for m in u]
        summary = self.summarizer.summarize(older)
        # Hard cap: a summary is useless if it can itself bloat the context.
        # Never let it exceed half the token budget (chars ≈ 4×tokens).
        max_summary_chars = max(1, self.max_tokens * 2)
        if len(summary) > max_summary_chars:
            summary = summary[:max_summary_chars] + (
                f"\n… [summary truncated, {len(summary) - max_summary_chars} chars dropped]"
            )
        self.compactions += 1
        compacted = list(system)
        compacted.append({
            "role": "system",
            "content": (
                "[context compacted] Summary of earlier conversation "
                f"({len(older)} messages omitted):\n{summary}"
            ),
        })
        return compacted + [m for u in keep_units for m in u]


class ModelSummarizer:
    """Optional model-backed summarizer; wraps an LLMClient + model name.
    Falls back to TruncatingSummarizer on any error."""

    def __init__(self, client: Any, model: str,
                 fallback: Optional[TruncatingSummarizer] = None,
                 model_chain: Optional[List[str]] = None,
                 max_transcript_chars: int = 160_000):
        self.client = client
        self.model = model
        self.fallback = fallback or TruncatingSummarizer()
        self.model_chain = model_chain or [model]
        # cap the transcript fed to the summarizer well below what most
        # endpoints accept; head is dropped (older context matters more)
        self.max_transcript_chars = max_transcript_chars

    def summarize(self, messages: List[Dict[str, Any]]) -> str:
        lines = []
        for m in messages:
            role = m.get("role", "?")
            c = m.get("content")
            text = c if isinstance(c, str) else str(c or "")
            tcs = m.get("tool_calls") or []
            if tcs:
                names = ", ".join(
                    tc.get("function", {}).get("name", "?") for tc in tcs)
                text = (text or "") + f" [tool calls: {names}]"
            lines.append(f"{role}: {text}")
        transcript = "\n".join(lines)
        if len(transcript) > self.max_transcript_chars:
            transcript = transcript[-self.max_transcript_chars:]
        prompt = [
            {"role": "system", "content":
                "Summarize the agent transcript so far. Keep goals, decisions, "
                "file paths, command outputs of note, and pending work. Be terse "
                "but complete; never drop a decision or an unresolved error."},
            {"role": "user", "content": transcript},
        ]
        try:
            resp = self.client.chat({"model": self.model, "messages": prompt,
                                     "temperature": 0.0},
                                    None, model_chain=self.model_chain)
            content = resp["choices"][0]["message"].get("content") or ""
            # reasoning models can return empty content on tiny max_tokens or
            # a length finish; treat empty as failure → fallback
            if not content.strip():
                return self.fallback.summarize(messages)
            return content
        except Exception:
            return self.fallback.summarize(messages)
