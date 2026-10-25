"""The deterministic agent loop: plan → act (tools) → verify → answer.

Talks to any OpenAI-compatible endpoint using stdlib only. Supports parallel
tool execution, context compaction, streaming, subagent spawning, and durable
JSONL traces with run metadata.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .budget import Budget
from .compaction import Compactor, ModelSummarizer, TruncatingSummarizer
from .config import AgentConfig
from .hooks import HookRunner
from .llm import LLMClient, LLMError
from .mcp import MCPManager, load_mcp_config
from .planning import Plan
from .skills import SkillTool, load_skills, skills_index
from .todos import TodoList, TodoTool, TodoError
from .tools import (
    Sandbox,
    SubagentTool,
    build_tools,
    execute_tools_parallel,
    execute_tool,
    tools_to_openai_schema,
)


class AgentRunError(Exception):
    pass


DEFAULT_COMPACTION_TOKENS = 24_000


class Agent:
    def __init__(self, config: AgentConfig, workspace: Path,
                 parent_tools: Optional[Dict[str, Any]] = None,
                 depth: int = 0,
                 compactor: Optional[Compactor] = None,
                 on_token: Optional[Callable[[str], None]] = None,
                 on_tool_delta: Optional[Callable[[str, str], None]] = None,
                 stream: bool = True):
        self.cfg = config
        self.workspace = workspace
        self.depth = depth
        self.on_token = on_token
        self.on_tool_delta = on_tool_delta
        self.session_budget: Optional[Budget] = None  # chat mode: set by CLI
        self.stream = stream and on_token is not None
        self.sandbox = Sandbox(workspace, strict=config.guardrails.sandbox_strict)
        self.plan = Plan(workspace / "plan.md")

        def _factory(tools: List[str], max_turns: int) -> "Agent":
            # child agents get an isolated workspace-scoped agent; no nested
            # subagent spawning beyond depth 1 to bound recursion
            child_cfg = AgentConfig(
                name=f"{config.name}.sub",
                model=config.model,
                system=config.system,
                tools=tools,
                guardrails=config.guardrails,
                base_url=config.base_url,
                api_key_env=config.api_key_env,
                temperature=config.temperature,
            )
            return Agent(child_cfg, workspace, depth=depth + 1,
                         parent_tools=self.tools, stream=False)

        self.tools = build_tools(config.tools, self.sandbox,
                                 config.guardrails.allow_shell,
                                 plan=self.plan,
                                 agent_factory=_factory if depth < 1 else None,
                                 parent_tools=parent_tools)
        # A subagent resolves its allowed child tools against its PARENT's tool
        # set. Root agents have no parent, so they must fall back to their own.
        for _t in self.tools.values():
            if isinstance(_t, SubagentTool) and not _t._parent_tools:
                _t._parent_tools = dict(self.tools)
        # skills: index in system prompt, body loaded just-in-time
        self.skills = load_skills(workspace)
        if self.skills and "skill" not in self.tools:
            self.tools["skill"] = SkillTool(self.skills, self.sandbox)
        # todos: model-maintained plan, persisted under .smerkil/todos.json
        self.todos = TodoList(workspace / ".smerkil" / "todos.json")
        self.todos.load()
        if "todo_write" not in self.tools:
            self.tools["todo_write"] = TodoTool(self.todos, self.sandbox)
        # MCP servers (mcp.json, Claude Code layout); failures are non-fatal
        self.mcp = MCPManager(load_mcp_config(workspace))
        self.mcp_errors = self.mcp.connect() if self.mcp.servers else {}
        if self.mcp_errors:
            for sname, err in self.mcp_errors.items():
                print(f"[smerkil] MCP server '{sname}' unavailable: {err[:200]}")
        self.mcp_tools = self.mcp.tool_specs() if self.mcp.servers else []
        self.hooks = HookRunner.load(workspace)
        self.client = LLMClient(config.base_url, config.resolve_api_key())
        if compactor is not None:
            self.compactor = compactor
        else:
            summarizer: Any = TruncatingSummarizer()
            if os.environ.get("SMERKIL_MODEL_SUMMARIZER", "").lower() in ("1", "true", "yes"):
                summarizer = ModelSummarizer(self.client, config.model)
            self.compactor = Compactor(summarizer,
                                       max_tokens=DEFAULT_COMPACTION_TOKENS)
        self.trace: Dict[str, Any] = {
            "run_id": "",
            "agent": config.name,
            "model": config.model,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "depth": depth,
            "steps": [],
            "tool_calls": [],
            "final": None,
            "stop_reason": None,
            "compactions": 0,
        }

    # ---------------- LLM I/O ----------------
    def _chat(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.cfg.model,
            "messages": messages,
            "temperature": self.cfg.temperature,
        }
        if self.tools:
            payload["tools"] = tools_to_openai_schema(self.tools)
            if self.mcp_tools:
                payload["tools"] = payload["tools"] + [
                    {"type": "function",
                     "function": {"name": t["name"],
                                  "description": t["description"],
                                  "parameters": t["parameters"]}}
                    for t in self.mcp_tools]
        model_chain = [self.cfg.model] + [m for m in self.cfg.fallback_models if m != self.cfg.model]
        try:
            if self.stream:
                return self.client.chat(payload, on_delta=self.on_token,
                                        on_tool=self.on_tool_delta,
                                        model_chain=model_chain)
            return self.client.chat(payload, None, model_chain=model_chain)
        except LLMError as e:
            raise AgentRunError(str(e)) from e

    # ---------------- loop ----------------
    def run(self, goal: str, max_turns: Optional[int] = None,
            history: Optional[List[Dict[str, Any]]] = None) -> str:
        """Run the agent loop. If `history` is a list, prior conversation
        turns are prepended (chat mode) and everything from this run is
        appended back into it, so successive calls share context."""
        guard = self.cfg.guardrails
        turns = max_turns or guard.max_turns
        budget = Budget(guard.max_tokens, guard.max_usd).model(self.cfg.model)
        self.trace["run_id"] = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-") + f"{os.getpid()}"

        system = (
            f"{self.cfg.system}\n\nWorkspace: {self.workspace}\n"
            "Loop discipline: (1) restate the goal in one line, (2) act with tools "
            "(issue independent tool calls together — they run in parallel), "
            "(3) verify the result satisfies the goal, (4) reply with the final answer only."
        )
        if self.skills:
            system += skills_index(self.skills)
        todo_block = self.todos.prompt_block()
        if todo_block:
            system += todo_block
        messages: List[Dict[str, Any]] = [{"role": "system", "content": system}]

        # durable memory injection
        mem_path = self.workspace / "memory.md"
        if mem_path.exists():
            notes = mem_path.read_text(encoding="utf-8").strip()
            if notes:
                messages.append({"role": "system", "content": f"Durable memory from previous runs:\n{notes}"})

        if history:
            # chat mode: replay prior turns between system preamble and the new goal
            messages.extend(json.loads(json.dumps(history)))  # deep copy
        messages.append({"role": "user", "content": goal})
        if history is not None:
            # record this run's turns incrementally (robust to compaction
            # reassigning `messages` mid-loop)
            history.append({"role": "user", "content": goal})
        final_text = ""

        # run_start hooks: output becomes part of the model's context
        hook_out = self.hooks.fire("run_start", env_extra={"SMERKIL_WORKSPACE": str(self.workspace)})
        if hook_out:
            messages.append({"role": "system", "content": f"[hooks/run_start]\n{hook_out}"})

        for turn in range(1, turns + 1):
            if budget.exhausted() or (
                    self.session_budget is not None
                    and self.session_budget.exhausted()):
                self.trace["stop_reason"] = "budget_exhausted"
                break

            messages = self.compactor.maybe_compact(messages)

            resp = self._chat(messages)
            usage = resp.get("usage", {}) or {}
            served_by = resp.get("model", self.cfg.model)
            budget.record(usage.get("prompt_tokens", 0) or 0,
                          usage.get("completion_tokens", 0) or 0,
                          model=served_by)
            if self.session_budget is not None:
                # chat mode: accumulate across runs (shared budget object)
                self.session_budget.record(usage.get("prompt_tokens", 0) or 0,
                                           usage.get("completion_tokens", 0) or 0,
                                           model=served_by)

            choice = resp["choices"][0]
            msg = choice["message"]
            content = msg.get("content") or ""
            calls = msg.get("tool_calls") or []
            finish = choice.get("finish_reason")

            self.trace["steps"].append({
                "turn": turn,
                "model": resp.get("model", self.cfg.model),
                "content": str(content)[:2000],
                "tool_calls": [
                    {"id": c.get("id", ""), "name": c["function"]["name"],
                     "args": str(c["function"].get("arguments", ""))[:500]}
                    for c in calls
                ],
                "tokens": {"prompt": usage.get("prompt_tokens", 0) or 0,
                           "completion": usage.get("completion_tokens", 0) or 0},
                "finish_reason": finish,
            })

            # A length-capped response is NOT a valid stopping point: the text
            # is truncated mid-thought and any tool-call JSON may be broken.
            # Inject a continuation nudge; bail with a clear stop_reason if it
            # happens repeatedly (guards against an infinite loop).
            if finish == "length":
                if self.trace.get("_length_nudges", 0) >= 2:
                    final_text = content or ""
                    self.trace["stop_reason"] = "max_output_tokens"
                    break
                self.trace["_length_nudges"] = self.trace.get("_length_nudges", 0) + 1
                messages.append(msg)
                messages.append({"role": "user", "content":
                    "[system notice] Your previous reply was cut off by the output "
                    "token limit. Continue exactly where you stopped; if you were "
                    "writing tool-call arguments, re-issue the complete tool call."})
                continue

            if not calls:
                # append the final assistant message so the transcript (and
                # chat history) contains it — previously it was dropped
                messages.append(msg)
                if history is not None:
                    history.append({"role": "assistant", "content": content})
                final_text = content
                self.trace["stop_reason"] = "completed"
                break

            messages.append(msg)
            if history is not None:
                # intermediate assistant turn with its tool calls (tool results
                # are appended right after, below)
                history.append(json.loads(json.dumps(msg)))
            # results appended in deterministic (call) order
            triplets = [(c.get("id", f"call_{i}"), c["function"]["name"],
                         str(c["function"].get("arguments", "")))
                        for i, c in enumerate(calls)]

            # tool_before hooks act as a GATE: non-zero exit blocks the call
            blocked: Dict[str, str] = {}
            for _cid, _name, _args in triplets:
                gate = self.hooks.fire("tool_before", tool_name=_name,
                                       env_extra={"SMERKIL_WORKSPACE": str(self.workspace),
                                                  "SMERKIL_TOOL_ARGS": _args[:500]})
                if gate is not None:
                    # convention: hook output starting with "BLOCK" denies the call
                    if gate.strip().upper().startswith("BLOCK"):
                        blocked[_cid] = f"blocked by hook: {gate.strip()[5:].strip() or 'denied'}"
            # split MCP calls from builtin calls
            mcp_calls = [(cid, n, a) for cid, n, a in triplets
                         if cid not in blocked and n.startswith("mcp__")]
            builtin_calls = [(cid, n, a) for cid, n, a in triplets
                             if cid not in blocked and not n.startswith("mcp__")]
            results = execute_tools_parallel(self.tools, builtin_calls)
            for cid, n, a in mcp_calls:
                try:
                    results.append((cid, self.mcp.call(n, json.loads(a or "{}"))))
                except (json.JSONDecodeError,) as e:
                    results.append((cid, json.dumps({"error": f"bad args JSON: {e}"})))
                except Exception as e:
                    results.append((cid, json.dumps({"error": f"{type(e).__name__}: {e}"})))
            for call_id, name, args in triplets:
                if call_id in blocked:
                    result = json.dumps({"error": blocked[call_id]})
                else:
                    result = next(r for cid, r in results if cid == call_id)
                self.hooks.fire("tool_after", tool_name=name,
                                env_extra={"SMERKIL_WORKSPACE": str(self.workspace),
                                           "SMERKIL_RESULT": str(result)[:1000]})
                self.trace["tool_calls"].append({
                    "turn": turn, "name": name, "args": args[:500],
                    "result": str(result)[:2000],
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": result,
                })
                if history is not None:
                    history.append({"role": "tool", "tool_call_id": call_id,
                                    "content": result})
            self.hooks.fire("turn_end", env_extra={"SMERKIL_WORKSPACE": str(self.workspace),
                                                   "SMERKIL_TURN": str(turn)})
        else:
            self.trace["stop_reason"] = "max_turns_reached"

        self.trace["final"] = str(final_text)[:4000]
        self.trace["budget"] = {
            "prompt_tokens": budget.prompt_tokens,
            "completion_tokens": budget.completion_tokens,
            "usd": round(budget.usd_spent, 6),
        }
        self.trace["compactions"] = self.compactor.compactions
        self.trace["finished_at"] = datetime.now(timezone.utc).isoformat()
        # chat history is maintained incrementally inside the loop (user goal,
        # assistant turns, tool results) — robust to compaction reassigning
        # `messages`; nothing to do here
        self.trace["totals"] = {
            "turns": len(self.trace["steps"]),
            "tool_calls": len(self.trace["tool_calls"]),
            "prompt_tokens": budget.prompt_tokens,
            "completion_tokens": budget.completion_tokens,
        }
        if self.depth == 0:
            self._save_trace()
        try:
            self.mcp.close()
        except Exception:
            pass
        self.hooks.fire("run_end", env_extra={
            "SMERKIL_WORKSPACE": str(self.workspace),
            "SMERKIL_STOP_REASON": str(self.trace["stop_reason"]),
            "SMERKIL_FINAL": str(final_text)[:1000],
        })
        return str(final_text) or f"[stopped: {self.trace['stop_reason']}]"

    # ---------------- trace persistence (JSONL) ----------------
    def _save_trace(self) -> Path:
        traces = self.workspace / "traces"
        traces.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        jsonl = traces / "runs.jsonl"
        with jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(self.trace, ensure_ascii=False) + "\n")
        # also write per-run JSON for easy replay/back-compat with 0.2 listers
        path = traces / f"run-{stamp}.json"
        path.write_text(json.dumps(self.trace, indent=2, ensure_ascii=False), encoding="utf-8")
        return path
