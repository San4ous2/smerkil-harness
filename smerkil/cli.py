"""smerkil CLI: new / run / eval / trace / chat / doctor. Colored ANSI output."""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

from . import __version__
from .agent import Agent
from .config import AgentConfig, load_config_text
from .mcp import load_mcp_config

# ---------------- ANSI colors ----------------
_IS_TTY = sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _IS_TTY else text


def bold(t: str) -> str:
    return _c("1", t)


def green(t: str) -> str:
    return _c("32", t)


def red(t: str) -> str:
    return _c("31", t)


def yellow(t: str) -> str:
    return _c("33", t)


def cyan(t: str) -> str:
    return _c("36", t)


def dim(t: str) -> str:
    return _c("2", t)


TEMPLATE_AGENT_YAML = """# SMERKIL agent definition
name: {name}
model: gpt-4o-mini
# fallback_models tried in order when the primary fails definitively
# fallback_models: [gpt-4o-mini, claude-3-5-haiku-latest]
# base_url: https://api.openai.com/v1     # any OpenAI-compatible endpoint
# api_key_env: SMERKIL_API_KEY
system: >
  You are {name}, an autonomous agent. Plan briefly, use tools, verify, answer.
tools: [read_file, write_file, list_dir, memory_note, plan_add, plan_list, plan_update, python_exec, web_fetch, git_status, git_log, subagent]
guardrails:
  max_turns: 40
  budget:
    tokens: 200000
    usd: 1.00
  sandbox: strict
  allow_shell: false
memory:
  episodic: memory.md
"""

# Role presets: `smerkil new coder --role coder` fills system + tools.
ROLE_PRESETS: Dict[str, Dict[str, Any]] = {
    "general": {},  # plain TEMPLATE_AGENT_YAML
    "coder": {
        "system": (
            "You are {name}, a senior software engineer. Prefer small, verifiable "
            "steps: write code, run it, fix failures. Always run/verify before "
            "claiming done. Never invent output."
        ),
        "tools": ["read_file", "write_file", "list_dir", "grep", "glob",
                  "python_exec", "run_shell", "plan_add", "plan_list",
                  "plan_update"],
        "allow_shell": True,
    },
    "researcher": {
        "system": (
            "You are {name}, a careful researcher. Gather sources first "
            "(web_fetch, files), cross-check claims, cite where each fact came "
            "from. Distinguish verified facts from inference in every answer."
        ),
        "tools": ["read_file", "write_file", "list_dir", "web_fetch",
                  "memory_note", "grep", "plan_add", "plan_list"],
    },
    "reviewer": {
        "system": (
            "You are {name}, a code reviewer. Read the code before judging. "
            "Report findings by severity (blocker / major / nit) with file:line "
            "references. Verify claims by running code where possible."
        ),
        "tools": ["read_file", "list_dir", "grep", "glob", "python_exec",
                  "run_shell", "git_status", "git_log"],
        "allow_shell": True,
    },
    "writer": {
        "system": (
            "You are {name}, a precise technical writer. No filler, no "
            "adjectives without evidence. Match the requested format exactly; "
            "verify every factual claim against sources in the workspace."
        ),
        "tools": ["read_file", "write_file", "list_dir", "web_fetch",
                  "memory_note"],
    },
}

TEMPLATE_EVALS_YAML = """# Eval scenarios — run with: smerkil eval {name}
scenarios:
  - name: greets and explains itself
    goal: Introduce yourself in one sentence.
    # terms cover every role preset (coder/researcher/reviewer/writer/general)
    expect:
      contains_any: ["agent", "help", "engineer", "researcher", "reviewer",
                     "writer", "assistant"]
      min_chars: 20
  - name: can write a file
    goal: Create a file note.txt containing the word SMERKIL, then confirm.
    expect:
      contains_all: ["note.txt"]
  - name: plans tasks with planning tools
    goal: >-
      Create a plan with exactly two tasks: 'first' and 'second'
      (plan_add or todo_write), mark the first done, then report the plan
      contents.
    # accept either planning tool's reporting style ("done" or [x] checkbox)
    expect:
      contains_all: ["first", "second"]
      contains_any: ["done", "[x]"]
  - name: parallel tool calls
    goal: >
      In ONE turn, issue three list_dir tool calls at the same time (they run in
      parallel), then confirm all three completed and name the tool you used.
    expect:
      contains_any: ["list_dir", "parallel"]
"""


def _agent_dir(name: str) -> Path:
    return Path.cwd() / name


def cmd_new(args: argparse.Namespace) -> int:
    d = _agent_dir(args.name)
    if d.exists() and any(d.iterdir()):
        print(red(f"error: '{args.name}' already exists and is not empty"), file=sys.stderr)
        return 1
    role = getattr(args, "role", None) or "general"
    if role not in ROLE_PRESETS:
        print(red(f"error: unknown role '{role}' (available: {', '.join(sorted(ROLE_PRESETS))})"), file=sys.stderr)
        return 1
    preset = ROLE_PRESETS[role]
    (d / "workspace").mkdir(parents=True)
    if preset:
        yaml_text = TEMPLATE_AGENT_YAML.format(name=args.name)
        if preset.get("system"):
            yaml_text = yaml_text.replace(
                "  You are {0}, an autonomous agent. Plan briefly, use tools, verify, answer.".format(args.name),
                "  " + preset["system"].format(name=args.name))
            # fallback: system line may span; do a robust regex replace instead
            import re as _re
            yaml_text = _re.sub(
                r"system: >\n  .*\n",
                "system: >\n  " + preset["system"].format(name=args.name) + "\n",
                yaml_text, count=1)
        if preset.get("tools"):
            yaml_text = _re.sub(r"^tools: \[.*\]$",
                                "tools: [" + ", ".join(preset["tools"]) + "]",
                                yaml_text, count=1, flags=_re.M)
        if preset.get("allow_shell"):
            yaml_text = yaml_text.replace("allow_shell: false", "allow_shell: true")
        (d / "agent.yaml").write_text(yaml_text, encoding="utf-8")
    else:
        (d / "agent.yaml").write_text(TEMPLATE_AGENT_YAML.format(name=args.name), encoding="utf-8")
    (d / "evals.yaml").write_text(TEMPLATE_EVALS_YAML.format(name=args.name), encoding="utf-8")
    (d / "workspace" / "memory.md").write_text("", encoding="utf-8")
    print(green("✓") + f" agent '{args.name}' scaffolded at {d}" +
          (f" (role: {role})" if role != "general" else ""))
    print(dim("  next: edit agent.yaml, export SMERKIL_API_KEY, then: smerkil run " + args.name + " --goal \"...\""))
    return 0


def _load(name: str) -> "tuple[Path, AgentConfig]":
    d = _agent_dir(name)
    cfg_path = d / "agent.yaml"
    if not cfg_path.exists():
        raise SystemExit(red(f"error: no agent.yaml in {d} (run: smerkil new {name})"))
    data = load_config_text(cfg_path.read_text(encoding="utf-8"))
    return d, AgentConfig.from_dict(data, name, d)


def cmd_run(args: argparse.Namespace) -> int:
    d, cfg = _load(args.name)
    agent = Agent(cfg, d / "workspace", on_token=None)
    result = agent.run(args.goal, max_turns=args.max_turns)
    print(result)
    if args.json:
        print(json.dumps(agent.trace.get("budget", {}), indent=2))
    return 0 if agent.trace.get("stop_reason") == "completed" else 2


def cmd_eval(args: argparse.Namespace) -> int:
    from .evals import run_evals
    d, cfg = _load(args.name)
    agent = Agent(cfg, d / "workspace")
    return run_evals(agent, d, verbose=args.verbose)


def cmd_trace(args: argparse.Namespace) -> int:
    traces = _agent_dir(args.name) / "workspace" / "traces"
    if args.show:
        if not traces.exists():
            print(red("no traces yet"))
            return 1
        return _trace_show(traces, args.show)
    if not traces.exists():
        print("no traces yet")
        return 0
    files = sorted(traces.glob("run-*.json"))
    for f in files[-20:]:
        data = json.loads(f.read_text(encoding="utf-8"))
        budget = data.get("budget", {})
        reason = data.get("stop_reason") or "?"
        color = green(reason) if reason == "completed" else yellow(reason)
        print(f"{f.name}  {color:18}  {budget.get('prompt_tokens', 0)}p/"
              f"{budget.get('completion_tokens', 0)}c  ${budget.get('usd', 0)}")
    return 0


def _trace_show(traces: Path, show: str) -> int:
    path = traces / show if show.endswith(".json") else traces / f"run-{show}.json"
    if not path.exists():
        matches = [f for f in sorted(traces.glob("run-*.json")) if show in f.name]
        path = matches[-1] if matches else path
    if not path.exists():
        print(red(f"trace not found: {show}"))
        return 1
    data = json.loads(path.read_text(encoding="utf-8"))
    print(bold(f"Trace {data.get('run_id') or path.stem}"))
    print(dim(f"  agent: {data.get('agent')}   model: {data.get('model')}   "
              f"depth: {data.get('depth', 0)}"))
    print(dim(f"  started: {data.get('started_at')}   finished: {data.get('finished_at')}"))
    print(dim(f"  stop_reason: {data.get('stop_reason')}   compactions: {data.get('compactions', 0)}"))
    totals = data.get("totals") or {}
    b = data.get("budget", {})
    print(f"  tokens: {b.get('prompt_tokens', 0)} prompt / {b.get('completion_tokens', 0)} completion"
          f"   turns: {totals.get('turns', len(data.get('steps', [])))}"
          f"   tool_calls: {totals.get('tool_calls', len(data.get('tool_calls', [])))}"
          f"   cost: ${b.get('usd', 0)}")
    print(bold("\nSteps:"))
    for step in data.get("steps", []):
        tc_names = ", ".join(c["name"] for c in step.get("tool_calls", [])) or "-"
        print(f"  turn {step['turn']:>2}  {dim('tokens ' + str(step.get('tokens', {}).get('prompt', 0)) + 'p')}"
              f"  tools: {cyan(tc_names)}")
        content = (step.get("content") or "").strip()
        if content:
            for line in content.splitlines()[:3]:
                print(dim(f"        | {line[:110]}"))
    print(bold("\nTool calls:"))
    for tc in data.get("tool_calls", []):
        res = str(tc.get("result", "")).replace("\n", " ")[:100]
        status = red("error") if res.startswith('{"error"') else green("ok")
        print(f"  turn {tc['turn']:>2}  {cyan(tc['name']):14} {status}  {dim(res)}")
    print(bold("\nFinal answer:"))
    final = data.get("final") or "(none)"
    for line in final.splitlines()[:20]:
        print(f"  {line}")
    return 0


# ---------------- chat (interactive REPL) ----------------
def cmd_chat(args: argparse.Namespace) -> int:
    d, cfg = _load(args.name)
    print(bold(f"smerkil chat — {cfg.name} ({cfg.model})"))
    print(dim("Type a message; /exit or Ctrl-D to quit, /reset to clear context, "
              "/tokens to show budget.\n"))
    workspace = d / "workspace"

    tool_stream = {"open": False}  # is a tool-call header currently printed?

    def on_delta(piece: str) -> None:
        if tool_stream["open"]:
            sys.stdout.write("\n")
            tool_stream["open"] = False
        sys.stdout.write(piece)
        sys.stdout.flush()

    def on_tool(name: str, args_fragment: str) -> None:
        # live tool-call stream: `⚙ name(args…` grows as arguments arrive;
        # on_delta closes the line before printing content tokens
        if not tool_stream["open"]:
            sys.stdout.write(dim(f"  ⚙ {name}("))
            tool_stream["open"] = True
        sys.stdout.write(dim(args_fragment))
        sys.stdout.flush()

    def on_shell_line(line: str) -> None:
        sys.stdout.write(dim(f"  | {line}"))
        sys.stdout.flush()

    agent = Agent(cfg, workspace, on_token=on_delta if _IS_TTY else None,
                  on_tool_delta=on_tool if _IS_TTY else None)
    # session budget: accumulate across chat turns (guardrails from agent.yaml)
    from .budget import Budget
    agent.session_budget = Budget(cfg.guardrails.max_tokens,
                                  cfg.guardrails.max_usd).model(cfg.model)
    # live shell output: wire the runner callback into the run_shell tool
    shell_tool = agent.tools.get("run_shell")
    if shell_tool is not None and _IS_TTY:
        shell_tool.on_line = on_shell_line
    history: list = []  # full API transcript carried across chat turns
    try:
        while True:
            try:
                user = input(green("you> ") if _IS_TTY else "you> ").strip()
            except EOFError:
                break
            if not user:
                continue
            if user in ("/exit", "/quit"):
                break
            if user == "/reset":
                history.clear()
                print(dim("context cleared"))
                continue
            if user == "/tokens":
                sb = agent.session_budget
                if sb is not None:
                    print(dim(f"  session: prompt={sb.prompt_tokens} "
                              f"completion={sb.completion_tokens} "
                              f"usd={sb.usd_spent:.4f} "
                              f"(limit {sb.max_tokens} tok / ${sb.max_usd})"))
                b = agent.trace.get("budget", {})
                print(dim(f"  last run: prompt={b.get('prompt_tokens', 0)} "
                          f"completion={b.get('completion_tokens', 0)}"))
                continue
            if _IS_TTY:
                print()
            result = agent.run(user, max_turns=args.max_turns, history=history)
            history.append({"role": "assistant", "content": result})
            if not _IS_TTY:
                print(result)
            else:
                print()  # streaming already printed deltas; newline after
    except KeyboardInterrupt:
        print(dim("\ninterrupted"))
    return 0


# ---------------- serve (HTTP API) ----------------
def cmd_serve(args: argparse.Namespace) -> int:
    from .serve import make_server
    d, cfg = _load(args.name)
    token = args.token or os.environ.get("SMERKIL_SERVE_TOKEN", "")
    httpd = make_server(cfg, d / "workspace", host=args.host, port=args.port,
                        token=token)
    port = httpd.server_address[1]
    print(bold(f"smerkil serve — {cfg.name} ({cfg.model})"))
    print(f"  listening on http://{args.host}:{port}/v1/chat")
    print(dim(f"  auth: {'bearer token' if token else 'open (no token set)'}   "
              f"sessions: /sessions   health: /health"))
    print(dim("Ctrl-C to stop"))
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print(dim("\nserver stopped"))
    finally:
        httpd.server_close()
    return 0


# ---------------- doctor ----------------
def cmd_doctor(args: argparse.Namespace) -> int:
    ok = True
    print(bold("smerkil doctor"))
    print(f"  version:   {green(__version__)}")
    print(f"  python:    {sys.version.split()[0]}")

    # env vars
    name = args.name
    key_env = "SMERKIL_API_KEY"
    base_url = os.environ.get("SMERKIL_BASE_URL", "")
    model = os.environ.get("SMERKIL_MODEL", "")
    if name:
        try:
            d, cfg = _load(name)
            key_env = cfg.api_key_env
            base_url = cfg.base_url or base_url
            model = cfg.model
            print(f"  agent:     {green(cfg.name)} at {d}")
            print(f"  tools:     {', '.join(cfg.tools)}")
            if cfg.fallback_models:
                print(f"  fallbacks: {', '.join(cfg.fallback_models)}")
            else:
                print(yellow("  fallbacks: none (single-model runs fail if the model fails)"))
        except SystemExit:
            print(yellow(f"  agent:     '{name}' not found — checking globals only"))
            ok = False

    # MCP servers in the current directory's mcp.json
    mcp_cfg = load_mcp_config(Path.cwd())
    if mcp_cfg:
        print(f"  mcp:       {len(mcp_cfg)} server(s) in ./mcp.json "
              f"({', '.join(str(s.get('name', '?')) for s in mcp_cfg)})")
    else:
        print("  mcp:       none (add mcp.json to attach MCP tools)")

    has_key = bool(os.environ.get(key_env))
    print(f"  api key:   {green(f'{key_env} set') if has_key else red(f'{key_env} NOT set')}")

    # model reachability
    if has_key and base_url:
        try:
            req = urllib.request.Request(
                base_url.rstrip("/") + "/models",
                headers={"Authorization": f"Bearer {os.environ[key_env]}"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                body = json.loads(resp.read().decode())
                n = len(body.get("data", []))
            print(f"  endpoint:  {green('reachable')} — {base_url} ({n} models listed)")
            if model:
                ids = {m.get("id") for m in body.get("data", [])}
                if model in ids:
                    print(f"  model:     {green(model + ' available')}")
                else:
                    print(yellow(f"  model:     '{model}' not in endpoint list (may still work)"))
        except Exception as e:
            print(red(f"  endpoint:  unreachable — {e}"))
            ok = False
    else:
        print(yellow("  endpoint:  skipped (set key + base_url)"))

    # workspace health
    if name:
        ws = Path.cwd() / name / "workspace"
        checks = {
            "memory.md exists": (ws / "memory.md").exists(),
            "plan.md present or creatable": (ws / "plan.md").exists() or ws.exists(),
            "traces dir": (ws / "traces").exists() or ws.exists(),
        }
        for label, passed in checks.items():
            print(f"  workspace: {green(label) if passed else yellow(label + ' (missing, will be created on first run)')}")

    print(green("\nall checks passed") if ok else yellow("\ncompleted with warnings"))
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(prog="smerkil", description="SMERKIL Agent harness")
    parser.add_argument("--version", action="version", version=f"smerkil {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_new = sub.add_parser("new", help="scaffold a new agent")
    p_new.add_argument("name")
    p_new.add_argument("--role", default=None,
                       choices=sorted(ROLE_PRESETS),
                       help="role preset: fills system prompt + tools (default: general)")
    p_new.set_defaults(fn=cmd_new)

    p_run = sub.add_parser("run", help="run an agent toward a goal")
    p_run.add_argument("name")
    p_run.add_argument("--goal", required=True)
    p_run.add_argument("--max-turns", type=int, default=None)
    p_run.add_argument("--json", action="store_true", help="also print budget JSON")
    p_run.set_defaults(fn=cmd_run)

    p_eval = sub.add_parser("eval", help="run eval scenarios (deploy gate)")
    p_eval.add_argument("name")
    p_eval.add_argument("-v", "--verbose", action="store_true")
    p_eval.set_defaults(fn=cmd_eval)

    p_trace = sub.add_parser("trace", help="list or show run traces")
    p_trace.add_argument("name")
    p_trace.add_argument("--show", default=None, help="pretty-print a specific trace (id or file)")
    p_trace.set_defaults(fn=cmd_trace)

    p_chat = sub.add_parser("chat", help="interactive chat with an agent (streaming)")
    p_chat.add_argument("name")
    p_chat.add_argument("--max-turns", type=int, default=8)
    p_chat.set_defaults(fn=cmd_chat)

    p_doc = sub.add_parser("doctor", help="check env vars, model reachability, workspace health")
    p_doc.add_argument("name", nargs="?", default=None)
    p_doc.set_defaults(fn=cmd_doctor)

    p_serve = sub.add_parser("serve", help="serve an agent over HTTP API")
    p_serve.add_argument("name")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8317)
    p_serve.add_argument("--token", default=None,
                         help="require 'Authorization: Bearer <token>' (or set SMERKIL_SERVE_TOKEN)")
    p_serve.set_defaults(fn=cmd_serve)

    args = parser.parse_args()
    try:
        return args.fn(args)
    except KeyboardInterrupt:
        print("\ninterrupted")
        return 130


if __name__ == "__main__":
    sys.exit(main())
