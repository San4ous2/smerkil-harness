"""smerkil CLI: new / run / eval / trace / chat / doctor. Colored ANSI output."""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional

from . import __version__
from .agent import Agent
from .config import AgentConfig, load_config_text
from .memory import MemoryStore
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
tools: [read_file, write_file, list_dir, memory_note, plan_add, plan_list, plan_update, python_exec, web_fetch, git_status, git_log, git_diff, git_commit, git_branch, subagent]
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
                  "plan_update", "git_status", "git_log", "git_diff",
                  "git_commit", "git_branch"],
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
                  "run_shell", "git_status", "git_log", "git_diff"],
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


def cmd_memory(args: argparse.Namespace) -> int:
    """Inspect/search/clear an agent's durable memory (memory.jsonl)."""
    d, _cfg = _load(args.name)
    store = MemoryStore(d / "workspace")
    if args.forget is not None:
        # --forget VALUE: exact id if a stored entry has it, else treat as query
        ids = {e["id"] for e in store.recent(10_000)}
        if args.forget in ids:
            n = store.forget(entry_id=args.forget)
        else:
            n = store.forget(query=args.forget)
        print(f"forgotten {n} note(s)")
        return 0
    if args.search is not None:
        hits = store.search(args.search, limit=args.limit)
        if not hits:
            print(dim("(no matching memories)"))
            return 0
        for e in hits:
            tags = f"  #{' #'.join(e['tags'])}" if e.get("tags") else ""
            print(f"{e['id']}  {e['ts'][:10]}  {e['text']}{tags}")
        return 0
    entries = store.recent(args.limit)
    if not entries:
        print(dim("(memory is empty — the agent saves notes via memory_save)"))
        return 0
    print(bold(f"durable memory — {args.name} ({len(entries)} shown)"))
    for e in reversed(entries):  # newest first
        tags = f"  #{' #'.join(e['tags'])}" if e.get("tags") else ""
        print(f"{e['id']}  {e['ts'][:10]}  {e['text']}{tags}")
    return 0


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


def cmd_replay(args: argparse.Namespace) -> int:
    """Replay a saved trace: turn-by-turn model output, tool calls, tokens.

    Reads the per-run JSON or runs.jsonl produced by `smerkil run` and prints
    what happened, in order. `--tools-only` skips model prose, `--json` emits
    the raw trace, `-q/--quiet` prints just the final answer.
    """
    d = _agent_dir(args.name)
    traces = _trace_dir(d)
    if traces is None:
        print(red("no traces yet for this agent"))
        return 1

    # locate the run: explicit id, or most recent
    target = None
    if args.run:
        rid = args.run.replace("run-", "").replace(".json", "")
        for p in sorted(traces.glob("run-*.json")):
            if rid in p.name:
                target = p
                break
        if target is None:
            # maybe it's in runs.jsonl
            rec = _find_run_in_jsonl(traces, args.run)
            if rec is None:
                print(red(f"no trace matching '{args.run}'"))
                return 1
            trace = rec
        else:
            trace = json.loads(target.read_text(encoding="utf-8"))
    else:
        files = sorted(traces.glob("run-*.json"))
        if not files:
            print(red("no traces yet"))
            return 1
        target = files[-1]
        trace = json.loads(target.read_text(encoding="utf-8"))

    if args.json:
        print(json.dumps(trace, indent=2))
        return 0

    if args.quiet:
        print(trace.get("final") or "")
        return 0

    steps = trace.get("steps") or []
    tool_calls = trace.get("tool_calls") or []
    budget = trace.get("budget") or {}
    totals = trace.get("totals") or {}

    print(bold(f"replay {trace.get('run_id') or (target.name if target else 'run')}"))
    print(dim(f"  agent {trace.get('agent','?')} · model {trace.get('model','?')}"))
    if trace.get("started_at"):
        print(dim(f"  started {trace['started_at']}"))
    print()

    if not steps and not tool_calls:
        print(dim("  (empty trace — no recorded turns)"))
    # merge by turn so tools appear where they happened
    by_turn = {}
    for tc in tool_calls:
        by_turn.setdefault(tc.get("turn", 0), []).append(tc)

    for st in steps:
        turn = st.get("turn", 0)
        toks = st.get("tokens") or {}
        head = f"  turn {turn}"
        p, c = toks.get("prompt", 0), toks.get("completion", 0)
        if p or c:
            head += f"  {p}p/{c}c"
        if st.get("finish_reason"):
            head += f"  {st['finish_reason']}"
        print(dim(head))
        content = (st.get("content") or "").strip()
        if content and not args.tools_only:
            for line in content.splitlines():
                print(f"    {line}")
        for tc in by_turn.get(turn, []):
            name = tc.get("name", "?")
            args_s = (tc.get("args") or "").strip()
            if len(args_s) > 160:
                args_s = args_s[:160] + "…"
            print(f"    {yellow('→')} {green(name)}({args_s})")
            res = (tc.get("result") or "").strip()
            if args.verbose and res:
                for line in res.splitlines()[:8]:
                    print(dim(f"      {line}"))
        print()

    # anything not in steps (shouldn't happen, but don't lose data)
    seen = set()
    for st in steps:
        seen.add(st.get("turn", 0))
    orphan = [t for t, v in by_turn.items() if t not in seen]
    for t in sorted(orphan):
        print(dim(f"  turn {t}"))
        for tc in by_turn[t]:
            print(f"    {yellow('→')} {green(tc.get('name','?'))}")

    final = (trace.get("final") or "").strip()
    if final and not args.tools_only:
        print(bold("final"))
        for line in final.splitlines():
            print(f"  {line}")
        print()

    reason = trace.get("stop_reason") or "?"
    rcolor = green(reason) if reason == "completed" else yellow(reason)
    print(f"  stop {rcolor}")
    print(dim(f"  turns {totals.get('turns', len(steps))} · "
              f"tool calls {totals.get('tool_calls', len(tool_calls))} · "
              f"{budget.get('prompt_tokens',0)}p/{budget.get('completion_tokens',0)}c · "
              f"${budget.get('usd',0):.4f}"))
    if trace.get("compactions"):
        print(dim(f"  compactions {trace['compactions']}"))
    return 0


def _find_run_in_jsonl(traces: "Path", run_id: str):
    """Look up a run inside runs.jsonl by run_id prefix."""
    f = traces / "runs.jsonl"
    if not f.exists():
        return None
    needle = run_id.replace("run-", "").replace(".json", "")
    for line in f.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        if needle in str(rec.get("run_id", "")):
            return rec
    return None


def cmd_push(args: argparse.Namespace) -> int:
    """Push this machine's usage report to a SMERKIL dashboard.

    Entirely opt-in and explicit: nothing is sent unless the user runs this
    with a target URL. The bearer token is read from a file or env var, never
    accepted on the command line (argv is world-readable via /proc).
    """
    import urllib.error
    import urllib.request

    url = args.url or os.environ.get("SMERKIL_DASHBOARD_URL") or load_dashboard_session().get("url")
    if not url:
        print(red("error: no dashboard URL (smerkil login, --url https://…, or $SMERKIL_DASHBOARD_URL)"))
        return 2
    if not url.startswith("https://") and not url.startswith("http://"):
        print(red("error: URL must start with http:// or https://"))
        return 2

    # token: --token-file > env; never a bare --token flag
    token = os.environ.get("SMERKIL_DASHBOARD_TOKEN", "")
    if args.token_file:
        try:
            token = Path(args.token_file).read_text(encoding="utf-8").strip()
        except OSError as e:
            print(red(f"error: cannot read token file: {e}"))
            return 2
    if not token:
        token = load_dashboard_session().get("token", "")
    if not token:
        print(red("error: no dashboard token (smerkil login, --token-file FILE, or $SMERKIL_DASHBOARD_TOKEN)"))
        return 2

    # build the same payload `smerkil report --json` prints
    import contextlib
    import io

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = cmd_report(_ArgsProxy(name=args.name, json=True, all=True, limit=args.limit))
    if rc != 0:
        return rc
    payload = buf.getvalue().encode("utf-8")

    endpoint = url.rstrip("/") + "/api/usage"
    req = urllib.request.Request(
        endpoint,
        data=payload,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
            "User-Agent": f"smerkil/{__version__}",
        },
    )
    print(f"pushing {len(payload)} bytes → {dim(endpoint)}")
    try:
        with urllib.request.urlopen(req, timeout=args.timeout) as resp:
            body = resp.read().decode("utf-8", "replace")
            if 200 <= resp.status < 300:
                print(green(f"ok ({resp.status})"))
                try:
                    info = json.loads(body)
                    if isinstance(info, dict):
                        print(dim(f"  agents {info.get('agents')}  history {info.get('history')}"))
                except Exception:
                    pass
                return 0
            print(red(f"failed ({resp.status}): {body[:300]}"))
            return 1
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:300]
        print(red(f"failed ({e.code}): {detail}"))
        return 1
    except Exception as e:
        print(red(f"error: {e}"))
        return 1


def cmd_status(args: argparse.Namespace) -> int:
    """Live status of a running `smerkil serve`.

    Finds the server via its heartbeat file, then asks GET /live for what
    sessions exist and which are actually running right now. Exits non-zero
    when nothing is serving, so it composes in scripts.
    """
    import urllib.error
    import urllib.request

    from .serve import read_heartbeat

    hb = read_heartbeat(max_age_s=args.max_age)
    if not hb and not args.url:
        print(dim("no agent serving (start one with: smerkil serve <name>)"))
        return 1

    url = (args.url or hb.get("url") or "").rstrip("/")
    if not url:
        print(red("error: heartbeat has no url; pass --url"))
        return 2
    if not url.startswith(("http://", "https://")):
        print(red("error: URL must start with http:// or https://"))
        return 2

    headers = {"User-Agent": f"smerkil/{__version__}"}
    if args.token:
        headers["Authorization"] = f"Bearer {args.token}"
    try:
        req = urllib.request.Request(url + "/live", headers=headers)
        with urllib.request.urlopen(req, timeout=args.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        print(red(f"server returned {e.code}"))
        return 1
    except Exception as e:
        print(red(f"error: {e}"))
        return 1

    if args.json:
        print(json.dumps({**data, "heartbeat": hb}, indent=2, ensure_ascii=False))
        return 0

    print(bold(f"smerkil status") + dim(f"  {data.get('agent')} · {data.get('model')}"))
    if hb:
        print(dim(f"  pid {hb.get('pid')} · {hb.get('url')} · up since {hb.get('updated_at')}"))
    active = data.get("active_runs", 0)
    sessions = data.get("sessions", [])
    badge = green(f"{active} running") if active else dim("idle")
    print(f"  {badge} · {len(sessions)} session(s)")

    if not sessions:
        print(dim("\n  no sessions yet — POST /v1/chat to create one"))
        return 0

    print()
    for s in sessions:
        if s.get("in_flight"):
            mark = green("● running")
        else:
            mark = dim("○ idle   ")
        stop = s.get("last_stop_reason") or "—"
        stop_s = dim(stop) if stop in ("completed", None, "—") else yellow(stop)
        print(f"  {mark}  {bold(s['session']):20} runs {s['runs']:3}  "
              f"{s['prompt_tokens']}p/{s['completion_tokens']}c  "
              f"${s['usd']}  {stop_s}")
    return 0


def cmd_login(args: argparse.Namespace) -> int:
    """Sign in to a SMERKIL dashboard and remember the session token.

    Prompts for the password without echoing it, then stores the session
    token in ~/.smerkil/dashboard.json (0600). Nothing else is persisted —
    no usage data leaves the machine until you run `smerkil push`.
    """
    import getpass
    import urllib.error
    import urllib.request

    url = (args.url or os.environ.get("SMERKIL_DASHBOARD_URL") or "").rstrip("/")
    if not url:
        print(red("error: no dashboard URL (--url https://… or $SMERKIL_DASHBOARD_URL)"))
        return 2
    if not url.startswith(("http://", "https://")):
        print(red("error: URL must start with http:// or https://"))
        return 2

    email = args.email or input("email: ").strip()
    if not email:
        print(red("error: no email"))
        return 2
    password = args.password or getpass.getpass("password: ")
    if not password:
        print(red("error: no password"))
        return 2

    payload = json.dumps({"email": email, "password": password, "mode": "signin"}).encode()
    req = urllib.request.Request(
        url + "/api/auth",
        data=payload,
        method="POST",
        headers={"Content-Type": "application/json", "User-Agent": f"smerkil/{__version__}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=args.timeout) as resp:
            raw = resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:200]
        print(red(f"login failed ({e.code}): {detail}"))
        return 1
    except Exception as e:
        print(red(f"error: {e}"))
        return 1

    token = ""
    # the API returns the token in the body and/or a Set-Cookie header
    try:
        body = json.loads(raw)
        token = body.get("token") or (body.get("session") or {}).get("token") or ""
    except Exception:
        body = {}
    for hdr in (resp.headers.get_all("Set-Cookie") or []):
        m = re.search(r"smerkil_session=([^;]+)", hdr)
        if m:
            token = m.group(1)
    if not token:
        print(red("error: server did not return a session token"))
        return 1

    # URL-decoded so Authorization headers work verbatim
    token = urllib.parse.unquote(token)
    path = dashboard_token_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "url": url,
        "email": email,
        "token": token,
        "logged_in_at": _utc_now(),
    }, indent=2), encoding="utf-8")
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    print(green(f"signed in as {email}"))
    print(dim(f"  token saved to {path} (0600)"))
    print(dim("  this machine can now push: smerkil push   /   live status via --publish-live"))
    return 0


def cmd_pair(args: argparse.Namespace) -> int:
    """Link this machine to a dashboard with a one-time pairing code.

    No account, no password, no stored credentials on the server side: the
    harness asks /api/pair for a short-lived code, the human claims it in the
    browser at /pair?code=…, and the harness picks up a self-contained signed
    token. Everything after that is Authorization: Bearer <token>.
    """
    import time
    import urllib.error
    import urllib.request

    url = (
        args.url
        or os.environ.get("SMERKIL_DASHBOARD_URL")
        or "https://smerkil-technologies.vercel.app"
    ).rstrip("/")

    try:
        req = urllib.request.Request(
            url + "/api/pair",
            data=b"{}",
            method="POST",
            headers={
                "Content-Type": "application/json",
                "User-Agent": f"smerkil/{__version__}",
            },
        )
        with urllib.request.urlopen(req, timeout=15.0) as resp:
            started = json.loads(resp.read().decode("utf-8", "replace"))
    except Exception as e:
        print(red(f"error: pairing failed: {e}"))
        return 1

    code = started.get("code") or ""
    pair_url = started.get("url") or f"{url}/pair?code={code}"
    if not code:
        print(red("error: dashboard did not return a pairing code"))
        return 1

    print(green("Pairing code: ") + f"  {code}")
    print(dim(f"Open: {pair_url}"))
    print(dim(f"Waiting up to {int(args.timeout)}s for you to claim it… (Ctrl-C to stop)"))

    deadline = time.time() + float(args.timeout)
    poll = 2.0
    token = ""
    try:
        while time.time() < deadline:
            try:
                preq = urllib.request.Request(
                    f"{url}/api/pair?code={urllib.parse.quote(code)}",
                    headers={"User-Agent": f"smerkil/{__version__}"},
                )
                with urllib.request.urlopen(preq, timeout=10.0) as resp:
                    st = json.loads(resp.read().decode("utf-8", "replace"))
                if st.get("token"):
                    token = st["token"]
                    break
                if st.get("error"):
                    print(red(f"error: {st.get('error')}"))
                    return 1
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    print(red("error: pairing code expired — run `smerkil pair` again"))
                    return 1
            except Exception:
                pass  # transient network hiccup: keep waiting
            time.sleep(poll)
    except KeyboardInterrupt:
        print(yellow("\ncancelled"))
        return 130

    if not token:
        print(red("error: timed out waiting for the code to be claimed"))
        return 1

    path = dashboard_token_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "url": url,
        "principal": "pair",
        "token": token,
        "paired_at": _utc_now(),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass

    if args.json:
        print(json.dumps({"ok": True, "url": url, "principal": "pair"}, indent=2))
    else:
        print(green("paired. this machine can now push live status."))
        print(dim(f"  token saved to {path} (0600)"))
        print(dim("  try: smerkil serve <name> --publish-live"))
    return 0


def dashboard_token_path() -> "Path":
    """Where `smerkil login` stores the dashboard session token."""
    from pathlib import Path

    root = Path(os.environ.get("SMERKIL_HOME", Path.home() / ".smerkil"))
    return root / "dashboard.json"


def load_dashboard_session() -> dict:
    """Read the stored dashboard session, or {} if not logged in."""
    try:
        data = json.loads(dashboard_token_path().read_text(encoding="utf-8"))
        if isinstance(data, dict) and data.get("token"):
            return data
    except Exception:
        pass
    return {}


def publish_live_loop(url: str, token: str, interval: float, snapshot_fn,
                      verbose: bool = True) -> "threading.Event":
    """Background-up-push `/live` to a dashboard every `interval` seconds.

    Returns a stop Event; setting it ends the loop after the current tick.
    Failures are silent by design — this must never take down a server, but
    the first failure is reported once so the user isn't left guessing.
    """
    import threading
    import urllib.error
    import urllib.request

    stop = threading.Event()
    warned = [False]

    def _once() -> None:
        snap = json.dumps(snapshot_fn()).encode("utf-8")
        req = urllib.request.Request(
            url.rstrip("/") + "/api/live",
            data=snap,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {token}",
                "User-Agent": f"smerkil/{__version__}",
            },
        )
        with urllib.request.urlopen(req, timeout=6.0) as resp:
            if resp.status >= 300 and not warned[0]:
                warned[0] = True
                if verbose:
                    print(yellow(f"  live publish: {resp.status}"))

    def _loop() -> None:
        while not stop.is_set():
            try:
                _once()
            except urllib.error.HTTPError as e:
                if not warned[0]:
                    warned[0] = True
                    if verbose:
                        detail = ""
                        try:
                            detail = e.read().decode("utf-8", "replace")[:120]
                        except Exception:
                            pass
                        print(yellow(f"  live publish: {e.code} {detail}"))
            except Exception as e:
                if not warned[0]:
                    warned[0] = True
                    if verbose:
                        print(yellow(f"  live publish: {e}"))
            stop.wait(interval)

    threading.Thread(target=_loop, daemon=True).start()
    return stop


class _ArgsProxy:
    """Minimal stand-in for argparse.Namespace when calling cmd_report."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


def cmd_report(args: argparse.Namespace) -> int:
    """Print a machine-readable usage report across all agents (or one).

    Walks ~/.smerkil/agents/*/workspace/traces/runs.jsonl and aggregates
    totals per agent: runs, tokens, cost, tools, stop reasons. With --json
    emits a single JSON document; otherwise a compact colored table.
    This is the payload the SMERKIL dashboard ingests.
    """
    agents_root = Path(os.environ.get("SMERKIL_HOME", Path.home() / ".smerkil")) / "agents"
    names: list[Path] = []
    if getattr(args, "name", None):
        d = _agent_dir(args.name)
        if d.exists():
            names = [d]
    elif agents_root.exists():
        names = [p for p in sorted(agents_root.iterdir()) if p.is_dir()]

    agents_out: list[dict] = []
    for d in names:
        runs = _load_trace_records(d)
        if not runs and not getattr(args, "all", False):
            continue
        agents_out.append(_aggregate_agent(d.name, runs, limit=args.limit))

    payload = {
        "version": __version__,
        "generated_at": _utc_now(),
        "agents": agents_out,
        "totals": _sum_totals(agents_out),
    }
    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0
    _print_report(payload)
    return 0


def _utc_now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def _trace_dir(agent_dir: Path) -> Optional[Path]:
    """Traces live at <agent>/workspace/traces in current scaffolds, but older
    layouts kept them at <agent>/traces. Accept either."""
    for cand in (agent_dir / "workspace" / "traces", agent_dir / "traces"):
        if cand.exists():
            return cand
    return None


def _load_trace_records(agent_dir: Path) -> list[dict]:
    """Merge per-run JSON files and the appended JSONL stream, newest last."""
    traces = _trace_dir(agent_dir)
    if traces is None:
        return []
    seen: dict[str, dict] = {}
    for f in sorted(traces.glob("run-*.json")):
        try:
            rec = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        seen[f.name] = rec
    jsonl = traces / "runs.jsonl"
    if jsonl.exists():
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            run_id = str(rec.get("run_id") or "")
            key = f"run-{run_id}.json" if run_id else f"jsonl-{len(seen)}"
            if key in seen:
                seen[key].update(rec)
            else:
                seen[key] = rec
    return list(seen.values())


def _aggregate_agent(name: str, runs: list[dict], limit: int = 20) -> dict:
    runs = sorted(runs, key=lambda r: str(r.get("finished_at") or r.get("started_at") or ""))
    recent = runs[-limit:]
    tools: dict[str, int] = {}
    stops: dict[str, int] = {}
    prompt = completion = 0
    usd = 0.0
    tool_calls = turns = 0
    for r in runs:
        prompt += int((r.get("budget") or {}).get("prompt_tokens", 0) or 0)
        completion += int((r.get("budget") or {}).get("completion_tokens", 0) or 0)
        usd += float((r.get("budget") or {}).get("usd", 0) or 0)
        tool_calls += int((r.get("totals") or {}).get("tool_calls", 0) or 0)
        turns += int((r.get("totals") or {}).get("turns", 0) or 0)
        stops[str(r.get("stop_reason") or "?")] = stops.get(str(r.get("stop_reason") or "?"), 0) + 1
        for t in r.get("tool_calls") or []:
            n = str(t.get("name") or "?") if isinstance(t, dict) else "?"
            tools[n] = tools.get(n, 0) + 1
    last = recent[-1] if recent else {}
    return {
        "agent": name,
        "runs": len(runs),
        "turns": turns,
        "tool_calls": tool_calls,
        "budget": {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "usd": round(usd, 6),
        },
        "stop_reasons": stops,
        "tools": dict(sorted(tools.items(), key=lambda kv: -kv[1])[:10]),
        "recent": [
            {
                "run_id": r.get("run_id"),
                "stop_reason": r.get("stop_reason"),
                "finished_at": r.get("finished_at"),
                "model": r.get("model"),
                "prompt_tokens": int((r.get("budget") or {}).get("prompt_tokens", 0) or 0),
                "completion_tokens": int((r.get("budget") or {}).get("completion_tokens", 0) or 0),
                "usd": round(float((r.get("budget") or {}).get("usd", 0) or 0), 6),
                "turns": int((r.get("totals") or {}).get("turns", 0) or 0),
                "tool_calls": int((r.get("totals") or {}).get("tool_calls", 0) or 0),
            }
            for r in reversed(recent)
        ],
        "last_run_at": last.get("finished_at") or last.get("started_at"),
    }


def _sum_totals(agents: list[dict]) -> dict:
    return {
        "agents": len(agents),
        "runs": sum(a["runs"] for a in agents),
        "turns": sum(a["turns"] for a in agents),
        "tool_calls": sum(a["tool_calls"] for a in agents),
        "prompt_tokens": sum(a["budget"]["prompt_tokens"] for a in agents),
        "completion_tokens": sum(a["budget"]["completion_tokens"] for a in agents),
        "usd": round(sum(a["budget"]["usd"] for a in agents), 6),
    }


def _print_report(payload: dict) -> None:
    t = payload["totals"]
    print(bold("SMERKIL usage report") + dim(f"  {payload['generated_at']}"))
    print(dim(f"  agents {t['agents']}   runs {t['runs']}   turns {t['turns']}   "
              f"tool calls {t['tool_calls']}"))
    print(f"  tokens: {t['prompt_tokens']} prompt / {t['completion_tokens']} completion"
          f"   cost: ${t['usd']}")
    if not payload["agents"]:
        print(dim("\n  no agents with traces yet"))
        return
    print()
    for a in payload["agents"]:
        reason = max(a["stop_reasons"].items(), key=lambda kv: kv[1])[0] if a["stop_reasons"] else "?"
        color = green(reason) if reason == "completed" else yellow(reason)
        print(f"  {bold(a['agent']):24} runs {a['runs']:3}  "
              f"{a['budget']['prompt_tokens']}p/{a['budget']['completion_tokens']}c  "
              f"${a['budget']['usd']}  {color}")
        if a["tools"]:
            print(dim("    tools: " + ", ".join(f"{k}×{v}" for k, v in a["tools"].items())))


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
    from .serve import make_server, write_heartbeat
    d, cfg = _load(args.name)
    token = args.token or os.environ.get("SMERKIL_SERVE_TOKEN", "")
    httpd = make_server(cfg, d / "workspace", host=args.host, port=args.port,
                        token=token)
    port = httpd.server_address[1]
    url = f"http://{args.host}:{port}"
    # publish reachability so `smerkil status` works without knowing the port
    stop = write_heartbeat(cfg, args.host, port, os.getpid(), url)
    print(bold(f"smerkil serve — {cfg.name} ({cfg.model})"))
    print(f"  listening on {url}/v1/chat")
    print(dim(f"  auth: {'bearer token' if token else 'open (no token set)'}   "
              f"sessions: /sessions   health: /health   live: /live"))
    print(dim(f"  pid {os.getpid()} · status via: smerkil status"))

    # Optional uplink: POST /live up to the dashboard every few seconds, so
    # the hosted Console shows this machine even though no browser can reach
    # localhost. Off unless asked for; failures never kill the server.
    up_url = args.publish_live or os.environ.get("SMERKIL_PUBLISH_LIVE", "")
    up_stop = None
    if up_url:
        sess = load_dashboard_session()
        up_token = args.publish_token or os.environ.get("SMERKIL_DASHBOARD_TOKEN", "") \
            or sess.get("token", "")
        up_url_final = up_url if up_url.startswith(("http://", "https://")) \
            else (sess.get("url", ""))
        if not up_token or not up_url_final:
            print(yellow("  --publish-live ignored: not logged in (run: smerkil login)"))
        else:
            store = httpd.smerkil_store
            up_stop = publish_live_loop(up_url_final, up_token,
                                        args.publish_interval, store.live)
            print(dim(f"  live status → {up_url_final} every {args.publish_interval}s"))

    print(dim("Ctrl-C to stop"))
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print(dim("\nserver stopped"))
    finally:
        stop.set()
        if up_stop:
            up_stop.set()
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

    p_mem = sub.add_parser("memory", help="inspect an agent's durable memory")
    p_mem.add_argument("name")
    p_mem.add_argument("--search", default=None, help="keyword search")
    p_mem.add_argument("--limit", type=int, default=20)
    p_mem.add_argument("--forget", default=None, help="forget by id or --search-style query")
    p_mem.set_defaults(fn=cmd_memory)

    p_doc = sub.add_parser("doctor", help="check env vars, model reachability, workspace health")
    p_doc.add_argument("name", nargs="?", default=None)
    p_doc.set_defaults(fn=cmd_doctor)

    p_rep = sub.add_parser("report", help="aggregate usage across agents (dashboard payload)")
    p_rep.add_argument("name", nargs="?", default=None)
    p_rep.add_argument("--json", action="store_true", help="emit a single JSON document")
    p_rep.add_argument("--all", action="store_true", help="include agents with no traces")
    p_rep.add_argument("--limit", type=int, default=20, help="recent runs per agent")
    p_rep.set_defaults(fn=cmd_report)

    p_push = sub.add_parser("push", help="send this machine's usage report to a dashboard")
    p_push.add_argument("name", nargs="?", default=None)
    p_push.add_argument("--url", default=None, help="dashboard base URL (or $SMERKIL_DASHBOARD_URL)")
    p_push.add_argument("--token-file", default=None,
                        help="file containing the dashboard token (never passed as a flag)")
    p_push.add_argument("--limit", type=int, default=20)
    p_push.add_argument("--timeout", type=float, default=20.0)
    p_push.set_defaults(fn=cmd_push)

    p_replay = sub.add_parser("replay", help="replay a saved trace turn by turn")
    p_replay.add_argument("name")
    p_replay.add_argument("run", nargs="?", default=None,
                          help="run id or file name (default: most recent)")
    p_replay.add_argument("--tools-only", action="store_true",
                          help="show tool calls only, skip model prose")
    p_replay.add_argument("--json", action="store_true", help="emit the raw trace")
    p_replay.add_argument("-q", "--quiet", action="store_true",
                          help="print only the final answer")
    p_replay.add_argument("-v", "--verbose", action="store_true",
                          help="also show tool results")
    p_replay.set_defaults(fn=cmd_replay)

    p_login = sub.add_parser("login", help="sign in to a dashboard and store the session token")
    p_login.add_argument("--url", default=None)
    p_login.add_argument("--email", default=None, help="prompted if omitted")
    p_login.add_argument("--password", default=None,
                         help="prompted without echo if omitted (prefer leaving it out)")
    p_login.add_argument("--timeout", type=float, default=15.0)
    p_login.set_defaults(fn=cmd_login)

    p_pair = sub.add_parser(
        "pair",
        help="link this machine to a dashboard with a one-time code (no account)",
    )
    p_pair.add_argument("--url", default=None,
                        help="dashboard URL (default: $SMERKIL_DASHBOARD_URL)")
    p_pair.add_argument("--timeout", type=float, default=600.0,
                        help="how long to wait for the code to be claimed")
    p_pair.add_argument("--json", action="store_true",
                        help="print the pairing result as JSON")
    p_pair.set_defaults(fn=cmd_pair)

    p_status = sub.add_parser("status", help="live status of a running 'smerkil serve'")
    p_status.add_argument("--url", default=None, help="override the discovered server URL")
    p_status.add_argument("--token", default=None, help="bearer token, if the server was started with one")
    p_status.add_argument("--max-age", type=float, default=20.0,
                          help="ignore a heartbeat older than N seconds")
    p_status.add_argument("--timeout", type=float, default=5.0)
    p_status.add_argument("--json", action="store_true")
    p_status.set_defaults(fn=cmd_status)

    p_serve = sub.add_parser("serve", help="serve an agent over HTTP API")
    p_serve.add_argument("name")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8317)
    p_serve.add_argument("--token", default=None,
                         help="require this bearer token (or $SMERKIL_SERVE_TOKEN)")
    p_serve.add_argument("--publish-live", default=None, nargs="?", const="1",
                         help="push /live snapshots to the logged-in dashboard so the "
                              "hosted Console can see this machine")
    p_serve.add_argument("--publish-token", default=None,
                         help="dashboard token for --publish-live (default: from smerkil login)")
    p_serve.add_argument("--publish-interval", type=float, default=5.0,
                         help="how often to push a live snapshot, in seconds")
    p_serve.set_defaults(fn=cmd_serve)

    args = parser.parse_args()
    try:
        return args.fn(args)
    except KeyboardInterrupt:
        print("\ninterrupted")
        return 130


if __name__ == "__main__":
    sys.exit(main())
