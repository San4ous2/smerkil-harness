# SMERKIL Agent

A production harness for building, evaluating and operating autonomous AI agents.
Zero dependencies — stdlib only. Model-agnostic (any OpenAI-compatible endpoint).

```bash
pip install smerkil-0.7.6.tar.gz

# scaffold an agent with a role preset
smerkil new my-agent --role coder   # general|coder|researcher|reviewer|writer

# edit agent.yaml (model, tools, guardrails), export a key...
export SMERKIL_API_KEY=sk-...
export SMERKIL_BASE_URL=https://api.openai.com/v1   # or any compatible endpoint

# run it
smerkil run my-agent --goal "Summarize ./notes/ and write brief.md"

# interactive session — streaming text AND tool-call args, persistent history
smerkil chat my-agent

# expose the agent as an HTTP API
smerkil serve my-agent --token $SERVE_TOKEN

# check environment, key, model reachability, workspace health
smerkil doctor my-agent

# check behavior before you ship
smerkil eval my-agent

# replay any run
smerkil trace list
smerkil trace my-agent --show <run-id>
```

## What's inside

- **Tool loop runtime** — deterministic loop, parallel tool execution (a thread pool runs all tool calls from one model turn concurrently, results appended in deterministic order), token & turn budgets, hard stop on runaway loops.
- **Multi-turn chat** — `Agent.run(goal, history=[...])` records user, assistant and tool turns incrementally; context compaction can never tear a session apart. `smerkil chat` is a streaming REPL (`/reset`, `/tokens`, `/exit`).
- **Role presets** — `--role coder|researcher|reviewer|writer|general` scaffolds a tuned system prompt and tool set (coder/reviewer get the shell).
- **HTTP API** — `smerkil serve` exposes the agent over a stdlib HTTP server: `POST /v1/chat` (multi-turn, per-session history + shared session budget), `GET /sessions`, `GET /health`, `GET /live`; optional bearer-token auth.
- **Live status** — `smerkil status` finds a running server via its heartbeat file (`~/.smerkil/serve.json`) and reports which sessions are actually executing right now; `--json` for scripting.
- **Usage report** — `smerkil report` aggregates every agent's on-disk traces into one document (runs, turns, tool calls, tokens, cost, stop reasons, top tools); `smerkil push` sends it to a dashboard, opt-in.
- **Subagents** — the `subagent` tool spawns isolated child agents with their own context and goal; the child's final answer returns as a tool result. Children cannot spawn further subagents (bounded depth).
- **Context compaction** — when the transcript exceeds a token budget, older messages are summarized (system prompt and recent window kept verbatim). Summarizers are pluggable (`smerkil.compaction.Summarizer` protocol); the default `TruncatingSummarizer` needs no model, `ModelSummarizer` uses the chat model when `SMERKIL_MODEL_SUMMARIZER=1`.
- **Streaming** — SSE streaming from any OpenAI-compatible endpoint over stdlib `http.client`: text deltas (`on_token`) and live tool-call argument fragments (`on_tool_delta`); automatic non-streaming fallback when the endpoint doesn't support SSE.
- **Sticky model failover** — `fallback_models` tried in order; once a fallback serves, later turns skip the dead primary. Each trace step records the model that actually answered.
- **Failover-aware budget** — cost priced per the model that actually served each turn; chat and serve sessions share one session budget (visible via `/tokens` or `GET /sessions`, enforced mid-session).
- **Retries & backoff** — exponential backoff with jitter on connection errors and 429/5xx, honoring `Retry-After`.
- **Structured planning** — `plan_add` / `plan_list` / `plan_update` tools backed by `workspace/plan.md` with `todo` / `in_progress` / `done` statuses.
- **More tools** — `web_fetch` (stdlib, size/time-limited, text extraction), `python_exec` (snippet in a sandboxed subprocess with timeout), `git_status` / `git_log` (read-only), plus the sandboxed file tools and opt-in whitelisted shell.
- **Memory & state** — per-agent working memory, durable notes (`memory.md`), carried across runs.
- **Evals** — scenario suites (`evals.yaml`) with grounded-expectation checks; non-zero exit on regression. Ships role-aware scenarios covering planning and parallel tool calls.
- **Traces** — every run saved as JSONL (`workspace/traces/runs.jsonl`) plus per-run JSON with metadata: run id, per-step serving model, timestamps, token totals, compaction count. Pretty-print with `smerkil trace <name> --show <id>`.

## CLI reference

| Command | Purpose |
|---|---|
| `smerkil new <name> [--role R]` | Scaffold an agent dir with a role preset |
| `smerkil run <name> --goal "..."` | Run one goal; exit 0 only on completed run |
| `smerkil chat <name>` | Streaming REPL: live text + tool args, session budget, `/reset`, `/tokens`, `/exit` |
| `smerkil serve <name>` | HTTP API: `POST /v1/chat`, `GET /sessions`, `GET /health`, `GET /live`; `--host`, `--port`, `--token` |
| `smerkil login` | sign in to a dashboard; stores the session token in `~/.smerkil/dashboard.json` (0600) |
| `smerkil status` | live status of a running server — active runs, sessions, budgets; `--url`, `--token`, `--json`, `--max-age` |
| `smerkil push` | upload `smerkil report --json` to a dashboard; `--url`, `--token-file` (opt-in) |
| `smerkil eval <name> [-v]` | Run eval scenarios; non-zero exit on regression |
| `smerkil trace <name> [--show <id>]` | List runs or pretty-print one trace |
| `smerkil doctor [name]` | Check API key env, endpoint/model reachability, workspace health |
| `smerkil --version` | Print version |

Output is ANSI-colored on TTYs, plain when piped.

## HTTP API (`smerkil serve`)

```bash
smerkil serve my-agent --host 127.0.0.1 --port 8317 --token "$SERVE_TOKEN"

# health
curl http://127.0.0.1:8317/health

# chat (multi-turn within a session id)
curl -X POST http://127.0.0.1:8317/v1/chat \
  -H "Authorization: Bearer $SERVE_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"message": "Summarize ./notes/", "session": "job-42"}'

# per-session token/cost accounting
curl -H "Authorization: Bearer $SERVE_TOKEN" http://127.0.0.1:8317/sessions
```

Each `session` id carries its own conversation history and session budget
(guardrails from `agent.yaml`); runs inside one session are serialized,
different sessions run concurrently.

### Live status

`smerkil serve` writes a heartbeat to `$SMERKIL_HOME/serve.json` every 5s, so
`smerkil status` can find it without knowing the port:

```bash
$ smerkil status
smerkil status  stub · stub-1
  pid 619248 · http://127.0.0.1:39567 · up since 2026-10-25T21:13:01+00:00
  1 running · 3 session(s)

  ○ idle     job-A                runs   1  0p/0c  $0.0  completed
  ○ idle     job-B                runs   1  0p/0c  $0.0  completed
  ● running  watch-me             runs   0  0p/0c  $0.0  —
```

`GET /live` returns the same data as JSON: per-session `in_flight`, `runs`,
`last_stop_reason` and budget, plus server-wide `active_runs`. Exits non-zero
when nothing is serving, and `--json` is there for scripts:

```bash
smerkil status --json | jq '.active_runs'
```

A heartbeat older than `--max-age` (default 20s), or one whose pid is gone, is
treated as "not running" — a crashed server can't masquerade as live.

### Seeing it from a hosted dashboard

Your machine is behind NAT; a hosted page can't open a socket to it. So the
direction is inverted — the harness pushes its status *up*, and the dashboard
just reads it:

```bash
smerkil login --url https://smerkil-web.vercel.app      # once, password not echoed
smerkil serve coder --publish-live                       # then every ~5s
```

`POST /api/live` carries the same snapshot `GET /live` returns. A snapshot
older than 45s means the server stopped pushing, so the dashboard reports it
as offline rather than showing a stale "running". Everything is opt-in:
without `--publish-live` nothing leaves the machine. `smerkil push` and
`smerkil status` reuse the token from `login`, so it only has to be typed once.

## Configuration (agent.yaml)

```yaml
name: my-agent
model: gpt-4o-mini
# tried in order when the primary fails definitively; once a fallback serves,
# later turns prefer it (sticky) until it too fails
fallback_models: [gpt-4o-mini, claude-haiku-4.5]
base_url: https://api.openai.com/v1     # any OpenAI-compatible endpoint
api_key_env: SMERKIL_API_KEY
system: You are my-agent. Plan, act with tools, verify, answer.
tools: [read_file, write_file, list_dir, memory_note,
        plan_add, plan_list, plan_update,
        python_exec, web_fetch, git_status, git_log, subagent]
guardrails:
  max_turns: 40
  budget:
    tokens: 200000
    usd: 1.00          # priced by the model that actually served each turn
  sandbox: strict
  allow_shell: false
```

Existing 0.2.x `agent.yaml` and `evals.yaml` files work unchanged; new tools are opt-in via the `tools:` list (unknown tool names are skipped with a warning rather than crashing the run).

## Available tools

`read_file`, `write_file`, `list_dir`, `run_shell` (whitelisted, opt-in), `memory_note`, `web_fetch`, `python_exec`, `git_status`, `git_log`, `plan_add`, `plan_list`, `plan_update` (alias: `todo`), `subagent`.

MIT-grade boring infrastructure for probabilistic workers.

© 2026 SMERKIL Technologies
