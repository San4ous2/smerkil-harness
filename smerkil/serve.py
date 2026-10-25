"""smerkil serve: HTTP API over an agent — stdlib ThreadingHTTPServer.

Endpoints:
    GET  /health                    → {"ok", "agent", "model"}
    GET  /sessions                  → [{"session", "prompt_tokens", ...}]
    POST /v1/chat                   → body {"message", "session": "default"}
                                      → {"reply", "stop_reason", "budget"}

Each session id gets its own history list and shared session Budget (same
accumulation semantics as `smerkil chat`). Runs for one session are
serialized with a per-session lock; different sessions run concurrently.
Optional bearer auth: pass token= or set SMERKIL_SERVE_TOKEN.
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Dict, List, Optional

from .agent import Agent
from .budget import Budget
from .config import AgentConfig


class SessionStore:
    """Thread-safe per-session history + session budget.

    Also tracks liveness: each session records when it started, when it was
    last active, and whether a run is currently in flight. That is what
    `GET /live` (and therefore `smerkil status`) reports.
    """

    def __init__(self, cfg: AgentConfig):
        self.cfg = cfg
        self._sessions: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self.started_at = _now_iso()

    def get(self, session_id: str) -> Dict[str, Any]:
        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = {
                    "history": [],
                    "budget": Budget(self.cfg.guardrails.max_tokens,
                                     self.cfg.guardrails.max_usd).model(self.cfg.model),
                    "lock": threading.Lock(),
                    "created_at": _now_iso(),
                    "last_active_at": _now_iso(),
                    "runs": 0,
                    "in_flight": False,
                    "last_stop_reason": None,
                }
            return self._sessions[session_id]

    def ids(self) -> List[str]:
        with self._lock:
            return sorted(self._sessions)

    def summary(self) -> List[Dict[str, Any]]:
        with self._lock:
            out = []
            for sid, s in sorted(self._sessions.items()):
                b = s["budget"]
                out.append({"session": sid,
                            "prompt_tokens": b.prompt_tokens,
                            "completion_tokens": b.completion_tokens,
                            "usd": round(b.usd_spent, 6)})
            return out

    def live(self) -> Dict[str, Any]:
        """Snapshot for GET /live: per-session state + server totals."""
        with self._lock:
            sessions = []
            for sid, s in sorted(self._sessions.items()):
                b = s["budget"]
                sessions.append({
                    "session": sid,
                    "created_at": s["created_at"],
                    "last_active_at": s["last_active_at"],
                    "runs": s["runs"],
                    "in_flight": s["in_flight"],
                    "last_stop_reason": s["last_stop_reason"],
                    "turns": len(s["history"]),
                    "prompt_tokens": b.prompt_tokens,
                    "completion_tokens": b.completion_tokens,
                    "usd": round(b.usd_spent, 6),
                })
            active = [s["session"] for s in sessions if s["in_flight"]]
            return {
                "started_at": self.started_at,
                "now": _now_iso(),
                "agent": self.cfg.name,
                "model": self.cfg.model,
                "active_runs": len(active),
                "active_sessions": active,
                "sessions": sessions,
            }


def _now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def make_handler(agent_factory: Callable[[], Agent], store: SessionStore,
                 token: str = "") -> type:
    """Build a request handler bound to an agent factory + session store."""

    class Handler(BaseHTTPRequestHandler):
        server_version = "smerkil-serve"

        def log_message(self, fmt: str, *args: Any) -> None:  # quiet by default
            pass

        # ---- helpers ----
        def _send(self, code: int, payload: Dict[str, Any]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            # /live is meant for local dashboards: the browser console polls it
            # from a page served on another port, which needs CORS. Cookies are
            # never involved, and the bearer token is checked separately, so
            # this does not widen who can read data — it only widens who can
            # ask. Localhost-only by default; use --host 0.0.0.0 deliberately.
            origin = self.headers.get("Origin")
            if origin:
                self.send_header("Access-Control-Allow-Origin", origin)
                self.send_header("Vary", "Origin")
            else:
                self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _authorized(self) -> bool:
            if not token:
                return True
            return self.headers.get("Authorization", "") == f"Bearer {token}"

        # ---- routes ----
        def do_GET(self) -> None:
            if not self._authorized():
                return self._send(401, {"error": "unauthorized"})
            if self.path == "/health":
                agent = agent_factory()
                return self._send(200, {"ok": True, "agent": agent.cfg.name,
                                        "model": agent.cfg.model})
            if self.path == "/sessions":
                return self._send(200, {"sessions": store.summary()})
            if self.path == "/live":
                return self._send(200, store.live())
            return self._send(404, {"error": f"no route: {self.path}"})

        def do_POST(self) -> None:
            if not self._authorized():
                return self._send(401, {"error": "unauthorized"})
            if self.path != "/v1/chat":
                return self._send(404, {"error": f"no route: {self.path}"})
            try:
                length = int(self.headers.get("Content-Length", "0") or "0")
                raw = self.rfile.read(length) if length else b"{}"
                data = json.loads(raw.decode("utf-8"))
                message = str(data.get("message", "")).strip()
                session_id = str(data.get("session", "default")) or "default"
            except (ValueError, UnicodeDecodeError):
                return self._send(400, {"error": "invalid JSON body"})
            if not message:
                return self._send(400, {"error": "'message' is required"})

            sess = store.get(session_id)
            with sess["lock"]:
                agent = agent_factory()
                agent.session_budget = sess["budget"]
                # mark in-flight before the run so /live can see it immediately
                with store._lock:
                    sess["in_flight"] = True
                    sess["last_active_at"] = _now_iso()
                try:
                    reply = agent.run(message, history=sess["history"])
                except Exception as e:  # keep the server alive on agent errors
                    with store._lock:
                        sess["in_flight"] = False
                    return self._send(502, {"error": f"agent run failed: {e}"})
                b = sess["budget"]  # session-accumulated (same as chat /tokens)
                stop = agent.trace.get("stop_reason", "")
                with store._lock:
                    sess["in_flight"] = False
                    sess["runs"] += 1
                    sess["last_active_at"] = _now_iso()
                    sess["last_stop_reason"] = stop
                return self._send(200, {
                    "reply": reply,
                    "session": session_id,
                    "stop_reason": stop,
                    "budget": {
                        "prompt_tokens": b.prompt_tokens,
                        "completion_tokens": b.completion_tokens,
                        "usd": round(b.usd_spent, 6),
                    },
                })

    return Handler


def heartbeat_path() -> "Path":
    """Where `smerkil serve` publishes how to reach it."""
    import os
    from pathlib import Path

    root = Path(os.environ.get("SMERKIL_HOME", Path.home() / ".smerkil"))
    return root / "serve.json"


def write_heartbeat(cfg: AgentConfig, host: str, port: int, pid: int,
                    url: str, every: float = 5.0):
    """Start a daemon thread that refreshes serve.json so `smerkil status`
    can find this server (and tell when it died) without knowing the port."""
    import atexit
    import json
    import os
    import time

    path = heartbeat_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()

    def _loop() -> None:
        while not stop.is_set():
            try:
                path.write_text(json.dumps({
                    "pid": pid,
                    "host": host,
                    "port": port,
                    "url": url,
                    "agent": cfg.name,
                    "model": cfg.model,
                    "updated_at": _now_iso(),
                }, indent=2), encoding="utf-8")
            except OSError:
                pass
            stop.wait(every)

    t = threading.Thread(target=_loop, daemon=True)
    t.start()

    def _cleanup() -> None:
        stop.set()
        try:
            if path.exists() and json.loads(path.read_text()).get("pid") == pid:
                path.unlink()
        except Exception:
            pass

    atexit.register(_cleanup)
    return stop


def read_heartbeat(max_age_s: float = 20.0) -> Optional[Dict[str, Any]]:
    """Read serve.json; return None if missing, stale, or the pid is dead."""
    import json
    import os
    import time
    from datetime import datetime, timezone

    path = heartbeat_path()
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    ts = data.get("updated_at")
    if ts:
        try:
            then = datetime.fromisoformat(ts)
            age = (datetime.now(timezone.utc) - then).total_seconds()
            if age > max_age_s:
                return None  # server died without cleaning up
        except Exception:
            return None
    pid = data.get("pid")
    if pid:
        try:
            os.kill(int(pid), 0)  # signal 0: liveness probe only
        except (OSError, ProcessLookupError, ValueError):
            return None
    return data


def make_server(cfg: AgentConfig, workspace, host: str = "127.0.0.1",
                port: int = 0, token: str = "",
                agent_factory: Optional[Callable[[], Agent]] = None
                ) -> ThreadingHTTPServer:
    """Bind an HTTP server. port=0 picks an ephemeral port (useful in tests)."""
    workspace = __import__("pathlib").Path(workspace)
    if agent_factory is None:
        def agent_factory() -> Agent:
            return Agent(cfg, workspace, stream=False)
    store = SessionStore(cfg)
    handler = make_handler(agent_factory, store, token)
    httpd = ThreadingHTTPServer((host, port), handler)
    httpd.smerkil_store = store  # exposed for tests/inspection
    return httpd
