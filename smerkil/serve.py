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
    """Thread-safe per-session history + session budget."""

    def __init__(self, cfg: AgentConfig):
        self.cfg = cfg
        self._sessions: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def get(self, session_id: str) -> Dict[str, Any]:
        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = {
                    "history": [],
                    "budget": Budget(self.cfg.guardrails.max_tokens,
                                     self.cfg.guardrails.max_usd).model(self.cfg.model),
                    "lock": threading.Lock(),
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
                try:
                    reply = agent.run(message, history=sess["history"])
                except Exception as e:  # keep the server alive on agent errors
                    return self._send(502, {"error": f"agent run failed: {e}"})
                b = sess["budget"]  # session-accumulated (same as chat /tokens)
                return self._send(200, {
                    "reply": reply,
                    "session": session_id,
                    "stop_reason": agent.trace.get("stop_reason", ""),
                    "budget": {
                        "prompt_tokens": b.prompt_tokens,
                        "completion_tokens": b.completion_tokens,
                        "usd": round(b.usd_spent, 6),
                    },
                })

    return Handler


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
