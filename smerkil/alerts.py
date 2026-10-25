"""Budget alerts: deterministic notifications before the guardrail trips.

The guardrail stops a run when it hits the cap. This tells you *before* that,
at configurable fractions of the budget (default 0.5, 0.8, 0.95), so a run
that is about to blow through $5 alerts you at $2.50 instead of silently
dying at the limit.

Two independent mechanisms, both opt-in:

  1. hooks.yaml — any shell command or webhook. Runs outside the LLM, so the
     model can't skip it. See hooks.py for the general mechanism; this module
     emits the event names `budget_warning` and `budget_exceeded`.

  2. SMERKIL_ALERT_URL — an HTTP endpoint that receives JSON. Set it in the
     environment, nothing else needed.

Alerts fire once per threshold per budget. They are best-effort: a broken
webhook logs to stderr and never fails the run.
"""
from __future__ import annotations

import json
import os
import subprocess
import threading
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

DEFAULT_THRESHOLDS = (0.5, 0.8, 0.95)

# env vars for the shell hook, mirroring hooks.py
_ENV_PREFIX = "SMERKIL_ALERT_"


def load_thresholds(workspace: Optional[Path] = None) -> List[float]:
    """Read alert thresholds from config, fall back to defaults.

    Looks at `agent.yaml` -> alerts.thresholds (a list of fractions). Anything
    outside (0, 1] is ignored rather than raising — a typo shouldn't kill runs.
    """
    vals = list(DEFAULT_THRESHOLDS)
    if workspace is None:
        return vals
    try:
        import yaml  # type: ignore
    except Exception:
        return vals
    cfg = workspace / "agent.yaml"
    if not cfg.exists():
        return vals
    try:
        data = yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}
    except Exception:
        return vals
    alerts = (data or {}).get("alerts") or {}
    raw = alerts.get("thresholds")
    if isinstance(raw, list) and raw:
        parsed = [float(x) for x in raw if isinstance(x, (int, float))
                  and 0 < float(x) <= 1]
        # floats like 0.8 come back as 0.8; keep sorted and unique
        seen: Set[float] = set()
        ordered = []
        for v in parsed:
            if v not in seen:
                seen.add(v)
                ordered.append(v)
        if ordered:
            return sorted(ordered)
    return vals


class BudgetAlerts:
    """Watch a Budget and fire alerts as it approaches its cap."""

    def __init__(self, budget, agent: str = "", run_id: str = "",
                 thresholds: Optional[List[float]] = None,
                 workspace: Optional[Path] = None,
                 verbose: bool = False):
        self.budget = budget
        self.agent = agent
        self.run_id = run_id
        self.workspace = workspace
        self.verbose = verbose
        self.thresholds = thresholds or load_thresholds(workspace)
        self._fired: Set[str] = set()
        self._lock = threading.Lock()

    # ---------- threshold math ----------
    def _fraction(self) -> float:
        """How much of the budget is used, as 0..1 (worst of tokens and USD)."""
        frac = 0.0
        try:
            tokens = getattr(self.budget, "total_tokens", 0) or 0
            maxt = getattr(self.budget, "max_tokens", 0) or 0
            if maxt > 0:
                frac = max(frac, tokens / maxt)
        except Exception:
            pass
        try:
            usd = getattr(self.budget, "usd_spent", 0) or 0
            maxu = getattr(self.budget, "max_usd", 0) or 0
            if maxu > 0:
                frac = max(frac, usd / maxu)
        except Exception:
            pass
        return min(frac, 1.0)

    def check(self) -> List[str]:
        """Fire any thresholds newly crossed. Returns the events emitted."""
        frac = self._fraction()
        emitted: List[str] = []
        for t in self.thresholds:
            key = f"{t:.2f}"
            if frac < t or key in self._fired:
                continue
            with self._lock:
                if key in self._fired:
                    continue
                self._fired.add(key)
            exceeded = frac >= 1.0
            event = "budget_exceeded" if exceeded else "budget_warning"
            payload = self._payload(event, frac, t)
            self._emit(event, payload)
            emitted.append(event)
        return emitted

    def _payload(self, event: str, frac: float, threshold: float) -> Dict[str, Any]:
        b = self.budget
        return {
            "event": event,
            "agent": self.agent,
            "run_id": self.run_id,
            "fraction": round(frac, 4),
            "threshold": threshold,
            "tokens": getattr(b, "total_tokens", 0),
            "max_tokens": getattr(b, "max_tokens", 0),
            "usd": round(getattr(b, "usd_spent", 0.0), 6),
            "max_usd": getattr(b, "max_usd", 0.0),
        }

    # ---------- delivery ----------
    def _emit(self, event: str, payload: Dict[str, Any]) -> None:
        if self.verbose:
            print(f"  ! {event} at {payload['fraction']*100:.0f}% "
                  f"(${payload['usd']:.4f} / ${payload['max_usd']:.2f})")
        self._to_url(event, payload)
        self._to_hook(event, payload)

    def _to_url(self, event: str, payload: Dict[str, Any]) -> None:
        url = os.environ.get("SMERKIL_ALERT_URL", "").strip()
        if not url:
            return
        try:
            req = urllib.request.Request(
                url, data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json",
                         "X-Smerkil-Event": event},
                method="POST")
            timeout = float(os.environ.get("SMERKIL_ALERT_TIMEOUT", "5"))
            urllib.request.urlopen(req, timeout=timeout).read()
        except (urllib.error.URLError, OSError, ValueError) as e:
            # never fail the run for a notification
            print(f"[smerkil] alert to {url} failed: {e}")

    def _to_hook(self, event: str, payload: Dict[str, Any]) -> None:
        """Run a per-event shell command from workspace/alerts.yaml if present."""
        ws = self.workspace
        if ws is None:
            return
        cfg = ws / "alerts.yaml"
        if not cfg.exists():
            return
        try:
            import yaml  # type: ignore
            data = yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}
        except Exception:
            return
        cmds = (data or {}).get("alerts") or {}
        cmd = cmds.get(event) or cmds.get("*")
        if not cmd:
            return
        env = dict(os.environ)
        for k, v in payload.items():
            env[_ENV_PREFIX + k.upper()] = str(v)
        try:
            subprocess.run(cmd, shell=True, env=env, timeout=30,
                           capture_output=True)
        except subprocess.SubprocessError as e:
            print(f"[smerkil] alert hook failed: {e}")


def alerts_from_config(workspace: Optional[Path], budget, agent: str = "",
                       run_id: str = "", verbose: bool = False) -> BudgetAlerts:
    return BudgetAlerts(budget, agent=agent, run_id=run_id,
                        workspace=workspace, verbose=verbose)
