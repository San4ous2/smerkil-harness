"""Eval runner: scenarios with grounded expectations. Non-zero exit on regression."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from .config import load_config_text


def load_scenarios(agent_dir: Path) -> List[Dict[str, Any]]:
    path = agent_dir / "evals.yaml"
    if not path.exists():
        return []
    data = load_config_text(path.read_text(encoding="utf-8"))
    scenarios = data.get("scenarios", [])
    return scenarios if isinstance(scenarios, list) else []


def check_expectations(result_text: str, scenario: Dict[str, Any],
                       agent: Any = None) -> Dict[str, Any]:
    checks = []
    expects = scenario.get("expect", {})
    if isinstance(expects, str):
        expects = {"contains_any": [expects]}
    if not isinstance(expects, dict):
        expects = {}

    contains_any = expects.get("contains_any") or []
    if contains_any:
        low = result_text.lower()
        hit = any(str(term).lower() in low for term in contains_any)
        checks.append({"rule": "contains_any", "terms": contains_any, "passed": hit})

    contains_all = expects.get("contains_all") or []
    if contains_all:
        low = result_text.lower()
        hit = all(str(term).lower() in low for term in contains_all)
        checks.append({"rule": "contains_all", "terms": contains_all, "passed": hit})

    contains_none = expects.get("contains_none") or []
    if contains_none:
        low = result_text.lower()
        bad = [t for t in contains_none if str(t).lower() in low]
        checks.append({"rule": "contains_none", "terms": contains_none,
                       "passed": not bad, "found": bad})

    regex = expects.get("regex_match")
    if regex:
        import re as _re
        try:
            hit = bool(_re.search(regex, result_text))
        except _re.error as e:
            hit = False
            checks.append({"rule": "regex_match", "pattern": regex,
                           "passed": False, "error": str(e)})
        else:
            checks.append({"rule": "regex_match", "pattern": regex, "passed": hit})

    min_chars = expects.get("min_chars")
    if min_chars:
        checks.append({"rule": "min_chars", "value": min_chars, "passed": len(result_text) >= int(min_chars)})

    tool_used = expects.get("tool_used") or []
    if tool_used:
        used = {tc.get("name") for tc in (getattr(agent, "trace", {}) or {}).get("tool_calls", [])}
        missing = [t for t in tool_used if t not in used]
        checks.append({"rule": "tool_used", "required": tool_used,
                       "passed": not missing, "missing": missing})

    if not checks:
        checks.append({"rule": "non_empty", "passed": bool(result_text.strip())})

    return {"passed": all(c["passed"] for c in checks), "checks": checks}


def run_evals(agent: Any, agent_dir: Path, verbose: bool = False) -> int:
    scenarios = load_scenarios(agent_dir)
    if not scenarios:
        print("no evals.yaml scenarios found — nothing to gate")
        return 0
    failures = 0
    for i, sc in enumerate(scenarios, 1):
        goal = str(sc.get("goal", "")).strip()
        if not goal:
            continue
        print(f"[{i}/{len(scenarios)}] {sc.get('name', goal[:60])} ... ", end="", flush=True)
        try:
            result = agent.run(goal)
            verdict = check_expectations(result, sc, agent=agent)
        except Exception as e:
            result, verdict = "", {"passed": False, "checks": [{"rule": "run_error", "error": str(e)}]}
        status = "PASS" if verdict["passed"] else "FAIL"
        if not verdict["passed"]:
            failures += 1
        print(status)
        if verbose or not verdict["passed"]:
            for c in verdict["checks"]:
                print(f"    - {c}")
            if not result:
                print("    (empty result)")
    print(f"\n{len(scenarios) - failures}/{len(scenarios)} passed")
    return 1 if failures else 0
