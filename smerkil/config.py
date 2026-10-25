"""Agent configuration: declarative agent.yaml + runtime defaults."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml  # optional; fallback parser below
except ImportError:  # pragma: no cover
    yaml = None


def _mini_yaml(text: str) -> Dict[str, Any]:
    """Tiny YAML subset parser (stdlib fallback): top-level keys, nested
    one-level maps, inline lists. Covers everything agent.yaml needs."""
    result: Dict[str, Any] = {}
    current_key = None
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        stripped = line.strip()
        if indent == 0:
            key, _, value = stripped.partition(":")
            key, value = key.strip(), value.strip()
            if value == "":
                result[key] = {}
                current_key = key
            elif value.startswith("[") and value.endswith("]"):
                items = [v.strip().strip("'\"") for v in value[1:-1].split(",") if v.strip()]
                result[key] = items
                current_key = None
            else:
                result[key] = _coerce(value)
                current_key = None
        elif indent > 0 and current_key is not None and isinstance(result.get(current_key), dict):
            key, _, value = stripped.partition(":")
            key, value = key.strip(), value.strip()
            if value.startswith("{") and value.endswith("}"):
                inner: Dict[str, Any] = {}
                for part in value[1:-1].split(","):
                    if ":" in part:
                        ik, _, iv = part.partition(":")
                        inner[ik.strip()] = _coerce(iv.strip())
                result[current_key][key] = inner
            else:
                result[current_key][key] = _coerce(value)
    return result


def _coerce(value: str) -> Any:
    value = value.strip().strip("'\"")
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def load_config_text(text: str) -> Dict[str, Any]:
    if yaml is not None:
        data = yaml.safe_load(text)
        return data if isinstance(data, dict) else {}
    return _mini_yaml(text)


@dataclass
class Guardrails:
    max_tokens: int = 100_000
    max_usd: float = 1.00
    max_turns: int = 40
    sandbox_strict: bool = True
    allow_shell: bool = False

    @classmethod
    def from_dict(cls, d: Dict[str, Any], strict_default: bool = True) -> "Guardrails":
        if not isinstance(d, dict):
            d = {}
        budget = d.get("budget") if isinstance(d.get("budget"), dict) else {}
        return cls(
            max_tokens=int(budget.get("tokens", cls.max_tokens)),
            max_usd=float(budget.get("usd", cls.max_usd)),
            max_turns=int(d.get("max_turns", cls.max_turns)),
            sandbox_strict=bool(d.get("sandbox", strict_default)),
            allow_shell=bool(d.get("allow_shell", False)),
        )


@dataclass
class AgentConfig:
    name: str
    model: str = "gpt-4o-mini"
    fallback_models: List[str] = field(default_factory=list)
    system: str = "You are a helpful autonomous agent. Plan briefly, act with tools, verify, then answer."
    tools: List[str] = field(default_factory=lambda: ["read_file", "write_file", "list_dir"])
    guardrails: Guardrails = field(default_factory=Guardrails)
    memory: Dict[str, str] = field(default_factory=dict)
    base_url: str = ""
    api_key_env: str = "SMERKIL_API_KEY"
    temperature: float = 0.2

    @classmethod
    def _defaults(cls) -> "AgentConfig":
        return cls(name="default")

    @classmethod
    def from_dict(cls, d: Dict[str, Any], name: str, agent_dir: Optional[Path] = None) -> "AgentConfig":
        base = cls._defaults()
        memory = d.get("memory") if isinstance(d.get("memory"), dict) else {}
        base_url = str(d.get("base_url", "") or os.environ.get("SMERKIL_BASE_URL", ""))
        if not base_url:
            # last resort: path-relative discovery of a local router config (no network)
            base_url = "https://api.openai.com/v1"
        return cls(
            name=name,
            model=str(d.get("model", base.model)),
            fallback_models=[str(m) for m in (d.get("fallback_models") or []) if str(m).strip()],
            system=str(d.get("system", base.system)),
            tools=list(d.get("tools", base.tools)),
            guardrails=Guardrails.from_dict(d.get("guardrails", {})),
            memory={str(k): str(v) for k, v in memory.items()},
            base_url=base_url,
            api_key_env=str(d.get("api_key_env", base.api_key_env)),
            temperature=float(d.get("temperature", base.temperature)),
        )

    def resolve_api_key(self) -> str:
        key = os.environ.get(self.api_key_env, "")
        if not key:
            raise RuntimeError(
                f"API key env '{self.api_key_env}' is not set. export {self.api_key_env}=<key>"
            )
        return key
