"""Token accounting + provider pricing table (editable, per-model)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


# USD per 1M tokens (prompt, completion). Defaults; override per model.
PRICING: Dict[str, tuple] = {
    "*": (0.15, 0.60),  # conservative default
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "claude-sonnet-4.6": (3.00, 15.00),
    "claude-haiku-4.5": (0.80, 4.00),
}


@dataclass
class Budget:
    max_tokens: int
    max_usd: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    usd_spent: float = 0.0
    events: list = field(default_factory=list)

    def model(self, name: str):
        # default pricing; record() re-prices per the actually-served model
        self._model = PRICING.get(name, PRICING["*"])
        return self

    @staticmethod
    def price_for(name: str) -> tuple:
        """(prompt, completion) USD/1M for a model id; prefix match supports
        provider-qualified ids like 'openai/gpt-4o-mini'."""
        if not name:
            return PRICING["*"]
        if name in PRICING:
            return PRICING[name]
        tail = name.rsplit("/", 1)[-1]
        if tail in PRICING:
            return PRICING[tail]
        for key, price in PRICING.items():
            if key != "*" and (key in name or name.endswith(key)):
                return price
        return PRICING["*"]

    def record(self, prompt: int, completion: int, model: str = "") -> None:
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        # price by the model that ACTUALLY served (failover may switch mid-run)
        p_price, c_price = self.price_for(model) if model else getattr(
            self, "_model", PRICING["*"])
        cost = (prompt * p_price + completion * c_price) / 1_000_000
        self.usd_spent += cost
        self.events.append({"model": model or "?", "prompt": prompt,
                            "completion": completion, "usd": round(cost, 6)})

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def exhausted(self) -> bool:
        return self.total_tokens >= self.max_tokens or self.usd_spent >= self.max_usd
