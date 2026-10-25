"""Per-model budget pricing (failover-aware) + session budget accumulation."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from smerkil.budget import Budget, PRICING


class TestPerModelPricing(unittest.TestCase):
    def test_price_for_exact(self):
        self.assertEqual(Budget.price_for("gpt-4o"), PRICING["gpt-4o"])

    def test_price_for_provider_qualified(self):
        self.assertEqual(Budget.price_for("openai/gpt-4o-mini"),
                         PRICING["gpt-4o-mini"])

    def test_price_for_unknown_uses_default(self):
        self.assertEqual(Budget.price_for("totally/unknown-model"), PRICING["*"])

    def test_record_prices_by_served_model(self):
        """Same token counts cost differently depending on which model served."""
        b = Budget(max_tokens=10**9, max_usd=1e9)
        b.record(1_000_000, 0, model="gpt-4o-mini")
        cheap = b.usd_spent
        b2 = Budget(max_tokens=10**9, max_usd=1e9)
        b2.record(1_000_000, 0, model="openai/gpt-4o")
        self.assertGreater(b2.usd_spent, cheap * 10)  # gpt-4o ≫ gpt-4o-mini
        self.assertEqual(b.events[0]["model"], "gpt-4o-mini")

    def test_failover_run_mixed_pricing(self):
        """Turn 1 on expensive primary, turn 2 on cheap fallback — each priced
        by the model that actually served."""
        b = Budget(max_tokens=10**9, max_usd=1e9)
        b.record(100_000, 1_000, model="gpt-4o")
        b.record(100_000, 1_000, model="openai/gpt-4o-mini")
        self.assertEqual(len(b.events), 2)
        self.assertGreater(b.events[0]["usd"], b.events[1]["usd"])


class TestSessionBudget(unittest.TestCase):
    def _agent(self):
        from smerkil.agent import Agent
        from smerkil.config import AgentConfig
        from tempfile import TemporaryDirectory
        import os
        os.environ["SMERKIL_TEST_NOPE"] = "test-key-not-real"
        cfg = AgentConfig.from_dict({
            "model": "m", "tools": [], "system": "sys",
            "api_key_env": "SMERKIL_TEST_NOPE",
            "guardrails": {"max_turns": 5, "max_tokens": 100000, "max_usd": 1.0},
        }, "t")
        td = TemporaryDirectory()
        a = Agent(cfg, Path(td.name), stream=False)
        a._td = td
        return a

    def test_session_budget_accumulates_across_runs(self):
        a = self._agent()
        from smerkil.budget import Budget
        sb = Budget(max_tokens=100000, max_usd=1.0).model("m")
        a.session_budget = sb
        resp = {"model": "m",
                "choices": [{"message": {"role": "assistant", "content": "ok"},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 10}}
        a._chat = lambda msgs: resp
        a.run("q1")
        a.run("q2")
        self.assertEqual(sb.prompt_tokens, 200)
        self.assertEqual(sb.completion_tokens, 20)

    def test_session_budget_exhaustion_stops_run(self):
        a = self._agent()
        from smerkil.budget import Budget
        # session limit already blown before the run starts
        sb = Budget(max_tokens=1, max_usd=1.0).model("m")
        sb.record(1000, 1000, model="m")
        a.session_budget = sb
        called = {"n": 0}
        def fake_chat(msgs):
            called["n"] += 1
            return {"model": "m", "choices": [{"message": {"role": "assistant",
                    "content": "x"}, "finish_reason": "stop"}], "usage": {}}
        a._chat = fake_chat
        r = a.run("q")
        self.assertEqual(called["n"], 0)  # loop never called the model
        self.assertEqual(a.trace["stop_reason"], "budget_exhausted")


if __name__ == "__main__":
    unittest.main()
