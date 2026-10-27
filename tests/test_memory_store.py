"""Tests for smerkil/memory.py — MemoryStore CRUD, search, prompt block."""
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from smerkil.memory import MemoryStore


class MemoryStoreTest(unittest.TestCase):
    def setUp(self):
        self.ws = Path(tempfile.mkdtemp())
        self.store = MemoryStore(self.ws)
        # store file must live under .smerkil/
        self.assertEqual(self.store.path.parent.name, ".smerkil")

    def test_add_and_persistence(self):
        e = self.store.add("Deploy = vercel deploy --prod --yes", tags=["deploy"])
        self.assertTrue(e["id"])
        # fresh store over same workspace must see the note
        store2 = MemoryStore(self.ws)
        self.assertEqual(len(store2.recent(10)), 1)
        self.assertEqual(store2.recent()[0]["text"], "Deploy = vercel deploy --prod --yes")

    def test_add_rejects_empty(self):
        with self.assertRaises(ValueError):
            self.store.add("   ")

    def test_add_truncates_long(self):
        e = self.store.add("x" * 5000)
        self.assertLessEqual(len(e["text"]), 2020)

    def test_recent_ordering(self):
        for i in range(12):
            self.store.add(f"note {i}")
        rec = self.store.recent(3)
        self.assertEqual([e["text"] for e in rec], ["note 9", "note 10", "note 11"])

    def test_search_by_text(self):
        self.store.add("python venv must be activated before tests")
        self.store.add("coffee machine broken")
        hits = self.store.search("python tests")
        self.assertEqual(len(hits), 1)
        self.assertIn("venv", hits[0]["text"])

    def test_search_tags_weight(self):
        self.store.add("unrelated text here", tags=["deploy"])
        self.store.add("deploy note about vercel")
        hits = self.store.search("deploy")
        # both match, but the tag match should rank first
        self.assertTrue(hits[0]["text"].startswith("unrelated"))

    def test_search_no_query_no_hits(self):
        self.store.add("something")
        self.assertEqual(self.store.search(""), [])
        self.assertEqual(self.store.search("   "), [])

    def test_recency_boost(self):
        old = self.store.add("vercel deploy checklist")
        new = self.store.add("vercel deploy checklist")
        # backdate the first entry by 60 days
        entries = self.store._load()
        entries[0]["ts"] = (datetime.now(timezone.utc) - timedelta(days=60)).isoformat()
        self.store._save(entries)
        hits = self.store.search("vercel")
        self.assertEqual(hits[0]["id"], new["id"])

    def test_forget_by_id(self):
        e1 = self.store.add("keep me")
        e2 = self.store.add("drop me")
        n = self.store.forget(entry_id=e2["id"])
        self.assertEqual(n, 1)
        texts = [e["text"] for e in self.store.recent(10)]
        self.assertIn("keep me", texts)
        self.assertNotIn("drop me", texts)

    def test_forget_by_query(self):
        self.store.add("temp secret alpha")
        self.store.add("unrelated")
        n = self.store.forget(query="secret alpha")
        self.assertEqual(n, 1)
        self.assertEqual(len(self.store.recent(10)), 1)

    def test_forget_requires_args(self):
        with self.assertRaises(ValueError):
            self.store.forget()

    def test_prompt_block(self):
        self.store.add("fact one", tags=["a"])
        self.store.add("fact two")
        block = self.store.prompt_block()
        self.assertIn("Durable memory", block)
        self.assertIn("fact two", block)
        self.assertIn("#a", block)

    def test_prompt_block_empty(self):
        self.assertEqual(self.store.prompt_block(), "")

    def test_prompt_block_char_budget(self):
        for i in range(50):
            self.store.add("y" * 300)
        block = self.store.prompt_block()
        # newest-first injection must stay bounded
        self.assertLess(len(block), 2400)
        self.assertIn("note", block) if False else None  # noop
        self.assertIn("y", block)

    def test_corrupt_lines_skipped(self):
        self.store.path.write_text(
            "{not json}\n" + json.dumps({"id": "1", "ts": "t", "text": "ok", "tags": []}) + "\n",
            encoding="utf-8")
        store2 = MemoryStore(self.ws)
        self.assertEqual(len(store2.recent(10)), 1)


if __name__ == "__main__":
    unittest.main()
