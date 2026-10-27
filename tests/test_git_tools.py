"""Tests for git tools: git_diff, git_commit, git_branch (run against a tmp repo)."""
import subprocess
import tempfile
import unittest
from pathlib import Path

from smerkil.tools import (
    Sandbox,
    GitBranch,
    GitCommit,
    GitDiff,
    GitLog,
    GitStatus,
    ToolError,
)


def _git(*args, cwd):
    r = subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return r.stdout


class GitToolsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        _git("init", "-q", cwd=self.tmp)
        _git("-c", "user.email=t@t", "-c", "user.name=T", "commit", "-q",
             "--allow-empty", "-m", "init", cwd=self.tmp)
        # repo-local identity so commits work without global git config
        _git("config", "user.email", "t@t", cwd=self.tmp)
        _git("config", "user.name", "T", cwd=self.tmp)
        self.sandbox = Sandbox(self.tmp)
        self.sw = False  # allow_shell unused by git tools

    def test_status_and_log(self):
        st = GitStatus(self.sandbox, self.sw).run()
        self.assertTrue(st.startswith("##") or st == "(clean)")
        log = GitLog(self.sandbox, self.sw).run()
        self.assertIn("init", log)

    def test_diff_empty(self):
        out = GitDiff(self.sandbox, self.sw).run()
        self.assertEqual(out, "(no changes)")

    def test_diff_worktree(self):
        (self.tmp / "a.txt").write_text("hello\n", encoding="utf-8")
        out = GitDiff(self.sandbox, self.sw).run()
        self.assertIn("+hello", out)

    def test_diff_commit(self):
        (self.tmp / "a.txt").write_text("hello\n", encoding="utf-8")
        _git("add", "a.txt", cwd=self.tmp)
        _git("commit", "-q", "-m", "add a", cwd=self.tmp)
        out = GitDiff(self.sandbox, self.sw).run(commit="HEAD")
        self.assertIn("+hello", out)

    def test_commit_requires_message(self):
        with self.assertRaises(ToolError):
            GitCommit(self.sandbox, self.sw).run(message="  ")

    def test_commit_nothing_staged(self):
        with self.assertRaises(ToolError):
            GitCommit(self.sandbox, self.sw).run(message="empty")

    def test_commit_stage_all(self):
        (self.tmp / "b.txt").write_text("data\n", encoding="utf-8")
        out = GitCommit(self.sandbox, self.sw).run(message="add b", add=["."])
        self.assertIn("add b", GitLog(self.sandbox, self.sw).run(max_count=1))
        self.assertTrue(GitStatus(self.sandbox, self.sw).run().startswith("##"))

    def test_commit_explicit_paths(self):
        (self.tmp / "c.txt").write_text("c\n", encoding="utf-8")
        (self.tmp / "d.txt").write_text("d\n", encoding="utf-8")
        GitCommit(self.sandbox, self.sw).run(message="add c only", add=["c.txt"])
        staged_after = _git("status", "--porcelain", cwd=self.tmp)
        self.assertIn("d.txt", staged_after)
        self.assertNotIn("c.txt", staged_after)

    def test_branch_lifecycle(self):
        br = GitBranch(self.sandbox, self.sw)
        out = br.run(action="list")
        self.assertTrue("main" in out or "master" in out)
        br.run(action="create", name="feature/x-1")
        cur = _git("branch", "--show-current", cwd=self.tmp).strip()
        self.assertEqual(cur, "feature/x-1")
        default = "master" if "master" in out else "main"
        br.run(action="switch", name=default)
        cur = _git("branch", "--show-current", cwd=self.tmp).strip()
        self.assertEqual(cur, default)

    def test_branch_name_validation(self):
        br = GitBranch(self.sandbox, self.sw)
        with self.assertRaises(ToolError):
            br.run(action="create", name="bad name")
        with self.assertRaises(ToolError):
            br.run(action="create", name="-rf")
        with self.assertRaises(ToolError):
            br.run(action="create", name="")


if __name__ == "__main__":
    unittest.main()
