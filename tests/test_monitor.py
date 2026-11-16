"""Tests for the decision logic, using fake GitHub data (no network, no tokens).

Run with:  python -m unittest discover -s tests -v
"""
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import monitor  # noqa: E402

CONFIG = {"repos": ["acme/shop"], "issues_per_repo": 5,
          "include_closed": False, "body_preview_chars": 50}


def issue(number, state="OPEN", title="Cart empties on refresh", body="Steps to reproduce..."):
    return {"number": number, "title": title, "url": f"https://github.com/acme/shop/issues/{number}",
            "state": state, "createdAt": "2026-11-16T10:00:00Z", "bodyText": body,
            "author": {"login": "alice"}, "labels": {"nodes": [{"name": "bug"}]}}


class Harness:
    """Runs monitor.run_once and records what would have been sent."""
    def __init__(self, state=None):
        self.state = state or {"repos": {}}
        self.sent, self.saves = [], 0

    def run(self, fetched, config=CONFIG, backfill=0):
        def save():
            self.saves += 1
        return monitor.run_once(config, self.state, fetched, self.sent.append, save, backfill)


class FirstRun(unittest.TestCase):
    def test_first_run_sends_nothing_and_records_newest(self):
        h = Harness()
        h.run({"acme/shop": [issue(12), issue(10), issue(11)]})
        self.assertEqual(h.sent, [])
        self.assertEqual(h.state["repos"]["acme/shop"]["last_seen"], 12)

    def test_backfill_sends_last_n_oldest_first(self):
        h = Harness()
        h.run({"acme/shop": [issue(12), issue(11), issue(10)]}, backfill=2)
        self.assertEqual(len(h.sent), 2)
        self.assertIn("#11", h.sent[0])
        self.assertIn("#12", h.sent[1])

    def test_repo_with_no_issues_starts_at_zero(self):
        h = Harness()
        h.run({"acme/shop": []})
        self.assertEqual(h.state["repos"]["acme/shop"]["last_seen"], 0)


class LaterRuns(unittest.TestCase):
    def test_only_issues_above_watermark_are_sent(self):
        h = Harness({"repos": {"acme/shop": {"last_seen": 10}}})
        sent = h.run({"acme/shop": [issue(12), issue(11), issue(10), issue(9)]})
        self.assertEqual(sent, 2)
        self.assertIn("#11", h.sent[0])  # oldest first
        self.assertEqual(h.state["repos"]["acme/shop"]["last_seen"], 12)

    def test_running_twice_never_sends_twice(self):
        h = Harness({"repos": {"acme/shop": {"last_seen": 10}}})
        data = {"acme/shop": [issue(11)]}
        h.run(data)
        h.run(data)
        self.assertEqual(len(h.sent), 1)

    def test_state_saved_after_each_issue(self):
        h = Harness({"repos": {"acme/shop": {"last_seen": 10}}})
        h.run({"acme/shop": [issue(12), issue(11)]})
        self.assertEqual(h.saves, 2)

    def test_crash_mid_run_resumes_after_last_sent(self):
        h = Harness({"repos": {"acme/shop": {"last_seen": 10}}})

        def flaky_send(text):
            if "#12" in text:
                raise RuntimeError("Telegram down")
            h.sent.append(text)
        with self.assertRaises(RuntimeError):
            monitor.run_once(CONFIG, h.state, {"acme/shop": [issue(12), issue(11)]},
                             flaky_send, lambda: None)
        self.assertEqual(h.state["repos"]["acme/shop"]["last_seen"], 11)  # #12 retried next run

    def test_closed_issues_skipped_but_watermark_advances(self):
        h = Harness({"repos": {"acme/shop": {"last_seen": 10}}})
        h.run({"acme/shop": [issue(11, state="CLOSED")]})
        self.assertEqual(h.sent, [])
        self.assertEqual(h.state["repos"]["acme/shop"]["last_seen"], 11)

    def test_closed_issues_sent_when_enabled(self):
        h = Harness({"repos": {"acme/shop": {"last_seen": 10}}})
        h.run({"acme/shop": [issue(11, state="CLOSED")]}, config={**CONFIG, "include_closed": True})
        self.assertIn("already closed", h.sent[0])

    def test_gap_warning_when_every_fetched_issue_is_new(self):
        h = Harness({"repos": {"acme/shop": {"last_seen": 1}}})
        h.run({"acme/shop": [issue(n) for n in range(10, 5, -1)]})  # 5 = issues_per_repo
        self.assertEqual(len(h.sent), 6)
        self.assertIn("some older ones may not have been sent", h.sent[-1])

    def test_missing_repo_is_skipped_not_fatal(self):
        h = Harness()
        config = {**CONFIG, "repos": ["acme/gone", "acme/shop"]}
        h.run({"acme/gone": None, "acme/shop": [issue(5)]}, config=config)
        self.assertNotIn("acme/gone", h.state["repos"])
        self.assertIn("acme/shop", h.state["repos"])


class Formatting(unittest.TestCase):
    NOW = datetime(2026, 11, 16, 10, 7, tzinfo=timezone.utc)

    def test_html_in_issue_is_escaped(self):
        text = monitor.format_issue("acme/shop", issue(1, title="<b>bold</b> & <script>"), 50, self.NOW)
        self.assertIn("&lt;b&gt;bold&lt;/b&gt; &amp; &lt;script&gt;", text)

    def test_long_body_is_truncated(self):
        text = monitor.format_issue("acme/shop", issue(1, body="x" * 500), 50, self.NOW)
        self.assertIn("x" * 50 + "…", text)
        self.assertNotIn("x" * 51, text)

    def test_age_and_deleted_author(self):
        data = issue(1)
        data["author"] = None
        text = monitor.format_issue("acme/shop", data, 50, self.NOW)
        self.assertIn("7 min ago", text)
        self.assertIn("ghost", text)


class Query(unittest.TestCase):
    def test_one_aliased_block_per_repo(self):
        q = monitor.build_query(["a/b", "c/d"], 20)
        self.assertIn('r0: repository(owner: "a", name: "b")', q)
        self.assertIn('r1: repository(owner: "c", name: "d")', q)
        self.assertIn("issues(first: 20", q)


if __name__ == "__main__":
    unittest.main()
