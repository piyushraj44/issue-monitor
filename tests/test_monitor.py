"""Tests for the decision logic, using fake GitHub data (no network, no tokens).

Run with:  python -m unittest discover -s tests -v
"""
import sys
import tempfile
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


class Phase2(unittest.TestCase):
    def test_duplicate_repos_in_config_are_watched_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            path.write_text("repos:\n  - acme/shop\n  - Acme/Shop\n  - acme/blog\n")
            with self.assertLogs("monitor", "WARNING"):
                config = monitor.load_config(path)
        self.assertEqual(config["repos"], ["acme/shop", "acme/blog"])
        self.assertEqual(config["heartbeat"], {"enabled": True, "hour": 9, "timezone": "UTC"})

    def test_removed_repo_is_forgotten(self):
        h = Harness({"repos": {"acme/shop": {"last_seen": 5}, "acme/old": {"last_seen": 9}}})
        h.run({"acme/shop": [issue(5)]})
        self.assertNotIn("acme/old", h.state["repos"])

    def test_alerts_are_counted_for_the_heartbeat(self):
        h = Harness({"repos": {"acme/shop": {"last_seen": 10}}, "alerts_since_heartbeat": 3})
        h.run({"acme/shop": [issue(11), issue(12)]})
        self.assertEqual(h.state["alerts_since_heartbeat"], 5)

    def test_large_repo_lists_are_fetched_in_batches(self):
        batches = []
        original = monitor._fetch_batch
        monitor._fetch_batch = lambda repos, per_repo, token: batches.append(len(repos)) or {}
        try:
            monitor.fetch_recent_issues([f"o/r{i}" for i in range(60)], 20, "token")
        finally:
            monitor._fetch_batch = original
        self.assertEqual(batches, [25, 25, 10])


class Heartbeat(unittest.TestCase):
    MORNING = datetime(2026, 11, 17, 9, 30, tzinfo=timezone.utc)

    def test_due_after_the_hour_once_per_day(self):
        self.assertTrue(monitor.heartbeat_due({}, self.MORNING, hour=9))
        self.assertFalse(monitor.heartbeat_due({}, self.MORNING, hour=10))
        self.assertFalse(monitor.heartbeat_due({"last_heartbeat": "2026-11-17"}, self.MORNING, 9))
        self.assertTrue(monitor.heartbeat_due({"last_heartbeat": "2026-11-16"}, self.MORNING, 9))

    def test_message_with_stats_and_missing_repo(self):
        text = monitor.format_heartbeat(50, 7, {"runs": 230, "failed": 2}, ["acme/gone"])
        self.assertIn("Watching 50 repos · 7 alert(s)", text)
        self.assertIn("230 runs, 2 failed", text)
        self.assertIn("Can't read: acme/gone", text)

    def test_message_without_stats(self):
        text = monitor.format_heartbeat(2, 0, None, [])
        self.assertIn("Run stats unavailable", text)
        self.assertNotIn("Can't read", text)


if __name__ == "__main__":
    unittest.main()
