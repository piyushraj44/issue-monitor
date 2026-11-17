"""Tests for similar-issue discovery, using fake data (no network, no tokens).

Run with:  python -m unittest discover -s tests -v
"""
import sys
import unittest
from datetime import date
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import discovery  # noqa: E402

DCFG = {**discovery.DEFAULTS}
CONFIG = {"repos": ["acme/shop"]}


def issue(number, title, body="", repo="other/store", state="OPEN", author="bob"):
    return {"number": number, "title": title, "bodyText": body, "state": state,
            "url": f"https://github.com/{repo}/issues/{number}", "author": {"login": author}}


# A watched e-commerce repo, as fetch_profiles would return it.
SHOP_PROFILE = {
    "acme/shop": {
        "text": "Open source e-commerce storefront with shopping cart, checkout and payments",
        "topics": ["ecommerce", "cart", "react"],
        "issues": [
            issue(41, "Cart contents lost when the page is reloaded",
                  "Shopping cart items saved in localStorage vanish after a browser reload; "
                  "cart state should persist across sessions", repo="acme/shop"),
            issue(42, "Checkout button misaligned on mobile Safari",
                  "The checkout button overlaps the footer on narrow screens in Safari", repo="acme/shop"),
        ],
    }
}
CART_BUG = issue(7, "Cart items disappear after refreshing the page",
                 "Items added to the shopping cart are gone after a browser refresh. "
                 "Looks like cart state is not restored from localStorage.")
DARK_MODE = issue(8, "Dark mode toggle uses wrong colours",
                  "Switching the theme toggle leaves some panels with light background colours "
                  "and unreadable grey text in the sidebar navigation.")


class Text(unittest.TestCase):
    def test_tokens_drop_stopwords_and_short_words(self):
        self.assertEqual(discovery.tokens("Steps to reproduce: the CART is empty on reload"),
                         ["cart", "empty", "reload"])

    def test_cosine_identical_and_disjoint(self):
        corpus = discovery.Corpus([["cart", "reload"], ["theme", "colour"]])
        a, b = corpus.vector(["cart", "reload"]), corpus.vector(["theme", "colour"])
        self.assertAlmostEqual(discovery.cosine(a, a), 1.0)
        self.assertEqual(discovery.cosine(a, b), 0.0)


class Ranking(unittest.TestCase):
    RELATED = {"other/store": ["acme/shop"]}

    def rank(self, *issues):
        return discovery.rank_candidates([("other/store", i) for i in issues],
                                         SHOP_PROFILE, self.RELATED, DCFG)

    def test_cart_bug_matches_and_explains_why(self):
        top = self.rank(DARK_MODE, CART_BUG)[0]
        self.assertEqual(top["issue"]["number"], 7)
        self.assertIsNotNone(top["tier"])
        self.assertEqual(top["closest"]["number"], 41)  # the "lost on reload" issue
        self.assertIn("cart", top["terms"])

    def test_unrelated_issue_is_not_a_match(self):
        scored = self.rank(DARK_MODE)
        self.assertTrue(all(m["tier"] is None for m in scored))

    def test_too_short_or_non_latin_issue_is_ignored(self):
        self.assertEqual(self.rank(issue(9, "购物车刷新后清空"), issue(10, "cart bug")), [])

    def test_repo_not_related_to_any_watched_repo_is_not_scored(self):
        scored = discovery.rank_candidates([("unrelated/x", CART_BUG)], SHOP_PROFILE, {}, DCFG)
        self.assertEqual(scored, [])


class Digest(unittest.TestCase):
    def test_digest_escapes_html_and_links_closest_issue(self):
        match = {"repo": "other/store", "issue": issue(7, "<b>Cart</b> & co"),
                 "watched": "acme/shop", "score": 0.4, "tier": "Strong",
                 "terms": ["cart", "localstorage"], "closest": SHOP_PROFILE["acme/shop"]["issues"][0]}
        text = discovery.format_digest([match])
        self.assertIn("&lt;b&gt;Cart&lt;/b&gt; &amp; co", text)
        self.assertIn("🟢 <b>Strong match</b>", text)
        self.assertIn("acme/shop/issues/41", text)
        self.assertIn("shared terms cart, localstorage", text)


class Neighbours(unittest.TestCase):
    def test_refresh_when_missing_changed_or_week_old(self):
        today = date(2026, 11, 17)
        fresh = {"neighbors": {"acme/shop": []}, "neighbors_refreshed": "2026-11-14"}
        self.assertTrue(discovery.neighbors_stale({}, ["acme/shop"], today))
        self.assertFalse(discovery.neighbors_stale(fresh, ["acme/shop"], today))
        self.assertTrue(discovery.neighbors_stale(fresh, ["acme/shop", "acme/blog"], today))
        old = {**fresh, "neighbors_refreshed": "2026-11-09"}
        self.assertTrue(discovery.neighbors_stale(old, ["acme/shop"], today))

    def test_neighbours_must_share_two_topics_and_skip_forks_and_watched(self):
        items = [
            {"full_name": "big/generic", "topics": ["react"], "stargazers_count": 90000},
            {"full_name": "good/store", "topics": ["ecommerce", "cart"], "stargazers_count": 500},
            {"full_name": "best/store", "topics": ["ecommerce", "cart", "react"], "stargazers_count": 100},
            {"full_name": "a/fork", "topics": ["ecommerce", "cart"], "stargazers_count": 999, "fork": True},
            {"full_name": "acme/shop", "topics": ["ecommerce", "cart"], "stargazers_count": 999},
        ]
        fake = mock.Mock(ok=True)
        fake.json.return_value = {"items": items}
        with mock.patch.object(discovery.monitor, "request_with_retry", return_value=fake), \
             mock.patch.object(discovery.time, "sleep"):
            found = discovery.find_neighbors("acme/shop", ["ecommerce", "cart", "react"],
                                             {"acme/shop"}, "token", limit=8, min_stars=50)
        self.assertEqual(found, ["best/store", "good/store"])  # 3 shared topics first

    def test_lists_tutorials_and_meta_topics_are_ignored(self):
        items = [
            {"full_name": "x/awesome-shops", "topics": ["ecommerce", "cart"], "stargazers_count": 9000},
            {"full_name": "x/shop-course", "topics": ["ecommerce", "cart", "tutorial"], "stargazers_count": 9000},
            {"full_name": "x/gsoc-orgs", "topics": ["gsoc", "hacktoberfest"], "stargazers_count": 9000},
            {"full_name": "tagged/everything", "topics": ["ecommerce", "cart"] + [f"t{i}" for i in range(18)],
             "stargazers_count": 9000},
            {"full_name": "focused/store", "topics": ["ecommerce", "cart"], "stargazers_count": 10},
        ]
        fake = mock.Mock(ok=True)
        fake.json.return_value = {"items": items}
        with mock.patch.object(discovery.monitor, "request_with_retry", return_value=fake) as call, \
             mock.patch.object(discovery.time, "sleep"):
            found = discovery.find_neighbors("acme/shop", ["gsoc", "ecommerce", "cart"],
                                             set(), "token", limit=8, min_stars=50)
        self.assertEqual(found, ["focused/store", "tagged/everything"])  # focused one first
        searched = [c.kwargs["params"]["q"] for c in call.call_args_list]
        self.assertFalse(any("topic:gsoc" in q for q in searched))


class RunDiscovery(unittest.TestCase):
    def setUp(self):
        self.sent = []
        self.state = {"neighbors": {"acme/shop": ["other/store"]}, "seen": {"other/store": 6}}

    def run_it(self, issues, send=None):
        return discovery.run_discovery(CONFIG, DCFG, self.state, SHOP_PROFILE,
                                       {"other/store": issues}, send or self.sent.append,
                                       lambda: None)

    def test_new_matching_issue_is_sent_once(self):
        self.run_it([CART_BUG, DARK_MODE])
        self.run_it([CART_BUG, DARK_MODE])
        self.assertEqual(len(self.sent), 1)
        self.assertIn("Cart items disappear", self.sent[0])
        self.assertNotIn("Dark mode", self.sent[0])
        self.assertEqual(self.state["seen"]["other/store"], 8)

    def test_new_neighbour_starts_quietly(self):
        self.state["seen"] = {}
        self.run_it([CART_BUG])
        self.assertEqual(self.sent, [])
        self.assertEqual(self.state["seen"]["other/store"], 7)

    def test_failed_send_keeps_watermark_for_retry(self):
        def broken(_):
            raise RuntimeError("Telegram down")
        with self.assertRaises(RuntimeError):
            self.run_it([CART_BUG], send=broken)
        self.assertEqual(self.state["seen"]["other/store"], 6)

    def test_bot_and_closed_issues_are_ignored(self):
        bot = {**CART_BUG, "number": 7, "author": {"login": "renovate[bot]"}}
        closed = {**CART_BUG, "number": 8, "state": "CLOSED"}
        self.run_it([bot, closed])
        self.assertEqual(self.sent, [])
        self.assertEqual(self.state["seen"]["other/store"], 8)  # still marked as seen

    def test_ex_neighbour_is_forgotten(self):
        self.state["seen"]["gone/repo"] = 3
        self.run_it([])
        self.assertNotIn("gone/repo", self.state["seen"])


if __name__ == "__main__":
    unittest.main()
