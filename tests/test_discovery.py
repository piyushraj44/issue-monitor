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


class NeighbourRefresh(unittest.TestCase):
    """Which watched repos get their neighbours re-searched in one run."""
    TODAY = date(2026, 11, 17)

    def pick(self, checked, watched, limit=12, force=False):
        return discovery.repos_to_refresh({"neighbors_checked": checked}, watched,
                                          self.TODAY, limit, force)

    def test_never_searched_repos_come_first_in_config_order(self):
        checked = {"a/old": "2026-11-01"}
        self.assertEqual(self.pick(checked, ["a/old", "a/new1", "a/new2"]),
                         ["a/new1", "a/new2", "a/old"])

    def test_only_week_old_repos_are_due_oldest_first(self):
        checked = {"a/6days": "2026-11-11", "a/7days": "2026-11-10", "a/20days": "2026-10-28"}
        self.assertEqual(self.pick(checked, list(checked)), ["a/20days", "a/7days"])

    def test_limit_caps_the_batch_and_the_rest_wait_for_later_runs(self):
        watched = [f"a/r{i}" for i in range(200)]
        self.assertEqual(self.pick({}, watched), watched[:12])
        self.assertEqual(self.pick({r: "2026-11-17" for r in watched[:12]}, watched),
                         watched[12:24])

    def test_nothing_to_do_when_everything_is_fresh(self):
        self.assertEqual(self.pick({"a/x": "2026-11-16"}, ["a/x"]), [])

    def test_force_takes_the_oldest_regardless_of_age(self):
        checked = {"a/yesterday": "2026-11-16", "a/today": "2026-11-17"}
        self.assertEqual(self.pick(checked, list(checked), limit=1, force=True), ["a/yesterday"])

    def test_removed_repos_are_dropped_from_state(self):
        state = {"neighbors": {"a/keep": ["n/1"], "a/gone": ["n/2"]},
                 "neighbors_checked": {"a/keep": "2026-11-16", "a/gone": "2026-11-16"}}
        discovery.tidy_neighbor_state(state, ["a/keep"])
        self.assertEqual(state["neighbors"], {"a/keep": ["n/1"]})
        self.assertEqual(state["neighbors_checked"], {"a/keep": "2026-11-16"})

    def test_old_state_with_one_date_is_migrated(self):
        state = {"neighbors": {"a/x": ["n/1"], "a/y": []}, "neighbors_refreshed": "2026-11-12"}
        discovery.tidy_neighbor_state(state, ["a/x", "a/y", "a/added"])
        self.assertNotIn("neighbors_refreshed", state)
        self.assertEqual(state["neighbors_checked"], {"a/x": "2026-11-12", "a/y": "2026-11-12"})
        # Only the newly added repo needs a search now; the others aren't a week old.
        self.assertEqual(discovery.repos_to_refresh(state, ["a/x", "a/y", "a/added"],
                                                    self.TODAY, 12), ["a/added"])

    def test_empty_state_gets_the_keys_it_needs(self):
        state = {}
        discovery.tidy_neighbor_state(state, ["a/x"])
        self.assertEqual(state, {"neighbors": {}, "neighbors_checked": {}})


class Neighbours(unittest.TestCase):
    def test_fetch_topics_asks_for_topics_only_in_batches_of_25(self):
        fake = mock.Mock(ok=True)
        fake.json.return_value = {"data": {
            "r0": {"repositoryTopics": {"nodes": [{"topic": {"name": "cart"}}]}},
            "r1": None, "rateLimit": {"cost": 1}}}
        with mock.patch.object(discovery.monitor, "request_with_retry", return_value=fake) as call:
            topics = discovery.fetch_topics([f"o/r{i}" for i in range(30)], "token")
        self.assertEqual(call.call_count, 2)
        self.assertEqual(topics["o/r0"], ["cart"])
        self.assertNotIn("o/r1", topics)  # unreadable repo is left out
        query = call.call_args_list[0].kwargs["json"]["query"]
        self.assertNotIn("readme", query)
        self.assertNotIn("issues", query)

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
        self.profile_requests = []  # each call to get_profiles: the repos it asked for
        self.state = {"neighbors": {"acme/shop": ["other/store"]}, "seen": {"other/store": 6}}

    def get_profiles(self, repos):
        self.profile_requests.append(repos)
        return {r: SHOP_PROFILE["acme/shop"] for r in repos}

    def run_it(self, issues, send=None, config=CONFIG, dcfg=DCFG, fetched=None):
        return discovery.run_discovery(config, dcfg, self.state,
                                       fetched or {"other/store": issues}, self.get_profiles,
                                       send or self.sent.append, lambda: None)

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

    # --- profiles are loaded only when, and for whom, they are needed -------

    def test_no_new_issues_means_no_profiles_are_loaded(self):
        self.state["seen"]["other/store"] = 8  # everything already seen
        self.run_it([CART_BUG, DARK_MODE])
        self.assertEqual(self.profile_requests, [])

    def test_new_neighbour_baseline_loads_no_profiles(self):
        self.state["seen"] = {}
        self.run_it([CART_BUG])
        self.assertEqual(self.profile_requests, [])

    def test_bot_closed_and_too_short_issues_load_no_profiles(self):
        bot = {**CART_BUG, "author": {"login": "dependabot"}}
        closed = {**DARK_MODE, "state": "CLOSED"}
        self.run_it([bot, closed, issue(9, "cart bug")])
        self.assertEqual(self.profile_requests, [])
        self.assertEqual(self.state["seen"]["other/store"], 9)

    def test_profiles_loaded_only_for_watched_repos_with_a_new_issue(self):
        config = {"repos": ["acme/shop", "acme/blog", "acme/docs"]}
        self.state = {"neighbors": {"acme/shop": ["other/store"], "acme/blog": ["other/cms"],
                                    "acme/docs": ["other/cms"]},
                      "seen": {"other/store": 6, "other/cms": 20}}
        self.run_it(None, config=config,
                    fetched={"other/store": [CART_BUG], "other/cms": [issue(20, "Old one")]})
        self.assertEqual(self.profile_requests, [["acme/shop"]])
        self.assertEqual(len(self.sent), 1)

    def test_neighbour_shared_by_two_watched_repos_loads_both(self):
        config = {"repos": ["acme/shop", "acme/blog"]}
        self.state = {"neighbors": {"acme/shop": ["other/store"], "acme/blog": ["other/store"]},
                      "seen": {"other/store": 6}}
        self.run_it([CART_BUG], config=config)
        self.assertEqual(self.profile_requests, [["acme/blog", "acme/shop"]])

    def test_extra_repo_issue_is_compared_with_every_watched_repo(self):
        config = {"repos": ["acme/shop", "acme/blog"]}
        dcfg = {**DCFG, "extra_repos": ["extra/repo"]}
        self.state = {"neighbors": {"acme/shop": ["other/store"], "acme/blog": []},
                      "seen": {"other/store": 6, "extra/repo": 6}}
        # Nothing new in the extra repo: only the related watched repo is loaded.
        self.run_it(None, config=config, dcfg=dcfg,
                    fetched={"other/store": [CART_BUG], "extra/repo": []})
        self.assertEqual(self.profile_requests, [["acme/shop"]])
        # A new issue in the extra repo: every watched repo is loaded.
        self.run_it(None, config=config, dcfg=dcfg,
                    fetched={"other/store": [CART_BUG], "extra/repo": [{**CART_BUG, "number": 9}]})
        self.assertEqual(self.profile_requests[-1], ["acme/blog", "acme/shop"])

    def test_failed_profile_load_keeps_watermark_for_retry(self):
        def broken(_):
            raise RuntimeError("GitHub 502")
        with self.assertRaises(RuntimeError):
            discovery.run_discovery(CONFIG, DCFG, self.state, {"other/store": [CART_BUG]},
                                    broken, self.sent.append, lambda: None)
        self.assertEqual(self.sent, [])
        self.assertEqual(self.state["seen"]["other/store"], 6)


if __name__ == "__main__":
    unittest.main()
