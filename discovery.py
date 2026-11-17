#!/usr/bin/env python3
"""GitHub issue monitor, Phase 3: similar-issue discovery.

Each run:
  1. re-searches the "neighbour" repos (popular repos sharing a watched repo's
     topics) for a few watched repos, so each one is refreshed about weekly
     without any single run taking long,
  2. fetches new issues from all neighbours (one GraphQL call per 15 repos),
  3. only if there are new issues: builds a profile (description, topics,
     README, 30 newest issues) of just the watched repos they relate to,
  4. scores each new issue against those repos' issues and profiles,
  5. sends ONE Telegram digest with the best matches and why they matched.

Scoring is plain TF-IDF word matching: free, instant, no model to download,
and it can say exactly which words matched. It finds "same words, same
problem", not deep paraphrases; that trade-off favours fewer, clearer alerts.
"""
from __future__ import annotations

import argparse
import html
import json
import logging
import math
import re
import sys
import time
from collections import Counter
from datetime import date, timedelta
from pathlib import Path
from typing import Callable

import monitor  # reuse config, retries, GitHub batching, Telegram, watermarks

log = logging.getLogger("discovery")

SEARCH_REPOS_URL = "https://api.github.com/search/repositories"
SEARCH_PAUSE_SECONDS = 2.5  # repo search allows 30 requests/minute
PROFILE_BATCH = 5           # READMEs + 30 issues each are big; 10 per request hit GitHub's 502 timeout
TOPICS_BATCH = 25           # topics alone are tiny
REFRESH_DAYS = 7            # re-search a watched repo's neighbours this often
MIN_WORDS = 8               # ignore issues with fewer meaningful words than this
MIN_SHARED_TERMS = 3        # a match needs at least this many words in common

DEFAULTS = {
    "enabled": True,
    "neighbors_per_repo": 8,   # neighbour repos found per watched repo
    "refresh_per_run": 12,     # watched repos whose neighbours are re-searched per run (~8 s each)
    "min_stars": 50,           # ignore tiny or abandoned repos
    "issues_per_neighbor": 10,
    "extra_repos": [],         # always scan these too
    "exclude_repos": [],       # never scan these
    "strong": 0.25,            # score at or above this: "Strong match"
    "possible": 0.15,          # at or above this: "Possible match"; below: dropped
    "max_per_digest": 5,
}


# --------------------------------------------------------------------------
# Text similarity: TF-IDF + cosine
# --------------------------------------------------------------------------

# Words too common to say anything about the problem. Includes the headings
# of GitHub issue templates ("steps to reproduce", "expected behavior", ...).
STOPWORDS = set("""
the and for are but not you your with this that have has had was were will would
can could should into from about when what which while where there their them they
then than also just only some any all more most other such very use using used
issue issues bug bugs problem error steps step reproduce expected behavior behaviour
actual description describe version versions environment please thanks thank
following does doesn did didn don get got make made like need want work works
working new one two after before here how why who its it's i'm i've we our out
same see seen yes because way still even http https www com github github.com
none null true false info log logs output code example run running ran try tried
kind type what's let let's sure think know really able each every many much
review reviewed reviewer reviews comment comments ups follow pull request merged
""".split())

WORD = re.compile(r"[a-z][a-z0-9]{2,}")


def tokens(text: str) -> list[str]:
    return [w for w in WORD.findall(text.lower()) if w not in STOPWORDS]


class Corpus:
    """IDF weights from all texts in this run: words that appear in few texts
    (e.g. "vram", "hydration") count more than words that appear everywhere."""

    def __init__(self, documents: list[list[str]]):
        doc_freq = Counter()
        for doc in documents:
            doc_freq.update(set(doc))
        n = len(documents)
        self.idf = {t: math.log((1 + n) / (1 + c)) + 1 for t, c in doc_freq.items()}
        self.unknown_idf = math.log(1 + n) + 1

    def vector(self, words: list[str]) -> dict[str, float]:
        counts = Counter(words)
        vec = {t: (1 + math.log(c)) * self.idf.get(t, self.unknown_idf)
               for t, c in counts.items()}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {t: v / norm for t, v in vec.items()}


def cosine(a: dict[str, float], b: dict[str, float]) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(v * b[t] for t, v in a.items() if t in b)


def shared_terms(a: dict[str, float], b: dict[str, float], k: int = 4) -> list[str]:
    """The words contributing most to the similarity: the "why it matched"."""
    common = [(a[t] * b[t], t) for t in a if t in b]
    return [t for _, t in sorted(common, reverse=True)[:k]]


def issue_text(issue: dict) -> str:
    # Title twice: it's the densest summary of the problem.
    body = (issue.get("bodyText") or "")[:1500]
    return f"{issue['title']} {issue['title']} {body}"


# --------------------------------------------------------------------------
# GitHub: watched-repo profiles and neighbour search
# --------------------------------------------------------------------------

def build_profile_query(repos: list[str]) -> str:
    blocks = []
    for i, full_name in enumerate(repos):
        owner, name = full_name.split("/")
        blocks.append(
            f"r{i}: repository(owner: {json.dumps(owner)}, name: {json.dumps(name)}) {{\n"
            "  description\n"
            "  repositoryTopics(first: 10) { nodes { topic { name } } }\n"
            '  readme: object(expression: "HEAD:README.md") { ... on Blob { text } }\n'
            "  issues(first: 30, orderBy: {field: CREATED_AT, direction: DESC}) {\n"
            "    nodes { number title url bodyText }\n"
            "  }\n"
            "}")
    return "query {\n" + "\n".join(blocks) + "\nrateLimit { cost }\n}"


def build_topics_query(repos: list[str]) -> str:
    blocks = []
    for i, full_name in enumerate(repos):
        owner, name = full_name.split("/")
        blocks.append(
            f"r{i}: repository(owner: {json.dumps(owner)}, name: {json.dumps(name)}) "
            "{ repositoryTopics(first: 10) { nodes { topic { name } } } }")
    return "query {\n" + "\n".join(blocks) + "\nrateLimit { cost }\n}"


def _graphql(query: str, token: str, label: str) -> dict:
    resp = monitor.request_with_retry(
        "POST", monitor.GITHUB_GRAPHQL_URL, label=label,
        headers={"Authorization": f"Bearer {token}"}, json={"query": query})
    if not resp.ok:
        raise RuntimeError(f"GitHub returned HTTP {resp.status_code}: {resp.text[:200]}")
    data = resp.json().get("data") or {}
    log.info("%s query cost %s point(s)", label, (data.get("rateLimit") or {}).get("cost"))
    return data


def fetch_topics(repos: list[str], token: str) -> dict[str, list[str]]:
    """{"owner/name": [topic, ...]}: all the neighbour search needs, so it
    doesn't pay for full profiles. Repos that can't be read are left out."""
    topics = {}
    for start in range(0, len(repos), TOPICS_BATCH):
        batch = repos[start:start + TOPICS_BATCH]
        data = _graphql(build_topics_query(batch), token, "GitHub topics")
        for i, full_name in enumerate(batch):
            if data.get(f"r{i}"):
                topics[full_name] = [n["topic"]["name"]
                                     for n in data[f"r{i}"]["repositoryTopics"]["nodes"]]
    return topics


def fetch_profiles(repos: list[str], token: str) -> dict:
    """{"owner/name": {"text": ..., "topics": [...], "issues": [...]}}; repos that
    can't be read are left out."""
    profiles = {}
    for start in range(0, len(repos), PROFILE_BATCH):
        batch = repos[start:start + PROFILE_BATCH]
        data = _graphql(build_profile_query(batch), token, "GitHub profiles")
        for i, full_name in enumerate(batch):
            repo = data.get(f"r{i}")
            if not repo:
                log.warning("%s: can't read profile, skipped", full_name)
                continue
            topics = [n["topic"]["name"] for n in repo["repositoryTopics"]["nodes"]]
            readme = ((repo.get("readme") or {}).get("text") or "")[:3000]
            profiles[full_name] = {
                "text": " ".join([repo.get("description") or "", " ".join(topics), readme]),
                "topics": topics,
                "issues": repo["issues"]["nodes"],
            }
    return profiles


def find_neighbors(repo: str, topics: list[str], skip: set[str], token: str,
                   limit: int, min_stars: int) -> list[str]:
    """Popular, active repos sharing this repo's topics.

    Searches the first 3 topics (to stay inside the search rate limit), then
    ranks every result by how many of ALL our topics it shares, using the
    topic list GitHub returns for each result. Sharing just one broad topic
    like "javascript" says little, so when we have 2+ topics, a neighbour
    must share at least 2."""
    topics = [t for t in topics if not is_meta_topic(t)]
    if not topics:
        log.warning("%s has no (useful) topics on GitHub: no neighbours found. "
                    "Add some manually under discovery.extra_repos.", repo)
        return []
    # Search the specific topics first ("physics-ml" before "deep-learning"):
    # broad ones mostly return the biggest generic repos and tutorials.
    topics = sorted(topics, key=lambda t: t.lower() in BROAD_TOPICS)
    ours = {t.lower() for t in topics}
    specific = ours - BROAD_TOPICS
    need = 2 if len(ours) >= 2 else 1
    since = (date.today() - timedelta(days=90)).isoformat()
    shared, focus, stars = {}, {}, {}
    for topic in topics[:3]:
        resp = monitor.request_with_retry(
            "GET", SEARCH_REPOS_URL, label="GitHub repo search",
            headers={"Authorization": f"Bearer {token}"},
            params={"q": f"topic:{topic} archived:false pushed:>={since} stars:>={min_stars}",
                    "sort": "stars", "per_page": 20})
        if not resp.ok:
            log.warning("Repo search for topic %r failed: HTTP %s", topic, resp.status_code)
            continue
        for item in resp.json().get("items", []):
            name = item["full_name"]
            theirs = {t.lower() for t in item.get("topics", [])}
            overlap = len(ours & theirs)
            if (name.lower() in skip or item.get("fork") or overlap < need
                    or is_list_repo(name, theirs)
                    or (specific and not specific & theirs)):  # "python" alone isn't enough
                continue
            shared[name] = overlap
            # Sharing 2 topics with a repo that has 4 says more than sharing
            # 2 with one tagged with 20 (some repos tag themselves with everything).
            focus[name] = overlap / len(theirs)
            stars[name] = item["stargazers_count"]
        time.sleep(SEARCH_PAUSE_SECONDS)
    return sorted(shared, key=lambda n: (-shared[n], -focus[n], -stars[n]))[:limit]


# Topics that describe a project's community, not what it does: searching
# "gsoc" finds GSoC *list* repos, not similar software.
META_TOPICS = {
    "gsoc", "google-summer-of-code", "hacktoberfest", "good-first-issue",
    "beginner-friendly", "first-timers-only", "open-source", "opensource", "help-wanted",
}


def is_meta_topic(topic: str) -> bool:
    t = topic.lower()  # also catches year variants like "gsoc-2026", "hacktoberfest2025"
    return t in META_TOPICS or t.startswith(("gsoc", "hacktoberfest", "google-summer-of-code"))
# Topics so broad that sharing only these says little about the problem space.
BROAD_TOPICS = {
    "python", "python3", "javascript", "typescript", "java", "go", "golang", "rust",
    "cpp", "c", "julia", "r", "kotlin", "dart", "ruby", "php", "scala",
    "machine-learning", "deep-learning", "ai", "artificial-intelligence", "ml",
    "data-science", "neural-network", "neural-networks", "pytorch", "tensorflow",
    "llm", "llms", "nlp", "computer-vision", "gpu", "cuda", "docker", "kubernetes",
    "cloud", "devops", "web", "frontend", "backend", "api", "framework", "library",
}
# Curated lists, tutorials and courses: lots of stars, no comparable issues.
LIST_TOPICS = {"awesome", "awesome-list", "list", "lists", "tutorial", "tutorials",
               "course", "courses", "book", "books", "roadmap", "interview",
               "cheatsheet", "learning-resources", "resources", "education"}
LIST_NAME_WORDS = ("awesome", "best-of", "for-beginners", "from-scratch", "tutorial",
                   "project-based", "30-days", "roadmap", "interview", "-course", "cheatsheet")


def is_list_repo(full_name: str, topics: set[str]) -> bool:
    name = full_name.split("/")[1].lower()
    return bool(topics & LIST_TOPICS) or any(w in name for w in LIST_NAME_WORDS)


def tidy_neighbor_state(state: dict, watched: list[str]) -> None:
    """Bring an older state file up to date and forget repos no longer watched."""
    neighbors = state.setdefault("neighbors", {})
    checked = state.setdefault("neighbors_checked", {})
    # Before the search was spread across runs, one date covered every repo.
    old_date = state.pop("neighbors_refreshed", None)
    if old_date:
        for repo in neighbors:
            checked.setdefault(repo, old_date)
    for entries in (neighbors, checked):
        for repo in list(entries):
            if repo not in watched:
                del entries[repo]


def repos_to_refresh(state: dict, watched: list[str], today: date, limit: int,
                     force: bool = False) -> list[str]:
    """The watched repos whose neighbours get re-searched this run: ones never
    searched first, then ones last searched REFRESH_DAYS or more ago, oldest
    first. Capped at `limit`, because each repo costs 3 rate-limited searches;
    200 repos in one run would take about 50 minutes. `force` ignores age."""
    checked = state.get("neighbors_checked", {})
    never = [r for r in watched if r not in checked]
    due = sorted((r for r in watched if r in checked and (
                      force or (today - date.fromisoformat(checked[r])).days >= REFRESH_DAYS)),
                 key=lambda r: checked[r])
    return (never + due)[:limit]


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

def rank_candidates(candidates: list[tuple[str, dict]], profiles: dict,
                    related: dict[str, list[str]], dcfg: dict) -> list[dict]:
    """Score each (neighbour repo, issue) against the watched repos it is
    related to. score = 0.7 × best match with one of the watched repo's issues
                       + 0.3 × match with the watched repo's profile.
    Issue-to-issue similarity dominates: "someone elsewhere hit the same
    problem" is what's useful; profile similarity only says "same field"."""
    if not candidates:
        return []
    # Texts too short to judge (or not in a Latin script, so few words are
    # recognised) produce wild scores: two 1-word texts look identical.
    candidates = [(r, i) for r, i in candidates if len(tokens(issue_text(i))) >= MIN_WORDS]
    if not candidates:
        return []
    texts = [tokens(issue_text(issue)) for _, issue in candidates]
    # The watched repos' issues, minus ones too short to judge: [(words, issue)]
    watched_issues = {w: [(t, i) for i in p["issues"]
                          if len(t := tokens(issue_text(i))) >= MIN_WORDS]
                      for w, p in profiles.items()}
    profile_tokens = {w: tokens(p["text"]) for w, p in profiles.items()}
    corpus = Corpus(texts + [t for pairs in watched_issues.values() for t, _ in pairs]
                    + list(profile_tokens.values()))
    # Vectors for the watched side, computed once and reused for every candidate.
    profile_vecs = {w: corpus.vector(t) for w, t in profile_tokens.items()}
    issue_vecs = {w: [(corpus.vector(t), issue) for t, issue in pairs]
                  for w, pairs in watched_issues.items()}

    matches = []
    for (repo, issue), words in zip(candidates, texts):
        cand = corpus.vector(words)
        best = None
        for watched in related.get(repo, []):
            if watched not in profiles:
                continue
            profile_sim = cosine(cand, profile_vecs[watched])
            issue_sim, closest_vec, closest = max(
                ((cosine(cand, vec), vec, i) for vec, i in issue_vecs[watched]),
                key=lambda x: x[0], default=(0.0, {}, None))
            # A match explained by one or two words isn't believable.
            if len(cand.keys() & closest_vec.keys()) < MIN_SHARED_TERMS:
                continue
            score = 0.7 * issue_sim + 0.3 * profile_sim
            if best is None or score > best["score"]:
                terms = shared_terms(cand, closest_vec)
                best = {"repo": repo, "issue": issue, "watched": watched,
                        "score": score, "closest": closest, "terms": terms}
        if best:
            best["tier"] = ("Strong" if best["score"] >= dcfg["strong"] else
                            "Possible" if best["score"] >= dcfg["possible"] else None)
            matches.append(best)
    return sorted(matches, key=lambda m: -m["score"])  # includes below-threshold ones


def format_digest(matches: list[dict]) -> str:
    lines = [f"🔎 <b>Similar issues elsewhere</b> ({len(matches)})"]
    for m in matches:
        issue, closest = m["issue"], m["closest"]
        icon = "🟢" if m["tier"] == "Strong" else "🟡"
        lines += ["",
                  f"{icon} <b>{m['tier']} match</b> · {html.escape(m['repo'])} #{issue['number']}",
                  f"<b>{html.escape(issue['title'][:150])}</b>",
                  f"Related to: {html.escape(m['watched'])}"]
        if m["terms"]:
            lines.append("Why: shared terms " + ", ".join(html.escape(t) for t in m["terms"]))
        if closest:
            lines.append(f'Like: <a href="{html.escape(closest["url"])}">'
                         f"#{closest['number']} {html.escape(closest['title'][:80])}</a>")
        lines.append(f'<a href="{html.escape(issue["url"])}">Open issue →</a>')
    return "\n".join(lines)


# --------------------------------------------------------------------------
# One discovery pass
# --------------------------------------------------------------------------

def is_bot(issue: dict) -> bool:
    login = ((issue.get("author") or {}).get("login") or "").lower()
    return login.endswith("[bot]") or login in {"dependabot", "renovate", "github-actions"}


def run_discovery(config: dict, dcfg: dict, state: dict, fetched: dict,
                  get_profiles: Callable[[list[str]], dict],
                  send: Callable[[str], None], save: Callable[[], None],
                  backfill: int = 0, show_scores: bool = False) -> int:
    """Pick new issues from neighbour repos, rank them, send one digest,
    THEN advance the watermarks (so a failed send is retried next run).
    `get_profiles(watched_repos)` is only called for the watched repos that
    have a new issue to compare against. Returns how many matches were sent."""
    related: dict[str, list[str]] = {}
    for watched, neighbors in state["neighbors"].items():
        for n in neighbors:
            related.setdefault(n, []).append(watched)
    for extra in dcfg["extra_repos"]:
        related[extra] = list(config["repos"])  # compare with every watched repo

    seen = state.setdefault("seen", {})
    candidates, new_marks = [], {}
    for repo, issues in fetched.items():
        if issues is None:
            continue
        if repo not in seen:  # new neighbour: start quietly, like Phase 1
            seen[repo] = monitor.baseline_number(issues, backfill)
        fresh = monitor.new_issues(issues, seen[repo])
        if fresh:
            new_marks[repo] = fresh[-1]["number"]
        candidates += [(repo, i) for i in fresh if i["state"] == "OPEN" and not is_bot(i)]

    # Profiles are the expensive part (README + 30 issues per repo), so load
    # them only for watched repos that a scoreable new issue relates to.
    needed = sorted({watched for repo, issue in candidates
                     if len(tokens(issue_text(issue))) >= MIN_WORDS
                     for watched in related.get(repo, [])})
    profiles = get_profiles(needed) if needed else {}
    scored = rank_candidates(candidates, profiles, related, dcfg)
    matches = [m for m in scored if m["tier"]]
    log.info("%d new issue(s) in neighbour repos, compared with %d watched repo(s), "
             "%d above the threshold", len(candidates), len(needed), len(matches))
    if show_scores:  # top 15 whatever their score, to help pick thresholds
        for m in scored[:15]:
            log.info("  %.3f %-8s %s #%s %s  [%s]", m["score"], m["tier"] or "-", m["repo"],
                     m["issue"]["number"], m["issue"]["title"][:60], ", ".join(m["terms"]))

    top = matches[:dcfg["max_per_digest"]]
    if top:
        send(format_digest(top))
    seen.update(new_marks)
    for repo in list(seen):  # forget repos that are no longer neighbours
        if repo not in related:
            del seen[repo]
    save()
    return len(top)


def main() -> int:
    parser = argparse.ArgumentParser(description="Find similar issues in other repos.")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--state", type=Path, default=Path("discovery_state.json"))
    parser.add_argument("--dry-run", action="store_true",
                        help="print the digest instead of sending; don't save state")
    parser.add_argument("--backfill", type=int, default=0, metavar="N",
                        help="treat the last N issues of each NEW neighbour as candidates")
    parser.add_argument("--show-scores", action="store_true",
                        help="log every match with its score (for tuning thresholds)")
    parser.add_argument("--refresh-neighbors", action="store_true",
                        help="re-search the next batch of neighbours now, even if under a week old")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    monitor.load_dotenv(Path(".env"))

    config = monitor.load_config(args.config)
    dcfg = {**DEFAULTS, **(config.get("discovery") or {})}
    if not dcfg["enabled"]:
        log.info("Discovery is disabled in config.yaml")
        return 0

    if args.dry_run:
        def send(text: str) -> None:
            print("─" * 60 + "\n" + text)
    else:
        send = monitor.make_telegram_sender(monitor.require_env("TELEGRAM_BOT_TOKEN"),
                                            monitor.require_env("TELEGRAM_CHAT_ID"))
    token = monitor.require_env("GITHUB_TOKEN")
    state = json.loads(args.state.read_text()) if args.state.exists() else {}
    save = (lambda: None) if args.dry_run else (lambda: monitor.save_state(args.state, state))

    watched = config["repos"]
    skip = {r.lower() for r in watched} | {r.lower() for r in dcfg["exclude_repos"]}
    tidy_neighbor_state(state, watched)
    batch = repos_to_refresh(state, watched, date.today(), dcfg["refresh_per_run"],
                             force=args.refresh_neighbors)
    if batch:
        topics = fetch_topics(batch, token)
        for repo in batch:
            if repo in topics:
                found = find_neighbors(repo, topics[repo], skip, token,
                                       dcfg["neighbors_per_repo"], dcfg["min_stars"])
            else:
                log.warning("%s: can't read its topics (renamed or private?)", repo)
                found = []
            state["neighbors"][repo] = found
            state["neighbors_checked"][repo] = date.today().isoformat()
            log.info("Neighbours of %s: %s", repo, ", ".join(found) or "none")
        save()
        waiting = sum(1 for r in watched if r not in state["neighbors_checked"])
        log.info("Neighbour search: %d repo(s) refreshed this run, %d not searched yet",
                 len(batch), waiting)

    to_scan = sorted({n for ns in state["neighbors"].values() for n in ns}
                     | set(dcfg["extra_repos"]))
    to_scan = [r for r in to_scan if r.lower() not in skip]
    log.info("Scanning %d neighbour repo(s) for new issues", len(to_scan))
    fetched = monitor.fetch_recent_issues(to_scan, dcfg["issues_per_neighbor"], token)

    sent = run_discovery(config, dcfg, state, fetched,
                         lambda repos: fetch_profiles(repos, token), send, save,
                         backfill=args.backfill, show_scores=args.show_scores)
    log.info("Done: %d match(es) %s", sent, "printed (dry run)" if args.dry_run else "sent")
    return 0


if __name__ == "__main__":
    sys.exit(main())
