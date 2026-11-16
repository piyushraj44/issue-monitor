#!/usr/bin/env python3
"""GitHub issue monitor, Phase 1: direct monitoring.

Each run:
  1. reads the repo list (config.yaml) and what we've already seen (state.json),
  2. asks GitHub for the newest issues of every repo in ONE GraphQL request,
  3. sends a Telegram message for each issue newer than what we've seen,
  4. records the newest issue number per repo so it is never sent twice.

It runs once and exits; a scheduler (GitHub Actions) runs it every few minutes.
"""
from __future__ import annotations

import argparse
import html
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import requests
import yaml

GITHUB_GRAPHQL_URL = "https://api.github.com/graphql"
TELEGRAM_SEND_URL = "https://api.telegram.org/bot{token}/sendMessage"

log = logging.getLogger("monitor")


# --------------------------------------------------------------------------
# Config, secrets and state
# --------------------------------------------------------------------------

def load_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text())
    for repo in config["repos"]:
        if repo.count("/") != 1:
            raise ValueError(f"Repo must look like owner/name, got: {repo!r}")
    config.setdefault("issues_per_repo", 20)
    config.setdefault("include_closed", False)
    config.setdefault("body_preview_chars", 300)
    return config


def load_dotenv(path: Path) -> None:
    """For local runs: read KEY=value lines from .env into the environment.
    Variables that are already set (e.g. GitHub Actions secrets) win."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        sys.exit(f"Missing environment variable {name}. Put it in .env locally, "
                 f"or in the repository's Actions secrets in the cloud.")
    return value


def load_state(path: Path) -> dict:
    # state.json looks like: {"repos": {"ollama/ollama": {"last_seen": 12345}}}
    if path.exists():
        return json.loads(path.read_text())
    return {"repos": {}}


def save_state(path: Path, state: dict) -> None:
    # Write to a temp file, then rename: a crash mid-write can't leave
    # a half-written (corrupt) state.json behind.
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


# --------------------------------------------------------------------------
# HTTP with retries
# --------------------------------------------------------------------------

def request_with_retry(method: str, url: str, *, label: str, attempts: int = 4,
                       **kwargs) -> requests.Response:
    """Send a request, retrying on network errors, 5xx and rate limits.

    `label` is what we log instead of the URL, because the Telegram URL
    contains the bot token and must never appear in logs."""
    for attempt in range(1, attempts + 1):
        resp = None
        try:
            resp = requests.request(method, url, timeout=30, **kwargs)
        except requests.RequestException as exc:
            # Log only the error type: the message can contain the URL.
            problem = type(exc).__name__
        else:
            rate_limited = resp.status_code == 429 or (
                resp.status_code == 403 and "rate limit" in resp.text.lower())
            if not (rate_limited or resp.status_code >= 500):
                return resp  # success, or an error retrying won't fix
            problem = f"HTTP {resp.status_code}"

        if attempt == attempts:
            break
        wait = _retry_delay(resp, attempt)
        log.warning("%s: %s, retrying in %ss (attempt %d/%d)",
                    label, problem, wait, attempt, attempts)
        time.sleep(wait)
    raise RuntimeError(f"{label}: gave up after {attempts} attempts ({problem})")


def _retry_delay(resp: requests.Response | None, attempt: int) -> int:
    if resp is not None:
        if resp.headers.get("Retry-After", "").isdigit():  # GitHub
            return min(int(resp.headers["Retry-After"]), 60)
        try:  # Telegram puts it in the JSON body
            return min(int(resp.json()["parameters"]["retry_after"]), 60)
        except (ValueError, KeyError, TypeError):
            pass
    return 2 ** attempt  # 2s, 4s, 8s


# --------------------------------------------------------------------------
# GitHub: newest issues for every repo in a single GraphQL request
# --------------------------------------------------------------------------

def build_query(repos: list[str], per_repo: int) -> str:
    """One query with one aliased block per repo (r0, r1, ...).
    `issues` in GraphQL never includes pull requests."""
    blocks = []
    for i, full_name in enumerate(repos):
        owner, name = full_name.split("/")
        blocks.append(
            f"r{i}: repository(owner: {json.dumps(owner)}, name: {json.dumps(name)}) {{\n"
            "  nameWithOwner\n"
            f"  issues(first: {per_repo}, orderBy: {{field: CREATED_AT, direction: DESC}}) {{\n"
            "    nodes { number title url state createdAt bodyText\n"
            "            author { login } labels(first: 5) { nodes { name } } }\n"
            "  }\n"
            "}")
    return "query {\n" + "\n".join(blocks) + "\nrateLimit { cost remaining }\n}"


def fetch_recent_issues(repos: list[str], per_repo: int, token: str) -> dict:
    """Returns {"owner/name": [issue, ...]} or {"owner/name": None} if that repo
    couldn't be read (renamed, deleted, made private)."""
    resp = request_with_retry(
        "POST", GITHUB_GRAPHQL_URL, label="GitHub",
        headers={"Authorization": f"Bearer {token}"},
        json={"query": build_query(repos, per_repo)})
    if not resp.ok:
        raise RuntimeError(f"GitHub returned HTTP {resp.status_code}: {resp.text[:200]}")

    payload = resp.json()
    # GraphQL can return partial data plus errors (e.g. one repo not found).
    for error in payload.get("errors") or []:
        log.warning("GitHub: %s", error.get("message"))
    data = payload.get("data")
    if not data:
        raise RuntimeError("GitHub query returned no data")

    limit = data.get("rateLimit") or {}
    log.info("GitHub query cost %s point(s), %s left this hour",
             limit.get("cost"), limit.get("remaining"))
    result = {}
    for i, full_name in enumerate(repos):
        repo_data = data.get(f"r{i}")  # None if GitHub couldn't find the repo
        if repo_data and repo_data["nameWithOwner"].lower() != full_name.lower():
            # GitHub follows renames/transfers, so this still works; just tell the user.
            log.warning("%s has moved to %s: update config.yaml", full_name,
                        repo_data["nameWithOwner"])
        result[full_name] = repo_data["issues"]["nodes"] if repo_data else None
    return result


# --------------------------------------------------------------------------
# Deciding what's new
# --------------------------------------------------------------------------

def baseline_number(issues: list[dict], backfill: int) -> int:
    """Starting point for a repo we've never seen. Normally the newest issue
    number (so nothing old is sent). With --backfill N, start N issues earlier."""
    numbers = sorted(issue["number"] for issue in issues)
    if not numbers:
        return 0
    if backfill <= 0:
        return numbers[-1]
    if backfill >= len(numbers):
        return numbers[0] - 1
    return numbers[-backfill - 1]


def new_issues(issues: list[dict], last_seen: int) -> list[dict]:
    """Issues numbered above the watermark, oldest first.
    Issue numbers only ever go up within a repo, so one number per repo is
    all the state we need."""
    return sorted((i for i in issues if i["number"] > last_seen),
                  key=lambda i: i["number"])


# --------------------------------------------------------------------------
# Telegram
# --------------------------------------------------------------------------

def _age(created_at: str, now: datetime) -> str:
    created = datetime.strptime(created_at, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    minutes = int((now - created).total_seconds() // 60)
    if minutes < 1:
        return "just now"
    if minutes < 60:
        return f"{minutes} min ago"
    if minutes < 48 * 60:
        return f"{minutes // 60} h ago"
    return created.strftime("%d %b %Y")


def format_issue(repo: str, issue: dict, preview_chars: int,
                 now: datetime | None = None) -> str:
    """Telegram message in HTML mode. Everything from GitHub is escaped, so an
    issue titled "<script>" can't break the formatting."""
    now = now or datetime.now(timezone.utc)
    author = (issue.get("author") or {}).get("login", "ghost")  # deleted users have no author
    labels = ", ".join(label["name"] for label in issue["labels"]["nodes"])
    body = " ".join((issue.get("bodyText") or "").split())  # collapse whitespace
    if len(body) > preview_chars:
        body = body[:preview_chars].rstrip() + "…"

    meta = f"👤 {html.escape(author)} · 🕒 {_age(issue['createdAt'], now)}"
    if labels:
        meta += f" · 🏷 {html.escape(labels)}"
    if issue.get("state") == "CLOSED":
        meta += " · (already closed)"

    lines = [f"🆕 <b>{html.escape(repo)}</b> #{issue['number']}",
             f"<b>{html.escape(issue['title'])}</b>",
             meta]
    if body:
        lines += ["", html.escape(body)]
    lines += ["", f'<a href="{html.escape(issue["url"])}">Open issue →</a>']
    return "\n".join(lines)


def make_telegram_sender(token: str, chat_id: str) -> Callable[[str], None]:
    def send(text: str) -> None:
        resp = request_with_retry(
            "POST", TELEGRAM_SEND_URL.format(token=token), label="Telegram",
            json={"chat_id": chat_id, "text": text, "parse_mode": "HTML",
                  "link_preview_options": {"is_disabled": True}})
        if not resp.ok:
            description = resp.json().get("description", "") if resp.content else ""
            raise RuntimeError(f"Telegram returned HTTP {resp.status_code}: {description}")
        time.sleep(1)  # Telegram allows about 1 message per second per chat
    return send


# --------------------------------------------------------------------------
# One monitoring pass
# --------------------------------------------------------------------------

def run_once(config: dict, state: dict, fetched: dict, send: Callable[[str], None],
             save: Callable[[], None], backfill: int = 0) -> int:
    """Compare fetched issues against state, send alerts, update state.
    Returns how many alerts were sent. Kept free of network code so the
    tests can drive it with fake data."""
    sent = 0
    per_repo = config["issues_per_repo"]
    for repo in config["repos"]:
        issues = fetched.get(repo)
        if issues is None:
            log.warning("%s: not found or not readable (renamed or private?), skipped", repo)
            continue

        entry = state["repos"].get(repo)
        if entry is None:
            start = baseline_number(issues, backfill)
            state["repos"][repo] = {"last_seen": start}
            save()
            log.info("%s: first time seen, starting after #%s", repo, start)
        last_seen = state["repos"][repo]["last_seen"]

        fresh = new_issues(issues, last_seen)
        for issue in fresh:
            if issue["state"] == "CLOSED" and not config["include_closed"]:
                log.info("%s #%s: opened and already closed, skipped", repo, issue["number"])
            else:
                send(format_issue(repo, issue, config["body_preview_chars"]))
                sent += 1
            # Save after EVERY issue: if the run dies halfway, the next run
            # resumes after the last issue that was actually sent.
            state["repos"][repo]["last_seen"] = issue["number"]
            save()

        # Every fetched issue was new: there may be even more we didn't fetch.
        if entry is not None and len(fresh) >= per_repo:
            send(f"⚠️ <b>{html.escape(repo)}</b>: {per_repo} or more new issues since the "
                 f"last check; some older ones may not have been sent.\n"
                 f'<a href="https://github.com/{html.escape(repo)}/issues">See all issues →</a>')
        log.info("%s: %d new issue(s)", repo, len(fresh))
    return sent


def main() -> int:
    parser = argparse.ArgumentParser(description="Send new GitHub issues to Telegram.")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--state", type=Path, default=Path("state.json"))
    parser.add_argument("--dry-run", action="store_true",
                        help="print messages instead of sending; don't save state")
    parser.add_argument("--test-telegram", action="store_true",
                        help="send one test message and exit")
    parser.add_argument("--backfill", type=int, default=0, metavar="N",
                        help="for repos seen for the first time, also send their last N issues")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    load_dotenv(Path(".env"))

    if args.dry_run:
        def send(text: str) -> None:
            print("─" * 60 + "\n" + text)
    else:
        send = make_telegram_sender(require_env("TELEGRAM_BOT_TOKEN"),
                                    require_env("TELEGRAM_CHAT_ID"))

    if args.test_telegram:
        send("✅ <b>Issue monitor</b>: test message. Delivery works.")
        log.info("Test message sent")
        return 0

    config = load_config(args.config)
    state = load_state(args.state)
    fetched = fetch_recent_issues(config["repos"], config["issues_per_repo"],
                                  require_env("GITHUB_TOKEN"))
    save = (lambda: None) if args.dry_run else (lambda: save_state(args.state, state))

    sent = run_once(config, state, fetched, send, save, backfill=args.backfill)
    log.info("Done: %d alert(s) %s", sent, "printed (dry run)" if args.dry_run else "sent")
    return 0


if __name__ == "__main__":
    sys.exit(main())
