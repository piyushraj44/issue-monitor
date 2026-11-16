# GitHub issue monitor

Sends a Telegram message when a new issue is opened in any repo listed in
[config.yaml](config.yaml). Runs free on GitHub Actions every 5 minutes.

## Files

| File | What it does |
|---|---|
| `config.yaml` | Repos to watch and a few settings. Edit this to add/remove repos. |
| `monitor.py` | The whole program: fetch issues, decide what's new, send to Telegram. |
| `.github/workflows/monitor.yml` | Tells GitHub Actions to run `monitor.py` every 5 minutes. |
| `tests/test_monitor.py` | Tests with fake data (no tokens needed). |
| `state.json` | Created at runtime: newest issue number seen per repo. Lives on the `state` branch in the cloud. |

## Run locally

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

Create a `.env` file (already ignored by git) with:

```
TELEGRAM_BOT_TOKEN=...
TELEGRAM_CHAT_ID=...
GITHUB_TOKEN=...        # e.g. the output of: gh auth token
```

```bash
.venv/bin/python monitor.py --dry-run        # print what would be sent, change nothing
.venv/bin/python monitor.py --test-telegram  # send one test message
.venv/bin/python monitor.py --backfill 2     # first run, but also send the last 2 issues per repo
.venv/bin/python monitor.py --heartbeat      # also send the daily check-in now
.venv/bin/python -m unittest discover -s tests -v
```

## Run in the cloud

Repository → Settings → Secrets and variables → Actions → add
`TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID`. `GITHUB_TOKEN` is provided automatically.
The Actions tab has a **Run workflow** button with three modes: `check` (a normal run),
`test-telegram` (send a test message) and `heartbeat-now` (a normal run plus the daily check-in).

A daily check-in message (see `heartbeat` in config.yaml) confirms the monitor is alive.
If it stops arriving, look at the Actions tab.

The repo must be public for unlimited free Actions minutes.
