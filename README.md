# Pythia

A self-grading market forecaster. Every day Pythia forms one specific,
falsifiable, time-boxed prediction about a liquid ETF, writes down its reasoning
and a probability **before** the outcome is known, logs it, and later grades
itself against what actually happened — keeping an honest, permanent track
record (Brier score) next to a ladder of dumb baselines.

The point is to find out whether a well-prompted LLM can produce *calibrated*
forecasts that beat trivial baselines — and to measure it, not assume it.

> **No autonomous real-money execution, ever.** Pythia forecasts, paper-trades,
> and grades itself fully automatically. It never places a live order. Real
> trades are surfaced for a human to review and place manually. See
> [CLAUDE.md](CLAUDE.md).

## The loop

```
forecast  ->  log  ->  (wait N trading days)  ->  resolve  ->  review
```

1. **forecast** — for each watchlist ETF, pull recent price data, ask the model
   for a probability + reasoning on a binary claim ("close on the resolution day
   will be >= the anchor close"), and log it as `pending`. The same claim is also
   scored by three baselines.
2. **resolve** — find matured forecasts, fetch the real close, mark each
   correct/incorrect, and compute the Brier score.
3. **review** — print the track record: every call, running hit-rate, and average
   Brier — for Pythia *and* each baseline, side by side.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12 (uv will fetch 3.12 for
you). Then:

```bash
uv sync                      # create the venv and install dependencies
cp .env.example .env         # then edit .env and add your ANTHROPIC_API_KEY
```

On Windows, `cp` is `Copy-Item .env.example .env`. If the `uv` command isn't
found after `pip install --user uv`, its folder (e.g.
`%APPDATA%\Python\Python3xx\Scripts`) isn't on your PATH — add it, or run the
commands below through the project venv at `.venv\Scripts\`.

`ANTHROPIC_API_KEY` is the only secret needed. It's required for `forecast`
(which calls the model); `resolve` and `review` work without it.

## Usage

```bash
uv run pythia forecast              # forecast the whole watchlist (SPY, QQQ, IWM)
uv run pythia forecast -t SPY       # just one ticker
uv run pythia forecast -h 10        # 10-trading-day horizon (default 5)

uv run pythia resolve               # settle everything matured as of today
uv run pythia resolve --date 2026-06-05   # settle as of a specific date (backfill)

uv run pythia review                # leaderboard + full call log
uv run pythia review -t SPY         # filter the call log to one ticker
uv run pythia review --no-log       # leaderboard only
```

Run `forecast` in the morning and `resolve` later (or the next day). The daily
loop is idempotent: running `forecast` twice the same day won't double-log.

## How grading works

- The claim is **"`ticker` closing price on the resolution day >= its anchor-day
  close."** The anchor is the most recent *completed* session at issue time, so
  the claim is known and falsifiable the moment it's logged.
- **Horizon is in trading sessions**, computed with a real NYSE calendar
  (`pandas-market-calendars`) — never naive calendar-day math. Resolution
  confirms the market was actually open on the resolution date.
- **Brier score** = `(probability - outcome)^2`, where `outcome` is 1.0 if the
  claim came true else 0.0. Lower is better; always predicting 0.50 scores 0.25.
- Prices use **raw (unadjusted) closes** so a logged claim always resolves to the
  same answer, regardless of later dividends.

## The baseline ladder

Every claim is also graded by three trivial forecasters, so the question becomes
"*what* did Pythia beat?":

| Baseline | Predicts | What beating it proves |
|---|---|---|
| `coin_flip` | always 0.50 | sanity floor — Brier is fixed at 0.25 |
| `drift` | "up" at the asset's historical up-day ratio (computed, not hardcoded) | the real bar: signal beyond the market's natural long drift |
| `naive_momentum` | direction of the last N sessions, at fixed confidence | if it matches Pythia, the model is just doing momentum |

## Testing

```bash
uv run pytest
```

The scoring core (Brier, outcome resolution, the `>=` boundary, the resolution
flow) and the trading-calendar math are covered by offline, deterministic tests —
no network or API key needed.

## Configuration

Everything tunable lives in [`pythia/config.py`](pythia/config.py): model IDs,
the watchlist, lookback windows, the default horizon, and the database path
(`PYTHIA_DB_PATH` overrides it). The track record is a local SQLite file
(`pythia.db`, gitignored).

## Automating later

v0 is run by hand. To automate the daily loop later, schedule two jobs — one for
`forecast` after the US market opens and one for `resolve` after it closes:

- **cron** (Linux/macOS) or **Task Scheduler** (Windows), or
- a **scheduled GitHub Action** (`on: schedule:`) that runs `uv run pythia
  forecast` / `resolve`, with `ANTHROPIC_API_KEY` stored as an encrypted repo
  secret and the SQLite file persisted (e.g. committed to a data branch or stored
  as an artifact/cache).

## Status

v0: predict -> log -> resolve -> review, with the baseline ladder. Later versions
(self-review, FRED macro, simulated options, alerting, dashboard) are sketched in
[CLAUDE.md](CLAUDE.md).
