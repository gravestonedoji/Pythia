# Pythia

A self-grading market forecaster. Every trading day Pythia forms one specific,
falsifiable, time-boxed prediction per watchlist ETF, writes down its reasoning
and a probability **before** the outcome is known, logs it, and later grades
itself against what actually happened — keeping an honest, permanent track
record (Brier score) next to a ladder of baselines that runs from a coin flip
up to a quant model and live market odds.

The point is to find out whether a well-prompted LLM can produce *calibrated*
forecasts that beat non-trivial baselines — and to measure it, not assume it.
Live since 2026-06-01.

> **No autonomous real-money execution, ever.** Pythia forecasts, paper-trades,
> and grades itself fully automatically. It never places a live order. Real
> trades are surfaced for a human to review and place manually. See
> [CLAUDE.md](CLAUDE.md).

## The loop

```
forecast  ->  log  ->  (wait N trading days)  ->  resolve  ->  review
              |                                      |
              +-> paper book (simulated options)     +-> weekly reflect -> lessons
              +-> high-conviction digest/alerts      +-> publish (dashboard)
```

1. **forecast** — for each of 30 watchlist ETFs, pull recent price data and log
   a probability + reasoning on a binary claim ("close on the resolution day
   will be >= the anchor close") as `pending`. Every claim is scored by every
   forecaster column at once (see the ladder below), simulated option positions
   open against the live quote book, and high-conviction calls are flagged into
   their own gradeable alert record.
2. **resolve** — find matured forecasts, fetch the real close, compute outcomes
   and Brier scores, and settle expired paper positions at intrinsic value.
3. **review / pnl / alerts / publish** — the track record side by side with the
   baselines; the simulated P&L ladder; the alert rule's own scoreboard; a
   static aggregate dashboard rendered to `docs/`.

## The forecaster columns

Every claim is graded across ten columns, so the question is never "did Pythia
win?" but "*what* did it beat?":

| Column | What it is |
|---|---|
| `pythia` | raw Claude Opus, price-only — the permanent control arm |
| `pythia_coached` | + lessons distilled from its own graded record (`reflect`) |
| `pythia_macro` | + point-in-time FRED macro (rates, breakevens, HY spread, VIX, dollar) |
| `pythia_iso`, `pythia_coached_iso` | isotonic remaps fitted on each arm's own resolved record |
| `coin_flip`, `drift`, `naive_momentum` | the floor ladder |
| `hmm_filter` | **the quant bar**: per-ticker Gaussian HMM, strictly point-in-time, own fit-health monitoring |
| `kalshi` | **the market bar**: live Kalshi order-book odds on mapped tickers (sparse by design) |

Each correction layer differs from the raw arm by exactly one ingredient, so
every layer is measured on identical claims — never assumed better.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12 (uv will fetch 3.12
for you). Then:

```bash
uv sync                      # create the venv and install dependencies
cp .env.example .env         # then edit .env (see below)
```

Model access is either an `ANTHROPIC_API_KEY`, or `PYTHIA_TRANSPORT=subscription`
to route the daily forecasts through a local Claude Code login instead (same
model, billed to the plan; rows are stamped with the transport they used).
Optional extras, each a clean no-op when unset: `FRED_API_KEY` (the macro arm)
and `PYTHIA_SMTP_*` (email digests). `resolve`, `review`, `pnl`, `alerts`, and
`publish` never need any key.

## Usage

```bash
uv run pythia forecast       # all arms + baselines + paper book + digest, one claim per ticker
uv run pythia resolve        # settle matured claims and expired paper positions
uv run pythia review         # leaderboard + call log
uv run pythia why            # Pythia's reasoning per call
uv run pythia reflect        # weekly self-review -> lessons.txt (coached arm)
uv run pythia pnl            # simulated-options P&L ladder (fixed/edge/kelly views)
uv run pythia alerts         # the high-conviction alert log + its scoreboard
uv run pythia publish        # render the aggregate dashboard to docs/ (never touches git)
uv run pytest                # offline test suite — no network, no key
```

The daily loop is idempotent (first write wins everywhere): running `forecast`
twice the same day won't double-log, re-running after a partial failure fills
only what's missing.

## How grading works

- The claim is **"`ticker` closing price on the resolution day >= its anchor-day
  close."** The anchor is the most recent *completed* session at issue time, so
  the claim is known and falsifiable the moment it's logged.
- **Horizon is in trading sessions**, computed with a real NYSE calendar
  (`pandas-market-calendars`) — never naive calendar-day math. Nothing grades
  against a session that hasn't closed yet.
- **Brier score** = `(probability - outcome)^2`. Lower is better; always
  predicting 0.50 scores 0.25.
- Prices use **raw (unadjusted) closes** so a logged claim always resolves to
  the same answer, regardless of later dividends.
- Values are logged before outcomes, never edited, and live-only records
  (market odds, option quotes, alerts) are never backfilled — a missing row
  beats a fudged one.

## Testing

```bash
uv run pytest
```

238 offline, deterministic tests — scoring, calendar math, the quote gates and
P&L replays, the digest rule, dashboard generation (including leak guards), and
the transport isolation. No network or API key needed.

## Configuration

Everything tunable lives in [`pythia/config.py`](pythia/config.py): model IDs
and transport, the watchlist, gates and thresholds, sizing policies, and the
database path (`PYTHIA_DB_PATH` overrides it). The track record is a local
SQLite file (`pythia.db`, gitignored — the dashboard publishes aggregates only).

## Status

| Phase | |
|---|---|
| v0 — forecast/resolve/review + baseline ladder | ✅ |
| v1 — self-review (coached arm), isotonic arms, FRED macro arm | ✅ |
| v2 — simulated ETF option positions (the paper book) | ✅ |
| v3 — high-conviction digest + gradeable alert record | ✅ |
| v4 — aggregate dashboard (`pythia publish`) | ✅ |
| Next — the news-reading arm; the first honest leaderboard read | ⏳ |

The full project journal — design rationale, the wind-tunnel findings that
shaped the architecture, and per-phase details — is in
[summary.md](summary.md); the hard constraints are in [CLAUDE.md](CLAUDE.md).
No leaderboard conclusions are drawn before the record can support them.
