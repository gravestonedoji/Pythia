# CLAUDE.md — Pythia

Read this every session. It is the contract for what Pythia is and the rules it
must not break. For the current project state, the experiment ladder, and the
Delphi findings that shaped it, read `summary.md` (the canonical handoff).

## What Pythia is

An autonomous, self-grading market forecaster. Every day it forms one specific,
falsifiable, time-boxed prediction about a liquid ETF, writes down its full
reasoning and a probability **before** the outcome is known, logs it, and later
grades itself against what actually happened. Over time it keeps an honest,
permanent track record (Brier score for calibration), and (later versions)
reviews its own record to improve its forecasting prompt.

The real question it answers: **can a well-prompted LLM produce *calibrated*
forecasts that beat a dumb baseline — and does richer data actually improve
calibration, or just make the forecasts sound smarter?** The version sequence is
built so each added data layer can be *measured* against the simpler one.

## Hard constraints — never violate these

1. **ETFs and liquid options only.** Broad-based ETFs (SPY, QQQ, IWM, ...) and
   liquid index/ETF option chains. No single stocks, no single-stock ETFs.
2. **No autonomous real-money execution. EVER.** This is a compliance
   requirement, not a preference. The system runs fully autonomously for
   **forecasting + simulated ("paper") execution + grading only**. It marks
   positions to market against *real* prices but never places a live order. For
   any real trade it only *surfaces* high-conviction calls for the human to
   review and place **manually**. Keep a hard architectural wall between
   "forecast/grade" (automated) and "execute real money" (human-only). Do not
   add any live-broker order placement.
3. **Secrets in `.env`, never committed.** `ANTHROPIC_API_KEY` is the only
   *required* secret (with `PYTHIA_TRANSPORT=subscription` the daily forecast
   instead rides the local Claude Code login — see summary.md §17 — and the
   key is only used by `reflect`). Optional email alerts add SMTP credentials
   (`PYTHIA_SMTP_*` / `PYTHIA_EMAIL_*`) — also `.env`-only, and a clean no-op
   when unset. `.env` and `*.db` are gitignored.

## v0 design notes (important)

- **v0 is price-only on purpose.** The forecaster sees only price/volume data
  (yfinance). The prompt explicitly **forbids** reasoning about rates,
  inflation, the Fed, or any macro it cannot see — stale pretraining macro is
  worse than none. This makes price-only Pythia a clean **ablation baseline** so
  that when v1 adds real macro (FRED), the macro-aware Brier can be *measured*
  against this version, not assumed better. Keep that A/B comparison possible.
- **Resolution must be provably correct.** The project's credibility rests on
  honest grading. The scoring core (`pythia/scoring.py`) is pure and unit-tested
  (`tests/`). Don't weaken it.
- **Trading-calendar date math, never naive.** `resolves_on` is computed by
  counting N *open* sessions forward via `pandas-market-calendars` (NYSE / XNYS).
  Resolution confirms the market was open on `resolves_on` before settling.
- **Raw (unadjusted) closes.** Anchor close and resolved close are both raw
  closes from the same source, so a logged claim resolves to the same answer no
  matter when it is graded (adjusted closes get rewritten on dividends).
- **The baseline ladder.** Every run scores the reference forecasters on the
  *same* claims as Pythia: `coin_flip` (0.50, the floor), `drift` (the asset's
  historical positive-day ratio), `naive_momentum` (direction of the last N
  sessions), and `hmm_filter` — a Gaussian HMM regime filter fitted per ticker,
  strictly point-in-time (only closes <= the anchor date ever reach the fit).
  The HMM is the **quant bar**: the strongest free, price-only statistical
  forecaster. The question is not "did Pythia win?" but "*what* did it beat?" —
  and the deploy-grade question is specifically "does it beat the quant bar?".
  `pythia backfill-hmm` reconstructs it onto pre-existing claims point-in-time.
- **The market bar (`kalshi`, kalshi.py).** Fifth column on mapped tickers only
  (SPY/QQQ/IBIT → S&P 500 / Nasdaq-100 / BTC contracts): the live Kalshi
  order-book mid, read at forecast time, escalating the gate to "beat the
  market". Read-only public data — no account, no orders, firmly on the
  forecast/grade side of the hard wall. **Never feed odds into the forecaster
  prompt** (Delphi's anchoring experiment showed the LLM parrots the crowd and
  subtracts value). Sparse by design: a claim only gets a row when a whitelisted
  contract settles within 2 trading sessions of `resolves_on` (the gap is
  recorded in the row's reasoning); no backfill, live books only.

## Tech stack

- Python 3.12 (managed by **uv**). SQLite via stdlib. **Anthropic** SDK for
  model calls. **yfinance** for prices. **pandas-market-calendars** + **pandas**
  for calendar/series. **python-dotenv**, **typer** + **rich** for the CLI.

## Model routing (in `pythia/config.py`, never hardcode elsewhere)

- **Daily forecast (the brain):** `claude-opus-4-8`. Calibration is the whole
  point and it runs once per ticker per day, so cost is trivial. Do not downgrade.
- **Weekly self-review (v1+):** `claude-opus-4-8`.
- **Data/news triage, extraction, tagging (v1+):** `claude-haiku-4-5`.
- **Transport:** forecasts reach the model via the metered API by default, or
  the Claude Code subscription login when `PYTHIA_TRANSPORT=subscription`
  (rows stamped `+via:claude-code`; same model either way).

## Run commands

```
uv run pythia forecast    # form + log one forecast per watchlist ticker (all arms + baselines)
uv run pythia resolve     # settle matured forecasts against the real close and score them
uv run pythia reflect     # weekly self-review -> lessons.txt (turns the coached arm on)
uv run pythia review      # print the track record, side by side with the baselines
uv run pythia notify      # email a forecast batch (re-send the latest, or --test SMTP)
uv run pythia pnl         # the paper book's P&L ladder (v2; simulated options only)
uv run pythia alerts      # the high-conviction alert log + scoreboard (v3)
uv run pythia publish     # render the aggregate dashboard to docs/ (v4; never touches git)
uv run pytest             # run the offline test suite (no network, no key)
```

`forecast` needs `ANTHROPIC_API_KEY` (copy `.env.example` to `.env`). `resolve`,
`review`, and `notify` do not (but `notify` needs the optional `PYTHIA_SMTP_*`
vars set, or it's a no-op). `forecast` also emails the batch automatically when
SMTP is configured; pass `--no-notify` to skip.

## Layout

```
pythia/
  config.py      model IDs, watchlist, lookback windows, DB path — single source of truth
  data.py        yfinance prices + trading-calendar math (shared, testable)
  storage.py     SQLite schema + CRUD (the forecasts table)
  forecaster.py  the brain: Anthropic call + price-only prompt (forbids macro)
  baselines.py   the reference-forecaster ladder
  scoring.py     resolution + Brier (pure core, unit-tested)
  notify.py      optional email alerts (surfacing-only; SMTP via stdlib)
  paper.py       v2: SIMULATED option positions (quote gates, intrinsic settle)
  pnl.py         v2: the read-time sizing ladder (fixed/edge/kelly replays)
  digest.py      v3: high-conviction flags + the alert record (pure)
  dashboard.py   v4: aggregate-only static dashboard (pure HTML/SVG generation)
  cli.py         typer CLI: forecast / resolve / review / why / notify / pnl / alerts / publish
tests/           offline test suite (no network, no key)
```

## Roadmap (build v0 first; don't jump ahead)

- **v1** — LLM self-review: DONE (`pythia reflect` + the `pythia_coached` /
  isotonic arms — see summary.md §4; the Delphi-validated correction stack,
  every layer measured against the raw arm on identical claims). Still to do:
  FRED macro as a *measured* experiment vs the v0 price-only baseline.
- **v2** — simulated liquid ETF option positions: DONE 2026-07-07 (summary.md
  §12; the paper book — entry at logged after-close mids, settlement at
  intrinsic off the official close, sizing as read-time views). Simulated
  only, forever.
- **v3** — daily digest + alerting: DONE 2026-07-07 (summary.md §13; the
  high-conviction rule logs its own gradeable record in `digest_alerts` —
  threshold revisit pre-registered at 30 resolved alerts). Other transports
  (Telegram/Discord) still open.
- **v4** — dashboard: BUILT 2026-07-07 (summary.md §14; `pythia publish`
  renders aggregates only and never touches git). Auto-publishing awaits the
  go-public owner call (public repo required for GitHub Pages).
- **Next** — the news-reading arm (the same measured-A/B design as the macro
  arm; live-only, post-cutoff data), and the month-3 first honest read.
