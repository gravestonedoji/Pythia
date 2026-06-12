# Pythia — Canonical Project Summary

*The authoritative handoff for Pythia sessions. Read CLAUDE.md for the hard
contract (what Pythia is and must never do); read this for where the project
stands and why. Sibling project:*
- **Delphi** (`C:\Users\user\Claude\Delphi`) — the synthetic **wind tunnel**
  (its own `summary.md` is the authority on everything below attributed to it).
- **Pythia** (`C:\Users\user\Claude\Pythia`, this repo) — the live real-ETF
  **test flight**. Never spend a Pythia-week on something Delphi can reject for
  $0.50; conversely, only Pythia's forward record is evidence that counts.

*Last updated 2026-06-10.*

---

## 1. What Pythia is (one breath)

An autonomous, self-grading forecaster: every trading day it logs one
falsifiable claim per watchlist ticker — "raw close in 5 sessions >= today's
raw close" — with a probability written down BEFORE the outcome, then grades
itself against real closes via trading-calendar math. Hard compliance wall: NO
autonomous real-money execution, ever (forecast + paper + grade only).
Live since 2026-06-01 (first anchors 2026-05-29). Subject: Claude Opus
(`claude-opus-4-8`), price-only by design (the prompt forbids macro/news — v0
is the clean ablation baseline for the later news-reading version).

## 2. Current state (as of 2026-06-10)

- **Watchlist: 30 tickers** (broad US, 8 sector SPDRs, industries, intl,
  bonds/credit, commodities, crypto). Breadth is the binding constraint on
  time-to-significance: ~30 graded claims per trading day once batches mature.
- **Record: 44 resolved + 30 pending per forecaster column.** Far too small to
  rank anything (the first resolved week was a single selloff week ≈ 2-3
  effective data points — claims within a day are correlated). Do not draw
  conclusions from the leaderboard yet; ~month 3 is the first honest read.
- **Nine forecaster columns on every claim** (identities in `config.py`):
  - `pythia` — raw Opus, the permanent control arm.
  - `pythia_coached` — Opus + distilled lessons (NEW, off until first reflect).
  - `pythia_iso` — isotonic remap of raw (NEW, auto-on at 150 resolved).
  - `pythia_coached_iso` — the full stack (NEW, needs 150 resolved coached).
  - `coin_flip`, `drift`, `naive_momentum` — the floor ladder.
  - `hmm_filter` — the QUANT BAR: per-ticker Gaussian HMM, strictly
    point-in-time, backfilled onto all claims. Deploy gate #1: beat this.
  - `kalshi` — the MARKET BAR (NEW): live Kalshi order-book mid on mapped
    tickers (SPY/QQQ/IBIT). Deploy gate #2: beat the market.

## 3. The Delphi findings that bind this repo

These were measured against a synthetic answer key (ground-truth p_true) and
are the reason Pythia's architecture looks the way it does:

1. **An uncorrected LLM is a confident overclaimer** — raw it captured ~22% of
   readable signal (one run: −2%, worse than a coin) while a free fitted HMM
   captured ~56%. Brier-vs-realized HIDES this; overconfidence is invisible in
   Brier and lethal in compounding (one run: $1→$0.025 at full Kelly).
2. **The correction stack works and stacks** (paired race, 10/10 seeds each):
   prose lessons +0.017 cal err, isotonic +0.022, together +0.025 ≈ the quant
   bar. Lessons transfer across models. THIS is why Pythia now runs 4 arms.
3. **Never feed market odds into the prompt** (anchoring experiment): an LLM
   shown a crowd price anchors to it (distance halves), lands BELOW the
   consensus it was shown, and follows the crowd off the cliff on the steps
   where the crowd is wrong. Odds are a baseline COLUMN, never a feature.
4. **Model choice**: corrected Haiku ≈ corrected Flash ≈ corrected anything
   reaching ~50% capture — the correction layers, not the model, carry most of
   the value. (Opus stays Pythia's subject: cost is trivial at 30-60
   calls/day and calibration is the product.)
5. **Reasoning effort doesn't drive overconfidence** (dial experiment: null).

## 4. The correction stack (NEW this session — how it works here)

- **`pythia reflect`** (weekly, run it manually): Opus reads the RESOLVED
  record only (leaderboard, confidence profile, worst 10 misses with their
  original reasoning) and distills ≤8 behavior-level lessons (no macro, no
  ticker rules) → `lessons.txt` + audit trail `lessons_history.jsonl`.
  Gate: ≥25 resolved raw claims (ALREADY MET at 44 — **the first reflect can
  run any time; it has deliberately NOT been run by the build session** since
  it switches the coached arm on and doubles daily Opus calls 30→60).
- **Coached arm**: identical claim, identical data; the ONLY difference is the
  lessons block in the system prompt. The lessons sha is recorded in the row's
  `model` column, so any record slice ties to the exact lessons it used. Raw
  `pythia` stays untouched forever — the coaching effect remains measurable on
  identical claims for the life of the project.
- **Isotonic arms** (free, no extra API calls): at forecast time, a PAV remap
  is fitted on the base arm's OWN resolved (probability → outcome) record and
  applied to today's probability; fit size recorded per row. Hard gate
  `ISO_MIN_RESOLVED = 150` because Delphi measured isotonic HURTING below
  ~1,000 training points and at n≈200-400/fold — below the gate the derived
  row is skipped (a missing row beats a fudged one). At ~30 resolutions/day
  the raw-arm gate clears within roughly a trading week; the coached-iso arm
  follows ~2 weeks after coaching starts.

## 5. The Kalshi market bar (NEW this session — `pythia/kalshi.py`)

- Read-only public API, no account/key/orders — strictly on the forecast/grade
  side of the wall. Mapped tickers only: SPY→^GSPC, QQQ→^NDX, IBIT→BTC-USD
  (claim level mapped through same-day raw closes).
- Implied P(up) from the live book: above-strike ladder interpolation
  (monotone-forced mids) or range-bucket CDF (mass-normalized); mid-quotes
  with a spread gate; refuses to extrapolate outside quoted strikes.
- **Sparse by design**: a claim gets a kalshi row only when a whitelisted
  close-settling contract settles within 2 trading sessions of `resolves_on`
  (gap recorded in the row's reasoning — it's the market's handicap, so it
  slightly EASES the "beat the market" gate; read close calls accordingly).
  With Kalshi's current listings (weekly index series dormant; dailies settle
  through Friday) roughly Mon/Tue anchors match. NO backfill: live books only,
  logged before outcomes. Hourly/intraday series deliberately excluded.

## 6. Run commands

```
uv run pythia forecast    # daily: all arms + baselines, one claim per ticker
uv run pythia resolve     # settle matured claims against real closes
uv run pythia reflect     # weekly self-review -> lessons.txt (coached arm on)
uv run pythia review      # leaderboard + call log
uv run pythia why         # Pythia's reasoning per call
uv run pythia backfill-hmm
uv run pytest             # 92 offline tests (scoring/calendar/kalshi/calibrate/reflect)
```

## 7. Roadmap and gates (gated, not scheduled)

- **now → month 3**: accumulate. Breadth (30 tickers) → ~1,400 graded claims
  in 6 months ≈ enough to tell edge from luck. Run `reflect` weekly; let the
  iso arms switch on by themselves.
- **month 3-6**: first honest leaderboard reads. The deploy-grade questions, in
  order: does any Pythia arm beat the quant bar (`hmm_filter`)? does it beat
  the market (`kalshi`) on the sparse matched subset? Expectation set by
  Delphi: the first deployable is likely the HMM, not the LLM.
- **month 6+**: IF gates pass — manual, small, human-placed positions on
  surfaced high-conviction calls only (the wall stays).
- **v1+ (build next)**: FRED macro as a measured A/B against the price-only
  arms; later the news-reading phase (the LLM's only plausible edge over a
  price model — testable only on post-cutoff data; Polymarket belongs here,
  not before).

## 8. Known limitations

- Claims within a day (and across overlapping 5-session windows) are
  correlated — effective sample size is far below row counts; significance
  arrives in months, not weeks.
- Raw closes mean dividend ex-dates dent the "close >= anchor" claim for
  heavy-distribution funds (handled by conservative NON_EQUITY priors).
- The kalshi column's settle-gap handicap (above) and the SPY/^GSPC dividend
  drift are known, documented slop — fine for a bar, not for execution.
- `hmm_filter`'s strength in Delphi is partly home-field advantage (that world
  IS an HMM); on real tapes it's a strong baseline, not a ceiling.

## 9. COMPLETED 2026-06-12 — HMM fit-health monitoring (the referee gets a referee)

Practitioner work on HMMs over real SPY found EM failing to converge on ~47%
of rolling windows and parameters swinging wildly between refits (a regime's
expected duration flipping days↔months) — a quant bar that does that corrupts
the leaderboard silently. The harness now detects and surfaces both; it does
NOT touch what the HMM emits.

- **Every fit is recorded** (`hmm_fits` table, one row per ticker+anchor):
  convergence (did EM meet tolerance / iterations used / final log-lik), the
  fitted parameters (per-regime mu, sig, transition matrix, pi as JSON), and
  the data window. Written live by `forecast`, by `backfill-hmm`, and by the
  retro pass below. First write wins, same as forecasts.
- **Stability check vs the previous fit** (`hmm_health.py`, pure + tested):
  states matched by sigma (labels are arbitrary), compared only when the prior
  fit is ≤7 calendar days old. Flags: `non_converged`; `duration_jump`
  (expected duration `1/(1-T_kk)` moves >3x); `sigma_jump` (>1.5x); `mu_jump`
  (>10 bps/day shift). `k_changed` is recorded but informational (the 2↔3
  switch at 750 sessions is by design). Thresholds in config.py — sized so one
  new observation on thousands can't trip them; a trip means EM landed in a
  different optimum. Strictly point-in-time: a fit is judged against its own
  data and STRICTLY EARLIER fits only.
- **Policy (decided): log + flag, never drop.** A tainted value stays on the
  leaderboard — skipping it would shrink the record exactly on the turbulent
  days where fitting is hard, flattering the bar and biasing deploy gate #1.
  The taint is stamped on the forecast row (`fit_flags`: NULL = pre-monitoring,
  '' = checked clean) and `review` annotates the board whenever any quant-bar
  value is tainted ("HMM bar: X/N values from flagged fits"), with the
  breakdown under `pythia hmm-health` (`--flagged`, `--ticker`).
- **Retro-annotation**: fits are seeded from (ticker, anchor) over raw closes,
  so `pythia hmm-health --backfill` reconstructs the fit behind every
  pre-monitoring row point-in-time — on the SAME window (trimmed to the n_obs
  in the row's model field; today's deep fetch starts later than the
  original's did). Bit-exactness is unattainable (Yahoo quietly revises old
  closes; a non-converged EM amplifies that to ~1e-4 of output drift —
  measured 2026-06-12), so a reconstruction is trusted within
  `HMM_RECONSTRUCTION_TOL` (1e-3); beyond it the row gets
  `reconstruction_mismatch` (tainting) instead of trust.
- Tests: 110 passing (18 new — forced non-convergence via capped EM iters,
  comparator flags incl. sigma-matching and the K-change carve-out, schema
  migration of a pre-monitoring DB, end-to-end taint into the DB and the
  review annotation). Run order note: the first `hmm-health --backfill`
  happened 2026-06-12, so consecutive-fit comparisons exist from the record's
  start; keep running plain `forecast` daily — health rows are written
  automatically from here on.
- **What it found, day one: 130/134 fits (97%) non-converged** — only LQD and
  XLK settled within 40 EM iterations at the 1e-7 relative tolerance. Worse
  than the practitioner's ~47%, and it means the quant bar's entire resolved
  record (52/52) currently rests on still-moving fits (which is also why
  reconstructions drift by ~1e-4: a fit that hasn't settled amplifies tiny
  close revisions). The bar's Brier still stands — log + flag, never drop —
  but "beat the quant bar" now visibly means "beat a referee that rarely
  converges". Whether to give EM more iterations / restarts (and re-baseline)
  is a METHODOLOGY decision deliberately not taken by the build session: it
  changes what the bar is, so it needs an explicit owner call, ideally
  Delphi-tested first.
