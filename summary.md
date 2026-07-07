# Pythia — Canonical Project Summary

*The authoritative handoff for Pythia sessions. Read CLAUDE.md for the hard
contract (what Pythia is and must never do); read this for where the project
stands and why. Sibling project:*
- **Delphi** (`C:\Users\user\Claude\Delphi`) — the synthetic **wind tunnel**
  (its own `summary.md` is the authority on everything below attributed to it).
- **Pythia** (`C:\Users\user\Claude\Pythia`, this repo) — the live real-ETF
  **test flight**. Never spend a Pythia-week on something Delphi can reject for
  $0.50; conversely, only Pythia's forward record is evidence that counts.

*Last updated 2026-07-07.*

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
- **Ten forecaster columns on every claim** (identities in `config.py`):
  - `pythia` — raw Opus, the permanent control arm.
  - `pythia_coached` — Opus + distilled lessons (NEW, off until first reflect).
  - `pythia_macro` — Opus + point-in-time FRED macro (NEW v1, §11; live-only,
    on when `FRED_API_KEY` is set).
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
uv run pythia forecast    # daily: all arms + baselines + paper book + digest
uv run pythia resolve     # settle matured claims + expired paper positions
uv run pythia reflect     # weekly self-review -> lessons.txt (coached arm on)
uv run pythia review      # leaderboard + call log
uv run pythia why         # Pythia's reasoning per call
uv run pythia pnl         # the simulated-options P&L ladder (v2, read-time views)
uv run pythia paper-trade # same-evening retry of the paper capture (v2)
uv run pythia alerts      # the high-conviction alert log + its scoreboard (v3)
uv run pythia publish     # render the dashboard to docs/ (v4, local-only)
uv run pythia backfill-hmm
uv run pytest             # 218 offline tests (no network, no key)
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
  arms — DONE 2026-06-22, see §11 (`pythia_macro`); later the news-reading
  phase (the LLM's only plausible remaining edge over a price model —
  testable only on post-cutoff data; Polymarket belongs here, not before).
- **v2 (simulated options)**: DONE 2026-07-07, see §12 — the paper book
  accumulates live-only from here.
- **v3 (digest + alerting)**: DONE 2026-07-07, see §13 — the alert record
  accumulates live-only from here.
- **v4 (dashboard)**: BUILT 2026-07-07, see §14 — `pythia publish` is
  local-only until the go-public owner call is made.
- **Next**: the news-reading arm (same measured-A/B design as macro), and the
  month-3 first honest read (~September 2026).

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
  Delphi-tested first. *(Both conditions met the same day — see §10.)*

## 10. COMPLETED 2026-06-12 — EM runs to convergence (the §9 owner call, Delphi-tested)

The §9 methodology question got its answer the hard way: Delphi found the
SAME bug in its own quant the same day — a too-small EM cap silently
truncating fits mid-climb — and fixing it there moved its quant bar from ~56%
to ~87% signal capture (Delphi summary.md §17–18). Audited Pythia's bar on
the live data and seeds BEFORE changing anything (`audit_em_convergence.py`,
numbers in `audit_em_convergence.json`):

- **28/30 watchlist fits truncated at iters=40** (§9's 97% reproduced on
  fresh fits) — but the case is MILD here: every ticker meets tolerance by
  108 iterations (median 72), the loglik left on the table is median 0.7 /
  max 22 (SLV), and the bar VALUE barely moves: |dP(up)| median 0.001, max
  0.028 (SOXX), 0/30 at or above 0.05. Unlike Delphi's 27 capture points,
  the truncation never seriously distorted this leaderboard — but a referee
  should meet its own tolerance, and §9's ~1e-4 reconstruction drift was a
  symptom of exactly this (an unsettled fit amplifies Yahoo's quiet close
  revisions). Converging costs median 1.6s → 2.8s per fit (~+35s/day).
- **The change** (`config.py`): `HMM_EM_MAX_ITERS = 1000` — a CAP, not a
  budget; the loop breaks at tolerance (~30–110 iters on the watchlist), so
  a converging fit never pays the headroom, and a fit that still hits the
  cap is genuinely stuck and keeps its `non_converged` taint. Determinism
  unchanged (same (ticker, anchor) seeds, same restarts).
- **Method-faithful reconstruction**: every hmm row's model descriptor now
  records the cap it was fitted under (`baseline:hmm(K=3,2512d,mc20k,em1000)`)
  and `hmm-health --backfill` refits each row under the row's OWN cap
  (`em_cap_from_model`; no token = pre-changeover row = `HMM_EM_LEGACY_ITERS`
  40). Without the pin, a retro health check of an old row would converge
  into a different optimum and read as a false `reconstruction_mismatch`.
- **Nothing logged was rewritten** (the wall stands): all resolved quant-bar
  values and their flags stay exactly as logged. CHANGEOVER NOTE for future
  pooled reads: hmm_filter rows are pre/post distinguishable PER ROW by the
  `em` token (and per fit by `hmm_fits.converged`/`n_iter`). The value shift
  across the boundary is ≤0.028, so pooled Brier ladders are fine; just
  don't read pre/post stability flags as one regime, and expect the
  `non_converged` taint rate to drop to ~0 from here on.
- Tests 110 → 113 (em-token round-trip + legacy default; em_iters provably
  reaches the fit; legacy reconstruction reproduces the truncated fit
  exactly).

## 11. COMPLETED 2026-06-22 — v1 macro arm (`pythia_macro`, FRED point-in-time)

The roadmap's "v1+ (build next): FRED macro as a measured A/B against the
price-only arms" is done. A tenth forecaster column, `pythia_macro`, runs
alongside raw `pythia` on every claim: identical ticker, identical price data,
identical horizon, identical anchor — the ONLY difference is a point-in-time
macro block in the prompt and a system prompt that lifts the v0 price-only
restriction for that arm alone. Raw `pythia` stays price-only forever, so the
macro effect is MEASURED on identical claims for the life of the project (same
design discipline as the coached arm, §4).

- **Six daily, market-implied FRED series** (`config.FRED_SERIES`): DGS10, DGS2
  (rates — and the 2s10s slope is DERIVED in the formatter from those two, not
  fetched, so it can't mismatch its own yields on vintage), T10YIE (breakeven
  inflation), BAMLH0A0HYM2 (HY credit spread), VIXCLS, DTWEXBGS (trade-weighted
  dollar). Chosen because all actually MOVE within a 5-session horizon (monthly
  macro is largely stale within the window) and all are exactly what the v0
  prompt forbids — so v1 lifts that ban for one arm and measures whether it
  helps calibration. The context block shows each series' latest value plus the
  value 5 and 20 sessions back (parallel to the price context's 5d/20d returns)
  and the change in native units.
- **Point-in-time via FRED's realtime API** (`macro.py`): every series fetched
  with `realtime_start=realtime_end=anchor_date` AND `observation_start/end`
  spanning the lookback, so each returned observation is the value KNOWN on the
  anchor day (its real-time period includes the anchor), not today's revised
  value. A backfilled anchor therefore sees the macro the forecaster actually
  had — no future-revision leakage. For these daily market-implied series
  revisions are near-nil anyway, but the realtime call is free and is the
  method-faithful choice. Unit-tested (`test_macro.py` pins both realtime params
  to the anchor date).
- **Audit trail**: one `macro_snapshots` row per anchor_date (macro is per-date,
  shared across all 30 tickers — the first ticker on a run pays the FRED fetch,
  the rest reuse it from the cache or DB). The series payload is content-hashed
  (`context_sha`) and the macro arm's `model` column carries `+macro:<sha>`, so
  any slice of the record ties to the exact macro data the arm saw — same
  pattern as the lessons sha (§4). First-write-wins like forecasts/hmm_fits;
  the vintage for a given anchor date never changes.
- **No `backfill-macro` command — deliberate.** Unlike `backfill-hmm` (a pure
  statistical model, clean to reconstruct point-in-time), the macro arm is an
  LLM call: backfilling would ask Opus *today* to forecast an old, possibly-
  resolved claim, risking training-cutoff leakage of the outcome. The macro arm
  is LIVE-ONLY, like raw `pythia` and `kalshi` — the A/B accumulates going
  forward, consistent with "logs before the outcome is known."
- **Optional + best-effort** (like email alerts): needs `FRED_API_KEY` in
  `.env`; when unset the macro arm is a clean no-op and the price-only arms run
  unaffected. A dead/missing series renders as n/a; if NO series returns
  usable data the arm skips that day (a missing row beats a fudged one — same
  rule as Kalshi).
- **No iso/coached-macro variants yet.** Base arm first, measured clean. A
  `pythia_macro_iso` can follow once the macro arm clears `ISO_MIN_RESOLVED`
  (150) — the calibrators are intentionally fitted on the base arms' own
  records only, and macro must not leak into the raw/coached iso fits.
- **Cost**: +1 Opus call per ticker per day. 30→60 with macro on (coached still
  off until first reflect); 30→90 once both macro and coached run. Within the
  "cost is trivial at 30-60 calls/day" line the project set — 90 is the upper
  bound it anticipated.
- Tests 113 → 137 (+24: formatter incl. the derived slope and missing-series
  handling, sha determinism + order-independence, FRED realtime point-in-time
  params, missing-value skipping, snapshot orchestration with a dead series,
  DB round-trip + idempotency + fresh-DB migration, forecaster prompt/tool
  variants).

The deploy question this arm answers, in order: does real point-in-time macro
beat the price-only arms (`pythia`, `hmm_filter`) on calibration? Delphi's
finding (§3.4) was that the correction layers, not the model, carry most of the
value — macro is the LLM's one plausible edge over a pure price model, and the
only honest way to test it is on claims logged before the outcome, which is now
running.

## 12. COMPLETED 2026-07-07 — v2: the paper book (`paper.py`/`pnl.py`, simulated options)

The Brier ladder says who is calibrated; the paper book says what that
calibration would have been *worth* — P&L is exactly where overconfidence
becomes visible (Delphi: $1 -> $0.025 at full Kelly behind a respectable
Brier). SIMULATED ONLY; the hard wall stands — no broker, no orders, nothing
that prepares one.

- **One contract pair per claim** (long ATM call + put, same strike, expiry
  within ±2 sessions of `resolves_on`, signed gap recorded — the kalshi
  settle-gap pattern). **Every arm and bar trades the same pair at the same
  logged quote**; only direction (its own p; exactly 0.5 sits out, which
  removes coin_flip) and read-time sizing differ, so the P&L ladder mirrors
  the Brier ladder row for row. Whitelist (conservative, gates are the real
  filter): SPY QQQ IWM DIA TLT GLD.
- **Entry honesty**: positions open only from probabilities READ BACK from
  logged forecast rows (never in-memory results); a time gate (capture before
  the next session's open) and an underlying-drift gate (≤0.5% off the anchor
  close) keep post-anchor information out of entries; kalshi-style book gates
  (two-sided uncrossed, relative-dominant spread cap max($0.05, 15% of mid),
  premium floor $0.20, OI ≥ 100) refuse-don't-fudge. BOTH sides must pass or
  nobody trades (a one-sided book would bias bullish vs bearish arms). Every
  selected book is logged usable-or-not (`option_quotes`, first-write-wins,
  gates-in-force stamped per row); claim-level refusals log nothing, like an
  unmatched kalshi claim.
- **P&L of record = entry at logged mid -> settle at INTRINSIC off the
  official raw close on expiry** (scoring.py's close). Deliberately NO daily
  marks — after-hours yfinance books are stale/zero-bid and a mark from one
  would be unauditable noise; open positions carry at cost. `--worst-fill`
  re-prices entries at the logged ask (the whole transaction-cost model).
- **Sizing is a measured READ-TIME ladder** (`pythia pnl`), never stored:
  fixed $100 / edge-scaled / kelly05 / kelly10 (compounding bankroll replay,
  deterministic from immutable rows; full Kelly exists precisely so an
  overconfident arm visibly compounds toward ruin). Changing the policy
  constants re-renders history — pnl stamps the params in force.
- **TIMING REALITY (watch week 1): ETF options stop trading 4:15 ET and Yahoo
  after-hours books are often one-sided/zero-bid — an evening run will log
  many honest usable=0 days. A usable paper book wants the capture in the
  ~4:00-4:15 ET window (`pythia paper-trade` is the same-evening retry).**
  Check the usable-rate after the first week before reading anything into
  book sizes.
- Live-only, no backfill (yfinance has no historical chains): the book
  accumulates forward from 2026-07-07. Tests 137 -> 184.

## 13. COMPLETED 2026-07-07 — v3: high-conviction digest + a gradeable alert record

The missing half of v3 (notify.py already emailed every batch): flag the calls
worth a human's attention, and make the alert rule itself accountable.

- **Rule**: any live LLM arm (raw/coached/macro) whose own p crosses
  |P-0.5| >= 0.15 flags the claim. Iso arms excluded (conviction is a remap
  artifact at this record size — the current PAV fit caps every output below
  0.5); bars excluded (the HMM's confident calls graded 38.9% hit on the early
  record — bars grade the product, they don't page the human).
- **The threshold was chosen POST-HOC** at the visible cliff of the early
  record (raw >=0.15: 11/12; coached: 6/6; ~1 flagged day/week — vs 60%
  hit at 0.10-0.15), n≈12 over correlated windows. So every flag is LOGGED
  (`digest_alerts`, first-write-wins, live-only, threshold AND gate arms in
  force stamped per row, `emailed_at` = delivery receipt, written even with
  SMTP unset): the alert log IS the rule's out-of-sample test. **Revisit
  pre-registered at 30 resolved alerts.** Outcomes never stored — the
  scoreboard (`pythia alerts`) derives them by joining to `forecasts` (one
  grading path), per-gating-arm alongside pooled (an OR-gate is three chances
  per claim).
- Flags are computed from rows READ BACK from the DB (an alert must never
  carry a p that matches no logged row). Digest sections prepend the existing
  batch email — quiet days say so explicitly; the subject gains
  "N HIGH-CONVICTION |" only on flagged days; overlapping-window repeats are
  called out inline ("same bet as the YYYY-MM-DD alert, not a new
  confirmation" — USO once flagged 4 batches running). Re-sends render logged
  alerts verbatim, never recomputed; pre-feature batches are labeled
  reconstructed. Wall language in every digest. Deferred: showing the kalshi
  mid to the human (unresolved anchoring argument), other transports.
  Tests 184 -> 204.

## 14. COMPLETED 2026-07-07 — v4: the dashboard (`pythia publish`, local-only for now)

`dashboard.py` is pure two-stage generation (rows -> aggregates -> HTML/JSON),
written to `docs/` as one self-contained page (inline CSS, hand-rolled inline
SVG, no scripts/CDN — works from file:// and GitHub Pages alike) plus
`data.json` (the same aggregates, machine-readable — keeps "aggregates only"
auditable).

- **Aggregates only, never rows**: no claim text, no reasoning, no per-claim
  probabilities; pending claims appear as COUNTS only (publishing live calls
  pre-resolution is an owner-level disclosure decision, not taken). Leak-guard
  tests plant sentinels in rows and assert they never reach the output.
- **Honesty furniture is first-class content**, rendered from data: the
  month-3 embargo banner ("do not rank yet", auto-demotes at 90 days), Wilson
  95% whiskers on equal-count calibration bins (min 25/bin, max 8 bins, NO
  curve below 100 resolved — "curve appears at N resolved"), the independence
  caveat on the panels themselves, the HMM taint line (hmm_health verbatim),
  the kalshi sparse-coverage note, per-section paper-book and alert-rule
  aggregates, and a research/no-advice/no-live-trading disclaimer strip.
- **Change detection**: `content_sha` hashes the aggregate payload EXCLUDING
  the generated-at stamp; unchanged records skip the write (`publishes` audit
  table). Brier-over-time renders as cumulative running averages (weekly
  buckets on overlapping claims would be noise).
- **`pythia publish` NEVER touches git.** OWNER CALL still open: going public
  needs a public repo (GitHub Pages free tier) — that exposes all code,
  lessons, and an unretractable public track record (which is also its
  scientific value). Alternatives: a separate dashboard-only public repo, or
  keep it local. No publish.bat exists yet ON PURPOSE — an unscheduled .bat
  that pushes is a loaded gun; write it only after the owner call, with the
  guards from the design review (abort on staged changes, never pull, never
  force, nonzero exit + log on push failure, no `pause`, ISO dates).
  Tests 204 -> 218.

## 15. Ops notes (2026-07-07)

- **Missed forecast day**: no anchors 2026-07-06 (Mon). Fri 7/3 was the
  July-4 holiday; the machine was likely off Monday. NOT backfilled — the
  LLM/kalshi/macro columns are live-only by design; the gap is honest. If
  this recurs, enable "Run task as soon as possible after a scheduled start
  is missed" on the forecast task in Windows Task Scheduler.
- `lessons.txt` + `lessons_history.jsonl` are now COMMITTED (the coached
  rows' `+lessons:<sha>` must stay resolvable to text even if this machine
  dies). The DB stays local/gitignored.
- The paper book and alert record both started 2026-07-07 — every day they
  run is record; gaps are permanent (no backfill anywhere).
