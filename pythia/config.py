"""Central configuration for Pythia.

Single source of truth for model IDs, the watchlist, lookback windows, and
defaults. Everything tunable lives here so it can be swapped in one place.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Load .env (if present) on import so ANTHROPIC_API_KEY and overrides are available.
load_dotenv()

# --- Model routing -----------------------------------------------------------
# Different jobs call different Claude models. See CLAUDE.md for the rationale.
# The daily forecast is the brain — calibration quality is the whole point, and
# it runs only once per ticker per day, so we never downgrade it.
MODEL_FORECAST = "claude-opus-4-8"   # the daily forecast (the brain)
MODEL_REVIEW = "claude-opus-4-8"     # weekly self-review (v1+), highest-leverage reasoning
MODEL_TRIAGE = "claude-haiku-4-5"    # high-volume extraction/tagging (v1+), cheap and fast

# --- Watchlist ---------------------------------------------------------------
# Broad-based / liquid ETFs only. No single stocks, no single-stock ETFs.
# Deliberately spread across distinct return drivers so the track record isn't
# a handful of correlated bets. 30 tickers: more graded forecasts per calendar
# week = statistical significance arrives sooner (the binding constraint on the
# whole project is evidence per week, not model quality).
WATCHLIST: list[str] = [
    # US broad equity
    "SPY", "QQQ", "IWM", "DIA",
    # US sectors (SPDRs)
    "XLF", "XLE", "XLK", "XLV", "XLI", "XLP", "XLU", "XLB",
    # US industry / theme
    "SOXX",                        # semiconductors
    "ITA",                         # aerospace & defense
    "CIBR",                        # cybersecurity
    "VNQ",                         # real estate (REITs)
    # International equity
    "EFA", "EEM", "EWJ",          # developed / emerging / Japan (broad baskets)
    "FXI", "EWZ",                 # China / Brazil (single-country)
    # Bonds / credit
    "TLT", "IEF",                 # 20+yr / 7-10yr Treasuries
    "HYG", "LQD",                 # high-yield / IG credit
    # Commodities
    "GLD", "SLV", "USO",          # gold / silver / crude oil
    # Crypto
    "IBIT", "ETHA",               # spot bitcoin / ether
]

# --- Asset classes -----------------------------------------------------------
# Used only to pick the forecaster's *directional prior* (see forecaster.py).
# Broad equity ETFs carry a real structural long-run up-drift (the equity risk
# premium) — a data-free, NON-macro prior worth giving the model. Commodities,
# bonds, and crypto funds have no such reliable drift, so handing them an equity
# up-bias would hurt calibration; their neutral anchor is 0.50. This is not a
# macro view and stays within the price-only ablation.
EQUITY = "equity"
NON_EQUITY = "non_equity"

# Conservative mapping. EQUITY (structural up-drift prior) only for broad,
# diversified equity baskets. Deliberately NON_EQUITY despite being stocks:
# FXI/EWZ (single-country EM — no reliable drift) and VNQ/HYG/LQD (heavy
# distributions — on RAW closes, ex-dividend drops eat much of the drift, so a
# price-only up-bias would be miscalibrated for the "close >= anchor" claim).
ASSET_CLASS: dict[str, str] = {
    "SPY": EQUITY, "QQQ": EQUITY, "IWM": EQUITY, "DIA": EQUITY,
    "XLF": EQUITY, "XLE": EQUITY, "XLK": EQUITY, "XLV": EQUITY,
    "XLI": EQUITY, "XLP": EQUITY, "XLU": EQUITY, "XLB": EQUITY,
    "SOXX": EQUITY, "ITA": EQUITY, "CIBR": EQUITY,
    "EFA": EQUITY, "EEM": EQUITY, "EWJ": EQUITY,
    "VNQ": NON_EQUITY, "FXI": NON_EQUITY, "EWZ": NON_EQUITY,
    "TLT": NON_EQUITY, "IEF": NON_EQUITY, "HYG": NON_EQUITY, "LQD": NON_EQUITY,
    "GLD": NON_EQUITY, "SLV": NON_EQUITY, "USO": NON_EQUITY,
    "IBIT": NON_EQUITY, "ETHA": NON_EQUITY,
}


def asset_class(ticker: str) -> str:
    """Asset class for the directional prior.

    Defaults to NON_EQUITY (the neutral, no-drift prior) for anything unmapped,
    so we only ever assert an up-drift where we know the asset is broad equity.
    """
    return ASSET_CLASS.get(ticker.upper(), NON_EQUITY)

# --- Forecast defaults -------------------------------------------------------
# Horizon is measured in *trading sessions*, resolved via a real exchange
# calendar (never naive calendar-day math).
DEFAULT_HORIZON_DAYS = 5  # one trading week

# How much daily price history to fetch and hand to the forecaster (calendar days).
PRICE_LOOKBACK_DAYS = 180

# Window (in trading sessions) used to compute the drift baseline's positive-day
# ratio. Computed from data, never hardcoded.
DRIFT_LOOKBACK_SESSIONS = 252  # ~one year of sessions

# Window (in trading sessions) for the naive-momentum baseline: "up" if the net
# move over the last N sessions was positive.
MOMENTUM_LOOKBACK_SESSIONS = 5

# Fixed confidence the naive-momentum rule places in its directional call. A
# Brier comparison needs a probability, not just a direction, so the rule
# predicts its chosen side at this confidence (and 1 - this on the other side).
# Kept modest on purpose — a dumb rule shouldn't be maximally confident.
MOMENTUM_CONFIDENCE = 0.60

# --- HMM quant-bar baseline (hmm_baseline.py) ---------------------------------
# The fourth baseline: a Gaussian HMM regime filter fitted per ticker, strictly
# point-in-time. It needs YEARS of history (unlike the other baselines), so it
# does its own deeper fetch rather than reusing the 180-day forecast window.
HMM_LOOKBACK_DAYS = 3650          # calendar days of history to fetch (~10y)
HMM_STATES = 3                    # regimes when history is rich (calm-up/churn/stress)
HMM_RICH_HISTORY_SESSIONS = 750   # below this many sessions, drop to 2 states
HMM_MIN_SESSIONS = 300            # below this, refuse to fit (young funds e.g. ETHA)
HMM_MC_PATHS = 20_000             # Monte-Carlo paths for the horizon probability

# EM iteration cap for the Baum-Welch fit. A CAP, not a budget: the loop breaks
# at its loglik tolerance, which the watchlist needs ~30-110 iterations to meet
# (audited 2026-06-12, audit_em_convergence.py) — so a converging fit never
# pays for the headroom, and a fit that still hits this cap is genuinely stuck
# and gets the non_converged taint. History: 40 until 2026-06-12, which
# truncated 97% of fits a hair short of tolerance (same failure mode Delphi
# found and fixed the same day, summary.md §18 there). Rows logged under the
# old cap carry no `em` token in their model descriptor and are reconstructed
# with HMM_EM_LEGACY_ITERS so retro health checks stay method-faithful.
HMM_EM_MAX_ITERS = 1000
HMM_EM_LEGACY_ITERS = 40          # the cap behind every row logged without an em token

# --- HMM fit-health monitoring (hmm_health.py) ---------------------------------
# The quant bar is the deploy gate, so a silently broken referee corrupts every
# leaderboard comparison against it. Every fit is recorded and judged: EM that
# exhausts its iterations without meeting tolerance, or parameters that jump
# beyond these thresholds between consecutive refits, taint that day's value.
# Policy (decided 2026-06-12): log + flag — a tainted value STAYS on the
# leaderboard, visibly annotated; dropping it would bias the bar toward the
# days where fitting is easy. Thresholds are sized so one new observation on
# thousands can't trip them — a trip means EM landed in a different optimum.
HMM_STABILITY_MAX_GAP_DAYS = 7    # only compare fits this close (calendar days)
HMM_DURATION_JUMP_RATIO = 3.0     # expected regime duration ratio between refits
HMM_SIGMA_JUMP_RATIO = 1.5        # per-state vol ratio between refits
HMM_MU_JUMP_ABS = 0.0010          # per-state mean daily log-return shift (10 bps)

# Retro-annotation only (`hmm-health --backfill`): a reconstructed fit must
# land within this much of the logged probability to be trusted as THE fit
# behind the row. Bit-exactness is unattainable — Yahoo quietly revises closes
# deep in history and a still-moving (non-converged) EM amplifies them into
# ~1e-4 output drift — while a different EM basin moves the output by whole
# percentage points, far past this line.
HMM_RECONSTRUCTION_TOL = 1e-3

# --- Kalshi market-implied baseline (kalshi.py) -------------------------------
# Read-only public market data; no account, no key, no orders — strictly on the
# forecast/grade side of the hard wall. Odds are a BASELINE COLUMN only and must
# never reach the forecaster prompt (Delphi anchoring experiment, §13.3).
KALSHI_API_BASE = "https://api.elections.kalshi.com/trade-api/v2"

# Whitelisted series per ticker, tried in order. Only series that settle on the
# OFFICIAL CLOSE (or, for BTC, the 5pm benchmark print) — hourly/intraday series
# are deliberately excluded. Dormant series cost one cheap empty request and are
# harmless to keep listed (Kalshi rotates weekly/monthly listings).
KALSHI_SERIES: dict[str, list[str]] = {
    "SPY": ["KXINXW", "KXINXM", "KXINX"],                    # S&P 500 weekly/monthly/daily
    "QQQ": ["KXNASDAQ100W", "KXNASDAQ100M", "KXNASDAQ100"],  # Nasdaq-100
    "IBIT": ["KXBTCW", "KXBTCD"],                            # BTC 5pm EDT benchmark
}

# Claim level -> contract underlying mapping (same-day raw closes, yfinance).
KALSHI_UNDERLYING: dict[str, str] = {
    "SPY": "^GSPC",
    "QQQ": "^NDX",
    "IBIT": "BTC-USD",
}

# A contract must settle within this many trading sessions of the claim's
# resolves_on to count. Sparse coverage is expected and accepted: with Kalshi's
# current listings (settles through Friday), roughly Mon/Tue anchors match.
KALSHI_MAX_SETTLE_GAP_SESSIONS = 2

# Widest acceptable yes bid/ask spread (in dollars) for a usable mid-quote.
KALSHI_MAX_SPREAD = 0.10

# Polite gap between unauthenticated API calls (the public limit trips easily).
KALSHI_THROTTLE_S = 1.0

# --- FRED macro context (v1: the macro-aware arm) ------------------------------
# Read-only public FRED API; needs FRED_API_KEY in .env (free key from
# https://fred.stlouisfed.org/docs/api/api_key.html). When unset, the
# pythia_macro arm is a clean no-op — the price-only arms run unaffected, just
# like email alerts without SMTP. Macro is fed ONLY to pythia_macro, never to
# the raw/coached arms, so the v0 price-only ablation stays clean and the macro
# effect is measured on IDENTICAL claims (same anchor, same price data, same
# horizon; the only difference is the macro block in the prompt).
#
# Point-in-time is the whole point of v1. Every series is fetched via FRED's
# realtime API (realtime_start=realtime_end=anchor_date) so the arm sees the
# values KNOWN on the anchor day, not today's revisions. Without this, a
# backfilled anchor would leak revised macro the forecaster never saw, and the
# A/B against price-only would be dishonest. For these daily market-implied
# series revisions are near-nil anyway, but the realtime call is free and is the
# methodologically faithful choice (and it future-proofs any monthly series).
FRED_API_BASE = "https://api.stlouisfed.org/fred/series/observations"
FRED_LOOKBACK_DAYS = 90          # calendar days of history to fetch (5d/20d change context)
FRED_THROTTLE_S = 0.2            # polite gap (free tier allows 120 req/min)


@dataclass(frozen=True)
class FredSeries:
    series_id: str
    label: str   # human label rendered in the macro context block
    unit: str    # "%" for yields/spreads; "" for index points / VIX
    fmt: str     # format spec for the value, e.g. ".2f"


# Six daily, market-implied series — all actually MOVE within a 5-session
# horizon (unlike monthly macro, which is stale within the window), and all are
# read point-in-time via the realtime API. Covers rates (2y/10y), the curve
# (the 2s10s slope is DERIVED in the formatter, not fetched), inflation
# expectations (10y breakeven), credit stress (HY spread), risk appetite (VIX),
# and the dollar. These are exactly the "rates, inflation, the Fed, the yield
# curve" the v0 price-only prompt forbids — v1 lifts that ban for the macro arm
# only and MEASURES whether it helps calibration.
FRED_SERIES: tuple[FredSeries, ...] = (
    FredSeries("DGS10", "10y Treasury yield", "%", ".2f"),
    FredSeries("DGS2", "2y Treasury yield", "%", ".2f"),
    FredSeries("T10YIE", "10y breakeven inflation", "%", ".2f"),
    FredSeries("BAMLH0A0HYM2", "HY credit spread", "%", ".2f"),
    FredSeries("VIXCLS", "VIX", "", ".1f"),
    FredSeries("DTWEXBGS", "Trade-weighted dollar", "", ".2f"),
)


def fred_api_key() -> str | None:
    """FRED API key from the env, or None when the macro arm should no-op."""
    key = os.environ.get("FRED_API_KEY", "").strip()
    return key or None

# --- Simulated option positions (paper.py / pnl.py; roadmap v2) ----------------
# SIMULATED ONLY — the hard wall stands. The paper book turns already-logged
# probabilities into simulated long single ATM options: entered at the logged
# after-close mid, settled at intrinsic value off the official raw close on
# expiry (the same close scoring.py grades with — provably correct, stable
# forever). There is deliberately NO daily mark-to-market of record: a mark
# from a delayed, often stale/zero-bid after-hours yfinance book would inject
# unauditable noise into a record whose whole value is honesty. No broker, no
# orders, no live execution anywhere in this layer.
#
# Every arm and bar trades the SAME contract pair at the SAME logged quote —
# only direction (its own p) and read-time sizing differ — so the P&L ladder
# is a true mirror of the Brier ladder. P&L is exactly where overconfidence
# becomes visible (Delphi: $1 -> $0.025 at full Kelly); that visibility is the
# point, so every forecast trades, not just high-conviction ones (conviction
# slices are a read-time filter in `pythia pnl`, never a logging gate).

# Watchlist subset with dense listed expiries and tight ATM books. The quote
# gates below are the real filter; this list just avoids chain fetches for
# tickers that would never pass them. Deliberately conservative to start —
# yfinance openInterest is previous-day and often 0 on thin weeklies, so the
# book will be index-ETF-dominated; widen only after the gates prove out.
OPTIONS_WHITELIST: list[str] = ["SPY", "QQQ", "IWM", "DIA", "TLT", "GLD"]

# Contract selection. Expiry must sit within this many trading sessions of the
# claim's resolves_on (kalshi settle-gap precedent; the signed gap is recorded
# on every position row, never hidden), and the nearest listed strike must sit
# within this fraction of the anchor close to count as ATM.
OPTIONS_MAX_EXPIRY_GAP_SESSIONS = 2
OPTIONS_MAX_ATM_DISTANCE = 0.02

# Quote gates — refuse, don't fudge (a failed gate logs the quote row with
# usable=0 and the reason; no positions). Spread gate is RELATIVE-dominant:
# an absolute-dominant cap would invert its intent on cheap premia (a $0.10
# spread on a $0.30 mid is 33% execution noise). The premium floor rejects
# books where even the relative gate leaves noise dominating ROI. After-hours
# books are often one-sided or zero-bid — usable=0 days are expected and
# honest; a usable paper book wants the capture near the close (~4:00-4:15 ET).
OPTIONS_MAX_SPREAD_ABS = 0.05     # spread <= max(this, ...) — floor for tiny mids ($)
OPTIONS_MAX_SPREAD_REL = 0.15     # ... max(..., this fraction of mid)
OPTIONS_MIN_PREMIUM = 0.20        # minimum usable mid ($)
OPTIONS_MIN_OPEN_INTEREST = 100   # a book nobody holds is not a market

# Entry honesty gates. The probabilities were computed as of the anchor close,
# so the entry quote must not embed post-anchor information: the underlying
# must sit within this fraction of the anchor close at fetch time, AND the
# fetch must happen before the next session's open (paper.entry_window_ok).
OPTIONS_MAX_UNDERLYING_DRIFT = 0.005

OPTIONS_THROTTLE_S = 1.0          # polite gap between yfinance chain calls

# The measured sizing ladder (pnl.py) — derived at READ time from immutable
# position rows, never stored, so the books are reproducible and immune to
# out-of-order-settlement state corruption. `fixed` and `edge` are
# non-compounding; the kelly-proxy books compound a bankroll and risk
# frac * 2|p-0.5| of available cash per trade ("proxy" because an ATM option
# is only approximately an even-money directional bet). kelly10 (full
# fraction) exists precisely so an overconfident arm visibly compounds toward
# ruin. NOTE: these constants parameterize a read-time VIEW — changing them
# re-renders history under the new policy (the logged rows never change);
# `pythia pnl` stamps the params in force on its output for that reason.
OPTIONS_FIXED_STAKE = 100.0       # $ premium per trade (fixed/edge books)
OPTIONS_BANKROLL_0 = 1000.0       # starting bankroll per compounding book
PAPER_POLICIES: tuple[str, ...] = ("fixed", "edge", "kelly05", "kelly10")
KELLY_FRACTIONS: dict[str, float] = {"kelly05": 0.5, "kelly10": 1.0}


def option_gates_descriptor() -> str:
    """Compact record of the gate values in force, stamped on every quote row.

    The gates will be tuned as the book accumulates; each row carrying the
    gates it was judged under keeps record slices honest across changeovers
    (same convention as the hmm `em` token and the lessons/macro shas).
    """
    return (
        f"spread<=max({OPTIONS_MAX_SPREAD_ABS},{OPTIONS_MAX_SPREAD_REL}m),"
        f"prem>={OPTIONS_MIN_PREMIUM},oi>={OPTIONS_MIN_OPEN_INTEREST},"
        f"atm<={OPTIONS_MAX_ATM_DISTANCE},drift<={OPTIONS_MAX_UNDERLYING_DRIFT},"
        f"expgap<={OPTIONS_MAX_EXPIRY_GAP_SESSIONS}"
    )

# --- Market / calendar -------------------------------------------------------
# NYSE calendar covers SPY/QQQ/IWM (US equity ETFs).
EXCHANGE = "XNYS"
MARKET_TZ = "America/New_York"

# --- Forecaster identities ---------------------------------------------------
# The label stored on each row so Pythia and every baseline can be graded
# side by side on the same claims.
PYTHIA = "pythia"
HMM_FILTER = "hmm_filter"
KALSHI = "kalshi"
BASELINES: list[str] = ["coin_flip", "drift", "naive_momentum", HMM_FILTER, KALSHI]

# The Delphi-validated correction stack (Delphi summary.md §8: lessons and
# isotonic calibration each help, and STACK), run as separate arms on identical
# claims so every layer stays measured, never assumed:
#   pythia             raw Opus — the permanent control arm
#   pythia_coached     Opus + distilled lessons in the prompt (reflect.py)
#   pythia_macro       Opus + real point-in-time macro (FRED) — the v1 A/B arm
#   pythia_iso         isotonic remap of pythia's raw probability (calibrate.py)
#   pythia_coached_iso isotonic remap of the coached probability (full stack)
PYTHIA_COACHED = "pythia_coached"
PYTHIA_MACRO = "pythia_macro"
PYTHIA_ISO = "pythia_iso"
PYTHIA_COACHED_ISO = "pythia_coached_iso"
PYTHIA_ARMS: list[str] = [PYTHIA, PYTHIA_COACHED, PYTHIA_MACRO, PYTHIA_ISO, PYTHIA_COACHED_ISO]
ALL_FORECASTERS: list[str] = [*PYTHIA_ARMS, *BASELINES]

# Human-friendly labels for review output.
FORECASTER_LABELS: dict[str, str] = {
    "pythia": "Pythia (raw)",
    "pythia_coached": "Pythia + lessons",
    "pythia_macro": "Pythia + macro",
    "pythia_iso": "Pythia + isotonic",
    "pythia_coached_iso": "Pythia + lessons + isotonic",
    "coin_flip": "Coin flip",
    "drift": "Drift (base rate)",
    "naive_momentum": "Naive momentum",
    "hmm_filter": "HMM filter (quant bar)",
    "kalshi": "Kalshi (market odds)",
}

# --- Self-review (reflect.py) --------------------------------------------------
# Don't review until the record can support generalizable lessons.
REFLECT_MIN_RESOLVED = 25
REFLECT_MAX_LESSONS = 8

# --- Isotonic calibration (calibrate.py) ----------------------------------------
# Below this many resolved rows the remap overfits noise (measured in Delphi:
# isotonic HURT at ~200-400 training points, shone at ~1,700) — the derived
# columns simply skip until the record is big enough.
ISO_MIN_RESOLVED = 150

# --- Storage -----------------------------------------------------------------
# The track record lives in a local SQLite file (gitignored). Override with
# PYTHIA_DB_PATH (used by tests to point at a temp file).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def db_path() -> Path:
    """Resolve the SQLite database path (env override or project-root default)."""
    override = os.environ.get("PYTHIA_DB_PATH")
    if override:
        return Path(override)
    return _PROJECT_ROOT / "pythia.db"


# Live lessons file (written by `pythia reflect`, read by the coached arm).
# Gitignore-adjacent state like the DB; history sits next to it.
LESSONS_PATH = _PROJECT_ROOT / "lessons.txt"


# --- Email alerts (optional; roadmap v3, pulled forward) ---------------------
# Surfacing-only: Pythia can email a summary of each batch of predictions right
# after it logs them, so a human can see what was called and in which direction.
# This stays on the forecasting side of the hard wall — it never places or
# suggests a live order. Every value comes from the environment so the SMTP
# password is never committed (see .env.example). Defaults target Gmail.
SMTP_HOST_DEFAULT = "smtp.gmail.com"
SMTP_PORT_DEFAULT = 587  # STARTTLS


@dataclass(frozen=True)
class EmailConfig:
    host: str
    port: int
    user: str
    password: str
    sender: str
    recipient: str


def email_config() -> EmailConfig | None:
    """Email settings from the environment, or None if not configured.

    Requires at least PYTHIA_SMTP_USER and PYTHIA_SMTP_PASSWORD; sender and
    recipient default to the SMTP user (i.e. email yourself) when left unset.
    Returning None lets callers treat alerting as an optional no-op.
    """
    user = os.environ.get("PYTHIA_SMTP_USER", "").strip()
    password = os.environ.get("PYTHIA_SMTP_PASSWORD", "").strip()
    if not user or not password:
        return None
    host = os.environ.get("PYTHIA_SMTP_HOST", "").strip() or SMTP_HOST_DEFAULT
    port_raw = os.environ.get("PYTHIA_SMTP_PORT", "").strip()
    port = int(port_raw) if port_raw else SMTP_PORT_DEFAULT
    sender = os.environ.get("PYTHIA_EMAIL_FROM", "").strip() or user
    recipient = os.environ.get("PYTHIA_EMAIL_TO", "").strip() or user
    return EmailConfig(
        host=host, port=port, user=user, password=password,
        sender=sender, recipient=recipient,
    )
