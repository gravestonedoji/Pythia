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
# three correlated bets: US large/tech/small, semis, defense, cyber, crypto
# (BTC + ETH), oil, gold, and long-duration rates.
WATCHLIST: list[str] = [
    "SPY", "QQQ", "IWM",          # US large / tech / small cap
    "SOXX",                        # semiconductors
    "ITA",                         # aerospace & defense
    "CIBR",                        # cybersecurity
    "IBIT", "ETHA",               # spot bitcoin / ether
    "USO",                         # crude oil
    "GLD",                         # gold
    "TLT",                         # 20+yr Treasuries (rates/duration)
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

ASSET_CLASS: dict[str, str] = {
    "SPY": EQUITY, "QQQ": EQUITY, "IWM": EQUITY,
    "SOXX": EQUITY, "ITA": EQUITY, "CIBR": EQUITY,
    "IBIT": NON_EQUITY, "ETHA": NON_EQUITY,
    "USO": NON_EQUITY, "GLD": NON_EQUITY, "TLT": NON_EQUITY,
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

# --- Market / calendar -------------------------------------------------------
# NYSE calendar covers SPY/QQQ/IWM (US equity ETFs).
EXCHANGE = "XNYS"
MARKET_TZ = "America/New_York"

# --- Forecaster identities ---------------------------------------------------
# The label stored on each row so Pythia and every baseline can be graded
# side by side on the same claims.
PYTHIA = "pythia"
BASELINES: list[str] = ["coin_flip", "drift", "naive_momentum"]
ALL_FORECASTERS: list[str] = [PYTHIA, *BASELINES]

# Human-friendly labels for review output.
FORECASTER_LABELS: dict[str, str] = {
    "pythia": "Pythia",
    "coin_flip": "Coin flip",
    "drift": "Drift (base rate)",
    "naive_momentum": "Naive momentum",
}

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
