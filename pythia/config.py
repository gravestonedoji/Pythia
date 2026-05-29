"""Central configuration for Pythia.

Single source of truth for model IDs, the watchlist, lookback windows, and
defaults. Everything tunable lives here so it can be swapped in one place.
"""

from __future__ import annotations

import os
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
# Broad-based, highly liquid ETFs only. No single stocks, no single-stock ETFs.
WATCHLIST: list[str] = ["SPY", "QQQ", "IWM"]

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
