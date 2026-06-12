"""Offline tests for the Kalshi baseline's pure core (no network, no key).

Market dicts mirror the real API shape: dollar-string quote fields
(yes_bid_dollars / yes_ask_dollars), strike_type in {greater, greater_or_equal,
between, less}, ISO-8601 close_time with Z suffix.
"""

from __future__ import annotations

from datetime import date

import pytest

from pythia import config, data
from pythia.kalshi import (
    ladder_prob_above, mid_prob, pick_event, prob_above, range_prob_above,
    settle_date_of,
)


def greater(strike, bid, ask, event="EV-1", close="2026-06-12T21:00:00Z"):
    return {
        "strike_type": "greater", "floor_strike": strike,
        "yes_bid_dollars": f"{bid:.4f}", "yes_ask_dollars": f"{ask:.4f}",
        "event_ticker": event, "close_time": close,
    }


def between(floor, cap, bid, ask, event="EV-1", close="2026-06-12T21:00:00Z"):
    return {
        "strike_type": "between", "floor_strike": floor, "cap_strike": cap,
        "yes_bid_dollars": f"{bid:.4f}", "yes_ask_dollars": f"{ask:.4f}",
        "event_ticker": event, "close_time": close,
    }


def less(cap, bid, ask, event="EV-1", close="2026-06-12T21:00:00Z"):
    return {
        "strike_type": "less", "cap_strike": cap,
        "yes_bid_dollars": f"{bid:.4f}", "yes_ask_dollars": f"{ask:.4f}",
        "event_ticker": event, "close_time": close,
    }


# --- mid_prob -------------------------------------------------------------------

def test_mid_prob_is_the_mid():
    assert mid_prob(greater(100, 0.40, 0.44)) == pytest.approx(0.42)


def test_mid_prob_rejects_wide_spreads():
    wide = config.KALSHI_MAX_SPREAD + 0.02
    assert mid_prob(greater(100, 0.40, 0.40 + wide)) is None


def test_mid_prob_rejects_empty_books():
    assert mid_prob({"yes_bid_dollars": None, "yes_ask_dollars": "0.5000"}) is None
    assert mid_prob({"yes_bid_dollars": "0.0000", "yes_ask_dollars": "0.0000"}) is None


def test_mid_prob_accepts_deep_otm():
    # bid 0.00 / ask 0.01 is a real (tiny) price, not a missing book
    assert mid_prob(greater(100, 0.00, 0.01)) == pytest.approx(0.005)


# --- ladder ---------------------------------------------------------------------

LADDER = [
    greater(100, 0.89, 0.91),   # P(X >= 100) = 0.90
    greater(110, 0.69, 0.71),   # 0.70
    greater(120, 0.29, 0.31),   # 0.30
    greater(130, 0.04, 0.06),   # 0.05
]


def test_ladder_exact_strike():
    assert ladder_prob_above(LADDER, 110) == pytest.approx(0.70)


def test_ladder_interpolates():
    assert ladder_prob_above(LADDER, 115) == pytest.approx(0.50)  # halfway 0.70->0.30


def test_ladder_refuses_extrapolation():
    assert ladder_prob_above(LADDER, 99) is None
    assert ladder_prob_above(LADDER, 131) is None


def test_ladder_forces_monotone():
    noisy = LADDER + [greater(125, 0.39, 0.41)]  # 0.40 > the 0.30 quoted at 120
    p = ladder_prob_above(noisy, 125)
    assert p is not None and p <= 0.30  # clipped, never rises with the strike


def test_ladder_needs_two_quotes():
    assert ladder_prob_above([LADDER[0]], 100) is None


# --- range buckets ----------------------------------------------------------------

BUCKETS = [
    less(100, 0.09, 0.11),            # P(X < 100)        = 0.10
    between(100, 110, 0.19, 0.21),    # P(100 <= X < 110) = 0.20
    between(110, 120, 0.39, 0.41),    # = 0.40
    between(120, 130, 0.19, 0.21),    # = 0.20
    greater(130, 0.09, 0.11),         # P(X >= 130)       = 0.10
]


def test_range_tail_sum():
    # P(X >= 110) = 0.40 + 0.20 + 0.10 = 0.70 (total mass = 1.0)
    assert range_prob_above(BUCKETS, 110) == pytest.approx(0.70)


def test_range_splits_the_containing_bucket():
    # halfway into [110, 120): 0.20 above-half + 0.20 + 0.10 = 0.50
    assert range_prob_above(BUCKETS, 115) == pytest.approx(0.50)


def test_range_normalizes_total_mass():
    doubled = [dict(b) for b in BUCKETS]
    for b in doubled:
        b["yes_bid_dollars"] = f"{float(b['yes_bid_dollars']) * 1.2:.4f}"
        b["yes_ask_dollars"] = f"{float(b['yes_ask_dollars']) * 1.2:.4f}"
    assert range_prob_above(doubled, 110) == pytest.approx(0.70, abs=0.01)


def test_range_rejects_sick_books():
    half = BUCKETS[:2]  # total quoted mass 0.30 — not a distribution
    assert range_prob_above(half, 105) is None


def test_range_rejects_unbounded_tail_levels():
    assert range_prob_above(BUCKETS, 95) is None    # inside the 'less' tail
    assert range_prob_above(BUCKETS, 135) is None   # inside the 'greater' tail


def test_prob_above_prefers_ladder():
    assert prob_above(LADDER, 115) == pytest.approx(0.50)
    assert prob_above(BUCKETS, 110) == pytest.approx(0.70)


# --- event picking ---------------------------------------------------------------

def test_settle_date_converts_to_market_tz():
    # 21:00Z on Jun 12 is 5pm EDT the SAME day
    assert settle_date_of(greater(100, 0.4, 0.42)) == date(2026, 6, 12)
    # 01:00Z on Jun 13 is 9pm EDT Jun 12 — still the 12th in market time
    m = greater(100, 0.4, 0.42, close="2026-06-13T01:00:00Z")
    assert settle_date_of(m) == date(2026, 6, 12)


def test_sessions_between_is_signed_and_calendar_aware():
    # Fri 2026-06-12 -> Mon 2026-06-15 is one trading session forward
    assert data.sessions_between(date(2026, 6, 12), date(2026, 6, 15)) == 1
    assert data.sessions_between(date(2026, 6, 15), date(2026, 6, 12)) == -1
    assert data.sessions_between(date(2026, 6, 12), date(2026, 6, 12)) == 0


def test_pick_event_prefers_smallest_session_gap():
    fri = [greater(100, 0.4, 0.42, event="FRI", close="2026-06-12T21:00:00Z")] * 2
    wed = [greater(100, 0.4, 0.42, event="WED", close="2026-06-10T21:00:00Z")] * 2
    picked = pick_event(fri + wed, resolves_on=date(2026, 6, 15), max_gap=3)
    assert picked is not None
    mkts, settle, gap = picked
    assert mkts[0]["event_ticker"] == "FRI"
    assert settle == date(2026, 6, 12)
    assert gap == -1  # settles one session before the claim resolves


def test_pick_event_enforces_the_gap_tolerance():
    wed = [greater(100, 0.4, 0.42, event="WED", close="2026-06-10T21:00:00Z")]
    assert pick_event(wed, resolves_on=date(2026, 6, 17), max_gap=2) is None


def test_pick_event_ignores_quoteless_events():
    dead = [greater(100, 0.0, 0.0, event="DEAD", close="2026-06-15T21:00:00Z")]
    live = [greater(100, 0.4, 0.42, event="LIVE", close="2026-06-12T21:00:00Z")]
    picked = pick_event(dead + live, resolves_on=date(2026, 6, 15), max_gap=3)
    assert picked is not None and picked[0][0]["event_ticker"] == "LIVE"
