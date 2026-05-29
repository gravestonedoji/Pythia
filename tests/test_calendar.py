"""Tests for trading-calendar math.

`resolves_on` is computed by counting *open* sessions forward, and resolution
must never fire on a weekend or holiday. These run offline (calendar only).
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from pythia import data


# --- is_market_open ----------------------------------------------------------

def test_market_closed_on_weekend():
    assert data.is_market_open(date(2026, 5, 30)) is False  # Saturday


def test_market_closed_on_holidays():
    assert data.is_market_open(date(2026, 5, 25)) is False   # Memorial Day 2026
    assert data.is_market_open(date(2025, 12, 25)) is False  # Christmas
    assert data.is_market_open(date(2026, 1, 1)) is False     # New Year's Day


def test_market_open_on_normal_weekday():
    assert data.is_market_open(date(2026, 5, 29)) is True    # ordinary Friday


# --- trading_days_forward ----------------------------------------------------

def test_forward_skips_memorial_day():
    # From Fri 5/22: Mon 5/25 is Memorial Day (closed), so +2 sessions = Wed 5/27.
    assert data.trading_days_forward(date(2026, 5, 22), 2) == date(2026, 5, 27)


def test_forward_skips_christmas():
    # From Tue 12/23: 12/24 (open, half day), 12/25 closed, 12/26 open, 12/29 open.
    assert data.trading_days_forward(date(2025, 12, 23), 3) == date(2025, 12, 29)


def test_forward_simple_one_session():
    # Mon 2026-01-05 + 1 session = Tue 2026-01-06.
    assert data.trading_days_forward(date(2026, 1, 5), 1) == date(2026, 1, 6)


def test_forward_over_a_weekend():
    # Fri 2026-05-29 + 1 session lands on the following Mon (6/1), skipping the weekend.
    assert data.trading_days_forward(date(2026, 5, 29), 1) == date(2026, 6, 1)


def test_forward_rejects_non_positive_horizon():
    with pytest.raises(ValueError):
        data.trading_days_forward(date(2026, 1, 5), 0)


# --- latest_completed_session ------------------------------------------------

def test_latest_completed_after_close():
    # 21:00 UTC on a normal Friday is past the US close -> that Friday is complete.
    now = pd.Timestamp("2026-05-29 21:00", tz="UTC")
    assert data.latest_completed_session(now) == date(2026, 5, 29)


def test_latest_completed_before_close():
    # 12:00 UTC (pre-market in the US) -> the most recent completed session is Thursday.
    now = pd.Timestamp("2026-05-29 12:00", tz="UTC")
    assert data.latest_completed_session(now) == date(2026, 5, 28)


def test_latest_completed_on_weekend():
    # Sunday -> the prior Friday's close is the latest completed session.
    now = pd.Timestamp("2026-05-31 12:00", tz="UTC")
    assert data.latest_completed_session(now) == date(2026, 5, 29)
