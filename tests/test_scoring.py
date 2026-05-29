"""Tests for the scoring core — Brier, outcome resolution, and aggregation.

These are the tests the whole project's credibility rests on, so they are
exhaustive about the grading math and run fully offline (a temp SQLite DB and
injected price lookups — no network, no API key).
"""

from __future__ import annotations

from datetime import date

import pytest

from pythia import scoring, storage
from pythia.storage import Forecast


# --- brier_score -------------------------------------------------------------

@pytest.mark.parametrize(
    "probability, outcome, expected",
    [
        (0.5, 1.0, 0.25),   # the coin-flip floor
        (0.5, 0.0, 0.25),
        (1.0, 1.0, 0.0),    # confident and right
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 1.0),    # confident and wrong — maximal penalty
        (0.0, 1.0, 1.0),
        (0.8, 1.0, 0.04),
        (0.3, 0.0, 0.09),
        (0.9, 0.0, 0.81),
    ],
)
def test_brier_score_known_values(probability, outcome, expected):
    assert scoring.brier_score(probability, outcome) == pytest.approx(expected)


@pytest.mark.parametrize("bad", [1.5, -0.1, 2.0])
def test_brier_rejects_out_of_range_probability(bad):
    with pytest.raises(ValueError):
        scoring.brier_score(bad, 1.0)


@pytest.mark.parametrize("bad", [0.5, 2.0, -1.0])
def test_brier_rejects_non_binary_outcome(bad):
    with pytest.raises(ValueError):
        scoring.brier_score(0.5, bad)


# --- compute_outcome (claim: close on resolution day >= anchor close) --------

def test_outcome_true_when_above():
    assert scoring.compute_outcome(100.0, 101.0) == 1.0


def test_outcome_true_when_equal():
    # ">=" — a flat close still satisfies the claim.
    assert scoring.compute_outcome(100.0, 100.0) == 1.0


def test_outcome_false_when_below():
    assert scoring.compute_outcome(100.0, 99.99) == 0.0


# --- is_hit ------------------------------------------------------------------

@pytest.mark.parametrize(
    "probability, outcome, hit",
    [
        (0.7, 1.0, True),
        (0.7, 0.0, False),
        (0.3, 0.0, True),
        (0.3, 1.0, False),
        (0.5, 1.0, True),   # P>=0.5 implies an "up" call
        (0.5, 0.0, False),
    ],
)
def test_is_hit(probability, outcome, hit):
    assert scoring.is_hit(probability, outcome) is hit


# --- resolution --------------------------------------------------------------

def _forecast(forecaster, probability, resolves_on, *, anchor_close=100.0, anchor_date="2026-01-02"):
    return Forecast(
        forecaster=forecaster, ticker="SPY", claim="test claim", horizon_days=5,
        anchor_date=anchor_date, anchor_close=anchor_close, resolves_on=resolves_on,
        probability=probability, reasoning=None, model=None,
    )


@pytest.fixture
def conn(tmp_path):
    c = storage.get_connection(tmp_path / "test.db")
    yield c
    c.close()


def test_resolve_scores_and_persists(conn):
    storage.insert_forecast(conn, _forecast("pythia", 0.8, "2026-01-09"))
    storage.insert_forecast(conn, _forecast("coin_flip", 0.5, "2026-01-09"))

    results = scoring.resolve_due(
        conn, today=date(2026, 1, 9),
        close_fetcher=lambda t, d: 105.0, is_open=lambda d: True,
    )

    by = {r.forecaster: r for r in results}
    assert by["pythia"].status == "resolved"
    assert by["pythia"].outcome == 1.0
    assert by["pythia"].brier == pytest.approx(0.04)   # (0.8 - 1)^2
    assert by["coin_flip"].brier == pytest.approx(0.25)

    persisted = {r["forecaster"]: r for r in storage.fetch_all(conn)}
    assert persisted["pythia"]["status"] == "resolved"
    assert persisted["pythia"]["resolved_close"] == 105.0
    assert persisted["pythia"]["brier"] == pytest.approx(0.04)


def test_resolve_boundary_equal_close_is_true(conn):
    storage.insert_forecast(conn, _forecast("pythia", 0.9, "2026-01-09", anchor_close=100.0))
    results = scoring.resolve_due(
        conn, today=date(2026, 1, 9),
        close_fetcher=lambda t, d: 100.0, is_open=lambda d: True,
    )
    assert results[0].outcome == 1.0
    assert results[0].brier == pytest.approx(0.01)  # (0.9 - 1)^2


def test_resolve_false_outcome(conn):
    storage.insert_forecast(conn, _forecast("pythia", 0.9, "2026-01-09", anchor_close=100.0))
    results = scoring.resolve_due(
        conn, today=date(2026, 1, 9),
        close_fetcher=lambda t, d: 95.0, is_open=lambda d: True,
    )
    assert results[0].outcome == 0.0
    assert results[0].brier == pytest.approx(0.81)  # (0.9 - 0)^2


def test_resolve_ignores_not_yet_due(conn):
    storage.insert_forecast(conn, _forecast("pythia", 0.8, "2026-02-01"))
    results = scoring.resolve_due(
        conn, today=date(2026, 1, 9),
        close_fetcher=lambda t, d: 105.0, is_open=lambda d: True,
    )
    assert results == []
    assert storage.fetch_all(conn)[0]["status"] == "pending"


def test_resolve_skips_when_market_closed(conn):
    storage.insert_forecast(conn, _forecast("pythia", 0.8, "2026-01-09"))
    results = scoring.resolve_due(
        conn, today=date(2026, 1, 9),
        close_fetcher=lambda t, d: 105.0, is_open=lambda d: False,
    )
    assert results[0].status == "skipped"
    assert storage.fetch_all(conn)[0]["status"] == "pending"  # left for retry


def test_resolve_skips_when_price_unavailable(conn):
    def boom(t, d):
        raise RuntimeError("not posted yet")

    storage.insert_forecast(conn, _forecast("pythia", 0.8, "2026-01-09"))
    results = scoring.resolve_due(
        conn, today=date(2026, 1, 9), close_fetcher=boom, is_open=lambda d: True,
    )
    assert results[0].status == "skipped"
    assert "not posted yet" in results[0].detail
    assert storage.fetch_all(conn)[0]["status"] == "pending"


def test_resolve_is_idempotent(conn):
    storage.insert_forecast(conn, _forecast("pythia", 0.8, "2026-01-09"))
    first = scoring.resolve_due(
        conn, today=date(2026, 1, 9),
        close_fetcher=lambda t, d: 105.0, is_open=lambda d: True,
    )
    second = scoring.resolve_due(
        conn, today=date(2026, 1, 9),
        close_fetcher=lambda t, d: 105.0, is_open=lambda d: True,
    )
    assert len(first) == 1
    assert second == []  # nothing pending and due anymore


def test_resolve_fetches_close_once_per_ticker_date(conn):
    calls = []

    def fetch(ticker, day):
        calls.append((ticker, day))
        return 105.0

    for fc in ["pythia", "coin_flip", "drift", "naive_momentum"]:
        storage.insert_forecast(conn, _forecast(fc, 0.5, "2026-01-09"))

    scoring.resolve_due(conn, today=date(2026, 1, 9), close_fetcher=fetch, is_open=lambda d: True)
    assert len(calls) == 1  # one shared close across all four forecasters


# --- summarize ---------------------------------------------------------------

def test_summarize_hit_rate_and_avg_brier(conn):
    pid1 = storage.insert_forecast(conn, _forecast("pythia", 0.8, "2026-01-09"))
    pid2 = storage.insert_forecast(conn, _forecast("pythia", 0.2, "2026-01-12", anchor_date="2026-01-05"))
    cid = storage.insert_forecast(conn, _forecast("coin_flip", 0.5, "2026-01-09"))
    storage.insert_forecast(conn, _forecast("coin_flip", 0.5, "2026-02-01", anchor_date="2026-01-26"))

    # pythia: one hit (0.8 -> true), one miss (0.2 -> true); coin_flip: one resolved + one pending
    storage.mark_resolved(conn, pid1, outcome=1.0, resolved_close=105.0, brier=scoring.brier_score(0.8, 1.0))
    storage.mark_resolved(conn, pid2, outcome=1.0, resolved_close=105.0, brier=scoring.brier_score(0.2, 1.0))
    storage.mark_resolved(conn, cid, outcome=0.0, resolved_close=95.0, brier=scoring.brier_score(0.5, 0.0))

    stats = scoring.summarize(storage.fetch_all(conn))

    p = stats["pythia"]
    assert p.resolved == 2
    assert p.hits == 1
    assert p.hit_rate == pytest.approx(0.5)
    assert p.avg_brier == pytest.approx((0.04 + 0.64) / 2)

    c = stats["coin_flip"]
    assert c.resolved == 1
    assert c.pending == 1
    assert c.avg_brier == pytest.approx(0.25)

    # Untouched baselines still appear, with empty stats.
    assert stats["drift"].resolved == 0
    assert stats["drift"].avg_brier is None


def test_insert_is_idempotent_per_day(conn):
    fc = _forecast("pythia", 0.8, "2026-01-09")
    first = storage.insert_forecast(conn, fc)
    dup = storage.insert_forecast(conn, fc)
    assert first is not None
    assert dup is None  # same (forecaster, ticker, anchor_date, horizon) is ignored
    assert len(storage.fetch_all(conn)) == 1
