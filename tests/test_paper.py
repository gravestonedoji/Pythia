"""Paper book (paper.py + the cli pass): gates, selection, symmetry, settlement.

Offline like the rest of the suite: chain fetches are faked, settlement closes
are injected (the scoring.resolve_due precedent), the DB is a temp file.
"""

from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from pythia import cli, config, paper, storage
from pythia.paper import PaperPosition

# Inside the entry window for the 2026-07-10 (Friday) anchor used below.
_NOW = datetime(2026, 7, 10, 21, 0, tzinfo=timezone.utc)


def quote(side="C", bid=1.00, ask=1.10, oi=5000, strike=620.0,
          expiry="2026-07-17", last=1.05, volume=250,
          underlying_last=620.0, quoted_at="2026-07-10T21:00:00+00:00"):
    """A plain quote dict shaped like paper._row_to_quote's output."""
    return {
        "side": side, "expiry_date": expiry, "strike": strike,
        "bid": bid, "ask": ask, "last": last, "volume": volume,
        "open_interest": oi, "underlying_last": underlying_last,
        "quoted_at": quoted_at, "last_trade_at": None,
    }


# --- quote gates -----------------------------------------------------------------

def test_usable_quote_passes_a_tight_two_sided_book():
    ok, why = paper.usable_quote(quote())
    assert ok and why == ""


def test_zero_bid_rejected():
    ok, why = paper.usable_quote(quote(bid=0.0))
    assert not ok and "bid" in why


def test_one_sided_book_rejected():
    ok, why = paper.usable_quote(quote(bid=None))
    assert not ok and "one-sided" in why


def test_crossed_book_rejected():
    ok, why = paper.usable_quote(quote(bid=1.20, ask=1.10))
    assert not ok and "crossed" in why


def test_premium_floor_rejects_cheap_mid():
    ok, why = paper.usable_quote(quote(bid=0.14, ask=0.18))  # mid 0.16 < 0.20
    assert not ok and "floor" in why


def test_spread_gate_is_relative_dominant():
    # mid 1.00: allowed spread = max(0.05, 0.15) = 0.15 — 0.16 fails, 0.14 passes
    ok, _ = paper.usable_quote(quote(bid=0.92, ask=1.08))
    assert not ok
    ok, _ = paper.usable_quote(quote(bid=0.93, ask=1.07))
    assert ok


def test_spread_gate_absolute_floor_covers_small_mids():
    # mid 0.25: allowed = max(0.05, 0.0375) = 0.05 — the absolute leg governs
    ok, _ = paper.usable_quote(quote(bid=0.23, ask=0.27))  # spread 0.04
    assert ok
    ok, why = paper.usable_quote(quote(bid=0.22, ask=0.28))  # spread 0.06
    assert not ok and "spread" in why


def test_open_interest_floor():
    ok, why = paper.usable_quote(quote(oi=99))
    assert not ok and "interest" in why
    ok, _ = paper.usable_quote(quote(oi=100))
    assert ok
    ok, why = paper.usable_quote(quote(oi=None))
    assert not ok


# --- direction / intrinsic ---------------------------------------------------------

def test_direction_sides_and_exact_half_sits_out():
    assert paper.direction(0.51) == "C"
    assert paper.direction(0.49) == "P"
    assert paper.direction(0.5) is None


def test_intrinsic_call_and_put():
    assert paper.intrinsic("C", 620.0, 625.0) == pytest.approx(5.0)
    assert paper.intrinsic("C", 620.0, 615.0) == 0.0
    assert paper.intrinsic("C", 620.0, 620.0) == 0.0
    assert paper.intrinsic("P", 620.0, 615.0) == pytest.approx(5.0)
    assert paper.intrinsic("P", 620.0, 625.0) == 0.0
    with pytest.raises(ValueError):
        paper.intrinsic("X", 620.0, 625.0)


# --- contract selection -------------------------------------------------------------

def test_pick_expiry_exact_match_wins():
    resolves = date(2026, 7, 17)  # a Friday session
    exp, gap = paper.pick_expiry(
        [date(2026, 7, 15), date(2026, 7, 17), date(2026, 7, 20)], resolves)
    assert exp == date(2026, 7, 17) and gap == 0


def test_pick_expiry_signed_gaps_within_tolerance():
    resolves = date(2026, 7, 17)
    exp, gap = paper.pick_expiry([date(2026, 7, 15)], resolves)
    assert exp == date(2026, 7, 15) and gap == -2
    exp, gap = paper.pick_expiry([date(2026, 7, 20)], resolves)
    assert exp == date(2026, 7, 20) and gap == 1


def test_pick_expiry_ties_break_earlier():
    resolves = date(2026, 7, 17)
    exp, gap = paper.pick_expiry([date(2026, 7, 20), date(2026, 7, 16)], resolves)
    assert exp == date(2026, 7, 16) and gap == -1


def test_pick_expiry_none_within_tolerance():
    resolves = date(2026, 7, 17)
    assert paper.pick_expiry([date(2026, 7, 10), date(2026, 7, 31)], resolves) is None
    assert paper.pick_expiry([], resolves) is None


def test_pick_strike_nearest_with_lower_tie():
    assert paper.pick_strike([618.0, 619.0, 620.0, 621.0], 620.3) == 620.0
    assert paper.pick_strike([620.0, 621.0], 620.5) == 620.0  # tie -> lower


def test_pick_strike_rejects_far_from_atm():
    # 2% gate: nearest strike 13% away is not the logged claim's bet
    assert paper.pick_strike([90.0, 113.0], 100.0) is None
    assert paper.pick_strike([], 100.0) is None


# --- entry-window gate ---------------------------------------------------------------

def test_entry_window_same_evening_ok():
    ok, _ = paper.entry_window_ok(
        date(2026, 7, 10), datetime(2026, 7, 10, 21, 0, tzinfo=timezone.utc))
    assert ok


def test_entry_window_next_preopen_ok_but_next_session_refused():
    # Anchor Friday 2026-07-10; next session Monday 07-13 opens 13:30 UTC (EDT).
    ok, _ = paper.entry_window_ok(
        date(2026, 7, 10), datetime(2026, 7, 13, 13, 0, tzinfo=timezone.utc))
    assert ok
    ok, why = paper.entry_window_ok(
        date(2026, 7, 10), datetime(2026, 7, 13, 14, 0, tzinfo=timezone.utc))
    assert not ok and "post-anchor" in why


# --- positions from logged rows -------------------------------------------------------

def _forecast_row(fc, p, ticker="SPY", anchor="2026-07-10", horizon=5,
                  resolves="2026-07-17", anchor_close=620.0):
    return {
        "forecaster": fc, "ticker": ticker, "anchor_date": anchor,
        "horizon_days": horizon, "resolves_on": resolves,
        "anchor_close": anchor_close, "probability": p,
    }


def test_positions_for_claim_directions_and_audit_copy():
    rows = [_forecast_row("pythia", 0.62), _forecast_row("hmm_filter", 0.41),
            _forecast_row("coin_flip", 0.5)]
    call_q, put_q = quote("C"), quote("P", bid=0.90, ask=1.00)
    positions = paper.positions_for_claim(rows, call_q, put_q, expiry_gap=0)
    assert len(positions) == 2  # coin_flip has no direction
    by_fc = {p.forecaster: p for p in positions}
    assert by_fc["pythia"].side == "C"
    assert by_fc["pythia"].entry_mid == pytest.approx(1.05)
    assert by_fc["pythia"].probability == 0.62  # audit copy of the LOGGED p
    assert by_fc["hmm_filter"].side == "P"
    assert by_fc["hmm_filter"].entry_mid == pytest.approx(0.95)
    assert all(p.status == "open" for p in positions)


# --- storage round-trip / migration ---------------------------------------------------

@pytest.fixture()
def conn(tmp_path):
    c = storage.get_connection(tmp_path / "test.db")
    yield c
    c.close()


def _position(fc="pythia", ticker="SPY", anchor="2026-07-10", horizon=5,
              side="C", strike=620.0, mid=1.05, expiry="2026-07-17", p=0.62):
    return PaperPosition(
        forecaster=fc, ticker=ticker, anchor_date=anchor, horizon_days=horizon,
        resolves_on="2026-07-17", probability=p, side=side,
        expiry_date=expiry, expiry_gap_sessions=0, strike=strike,
        entry_bid=mid - 0.05, entry_ask=mid + 0.05, entry_mid=mid,
        entry_quoted_at="2026-07-10T21:00:00+00:00",
    )


def test_fresh_db_has_paper_tables(conn):
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"option_quotes", "paper_positions"} <= tables


def test_pre_paper_db_gains_tables(tmp_path):
    import sqlite3
    db = tmp_path / "old.db"
    raw = sqlite3.connect(db)
    # A pre-paper DB has the full forecasts table (the schema's indexes need
    # status/resolves_on), just none of the paper tables.
    raw.execute(
        "CREATE TABLE forecasts (id INTEGER PRIMARY KEY, forecaster TEXT, "
        "status TEXT, resolves_on TEXT, fit_flags TEXT)")
    raw.commit()
    raw.close()
    c = storage.get_connection(db)
    tables = {r[0] for r in c.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"option_quotes", "paper_positions"} <= tables
    c.close()


def test_quote_and_position_inserts_are_first_write_wins(conn):
    rid = storage.insert_option_quote(
        conn, "SPY", "2026-07-10", 5, quote(), usable=True,
        reject_reason=None, gates=config.option_gates_descriptor())
    assert rid is not None
    dup = storage.insert_option_quote(
        conn, "SPY", "2026-07-10", 5, quote(bid=9.99, ask=10.0), usable=False,
        reject_reason="different book", gates="")
    assert dup is None
    row = storage.fetch_quote_pair(conn, "SPY", "2026-07-10", 5)["C"]
    assert row["bid"] == 1.00 and row["usable"] == 1  # the first book stayed

    assert storage.insert_paper_position(conn, _position()) is not None
    assert storage.insert_paper_position(conn, _position(mid=9.0)) is None
    rows = storage.fetch_paper_positions(conn)
    assert len(rows) == 1 and rows[0]["entry_mid"] == pytest.approx(1.05)


def test_position_row_joins_back_to_its_forecast_row(conn):
    from pythia.storage import Forecast
    storage.insert_forecast(conn, Forecast(
        forecaster="pythia", ticker="SPY", claim="c", horizon_days=5,
        anchor_date="2026-07-10", anchor_close=620.0, resolves_on="2026-07-17",
        probability=0.62, reasoning=None, model=None))
    storage.insert_paper_position(conn, _position())
    joined = conn.execute(
        """
        SELECT f.probability AS fp, p.probability AS pp FROM paper_positions p
        JOIN forecasts f ON f.forecaster = p.forecaster AND f.ticker = p.ticker
          AND f.anchor_date = p.anchor_date AND f.horizon_days = p.horizon_days
        """).fetchall()
    assert len(joined) == 1 and joined[0]["fp"] == joined[0]["pp"]


# --- settlement ------------------------------------------------------------------------

def test_settle_call_and_put_at_intrinsic(conn):
    storage.insert_paper_position(conn, _position(fc="pythia", side="C"))
    storage.insert_paper_position(conn, _position(fc="drift", side="P", mid=0.95))
    results = paper.settle_due(
        conn, today=date(2026, 7, 17), last_completed=date(2026, 7, 17),
        close_fetcher=lambda t, d: 625.0, is_open=lambda d: True)
    assert {r.status for r in results} == {"settled"}
    rows = {r["forecaster"]: r for r in storage.fetch_paper_positions(conn)}
    call = rows["pythia"]
    assert call["status"] == "settled"
    assert call["settle_close"] == 625.0
    assert call["intrinsic"] == pytest.approx(5.0)
    assert call["pnl_per_unit"] == pytest.approx(5.0 / 1.05 - 1.0)
    put = rows["drift"]
    assert put["intrinsic"] == 0.0
    assert put["pnl_per_unit"] == pytest.approx(-1.0)  # premium fully lost


def test_settle_waits_for_unavailable_close(conn):
    storage.insert_paper_position(conn, _position())

    def no_data(t, d):
        raise RuntimeError("no close yet")

    results = paper.settle_due(conn, today=date(2026, 7, 17),
                               last_completed=date(2026, 7, 17),
                               close_fetcher=no_data, is_open=lambda d: True)
    assert results[0].status == "skipped"
    assert storage.fetch_paper_positions(conn)[0]["status"] == "open"  # retries


def test_settle_skips_non_session_expiry_without_guessing(conn):
    storage.insert_paper_position(conn, _position())
    results = paper.settle_due(conn, today=date(2026, 7, 17),
                               last_completed=date(2026, 7, 17),
                               close_fetcher=lambda t, d: 625.0,
                               is_open=lambda d: False)
    assert results[0].status == "skipped"
    assert "not a trading session" in results[0].detail
    assert storage.fetch_paper_positions(conn)[0]["status"] == "open"


def test_settle_ignores_unexpired_positions(conn):
    storage.insert_paper_position(conn, _position())
    results = paper.settle_due(conn, today=date(2026, 7, 16),
                               last_completed=date(2026, 7, 16),
                               close_fetcher=lambda t, d: 625.0,
                               is_open=lambda d: True)
    assert results == []


def test_settle_never_uses_an_uncompleted_session(conn):
    # Expiry day has arrived but the session hasn't CLOSED: an intraday run
    # must not settle against the in-progress bar (Yahoo serves the live
    # price as today's "close"; a settled row is permanent).
    storage.insert_paper_position(conn, _position())
    results = paper.settle_due(conn, today=date(2026, 7, 17),
                               last_completed=date(2026, 7, 16),
                               close_fetcher=lambda t, d: 625.0,
                               is_open=lambda d: True)
    assert results == []
    assert storage.fetch_paper_positions(conn)[0]["status"] == "open"


# --- the cli pass: symmetry + logged-rows-only + idempotency -----------------------------

def _log_claim(conn, probs: dict[str, float]):
    from pythia.storage import Forecast
    for fc, p in probs.items():
        storage.insert_forecast(conn, Forecast(
            forecaster=fc, ticker="SPY", claim="c", horizon_days=5,
            anchor_date="2026-07-10", anchor_close=620.0,
            resolves_on="2026-07-17", probability=p, reasoning=None, model=None))


def test_paper_pass_symmetric_refusal_logs_quotes_but_no_positions(conn, monkeypatch):
    _log_claim(conn, {"pythia": 0.62, "drift": 0.45})
    # Call book fine, put book zero-bid: NOBODY trades (a one-sided claim
    # would let bullish arms trade while bearish arms skip).
    monkeypatch.setattr(paper, "fetch_chain_pair",
                        lambda *a, **k: (quote("C"), quote("P", bid=0.0), 0))
    cli._paper_pass(conn, "SPY", "2026-07-10", 5, now=_NOW)
    pair = storage.fetch_quote_pair(conn, "SPY", "2026-07-10", 5)
    assert pair["C"]["usable"] == 1 and pair["P"]["usable"] == 0
    assert "bid" in pair["P"]["reject_reason"]
    assert pair["C"]["gates"] == config.option_gates_descriptor()
    assert storage.fetch_paper_positions(conn) == []


def test_paper_pass_opens_positions_from_logged_rows_and_is_idempotent(conn, monkeypatch):
    _log_claim(conn, {"pythia": 0.62, "drift": 0.45, "coin_flip": 0.5})
    calls = [0]

    def fake_fetch(*a, **k):
        calls[0] += 1
        return quote("C"), quote("P", bid=0.90, ask=1.00), 0

    monkeypatch.setattr(paper, "fetch_chain_pair", fake_fetch)
    cli._paper_pass(conn, "SPY", "2026-07-10", 5, now=_NOW)
    rows = storage.fetch_paper_positions(conn)
    assert {r["forecaster"] for r in rows} == {"pythia", "drift"}  # 0.5 sits out
    by_fc = {r["forecaster"]: r for r in rows}
    assert by_fc["pythia"]["side"] == "C" and by_fc["drift"]["side"] == "P"
    assert by_fc["pythia"]["probability"] == 0.62  # the LOGGED p, audit-copied

    # Second run: the logged book is reused (no refetch), positions unchanged.
    cli._paper_pass(conn, "SPY", "2026-07-10", 5, now=_NOW)
    assert calls[0] == 1
    assert len(storage.fetch_paper_positions(conn)) == 2


def test_paper_pass_unusable_logged_book_never_refetches(conn, monkeypatch):
    _log_claim(conn, {"pythia": 0.62})
    storage.insert_option_quote(conn, "SPY", "2026-07-10", 5, quote("C", bid=0.0),
                                usable=False, reject_reason="zero bid", gates="")
    storage.insert_option_quote(conn, "SPY", "2026-07-10", 5, quote("P"),
                                usable=True, reject_reason=None, gates="")

    def boom(*a, **k):
        raise AssertionError("must not refetch a logged book (first write wins)")

    monkeypatch.setattr(paper, "fetch_chain_pair", boom)
    cli._paper_pass(conn, "SPY", "2026-07-10", 5, now=_NOW)
    assert storage.fetch_paper_positions(conn) == []


def test_paper_pass_requires_logged_claim_rows(conn):
    with pytest.raises(RuntimeError, match="no logged forecast rows"):
        cli._paper_pass(conn, "SPY", "2026-07-10", 5, now=_NOW)


def test_paper_pass_refuses_positions_after_the_entry_window(conn, monkeypatch):
    # A usable pair is already LOGGED, but the next session has opened: the
    # position inserts must refuse too — a late run that has glimpsed (or
    # knows) the outcome could otherwise choose which claims to "enter".
    _log_claim(conn, {"pythia": 0.62})
    storage.insert_option_quote(conn, "SPY", "2026-07-10", 5, quote("C"),
                                usable=True, reject_reason=None, gates="")
    storage.insert_option_quote(conn, "SPY", "2026-07-10", 5,
                                quote("P", bid=0.90, ask=1.00),
                                usable=True, reject_reason=None, gates="")
    late = datetime(2026, 7, 13, 15, 0, tzinfo=timezone.utc)  # Mon, post-open
    with pytest.raises(RuntimeError, match="post-anchor"):
        cli._paper_pass(conn, "SPY", "2026-07-10", 5, now=late)
    assert storage.fetch_paper_positions(conn) == []


def test_paper_pass_recovers_a_half_logged_pair(conn, monkeypatch):
    # Legacy hazard: a crash between the two quote inserts (they are atomic
    # now) left one side logged. While the window is open, the pass refetches
    # and fills ONLY the missing side (first write wins on the logged one).
    _log_claim(conn, {"pythia": 0.62})
    storage.insert_option_quote(conn, "SPY", "2026-07-10", 5, quote("C"),
                                usable=True, reject_reason=None, gates="")
    monkeypatch.setattr(paper, "fetch_chain_pair",
                        lambda *a, **k: (quote("C", bid=9.0, ask=9.1),
                                         quote("P", bid=0.90, ask=1.00), 0))
    cli._paper_pass(conn, "SPY", "2026-07-10", 5, now=_NOW)
    pair = storage.fetch_quote_pair(conn, "SPY", "2026-07-10", 5)
    assert pair["C"]["bid"] == 1.00  # the originally logged side stayed
    assert pair["P"]["bid"] == 0.90  # the missing side got filled
    assert len(storage.fetch_paper_positions(conn)) == 1
