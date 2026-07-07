"""P&L replay (pnl.py): flat books, kelly-proxy compounding, the ruin regression.

Pure module, pure tests — position rows are plain dicts shaped like
paper_positions rows.
"""

from __future__ import annotations

import pytest

from pythia import config, pnl


_IDS = iter(range(1, 10_000))


def pos(fc="pythia", p=0.75, anchor="2026-07-06", expiry="2026-07-13",
        mid=1.00, ask=None, intrinsic=None, pid=None):
    """One paper_positions row. intrinsic=None means still open."""
    return {
        "id": pid if pid is not None else next(_IDS),
        "forecaster": fc, "probability": p,
        "anchor_date": anchor, "expiry_date": expiry,
        "entry_mid": mid, "entry_ask": ask if ask is not None else mid + 0.05,
        "intrinsic": intrinsic,
        "status": "open" if intrinsic is None else "settled",
    }


# --- per-trade return ---------------------------------------------------------------

def test_trade_roi_mid_and_worst_fill():
    r = pos(mid=1.00, ask=1.10, intrinsic=2.20)
    assert pnl.trade_roi(r) == pytest.approx(1.20)
    assert pnl.trade_roi(r, worst_fill=True) == pytest.approx(1.00)


def test_trade_roi_open_position_is_none():
    assert pnl.trade_roi(pos(intrinsic=None)) is None


# --- flat books ----------------------------------------------------------------------

def test_fixed_book_sums_stake_times_roi():
    rows = [pos(intrinsic=2.0),   # roi +1.0
            pos(intrinsic=0.0)]   # roi -1.0
    b = pnl.replay_book(rows, "fixed", "pythia")
    assert b.trades == 2
    assert b.staked == pytest.approx(2 * config.OPTIONS_FIXED_STAKE)
    assert b.pnl == pytest.approx(0.0)
    assert b.terminal is None  # non-compounding books have no bankroll


def test_edge_book_scales_stake_by_conviction():
    rows = [pos(p=0.75, intrinsic=0.0)]  # edge 0.5 -> stake $50, all lost
    b = pnl.replay_book(rows, "edge", "pythia")
    assert b.staked == pytest.approx(0.5 * config.OPTIONS_FIXED_STAKE)
    assert b.pnl == pytest.approx(-0.5 * config.OPTIONS_FIXED_STAKE)


def test_flat_books_carry_open_positions_separately():
    rows = [pos(intrinsic=2.0), pos(intrinsic=None)]
    b = pnl.replay_book(rows, "fixed", "pythia")
    assert b.trades == 1 and b.open_trades == 1
    assert b.pnl == pytest.approx(config.OPTIONS_FIXED_STAKE)


# --- kelly-proxy replay -----------------------------------------------------------------

def test_kelly_single_trade_arithmetic():
    # p 0.75 -> edge 0.5; kelly10 stakes 1.0 * 0.5 * $1000 = $500; roi +1
    rows = [pos(p=0.75, anchor="2026-07-06", expiry="2026-07-13", intrinsic=2.0)]
    b = pnl.replay_book(rows, "kelly10", "pythia")
    assert b.staked == pytest.approx(500.0)
    assert b.pnl == pytest.approx(500.0)
    assert b.terminal == pytest.approx(1500.0)


def test_kelly_replay_is_input_order_independent():
    rows = [
        pos(p=0.75, anchor="2026-07-08", expiry="2026-07-15", intrinsic=0.0),
        pos(p=0.75, anchor="2026-07-06", expiry="2026-07-13", intrinsic=2.0),
    ]
    a = pnl.replay_book(rows, "kelly10", "pythia")
    b = pnl.replay_book(list(reversed(rows)), "kelly10", "pythia")
    assert a.terminal == pytest.approx(b.terminal)
    assert a.max_drawdown == pytest.approx(b.max_drawdown)


def test_kelly_open_premium_reduces_available_cash():
    # Day 1: stake 500 (cash 500 left). Day 2 entry sizes off the REMAINING
    # cash: 1.0 * 0.5 * 500 = 250, not 500.
    rows = [
        pos(p=0.75, anchor="2026-07-06", expiry="2026-07-20", intrinsic=None),
        pos(p=0.75, anchor="2026-07-07", expiry="2026-07-21", intrinsic=None),
    ]
    b = pnl.replay_book(rows, "kelly10", "pythia")
    assert b.staked == pytest.approx(500.0 + 250.0)
    assert b.terminal == pytest.approx(1000.0)  # all still carried at cost


def test_kelly_same_day_entries_pro_rata_when_oversubscribed():
    # Two p=0.9 entries same day: each wants 0.8 * $1000 = $800, sum $1600 >
    # $1000 cash -> scaled to $500 each; cash exactly zero, equity unchanged.
    rows = [
        pos(p=0.90, anchor="2026-07-06", expiry="2026-07-13", intrinsic=None),
        pos(p=0.90, anchor="2026-07-06", expiry="2026-07-13", intrinsic=None),
    ]
    b = pnl.replay_book(rows, "kelly10", "pythia")
    assert b.staked == pytest.approx(1000.0)
    assert b.terminal == pytest.approx(1000.0)


def test_kelly_settles_before_entering_on_the_same_day():
    # A settles on 07-13 (roi +1, freeing 500 -> cash 1500) BEFORE B enters
    # that evening: B stakes 0.5 * 1500 = 750, then doubles -> terminal 2250.
    # (Enter-first ordering would give 1750 — this pins the convention.)
    rows = [
        pos(p=0.75, anchor="2026-07-06", expiry="2026-07-13", intrinsic=2.0),
        pos(p=0.75, anchor="2026-07-13", expiry="2026-07-20", intrinsic=2.0),
    ]
    b = pnl.replay_book(rows, "kelly10", "pythia")
    assert b.terminal == pytest.approx(2250.0)


def test_kelly_ruin_regression_overconfidence_is_visible():
    """The Delphi-visibility test: an overconfident arm that keeps losing must
    visibly compound toward ruin under kelly10 while the fixed book bleeds
    linearly and kelly05 survives longer."""
    rows = [
        pos(p=0.95, anchor="2026-07-06", expiry="2026-07-07", intrinsic=0.0),
        pos(p=0.95, anchor="2026-07-08", expiry="2026-07-09", intrinsic=0.0),
        pos(p=0.95, anchor="2026-07-10", expiry="2026-07-11", intrinsic=0.0),
    ]
    k10 = pnl.replay_book(rows, "kelly10", "pythia")
    k05 = pnl.replay_book(rows, "kelly05", "pythia")
    fixed = pnl.replay_book(rows, "fixed", "pythia")
    # cash *= (1 - 0.9) per loss under kelly10 -> $1000 * 0.1^3 = $1
    assert k10.terminal == pytest.approx(1.0)
    assert k10.max_drawdown == pytest.approx(0.999)
    # kelly05 risks 0.45 per loss -> $1000 * 0.55^3 ~ $166
    assert k05.terminal == pytest.approx(1000 * 0.55**3)
    assert k05.terminal > k10.terminal
    # the fixed book loses the same three stakes linearly
    assert fixed.pnl == pytest.approx(-3 * config.OPTIONS_FIXED_STAKE)


def test_kelly_worst_fill_prices_entries_at_the_ask():
    rows = [pos(p=0.75, mid=1.00, ask=1.25, intrinsic=2.0,
                anchor="2026-07-06", expiry="2026-07-13")]
    mid_book = pnl.replay_book(rows, "kelly10", "pythia")
    wf_book = pnl.replay_book(rows, "kelly10", "pythia", worst_fill=True)
    assert mid_book.pnl == pytest.approx(500.0)          # roi +1.0
    assert wf_book.pnl == pytest.approx(500.0 * 0.6)     # roi 2.0/1.25 - 1 = 0.6


# --- the ladder ---------------------------------------------------------------------------

def test_ladder_groups_by_forecaster_and_policy():
    rows = [pos(fc="pythia", intrinsic=2.0), pos(fc="drift", intrinsic=0.0)]
    books = pnl.ladder(rows)
    assert set(books) == {(fc, pol) for fc in ("pythia", "drift")
                          for pol in config.PAPER_POLICIES}
    assert books[("pythia", "fixed")].pnl > 0 > books[("drift", "fixed")].pnl


def test_ladder_min_edge_slices_at_read_time():
    # min_edge is |p - 0.5| (as documented), NOT the 2|p-0.5| stake
    # multiplier: p=0.60 has |p-0.5| = 0.10 and must be excluded at 0.15
    # (under the doubled scale it would sneak in at 0.20).
    rows = [pos(fc="pythia", p=0.55, intrinsic=2.0),
            pos(fc="pythia", p=0.60, intrinsic=2.0),
            pos(fc="pythia", p=0.70, intrinsic=0.0)]
    books = pnl.ladder(rows, min_edge=0.15)
    b = books[("pythia", "fixed")]
    assert b.trades == 1
    assert b.pnl == pytest.approx(-config.OPTIONS_FIXED_STAKE)  # only the 0.70 row


def test_unknown_policy_raises():
    with pytest.raises(ValueError):
        pnl.replay_book([], "martingale", "pythia")


def test_policy_descriptor_stamps_the_params_in_force():
    d = pnl.policy_descriptor()
    assert f"stake=${config.OPTIONS_FIXED_STAKE:.0f}" in d
    assert "kelly10=1.0" in d
