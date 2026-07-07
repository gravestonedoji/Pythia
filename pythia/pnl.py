"""Paper-book P&L — the measured sizing ladder (pure, no I/O).

Sizing is deliberately NOT stored on position rows: every book below is a pure
function of the immutable pre-outcome fields (probability, entry quote,
intrinsic settlement), replayed deterministically at read time. That makes the
books reproducible, immune to out-of-order-settlement state corruption, and
lets the same logged record answer new sizing questions later. The flip side
(worth stating plainly): the policy constants in config.py parameterize a
VIEW — changing them re-renders history under the new policy — which is why
`pythia pnl` stamps the params in force on its output.

The ladder (config.PAPER_POLICIES):
- fixed    $OPTIONS_FIXED_STAKE premium per trade, non-compounding. The
           cleanest "was the direction worth anything" read.
- edge     $OPTIONS_FIXED_STAKE * 2|p-0.5| per trade, non-compounding —
           conviction-weighted, still ruin-proof.
- kelly05/ compounding: risk frac * 2|p-0.5| of AVAILABLE cash as premium
  kelly10  (frac 0.5 / 1.0). A *proxy* for Kelly — an ATM option is only
           approximately an even-money directional bet, so cross-policy
           drawdown comparisons are qualitative. kelly10 exists precisely so
           an overconfident arm visibly compounds toward ruin (Delphi:
           $1 -> $0.025 at full Kelly) instead of hiding behind a fixed stake.

Replay conventions (documented because they ARE the definition):
- Events process in date order; on a date that both settles and opens
  positions, settles happen FIRST (an expiring position frees its cash before
  that evening's entries are sized).
- Same-day entries are sized from the same pre-entry cash; if their desired
  stakes sum past available cash they are pro-rata scaled (cash can hit zero —
  ruin — but never goes negative; max loss on a long option is its premium).
- Open (unsettled) positions are carried at COST in equity — no unrealized
  marks, by design (see paper.py).
- Returns use the logged mid by default; `worst_fill=True` re-prices every
  entry at the logged ask — the whole transaction-cost model, recomputable
  for free because both sides of the book were logged.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from . import config


def _edge(p: float) -> float:
    return 2.0 * abs(p - 0.5)


def trade_roi(row, *, worst_fill: bool = False) -> float | None:
    """Return on premium for one SETTLED position row, or None if open."""
    if row["intrinsic"] is None:
        return None
    entry = row["entry_ask"] if worst_fill else row["entry_mid"]
    if entry is None or entry <= 0:
        return None
    return row["intrinsic"] / entry - 1.0


@dataclass
class BookStats:
    """One (forecaster, policy) book, replayed from its position rows."""

    forecaster: str
    policy: str
    trades: int                 # settled positions in the book
    open_trades: int            # entered, not yet expired (carried at cost)
    staked: float               # premium deployed across settled trades ($)
    pnl: float                  # settled P&L ($)
    terminal: float | None = None      # compounding books: final equity ($)
    max_drawdown: float | None = None  # compounding: worst peak-to-trough fraction

    @property
    def roi(self) -> float | None:
        return self.pnl / self.staked if self.staked > 0 else None


def _flat_book(rows, policy: str, forecaster: str, *, worst_fill: bool) -> BookStats:
    """Non-compounding books: a fixed dollar stake (flat or edge-scaled) per trade."""
    trades = open_trades = 0
    staked = pnl = 0.0
    for r in rows:
        stake = config.OPTIONS_FIXED_STAKE
        if policy == "edge":
            stake *= _edge(r["probability"])
        if stake <= 0:
            continue
        roi = trade_roi(r, worst_fill=worst_fill)
        if roi is None:
            open_trades += 1
            continue
        trades += 1
        staked += stake
        pnl += stake * roi
    return BookStats(forecaster, policy, trades, open_trades, staked, pnl)


def _kelly_book(rows, policy: str, forecaster: str, *, worst_fill: bool) -> BookStats:
    """Compounding kelly-proxy replay. Deterministic from the immutable rows."""
    frac = config.KELLY_FRACTIONS[policy]
    entries_by_date: dict[str, list] = defaultdict(list)
    settles_by_date: dict[str, list] = defaultdict(list)
    for r in rows:
        entries_by_date[r["anchor_date"]].append(r)
        if r["intrinsic"] is not None:
            settles_by_date[r["expiry_date"]].append(r)

    cash = config.OPTIONS_BANKROLL_0
    open_cost: dict[int, float] = {}   # position id -> staked premium
    peak = cash
    max_dd = 0.0
    trades = open_trades = 0
    staked = pnl = 0.0

    for day in sorted(set(entries_by_date) | set(settles_by_date)):
        # Settle first: an expiring position frees its cash before that
        # evening's entries are sized.
        for r in settles_by_date.get(day, ()):
            stake = open_cost.pop(r["id"], None)
            if stake is None:
                continue  # entry was skipped (ruin) — nothing to settle
            roi = trade_roi(r, worst_fill=worst_fill)
            cash += stake * (1.0 + roi)
            pnl += stake * roi
        # Enter: same-day stakes sized from the same pre-entry cash, pro-rata
        # scaled if they sum past what is available.
        todays = entries_by_date.get(day, ())
        desired = [(r, cash * frac * _edge(r["probability"])) for r in todays]
        total = sum(d for _, d in desired)
        scale = 1.0 if total <= cash or total == 0 else cash / total
        for r, want in desired:
            stake = want * scale
            if stake <= 0:
                continue
            open_cost[r["id"]] = stake
            cash -= stake
            staked += stake
            if r["intrinsic"] is None:
                open_trades += 1
            else:
                trades += 1
        equity = cash + sum(open_cost.values())
        peak = max(peak, equity)
        if peak > 0:
            max_dd = max(max_dd, 1.0 - equity / peak)

    terminal = cash + sum(open_cost.values())
    return BookStats(forecaster, policy, trades, open_trades, staked, pnl,
                     terminal=terminal, max_drawdown=max_dd)


def replay_book(rows, policy: str, forecaster: str, *,
                worst_fill: bool = False) -> BookStats:
    """One (forecaster, policy) book from that forecaster's position rows.

    `rows` are paper_positions rows (sqlite Rows or dicts) for ONE forecaster;
    order does not matter — compounding books sort events internally.
    """
    if policy in config.KELLY_FRACTIONS:
        return _kelly_book(rows, policy, forecaster, worst_fill=worst_fill)
    if policy in ("fixed", "edge"):
        return _flat_book(rows, policy, forecaster, worst_fill=worst_fill)
    raise ValueError(f"unknown sizing policy {policy!r}")


def ladder(rows, *, policies: tuple[str, ...] | None = None,
           min_edge: float = 0.0, worst_fill: bool = False
           ) -> dict[tuple[str, str], BookStats]:
    """Every (forecaster, policy) book from the full position record.

    `min_edge` keeps only positions with |p-0.5| >= min_edge — the read-time
    high-conviction slice (never a logging gate). The interesting value to try
    is config.ALERT_CONVICTION_MIN once v3 defines it.
    """
    policies = policies or config.PAPER_POLICIES
    by_fc: dict[str, list] = defaultdict(list)
    for r in rows:
        if _edge(r["probability"]) < min_edge:
            continue
        by_fc[r["forecaster"]].append(r)
    out: dict[tuple[str, str], BookStats] = {}
    for fc, fc_rows in by_fc.items():
        for pol in policies:
            out[(fc, pol)] = replay_book(fc_rows, pol, fc, worst_fill=worst_fill)
    return out


def policy_descriptor() -> str:
    """The sizing params in force, stamped on `pythia pnl` output (the books
    are read-time views; the descriptor dates any number quoted elsewhere)."""
    fr = ",".join(f"{k}={v}" for k, v in sorted(config.KELLY_FRACTIONS.items()))
    return (f"stake=${config.OPTIONS_FIXED_STAKE:.0f},"
            f"bankroll0=${config.OPTIONS_BANKROLL_0:.0f},{fr}")
