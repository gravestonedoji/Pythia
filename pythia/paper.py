"""Simulated option positions — the paper book (roadmap v2). SIMULATED ONLY.

WHY: the Brier ladder says who is *calibrated*; the paper book says what that
calibration would have been *worth* — and P&L is exactly where overconfidence
becomes visible (Delphi measured a raw LLM compounding $1 -> $0.025 at full
Kelly while its Brier looked respectable). Every forecaster arm and bar trades
the SAME contract pair at the SAME logged quote — only direction (its own p)
and read-time sizing differ — so the P&L ladder is a true mirror of the Brier
ladder: differences between arms are forecaster differences, never execution
differences.

THE HARD WALL: everything here is simulation against real quotes. No broker,
no orders, no order preparation, no live execution — and none may ever be
added. Positions are opened from probabilities ALREADY LOGGED in `forecasts`
(read back from the DB, never from in-memory results, so a re-run can never
open a position whose probability matches no logged row).

HONESTY RULES (kalshi.py's, applied to options):
- Live-only, no backfill. yfinance has no historical chains; a claim either
  got its quotes logged at entry time or it never trades. Sparse coverage is
  expected and fine — a missing row beats a fudged one.
- P&L of record = enter at the logged mid -> settle at INTRINSIC value off the
  official raw close on expiry (max(0, S-K) / max(0, K-S), the same raw close
  scoring.py grades with). Deliberately NO daily mark-to-market: a mark from a
  delayed, often stale/zero-bid after-hours book would inject unauditable
  noise. Open positions are carried at cost until expiry.
- Refuse, don't fudge. A selected contract whose book fails a gate logs its
  quote row with usable=0 and the reason (the audit trail behind every refusal)
  and NO positions. Claim-level refusals (no expiry in tolerance, no ATM
  strike, drift/time gate) log nothing, like a kalshi claim with no matching
  contract — the missing row is the record.
- BOTH sides (call and put) must pass the gates or NO arm trades the claim.
  If only the call book were usable, bullish arms would trade while bearish
  arms skipped — an asymmetry that biases the ladder.
- Entry quotes must not embed post-anchor information: the capture must happen
  before the next session's open (entry_window_ok) and the underlying must sit
  within OPTIONS_MAX_UNDERLYING_DRIFT of the anchor close. The probabilities
  were computed as of the anchor close; a quote from the middle of the next
  session is a different experiment.
- Expiry alignment mirrors the kalshi settle-gap: the listed expiry closest to
  `resolves_on` within OPTIONS_MAX_EXPIRY_GAP_SESSIONS trading sessions, the
  signed gap recorded on every position row. Recorded, never hidden.
- First write wins everywhere (same natural keys as forecasts): re-running the
  capture later the same evening sees a different book, and the book the
  positions were opened on is the one that stays.

TIMING REALITY (measured expectation, not a bug): ETF options stop trading at
4:15 ET and Yahoo's after-hours books are frequently one-sided or zero-bid.
A capture run long after the close will mostly log usable=0 rows — honestly.
A usable paper book wants the capture in the ~4:00-4:15 ET window. Watch the
usable-rate the first week before drawing any conclusion from the book's size.

Simulation conventions: positions are sized in fractional premium dollars (a
measurement instrument, not an execution rehearsal — whole-contract rounding
would make small books lumpy and arm-dependent). Early exercise, dividends
into expiry, and greeks are ignored: long-only ~5-session ATM positions
settled at intrinsic make these second-order, the same documented-slop class
as the kalshi SPY/^GSPC dividend drift.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Callable

from . import config, data
from .storage import now_iso

_LAST_CALL = [0.0]  # throttle clock, shared across chain calls in this process


# --- pure core (offline, unit-tested) ------------------------------------------

def direction(probability: float) -> str | None:
    """The side an arm trades: call above 0.5, put below, None at exactly 0.5.

    p == 0.5 has no direction, so the arm deterministically sits out — which
    cleanly removes coin_flip from the book.
    """
    if probability > 0.5:
        return "C"
    if probability < 0.5:
        return "P"
    return None


def intrinsic(side: str, strike: float, close: float) -> float:
    """Settlement value of one long unit at expiry, off the official raw close."""
    if side == "C":
        return max(0.0, close - strike)
    if side == "P":
        return max(0.0, strike - close)
    raise ValueError(f"side must be 'C' or 'P', got {side!r}")


def usable_quote(q: dict) -> tuple[bool, str]:
    """Judge one side's book against the quote gates. Returns (ok, reason).

    Gates (config.py): two-sided uncrossed book, premium floor, RELATIVE-
    dominant spread cap, open-interest floor. The reason string is stored on
    the quote row so every refusal is auditable.
    """
    bid, ask = q.get("bid"), q.get("ask")
    if bid is None or ask is None:
        return False, "one-sided book (missing bid or ask)"
    if bid <= 0:
        return False, f"zero/negative bid ({bid})"
    if bid > ask:
        return False, f"crossed book (bid {bid} > ask {ask})"
    mid = (bid + ask) / 2.0
    if mid < config.OPTIONS_MIN_PREMIUM:
        return False, (f"premium {mid:.2f} below floor "
                       f"{config.OPTIONS_MIN_PREMIUM:.2f} (noise would dominate ROI)")
    max_spread = max(config.OPTIONS_MAX_SPREAD_ABS,
                     config.OPTIONS_MAX_SPREAD_REL * mid)
    if ask - bid > max_spread:
        return False, f"spread {ask - bid:.2f} > {max_spread:.2f} allowed at mid {mid:.2f}"
    oi = q.get("open_interest")
    if oi is None or oi < config.OPTIONS_MIN_OPEN_INTEREST:
        return False, f"open interest {oi} < {config.OPTIONS_MIN_OPEN_INTEREST}"
    return True, ""


def pick_expiry(expiries: list[date], resolves_on: date) -> tuple[date, int] | None:
    """The listed expiry best aligned with the claim, or None.

    Minimizes |gap| in trading sessions from `resolves_on` (signed gap
    returned; positive = expiry after resolution, kalshi convention), rejects
    beyond OPTIONS_MAX_EXPIRY_GAP_SESSIONS, breaks ties toward the EARLIER
    expiry (pinned before the first row was logged, per the design review).
    """
    best: tuple[int, date, int] | None = None
    for exp in expiries:
        gap = data.sessions_between(resolves_on, exp)
        if abs(gap) > config.OPTIONS_MAX_EXPIRY_GAP_SESSIONS:
            continue
        key = (abs(gap), exp)
        if best is None or key < (best[0], best[1]):
            best = (abs(gap), exp, gap)
    if best is None:
        return None
    return best[1], best[2]


def pick_strike(strikes: list[float], anchor_close: float) -> float | None:
    """The nearest listed strike to the anchor close, or None if none is ATM.

    Ties break toward the LOWER strike (deterministic). Rejects strikes
    further than OPTIONS_MAX_ATM_DISTANCE from the anchor — trading a far
    strike is a different bet than the logged claim.
    """
    best: float | None = None
    for s in strikes:
        if best is None:
            best = s
            continue
        d, bd = abs(s - anchor_close), abs(best - anchor_close)
        if d < bd or (d == bd and s < best):
            best = s
    if best is None:
        return None
    if abs(best / anchor_close - 1.0) > config.OPTIONS_MAX_ATM_DISTANCE:
        return None
    return best


def entry_window_ok(anchor_date: date, quoted_at: datetime) -> tuple[bool, str]:
    """The entry-honesty time gate.

    The arms' probabilities are as-of the anchor close; a quote captured at or
    after the NEXT session's open embeds a new session's information under an
    anchor-dated p. Before that open (evening of the anchor day, or pre-open
    the next morning) the regular-session tape the forecasts saw is still the
    latest one.
    """
    next_open = data.session_open_utc(data.trading_days_forward(anchor_date, 1))
    if quoted_at.tzinfo is None:
        quoted_at = quoted_at.replace(tzinfo=timezone.utc)
    if quoted_at >= next_open.to_pydatetime():
        return False, (
            f"quote time {quoted_at.isoformat()} is past the next session's open "
            f"({next_open.isoformat()}) — entry would embed post-anchor information"
        )
    return True, ""


@dataclass
class PaperPosition:
    """One simulated long-option position (one arm, one claim)."""

    forecaster: str
    ticker: str
    anchor_date: str
    horizon_days: int
    resolves_on: str
    probability: float          # the arm's p at entry (audit copy of the forecast row)
    side: str                   # 'C' | 'P'
    expiry_date: str
    expiry_gap_sessions: int    # signed, kalshi convention; recorded, never hidden
    strike: float
    entry_bid: float
    entry_ask: float
    entry_mid: float
    entry_quoted_at: str
    opened_at: str = field(default_factory=now_iso)
    status: str = "open"
    id: int | None = None
    settled_at: str | None = None
    settle_close: float | None = None
    intrinsic: float | None = None
    pnl_per_unit: float | None = None   # return on premium: intrinsic/entry_mid - 1


def positions_for_claim(
    forecast_rows, call_q: dict, put_q: dict, expiry_gap: int
) -> list[PaperPosition]:
    """Build one position per LOGGED forecast row with a direction.

    `forecast_rows` must be rows read back from the forecasts table (never
    in-memory results): the position's probability is an audit copy of the
    logged one, and the P&L ladder must mirror the Brier ladder row for row.
    """
    out: list[PaperPosition] = []
    for row in forecast_rows:
        side = direction(row["probability"])
        if side is None:
            continue
        q = call_q if side == "C" else put_q
        out.append(PaperPosition(
            forecaster=row["forecaster"], ticker=row["ticker"],
            anchor_date=row["anchor_date"], horizon_days=row["horizon_days"],
            resolves_on=row["resolves_on"], probability=row["probability"],
            side=side, expiry_date=q["expiry_date"], expiry_gap_sessions=expiry_gap,
            strike=q["strike"], entry_bid=q["bid"], entry_ask=q["ask"],
            entry_mid=(q["bid"] + q["ask"]) / 2.0, entry_quoted_at=q["quoted_at"],
        ))
    return out


# --- network (yfinance chain fetch, throttled) ----------------------------------

def _import_yf():
    # Lazy, like data.py: the pure core and tests never pay the import.
    import yfinance as yf

    return yf


def _throttle() -> None:
    wait = config.OPTIONS_THROTTLE_S - (time.monotonic() - _LAST_CALL[0])
    if wait > 0:
        time.sleep(wait)
    _LAST_CALL[0] = time.monotonic()


def _clean(v) -> float | None:
    """A yfinance cell as float, or None (NaN/absent/garbage)."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def _row_to_quote(df, strike: float, side: str, expiry: date,
                  underlying_last: float, quoted_at: datetime) -> dict:
    """One chain DataFrame row -> the plain quote dict the pure core judges."""
    sub = df[df["strike"] == strike]
    if sub.empty:
        raise RuntimeError(f"strike {strike} vanished from the {side} chain")
    r = sub.iloc[0]

    def col(name):
        try:
            return r[name]
        except (KeyError, IndexError):
            return None

    oi = _clean(col("openInterest"))
    vol = _clean(col("volume"))
    ltd = col("lastTradeDate")
    return {
        "side": side,
        "expiry_date": expiry.isoformat(),
        "strike": float(strike),
        "bid": _clean(col("bid")),
        "ask": _clean(col("ask")),
        "last": _clean(col("lastPrice")),
        "volume": int(vol) if vol is not None else None,
        "open_interest": int(oi) if oi is not None else None,
        "underlying_last": underlying_last,
        "quoted_at": quoted_at.isoformat(timespec="seconds"),
        "last_trade_at": (ltd.isoformat() if hasattr(ltd, "isoformat") else None),
    }


def _underlying_last(tkr, chain) -> float:
    """Best available underlying price at fetch time (for the drift gate)."""
    und = getattr(chain, "underlying", None)
    if isinstance(und, dict):
        v = _clean(und.get("regularMarketPrice"))
        if v is not None:
            return v
    try:
        v = _clean(tkr.fast_info["last_price"])
        if v is not None:
            return v
    except Exception:  # noqa: BLE001 - fast_info shape varies across versions
        pass
    raise RuntimeError("no underlying price available from the chain")


def fetch_chain_pair(
    ticker: str, anchor_date: date, anchor_close: float, resolves_on: date,
    *, now: datetime | None = None,
) -> tuple[dict, dict, int]:
    """Select and quote the claim's ATM contract pair from the live chain.

    Returns (call_quote, put_quote, expiry_gap_sessions) as plain dicts — the
    book as seen, gated or not; the caller judges each side with usable_quote
    and logs both rows either way. Raises RuntimeError (readable message) on
    CLAIM-level refusals — time gate, no aligned expiry, underlying drift, no
    ATM strike — where there is no selected contract to log (kalshi precedent:
    the missing row is the record).
    """
    quoted_at = now or datetime.now(timezone.utc)
    ok, why = entry_window_ok(anchor_date, quoted_at)
    if not ok:
        raise RuntimeError(why)

    yf = _import_yf()
    tkr = yf.Ticker(ticker)
    _throttle()
    expiries = []
    for e in tkr.options or ():
        try:
            expiries.append(date.fromisoformat(e))
        except ValueError:
            continue
    picked = pick_expiry(expiries, resolves_on)
    if picked is None:
        raise RuntimeError(
            f"no listed expiry within {config.OPTIONS_MAX_EXPIRY_GAP_SESSIONS} "
            f"sessions of {resolves_on.isoformat()} "
            f"({len(expiries)} expiries listed)")
    expiry, gap = picked

    _throttle()
    chain = tkr.option_chain(expiry.isoformat())
    und = _underlying_last(tkr, chain)
    drift = abs(und / anchor_close - 1.0)
    if drift > config.OPTIONS_MAX_UNDERLYING_DRIFT:
        raise RuntimeError(
            f"underlying {und:.2f} has drifted {drift * 100:.2f}% from the anchor "
            f"close {anchor_close:.2f} (> {config.OPTIONS_MAX_UNDERLYING_DRIFT * 100:.1f}% "
            "— entry would embed post-anchor information)")

    # The pair must be the SAME strike on both sides, so pick from the
    # intersection of listed call and put strikes.
    call_strikes = {float(s) for s in chain.calls["strike"].dropna()}
    put_strikes = {float(s) for s in chain.puts["strike"].dropna()}
    strike = pick_strike(sorted(call_strikes & put_strikes), anchor_close)
    if strike is None:
        raise RuntimeError(
            f"no strike within {config.OPTIONS_MAX_ATM_DISTANCE * 100:.0f}% of the "
            f"anchor close {anchor_close:.2f} listed on both sides of {expiry.isoformat()}")

    call_q = _row_to_quote(chain.calls, strike, "C", expiry, und, quoted_at)
    put_q = _row_to_quote(chain.puts, strike, "P", expiry, und, quoted_at)
    return call_q, put_q, gap


# --- settlement (mirrors scoring.resolve_due) ------------------------------------

CloseFetcher = Callable[[str, date], float]
OpenChecker = Callable[[date], bool]


@dataclass
class SettleResult:
    position_id: int
    ticker: str
    forecaster: str
    expiry_date: str
    status: str  # "settled" | "skipped"
    detail: str
    pnl_per_unit: float | None = None


def settle_due(
    conn,
    *,
    today: date | None = None,
    close_fetcher: CloseFetcher | None = None,
    is_open: OpenChecker | None = None,
) -> list[SettleResult]:
    """Settle every expired, still-open paper position at intrinsic value.

    Same discipline as scoring.resolve_due: the settle close is the official
    raw close on the expiry date (injectable for offline tests); positions
    whose close is not yet available stay open and retry on the next run. A
    recorded expiry that turns out not to be a trading session is reported and
    left open — guessing a different settle date would be a fudged mark.
    """
    from . import storage  # local import: storage has no reason to import paper

    today = today or date.today()
    close_fetcher = close_fetcher or data.close_on
    is_open = is_open or data.is_market_open

    due = storage.fetch_open_positions_due(conn, today)
    results: list[SettleResult] = []
    close_cache: dict[tuple[str, date], float] = {}

    for row in due:
        pid = row["id"]
        ticker = row["ticker"]
        expiry = date.fromisoformat(row["expiry_date"])

        if not is_open(expiry):
            results.append(SettleResult(
                pid, ticker, row["forecaster"], row["expiry_date"],
                "skipped", "recorded expiry is not a trading session",
            ))
            continue

        key = (ticker, expiry)
        if key not in close_cache:
            try:
                close_cache[key] = close_fetcher(ticker, expiry)
            except Exception as exc:  # noqa: BLE001 - keep open, retry next run
                results.append(SettleResult(
                    pid, ticker, row["forecaster"], row["expiry_date"],
                    "skipped", f"settle close not available yet ({exc})",
                ))
                continue
        close = close_cache[key]

        value = intrinsic(row["side"], row["strike"], close)
        roi = value / row["entry_mid"] - 1.0
        storage.mark_position_settled(
            conn, pid, settle_close=close, intrinsic=value, pnl_per_unit=roi,
        )
        results.append(SettleResult(
            pid, ticker, row["forecaster"], row["expiry_date"],
            "settled",
            f"{row['side']} {row['strike']:.2f} vs close {close:.2f} -> "
            f"intrinsic {value:.2f} on {row['entry_mid']:.2f} premium",
            pnl_per_unit=roi,
        ))

    return results
