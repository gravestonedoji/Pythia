"""Kalshi market-implied odds — the fifth baseline column (read-only, $0).

WHY: escalates the deploy gate from "beat the quant bar" to "beat the market".
A real prediction market's price for ~the same question benchmarks every
forecaster, including the HMM. Per the Delphi anchoring experiment (Delphi
summary.md §13.3 — an LLM shown a crowd price anchors to it and lands BELOW
it), odds enter Pythia as a BASELINE COLUMN ONLY. They must never be fed into
the forecaster's prompt.

HONESTY RULES (same spirit as the rest of Pythia):
- Live-only. The implied probability is read from the live order book at
  forecast time and logged BEFORE the outcome, like every other row. There is
  deliberately NO backfill: reconstructing what a now-expired book showed on a
  past date from public data is not provably point-in-time honest.
- Mid-quotes of a reasonably tight two-sided book only; one-sided or wide
  (> KALSHI_MAX_SPREAD) quotes are excluded market-by-market.
- Settle-gap handicap, recorded per row: Kalshi rarely lists a contract that
  settles exactly on a claim's `resolves_on`, so we take the open contract set
  settling CLOSEST to it, within KALSHI_MAX_SETTLE_GAP_SESSIONS trading
  sessions, and grade it against the claim anyway. The market is pricing a
  window that overlaps but may not coincide with the claim's. That handicap is
  the market's, not Pythia's — it slightly EASES the "beat the market" gate, so
  close calls should be read accordingly. The matched contract and gap are
  recorded in the row's reasoning. In practice (Kalshi's current listings:
  daily/weekly index settles through Friday), roughly Mon/Tue-anchored claims
  find a match and later anchors skip — sparse coverage is expected and fine.
- Only series that settle on the OFFICIAL CLOSE are whitelisted
  (config.KALSHI_SERIES). Hourly/intraday series are excluded on purpose: a
  10am settle is not the claim's close-to-close question.

LEVEL MAPPING: the claim is about an ETF's raw close, but Kalshi contracts are
struck on the underlying index/asset (S&P 500 level, BTC price). We map through
same-day raw closes: P(SPY_resolve >= SPY_anchor) ≈ P(index_settle >=
index_anchor_close), with the index anchor close fetched point-in-time
(SPY→^GSPC, QQQ→^NDX, IBIT→BTC-USD). The ratio drifts slightly (dividends;
BTC's 24h clock vs the 4pm ETF close) — acceptable slop for a baseline column,
and recorded in the reasoning.

No new dependencies: stdlib urllib for HTTP; the pure ladder/CDF math is
offline unit-tested (tests/test_kalshi.py).
"""

from __future__ import annotations

import json
import time
import urllib.request
from datetime import date, datetime
from zoneinfo import ZoneInfo

from . import config, data
from .baselines import BaselinePrediction

_MARKET_TZ = ZoneInfo(config.MARKET_TZ)
_LAST_CALL = [0.0]  # throttle clock, shared across calls in this process


# --- pure core (offline, unit-tested) -----------------------------------------

def _dollars(x) -> float | None:
    """Parse a Kalshi dollar-string field ('0.0700') to float, None if absent."""
    if x is None or x == "":
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def mid_prob(market: dict) -> float | None:
    """Mid of the YES book in [0, 1], or None when the book is unusable.

    Requires both sides present and a spread <= KALSHI_MAX_SPREAD. A book like
    bid 0.00 / ask 0.01 (deep out-of-the-money) is fine; bid 0.00 / ask 1.00
    (no real market) is not.
    """
    bid = _dollars(market.get("yes_bid_dollars"))
    ask = _dollars(market.get("yes_ask_dollars"))
    if bid is None or ask is None:
        return None
    if ask <= 0 and bid <= 0:
        return None
    if ask - bid > config.KALSHI_MAX_SPREAD:
        return None
    return max(0.0, min(1.0, (bid + ask) / 2.0))


def ladder_prob_above(markets: list[dict], level: float) -> float | None:
    """P(X >= level) interpolated from an above-strike ladder.

    Uses markets with strike_type greater/greater_or_equal (P(X >= strike) =
    mid). Mids are forced monotone non-increasing in strike (tiny book noise
    otherwise produces a non-CDF), then linearly interpolated. Returns None if
    `level` falls outside the quoted strikes — extrapolating a tail from the
    last quote would be invention, not measurement.
    """
    pts = []
    for m in markets:
        if m.get("strike_type") not in ("greater", "greater_or_equal"):
            continue
        s = m.get("floor_strike")
        p = mid_prob(m)
        if s is not None and p is not None:
            pts.append((float(s), p))
    if len(pts) < 2:
        return None
    pts.sort()
    strikes = [s for s, _ in pts]
    probs = []
    for _, p in pts:
        probs.append(min(p, probs[-1]) if probs else p)
    if not (strikes[0] <= level <= strikes[-1]):
        return None
    for i in range(1, len(strikes)):
        if level <= strikes[i]:
            lo, hi = strikes[i - 1], strikes[i]
            w = 0.0 if hi == lo else (level - lo) / (hi - lo)
            return probs[i - 1] + w * (probs[i] - probs[i - 1])
    return probs[-1]


def range_prob_above(markets: list[dict], level: float) -> float | None:
    """P(X >= level) from a bucket ('range') event: less / between / greater.

    Sums the quoted mass strictly above `level` (splitting the containing
    bucket uniformly) and normalizes by the total quoted mass, which absorbs
    the book-wide bid/ask skew. Returns None when the book is too sick to be a
    distribution (total mass far from 1) or `level` falls in an unbounded tail
    bucket (an infinite bucket cannot be split).
    """
    buckets = []  # (floor or None, cap or None, prob)
    for m in markets:
        p = mid_prob(m)
        if p is None:
            continue
        st = m.get("strike_type")
        if st == "between":
            buckets.append((float(m["floor_strike"]), float(m["cap_strike"]), p))
        elif st in ("less", "less_or_equal"):
            buckets.append((None, float(m["cap_strike"]), p))
        elif st in ("greater", "greater_or_equal"):
            buckets.append((float(m["floor_strike"]), None, p))
    if len(buckets) < 3:
        return None
    total = sum(p for _, _, p in buckets)
    if not (0.7 <= total <= 1.3):
        return None

    above = 0.0
    for floor, cap, p in buckets:
        if floor is not None and level <= floor:
            above += p          # bucket entirely above the level
        elif cap is not None and level >= cap:
            pass                # bucket entirely below
        elif floor is not None and cap is not None:
            above += p * (cap - level) / (cap - floor)  # split the container
        else:
            return None         # level inside an unbounded tail bucket
    return above / total


def prob_above(markets: list[dict], level: float) -> float | None:
    """Implied P(X >= level) from one event's markets: ladder if quoted, else
    bucket CDF."""
    p = ladder_prob_above(markets, level)
    if p is None:
        p = range_prob_above(markets, level)
    return p


def settle_date_of(market: dict) -> date | None:
    """The settle date implied by close_time, in exchange-local time."""
    ct = market.get("close_time")
    if not ct:
        return None
    try:
        dt = datetime.fromisoformat(ct.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt.astimezone(_MARKET_TZ).date()


def pick_event(markets: list[dict], resolves_on: date,
               max_gap: int | None = None) -> tuple[list[dict], date, int] | None:
    """Choose the event whose settle date best matches the claim.

    Groups open markets by event_ticker, computes the signed gap in TRADING
    SESSIONS between each event's settle date and `resolves_on`, keeps events
    with |gap| <= max_gap, and prefers (smallest |gap|, most usable quotes).
    Returns (event_markets, settle_date, gap_sessions) or None.
    """
    if max_gap is None:
        max_gap = config.KALSHI_MAX_SETTLE_GAP_SESSIONS
    groups: dict[str, list[dict]] = {}
    for m in markets:
        ev = m.get("event_ticker")
        if ev and settle_date_of(m) is not None:
            groups.setdefault(ev, []).append(m)

    candidates = []
    for ev, mkts in groups.items():
        settle = settle_date_of(mkts[0])
        # signed so that gap < 0 means "settles BEFORE the claim resolves"
        gap = data.sessions_between(resolves_on, settle)
        if abs(gap) > max_gap:
            continue
        n_quoted = sum(1 for m in mkts if mid_prob(m) is not None)
        if n_quoted == 0:
            continue
        candidates.append((abs(gap), -n_quoted, ev, mkts, settle, gap))
    if not candidates:
        return None
    candidates.sort(key=lambda c: c[:3])
    _, _, _, mkts, settle, gap = candidates[0]
    return mkts, settle, gap


# --- network (read-only, throttled, unauthenticated) ---------------------------

def _get(url: str) -> dict:
    # Public API rate limit is easy to trip; keep a polite gap between calls.
    wait = config.KALSHI_THROTTLE_S - (time.monotonic() - _LAST_CALL[0])
    if wait > 0:
        time.sleep(wait)
    req = urllib.request.Request(
        url, headers={"User-Agent": "pythia/0.1 (read-only baseline)",
                      "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=20) as r:
        payload = json.loads(r.read().decode())
    _LAST_CALL[0] = time.monotonic()
    return payload


def fetch_open_markets(series_ticker: str) -> list[dict]:
    """All open markets of one series (paginated)."""
    out: list[dict] = []
    cursor = ""
    for _ in range(5):  # safety cap; real series fit in 1-2 pages
        url = (f"{config.KALSHI_API_BASE}/markets?series_ticker={series_ticker}"
               f"&status=open&limit=1000")
        if cursor:
            url += f"&cursor={cursor}"
        d = _get(url)
        out.extend(d.get("markets", []))
        cursor = d.get("cursor") or ""
        if not cursor:
            break
    return out


def kalshi_prediction(ticker: str, anchor_date: date, anchor_close: float,
                      resolves_on: date) -> BaselinePrediction:
    """Build the market-implied baseline row for one claim, from the live book.

    Raises (with a readable message) whenever no honest number exists — the
    caller logs the skip; a missing row is always preferable to a fudged one.
    """
    series_list = config.KALSHI_SERIES.get(ticker.upper())
    if not series_list:
        raise RuntimeError(f"no Kalshi series mapped for {ticker}")

    underlying = config.KALSHI_UNDERLYING.get(ticker.upper())
    if underlying and underlying.upper() != ticker.upper():
        level = data.close_on(underlying, anchor_date)
        level_desc = (f"{underlying} anchor close {level:.2f} "
                      f"(maps the {ticker} claim onto the contract underlying)")
    else:
        level = anchor_close
        level_desc = f"anchor close {level:.2f}"

    markets: list[dict] = []
    for ser in series_list:
        try:
            markets.extend(fetch_open_markets(ser))
        except Exception:  # noqa: BLE001 — a dormant series must not kill the rest
            continue

    picked = pick_event(markets, resolves_on)
    if picked is None:
        raise RuntimeError(
            f"no Kalshi contract settles within "
            f"{config.KALSHI_MAX_SETTLE_GAP_SESSIONS} sessions of "
            f"{resolves_on.isoformat()} (series: {', '.join(series_list)})")
    ev_markets, settle, gap = picked

    p = prob_above(ev_markets, level)
    if p is None:
        raise RuntimeError(
            f"event {ev_markets[0].get('event_ticker')} has no usable book "
            f"around level {level:.2f}")
    p = float(min(0.99, max(0.01, p)))

    event_ticker = ev_markets[0].get("event_ticker")
    n_quoted = sum(1 for m in ev_markets if mid_prob(m) is not None)
    gap_note = ("settles ON the resolution date" if gap == 0 else
                f"settles {abs(gap)} session(s) {'before' if gap < 0 else 'after'} "
                f"the claim resolves (known handicap, see kalshi.py)")
    return BaselinePrediction(
        forecaster=config.KALSHI,
        probability=p,
        model=f"baseline:kalshi({event_ticker},gap={gap:+d}s)",
        reasoning=(
            f"Kalshi event {event_ticker} ({n_quoted} quoted markets, mid-book): "
            f"implied P(settle >= {level:.2f}) = {p:.2f}, where the level is the "
            f"{level_desc}. Contract {gap_note}: settle {settle.isoformat()} vs "
            f"claim resolution {resolves_on.isoformat()}. Read live at forecast "
            f"time; market odds are a baseline column, never a prompt input."
        ),
    )
