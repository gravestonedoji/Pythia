"""Price data access and trading-calendar math.

Two responsibilities, both kept here so they are shared and unit-testable:

1. Price data via yfinance (raw, unadjusted closes — see note below).
2. Trading-day arithmetic via a real exchange calendar
   (pandas-market-calendars). "N trading days" is never naive calendar math.

Note on adjustment: we fetch with ``auto_adjust=False`` and compare *raw*
closing prices. The anchor close is captured at issue time and compared to the
raw close on the resolution day. Raw closes are stable over time (adjusted
closes get rewritten whenever a new dividend goes ex), which keeps a logged
claim resolvable to exactly the same answer no matter when we grade it — and a
claim about "closing price" should mean the literal close.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pandas_market_calendars as mcal

from . import config

_calendar = mcal.get_calendar(config.EXCHANGE)


# --- Trading-calendar math (offline, deterministic) --------------------------

def is_market_open(day: date) -> bool:
    """True if `day` was/will be a trading session on the exchange calendar."""
    sessions = _calendar.valid_days(start_date=day, end_date=day)
    return len(sessions) == 1


def latest_completed_session(now: pd.Timestamp | None = None) -> date:
    """The most recent trading session whose closing bell has already passed.

    Used as the anchor reference: a session is only usable as an anchor once it
    has actually closed, so we never anchor to a partial intraday bar.
    """
    if now is None:
        now = pd.Timestamp.now(tz="UTC")
    elif now.tzinfo is None:
        now = now.tz_localize("UTC")
    else:
        now = now.tz_convert("UTC")

    sched = _calendar.schedule(
        start_date=(now - pd.Timedelta(days=10)).date(),
        end_date=now.date(),
    )
    closes = sched["market_close"]
    if closes.dt.tz is None:
        closes = closes.dt.tz_localize("UTC")
    completed = closes[closes <= now]
    if completed.empty:
        # Extremely unlikely (would need >10 days with no closed session); widen.
        sched = _calendar.schedule(
            start_date=(now - pd.Timedelta(days=30)).date(),
            end_date=now.date(),
        )
        closes = sched["market_close"]
        if closes.dt.tz is None:
            closes = closes.dt.tz_localize("UTC")
        completed = closes[closes <= now]
    return completed.index[-1].date()


def trading_days_forward(start: date, n: int) -> date:
    """The date that is `n` trading sessions strictly after `start`.

    Counts forward over *open* sessions only (skips weekends and holidays),
    expanding the search window as needed so holidays never short the count.
    """
    if n < 1:
        raise ValueError(f"horizon must be >= 1 trading day, got {n}")
    window = n * 2 + 15  # generous buffer for weekends/holidays
    while True:
        sessions = _calendar.valid_days(
            start_date=start + timedelta(days=1),
            end_date=start + timedelta(days=window),
        )
        if len(sessions) >= n:
            return sessions[n - 1].date()
        window *= 2  # holiday-dense stretch; widen and retry


def session_open_utc(day: date) -> pd.Timestamp:
    """The market-open timestamp (UTC) of the trading session `day`.

    Used by the paper book's entry-window gate: a quote captured at or after
    the NEXT session's open embeds post-anchor information and must be refused.
    """
    sched = _calendar.schedule(start_date=day, end_date=day)
    if sched.empty:
        raise ValueError(f"{day.isoformat()} is not a trading session for {config.EXCHANGE}")
    open_ts = sched["market_open"].iloc[0]
    if open_ts.tzinfo is None:
        open_ts = open_ts.tz_localize("UTC")
    return open_ts.tz_convert("UTC")


def sessions_between(start: date, end: date) -> int:
    """Signed count of trading sessions in (start, end] — 0 when equal,
    negative when `end` is before `start`. Neither endpoint needs to be a
    session itself (e.g. a weekend crypto-contract settle date).
    """
    if start == end:
        return 0
    a, b = (start, end) if start < end else (end, start)
    n = len(_calendar.valid_days(start_date=a + timedelta(days=1), end_date=b))
    return n if start < end else -n


# --- Price data (network via yfinance) ---------------------------------------

def _import_yf():
    # Imported lazily so calendar math and tests don't pay the yfinance import cost.
    import yfinance as yf

    return yf


def get_price_history(ticker: str, lookback_days: int = config.PRICE_LOOKBACK_DAYS) -> pd.DataFrame:
    """Fetch recent daily OHLCV for `ticker` (raw, unadjusted).

    Returns a DataFrame indexed by trading date with Open/High/Low/Close/Volume.
    """
    yf = _import_yf()
    end = date.today() + timedelta(days=1)  # end is exclusive; include today
    start = date.today() - timedelta(days=lookback_days)
    hist = yf.Ticker(ticker).history(
        start=start.isoformat(),
        end=end.isoformat(),
        interval="1d",
        auto_adjust=False,
        actions=False,
    )
    if hist is None or hist.empty:
        raise RuntimeError(f"No price data returned for {ticker!r}")
    return hist


def close_on(ticker: str, day: date) -> float:
    """The raw closing price for `ticker` on the trading session `day`.

    Raises if `day` is not an exchange session or has no available data.
    """
    if not is_market_open(day):
        raise ValueError(f"{day.isoformat()} is not a trading session for {config.EXCHANGE}")
    yf = _import_yf()
    hist = yf.Ticker(ticker).history(
        start=(day - timedelta(days=4)).isoformat(),
        end=(day + timedelta(days=2)).isoformat(),
        interval="1d",
        auto_adjust=False,
        actions=False,
    )
    if hist is None or hist.empty:
        raise RuntimeError(f"No price data for {ticker!r} around {day.isoformat()}")
    for idx, row in hist.iterrows():
        if idx.date() == day:
            return float(row["Close"])
    raise RuntimeError(f"No close for {ticker!r} on {day.isoformat()} (data not yet available?)")


def anchor_from_history(
    history: pd.DataFrame, now: pd.Timestamp | None = None
) -> tuple[date, float]:
    """Pick the anchor (reference) session and close from fetched history.

    The anchor is the latest *completed* session present in the data — this both
    guarantees the close exists locally and avoids anchoring to a partial
    intraday bar.
    """
    cutoff = latest_completed_session(now)
    completed = history[[d.date() <= cutoff for d in history.index]]
    if completed.empty:
        raise RuntimeError("No completed trading session found in price history")
    last_idx = completed.index[-1]
    return last_idx.date(), float(completed.loc[last_idx, "Close"])


# --- Pure price computations (shared by forecaster + baselines) --------------

def daily_returns(history: pd.DataFrame) -> pd.Series:
    """Daily simple returns from the close series (no fill)."""
    return history["Close"].dropna().pct_change().dropna()


def positive_day_ratio(history: pd.DataFrame, window: int | None = None) -> float:
    """Fraction of up days over (the last `window` sessions of) the history.

    This is the drift baseline's probability — computed from data, not hardcoded.
    """
    rets = daily_returns(history)
    if window is not None:
        rets = rets.tail(window)
    if rets.empty:
        raise RuntimeError("Not enough price history to compute positive-day ratio")
    return float((rets > 0).mean())


def momentum_up(history: pd.DataFrame, n: int = config.MOMENTUM_LOOKBACK_SESSIONS) -> bool:
    """True if the net move over the last `n` sessions was positive."""
    closes = history["Close"].dropna()
    if len(closes) < n + 1:
        raise RuntimeError(f"Need at least {n + 1} closes for momentum, have {len(closes)}")
    return bool(closes.iloc[-1] > closes.iloc[-1 - n])


def price_features(history: pd.DataFrame) -> dict:
    """Compute a compact set of price/volume-derived features for context.

    Strictly price-only: nothing here references macro, rates, or news.
    """
    closes = history["Close"].dropna()
    rets = closes.pct_change().dropna()
    last = float(closes.iloc[-1])

    def ret_over(n: int) -> float | None:
        if len(closes) < n + 1:
            return None
        return float(closes.iloc[-1] / closes.iloc[-1 - n] - 1.0)

    def sma(n: int) -> float | None:
        if len(closes) < n:
            return None
        return float(closes.tail(n).mean())

    sma20 = sma(20)
    sma50 = sma(50)
    window = closes.tail(min(len(closes), config.DRIFT_LOOKBACK_SESSIONS))
    vol20 = float(rets.tail(20).std()) if len(rets) >= 20 else None

    return {
        "last_close": last,
        "ret_1d": ret_over(1),
        "ret_5d": ret_over(5),
        "ret_20d": ret_over(20),
        "ret_60d": ret_over(60),
        "sma20": sma20,
        "sma50": sma50,
        "pct_vs_sma20": (last / sma20 - 1.0) if sma20 else None,
        "pct_vs_sma50": (last / sma50 - 1.0) if sma50 else None,
        "vol_20d_daily": vol20,
        "high_window": float(window.max()),
        "low_window": float(window.min()),
        "pct_from_high": float(last / window.max() - 1.0),
        "pct_from_low": float(last / window.min() - 1.0),
        "n_sessions": int(len(closes)),
    }


def format_price_context(history: pd.DataFrame, max_rows: int = 30) -> str:
    """Render a compact, price-only context block for the forecasting prompt."""
    closes = history["Close"].dropna()
    vols = history["Volume"] if "Volume" in history.columns else None
    rets = closes.pct_change()
    avg_vol = float(vols.tail(60).mean()) if vols is not None and len(vols) else None

    rows = []
    tail_idx = closes.index[-max_rows:]
    for idx in tail_idx:
        c = float(closes.loc[idx])
        r = rets.loc[idx]
        rstr = f"{r * 100:+.2f}%" if pd.notna(r) else "   n/a"
        if vols is not None and idx in vols.index and pd.notna(vols.loc[idx]):
            v = float(vols.loc[idx])
            vstr = f"{v / 1e6:8.1f}M"
            if avg_vol:
                vstr += f" ({v / avg_vol:.2f}x avg)"
        else:
            vstr = "n/a"
        rows.append(f"  {idx.date().isoformat()}  close={c:10.2f}  chg={rstr}  vol={vstr}")

    f = price_features(history)

    def pct(x: float | None) -> str:
        return f"{x * 100:+.2f}%" if x is not None else "n/a"

    def num(x: float | None) -> str:
        return f"{x:.2f}" if x is not None else "n/a"

    summary = (
        f"Sessions available: {f['n_sessions']}\n"
        f"Last close: {f['last_close']:.2f}\n"
        f"Return 1d / 5d / 20d / 60d: {pct(f['ret_1d'])} / {pct(f['ret_5d'])} / "
        f"{pct(f['ret_20d'])} / {pct(f['ret_60d'])}\n"
        f"SMA20 / SMA50: {num(f['sma20'])} / {num(f['sma50'])}\n"
        f"Price vs SMA20 / SMA50: {pct(f['pct_vs_sma20'])} / {pct(f['pct_vs_sma50'])}\n"
        f"20d daily volatility (std of returns): {pct(f['vol_20d_daily'])}\n"
        f"Window high / low: {num(f['high_window'])} / {num(f['low_window'])}  "
        f"(now {pct(f['pct_from_high'])} from high, {pct(f['pct_from_low'])} from low)"
    )

    return (
        "RECENT DAILY PRICE ACTION (most recent last):\n"
        + "\n".join(rows)
        + "\n\nPRICE SUMMARY:\n"
        + summary
    )
