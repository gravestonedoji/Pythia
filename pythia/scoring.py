"""Resolution and scoring — the honest-grading core of Pythia.

The pure functions here (``brier_score``, ``compute_outcome``, ``is_hit``,
``summarize``) carry no I/O and are unit-tested directly: the whole project's
credibility rests on grading being provably correct. ``resolve_due`` wires them
to the database and price data, with the price lookup injectable so resolution
can be tested without the network.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from typing import Callable

from . import config, data, storage

# Type of a function that returns the close for a ticker on a given date.
CloseFetcher = Callable[[str, date], float]
OpenChecker = Callable[[date], bool]


# --- Pure scoring primitives -------------------------------------------------

def compute_outcome(anchor_close: float, resolved_close: float) -> float:
    """Resolve the claim "close on resolution day >= anchor close".

    Returns 1.0 if the claim is true, else 0.0. Equality counts as true (>=).
    """
    return 1.0 if resolved_close >= anchor_close else 0.0


def brier_score(probability: float, outcome: float) -> float:
    """Brier score for a single binary forecast: ``(probability - outcome)^2``.

    `outcome` must be 1.0 (claim true) or 0.0 (claim false). Lower is better;
    0.0 is a perfect confident-correct call, 1.0 a perfect confident-wrong one,
    and 0.25 is what predicting 0.5 always yields.
    """
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"probability must be in [0, 1], got {probability}")
    if outcome not in (0.0, 1.0):
        raise ValueError(f"outcome must be 0.0 or 1.0, got {outcome}")
    return (probability - outcome) ** 2


def is_hit(probability: float, outcome: float) -> bool:
    """Whether the forecaster's directional call was correct.

    A forecast is a "hit" if its implied direction (P >= 0.5 means it expects
    the claim to come true) matches what actually happened. This is a coarse
    accuracy view; the Brier score is the real calibration measure.
    """
    return (probability >= 0.5) == (outcome >= 0.5)


# --- Resolution --------------------------------------------------------------

@dataclass
class ResolveResult:
    forecast_id: int
    ticker: str
    forecaster: str
    resolves_on: str
    status: str  # "resolved" | "skipped"
    detail: str
    outcome: float | None = None
    brier: float | None = None


def resolve_due(
    conn,
    *,
    today: date | None = None,
    close_fetcher: CloseFetcher | None = None,
    is_open: OpenChecker | None = None,
    last_completed: date | None = None,
) -> list[ResolveResult]:
    """Resolve every matured, still-pending forecast.

    For each forecast whose ``resolves_on`` has arrived: confirm the market was
    open that day, fetch the actual close, compute the outcome and Brier score,
    and persist the result. Forecasts whose price data is not yet available
    (e.g. resolving the same day before data posts) are left pending to retry.

    A resolved row is permanent, so a session that has not CLOSED yet is never
    resolvable: during market hours Yahoo serves the in-progress bar as
    today's "close", and an intraday run would grade against a live price.
    The cutoff is clamped to the last completed session — a claim resolving
    today grades on the first run after the bell.

    `close_fetcher`, `is_open`, and `last_completed` are injectable for
    testing without network.
    """
    today = today or date.today()
    close_fetcher = close_fetcher or data.close_on
    is_open = is_open or data.is_market_open
    last_completed = last_completed or data.latest_completed_session()

    due = storage.fetch_pending_due(conn, min(today, last_completed))
    results: list[ResolveResult] = []
    # All forecasters share a (ticker, resolves_on) close, so fetch each once.
    close_cache: dict[tuple[str, date], float] = {}

    for row in due:
        rid = row["id"]
        ticker = row["ticker"]
        resolves_on = date.fromisoformat(row["resolves_on"])

        if not is_open(resolves_on):
            results.append(ResolveResult(
                rid, ticker, row["forecaster"], row["resolves_on"],
                "skipped", "resolution date was not a trading session",
            ))
            continue

        key = (ticker, resolves_on)
        if key not in close_cache:
            try:
                close_cache[key] = close_fetcher(ticker, resolves_on)
            except Exception as exc:  # noqa: BLE001 - report and keep pending
                results.append(ResolveResult(
                    rid, ticker, row["forecaster"], row["resolves_on"],
                    "skipped", f"price data not available yet ({exc})",
                ))
                continue
        resolved_close = close_cache[key]

        outcome = compute_outcome(row["anchor_close"], resolved_close)
        brier = brier_score(row["probability"], outcome)
        storage.mark_resolved(
            conn, rid, outcome=outcome, resolved_close=resolved_close, brier=brier,
        )
        results.append(ResolveResult(
            rid, ticker, row["forecaster"], row["resolves_on"],
            "resolved", f"close {resolved_close:.2f} vs anchor {row['anchor_close']:.2f}",
            outcome=outcome, brier=brier,
        ))

    return results


# --- Aggregation for the review report ---------------------------------------

@dataclass
class ForecasterStats:
    forecaster: str
    resolved: int = 0
    pending: int = 0
    hits: int = 0
    brier_sum: float = 0.0

    @property
    def hit_rate(self) -> float | None:
        return self.hits / self.resolved if self.resolved else None

    @property
    def avg_brier(self) -> float | None:
        return self.brier_sum / self.resolved if self.resolved else None


def summarize(rows) -> dict[str, ForecasterStats]:
    """Aggregate per-forecaster running hit-rate and average Brier score."""
    stats: dict[str, ForecasterStats] = {
        fc: ForecasterStats(fc) for fc in config.ALL_FORECASTERS
    }
    for row in rows:
        fc = row["forecaster"]
        if fc not in stats:
            stats[fc] = ForecasterStats(fc)
        s = stats[fc]
        if row["status"] == "resolved" and row["brier"] is not None:
            s.resolved += 1
            s.brier_sum += row["brier"]
            if is_hit(row["probability"], row["outcome"]):
                s.hits += 1
        elif row["status"] == "pending":
            s.pending += 1
    return stats
