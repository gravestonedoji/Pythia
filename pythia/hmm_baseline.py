"""The quant bar — a fitted HMM regime filter as the fourth baseline.

A Gaussian hidden-Markov-model forecaster: it learns the ticker's hidden
"moods" (regimes — e.g. calm-up / churn / selloff) from its own daily return
history, tracks a running belief over the current mood with a Bayes forward
filter, then Monte-Carlo simulates the horizon to get P(close >= anchor).

Why it exists: the coin/drift/momentum ladder is a floor, not a bar. This is
the strongest price-only statistical forecaster we can run for free, so the
question "is the LLM worth deploying?" becomes "does it beat THIS?". If it
doesn't, the deployable model is this one.

Honesty rules (same spirit as the rest of Pythia):
- Strictly point-in-time: fitted ONLY on closes up to and including the anchor
  date. The backfill path slices history to the anchor before fitting, so a
  backfilled forecast is identical to what would have been logged live that day.
- Deterministic: EM restarts and the MC are seeded from (ticker, anchor_date),
  so re-running produces the same probability (idempotent like the daily loop).
- No new dependencies: numpy only (no scipy), pure functions unit-testable
  offline.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from . import config, data
from .baselines import BaselinePrediction
from .hmm_health import FitRecord

_SIGMA_FLOOR = 1e-5  # daily-return scale; well below any real ETF's vol


# --- pure core (offline, unit-tested) -----------------------------------------

@dataclass
class HMMFit:
    T: np.ndarray       # KxK transition matrix
    mu: np.ndarray      # per-state mean daily log-return
    sig: np.ndarray     # per-state std
    pi: np.ndarray      # initial state distribution
    loglik: float
    n_obs: int
    K: int
    # Health telemetry for the chosen restart (hmm_health.py). Defaults keep
    # direct construction (and older callers) working unchanged.
    converged: bool = True   # did EM meet tolerance before exhausting `iters`?
    n_iter: int = 0          # EM iterations actually run


def _emissions(r: np.ndarray, mu: np.ndarray, sig: np.ndarray) -> np.ndarray:
    """Normal pdf of each observation under each state, shape (n, K)."""
    var = sig**2
    e = np.exp(-0.5 * (r[:, None] - mu) ** 2 / var) / np.sqrt(2 * np.pi * var)
    return np.maximum(e, 1e-300)


def fit_hmm(returns: np.ndarray, K: int, *, iters: int = config.HMM_EM_MAX_ITERS,
            restarts: int = 2, seed: int = 0) -> HMMFit:
    """Baum-Welch EM for a Gaussian-emission HMM (scaled forward-backward).

    ``iters`` is a cap, not a budget — the loop breaks at the loglik tolerance
    (the watchlist needs ~30-110 iterations; see config.HMM_EM_MAX_ITERS).
    The cap was 40 until 2026-06-12, which truncated 97% of recorded fits just
    short of tolerance; reconstruction of those rows pins the old cap via the
    `em` model-descriptor token (em_cap_from_model)."""
    r = np.asarray(returns, float)
    n = len(r)
    if n < 10 * K:
        raise ValueError(f"need at least {10 * K} returns to fit K={K} states, have {n}")
    rng = np.random.default_rng(seed)
    best: HMMFit | None = None

    for _ in range(restarts):
        # init: quantile-split means/stds (low state = selloff, high = rally)
        chunks = np.array_split(np.sort(r), K)
        mu = np.array([c.mean() for c in chunks]) + rng.normal(0, 1e-4, K)
        sig = np.maximum([c.std() for c in chunks], 5 * _SIGMA_FLOOR)
        T = np.full((K, K), 0.1 / max(K - 1, 1))
        np.fill_diagonal(T, 0.9)
        pi = np.full(K, 1.0 / K)
        prev_ll = -np.inf
        ll = prev_ll
        converged = False
        used = 0

        for _ in range(iters):
            used += 1
            e = _emissions(r, mu, sig)

            alpha = np.empty((n, K)); c = np.empty(n)
            a = pi * e[0]; c[0] = a.sum(); alpha[0] = a / c[0]
            for t in range(1, n):
                a = (alpha[t - 1] @ T) * e[t]
                c[t] = a.sum(); alpha[t] = a / c[t]
            ll = float(np.log(c).sum())

            beta = np.empty((n, K)); beta[-1] = 1.0
            for t in range(n - 2, -1, -1):
                beta[t] = (T @ (e[t + 1] * beta[t + 1])) / c[t + 1]

            gamma = alpha * beta
            gamma /= np.maximum(gamma.sum(axis=1, keepdims=True), 1e-300)

            num = np.zeros((K, K))
            for t in range(n - 1):
                num += np.outer(alpha[t], e[t + 1] * beta[t + 1]) * T / c[t + 1]
            T = num / np.maximum(num.sum(axis=1, keepdims=True), 1e-300)

            w = gamma.sum(axis=0)
            mu = (gamma * r[:, None]).sum(axis=0) / np.maximum(w, 1e-300)
            var = (gamma * (r[:, None] - mu) ** 2).sum(axis=0) / np.maximum(w, 1e-300)
            sig = np.maximum(np.sqrt(var), _SIGMA_FLOOR)
            pi = gamma[0]

            if abs(ll - prev_ll) < 1e-7 * max(1.0, abs(ll)):
                converged = True
                break
            prev_ll = ll

        if best is None or ll > best.loglik:
            best = HMMFit(T=T, mu=mu, sig=sig, pi=pi, loglik=ll, n_obs=n, K=K,
                          converged=converged, n_iter=used)
    return best


def filter_belief(returns: np.ndarray, fit: HMMFit) -> np.ndarray:
    """Forward-filtered belief over the current state after the last return."""
    e = _emissions(np.asarray(returns, float), fit.mu, fit.sig)
    b = fit.pi * e[0]
    b /= max(b.sum(), 1e-300)
    for t in range(1, len(returns)):
        b = (b @ fit.T) * e[t]
        s = b.sum()
        b = b / s if s > 0 else np.full(fit.K, 1.0 / fit.K)
    return b


def prob_up_over_horizon(fit: HMMFit, belief: np.ndarray, horizon: int,
                         *, n_paths: int = 20_000, seed: int = 0) -> float:
    """P(sum of the next `horizon` daily log-returns >= 0), by Monte Carlo.

    This is a forecaster, not an answer key, so MC is legitimate here; with
    20k paths the standard error is ~0.4 percentage points.
    """
    rng = np.random.default_rng(seed)
    cum = np.cumsum(fit.T, axis=1)
    states = rng.choice(fit.K, size=n_paths, p=belief / belief.sum())
    total = np.zeros(n_paths)
    for _ in range(horizon):
        u = rng.random(n_paths)
        states = (u[:, None] > cum[states]).sum(axis=1)  # transition first...
        total += rng.normal(fit.mu[states], fit.sig[states])  # ...then emit
    return float((total >= 0.0).mean())


def hmm_probability_up(log_returns: np.ndarray, horizon: int, *, seed: int = 0,
                       em_iters: int = config.HMM_EM_MAX_ITERS,
                       ) -> tuple[float, HMMFit, np.ndarray]:
    """Fit + filter + simulate: the full pure pipeline from returns to P(up).

    ``em_iters`` exists for method-faithful reconstruction of rows logged under
    an older cap (hmm-health --backfill); live forecasts use the default."""
    n = len(log_returns)
    if n < config.HMM_MIN_SESSIONS:
        raise RuntimeError(
            f"only {n} sessions of history; need >= {config.HMM_MIN_SESSIONS} "
            f"to fit the HMM honestly"
        )
    k = config.HMM_STATES if n >= config.HMM_RICH_HISTORY_SESSIONS else 2
    fit = fit_hmm(log_returns, K=k, iters=em_iters, seed=seed)
    belief = filter_belief(log_returns, fit)
    p = prob_up_over_horizon(fit, belief, horizon,
                             n_paths=config.HMM_MC_PATHS, seed=seed + 1)
    # never 0.00/1.00 — a finite-sample statistical model has no business
    # claiming certainty about a 5-day move
    return float(np.clip(p, 0.01, 0.99)), fit, belief


# --- wiring (history in, BaselinePrediction out) -------------------------------

def _seed_for(ticker: str, anchor_date: date) -> int:
    h = hashlib.sha1(f"{ticker}|{anchor_date.isoformat()}".encode()).hexdigest()
    return int(h[:8], 16)


def em_cap_from_model(model: str | None) -> int:
    """The EM iteration cap a logged row was fitted under, from its model
    descriptor's `em` token. Rows logged before the cap was raised
    (2026-06-12) carry no token and reconstruct under the legacy cap — the
    method that produced them, not today's."""
    m = re.search(r",em(\d+)\)", model or "")
    return int(m.group(1)) if m else config.HMM_EM_LEGACY_ITERS


def hmm_prediction_with_health(
    ticker: str, history: pd.DataFrame, anchor_date: date, horizon: int,
    *, em_iters: int = config.HMM_EM_MAX_ITERS,
) -> tuple[BaselinePrediction, FitRecord]:
    """Build the HMM forecast from pre-fetched history, sliced to the anchor.

    The slice is what makes backfill honest: only closes dated <= anchor_date
    ever reach the fit, so a reconstructed forecast equals the live one.

    Alongside the prediction, returns the fit's health telemetry (convergence,
    parameters, data window) for hmm_health.health_check + persistence — the
    record is raw here; flags are filled in by the caller against the previous
    stored fit.
    """
    past = history[[d.date() <= anchor_date for d in history.index]]
    closes = past["Close"].dropna()
    log_returns = np.diff(np.log(closes.values))
    seed = _seed_for(ticker, anchor_date)
    p, fit, belief = hmm_probability_up(log_returns, horizon, seed=seed,
                                        em_iters=em_iters)

    order = np.argsort(fit.mu)  # lowest-drift state first, for readability
    desc = ", ".join(
        f"state{i}: mu={fit.mu[j] * 100:+.3f}%/d sig={fit.sig[j] * 100:.2f}% "
        f"belief={belief[j]:.2f}"
        for i, j in enumerate(order)
    )
    prediction = BaselinePrediction(
        forecaster=config.HMM_FILTER,
        probability=p,
        model=(f"baseline:hmm(K={fit.K},{fit.n_obs}d,"
               f"mc{config.HMM_MC_PATHS // 1000}k,em{em_iters})"),
        reasoning=(
            f"Gaussian HMM fitted on {fit.n_obs} sessions of {ticker} daily "
            f"log-returns up to {anchor_date.isoformat()} ({desc}). "
            f"{config.HMM_MC_PATHS:,}-path Monte Carlo over {horizon} sessions "
            f"gives P(up) = {p:.2f}."
        ),
    )
    record = FitRecord(
        ticker=ticker, anchor_date=anchor_date.isoformat(),
        converged=fit.converged, n_iter=fit.n_iter, loglik=fit.loglik,
        n_states=fit.K, n_obs=fit.n_obs,
        window_start=closes.index[0].date().isoformat(),
        window_end=closes.index[-1].date().isoformat(),
        mu=fit.mu.tolist(), sig=fit.sig.tolist(),
        transition=fit.T.tolist(), pi=fit.pi.tolist(),
    )
    return prediction, record


def hmm_prediction_from_history(
    ticker: str, history: pd.DataFrame, anchor_date: date, horizon: int
) -> BaselinePrediction:
    """The prediction alone (see hmm_prediction_with_health for telemetry)."""
    return hmm_prediction_with_health(ticker, history, anchor_date, horizon)[0]


def hmm_prediction(ticker: str, anchor_date: date, horizon: int) -> BaselinePrediction:
    """Fetch deep history (network) and build the HMM forecast for a live claim."""
    history = data.get_price_history(ticker, lookback_days=config.HMM_LOOKBACK_DAYS)
    return hmm_prediction_from_history(ticker, history, anchor_date, horizon)
