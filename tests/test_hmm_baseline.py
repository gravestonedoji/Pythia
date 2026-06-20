"""Offline tests for the HMM quant-bar baseline (pure core, no network).

The credibility rule applies to baselines too: a quant bar that leaks future
data or wobbles between runs would corrupt every comparison against it.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from pythia import config
from pythia.hmm_baseline import (
    em_cap_from_model,
    fit_hmm,
    filter_belief,
    hmm_prediction_from_history,
    hmm_prediction_with_health,
    hmm_probability_up,
    prob_up_over_horizon,
)


def _two_regime_returns(n: int = 2000, seed: int = 7) -> np.ndarray:
    """Synthetic returns from a known two-mood process: a calm up-drift state
    and a volatile down-drift state, switching rarely."""
    rng = np.random.default_rng(seed)
    out = np.empty(n)
    state = 0
    for i in range(n):
        if rng.random() < 0.02:
            state = 1 - state
        out[i] = rng.normal(0.0008, 0.008) if state == 0 else rng.normal(-0.002, 0.025)
    return out


def test_fit_recovers_two_regimes():
    r = _two_regime_returns()
    fit = fit_hmm(r, K=2, seed=1)
    # the two fitted states must actually separate: one calm, one volatile
    assert fit.sig.max() / fit.sig.min() > 1.8
    # the volatile state should carry the lower (negative) drift
    assert fit.mu[np.argmax(fit.sig)] < fit.mu[np.argmin(fit.sig)]


def test_filter_belief_is_a_distribution():
    r = _two_regime_returns(800)
    fit = fit_hmm(r, K=2, seed=1)
    b = filter_belief(r, fit)
    assert b.shape == (2,)
    assert abs(b.sum() - 1.0) < 1e-9
    assert (b >= 0).all()


def test_probability_respects_drift_direction():
    rng = np.random.default_rng(3)
    up = rng.normal(0.003, 0.008, 1500)    # strong steady up-drift
    down = rng.normal(-0.003, 0.008, 1500)  # strong steady down-drift
    p_up, _, _ = hmm_probability_up(up, horizon=5, seed=11)
    p_down, _, _ = hmm_probability_up(down, horizon=5, seed=11)
    assert p_up > 0.60
    assert p_down < 0.40


def test_probability_is_deterministic():
    r = _two_regime_returns(1200)
    a, _, _ = hmm_probability_up(r, horizon=5, seed=42)
    b, _, _ = hmm_probability_up(r, horizon=5, seed=42)
    assert a == b


def test_mc_probability_in_bounds():
    r = _two_regime_returns(900)
    fit = fit_hmm(r, K=2, seed=1)
    belief = filter_belief(r, fit)
    p = prob_up_over_horizon(fit, belief, horizon=5, n_paths=5000, seed=0)
    assert 0.0 <= p <= 1.0


def test_point_in_time_slice_ignores_future_rows():
    """The forecast built from history truncated at the anchor must equal the
    forecast built from history that CONTAINS future rows — i.e. future data
    can never leak into the fit."""
    r = _two_regime_returns(1300, seed=9)
    idx = pd.bdate_range("2020-01-01", periods=1301)
    closes = 100 * np.exp(np.concatenate([[0.0], np.cumsum(r)]))
    df_full = pd.DataFrame({"Close": closes}, index=idx)

    anchor = idx[1000].date()
    df_truncated = df_full.iloc[:1001]  # ends exactly at the anchor

    a = hmm_prediction_from_history("TEST", df_full, anchor, horizon=5)
    b = hmm_prediction_from_history("TEST", df_truncated, anchor, horizon=5)
    assert a.probability == b.probability


def test_refuses_thin_history():
    rng = np.random.default_rng(0)
    thin = rng.normal(0, 0.01, config.HMM_MIN_SESSIONS - 50)
    with pytest.raises(RuntimeError, match="sessions"):
        hmm_probability_up(thin, horizon=5, seed=0)


def test_probability_never_certain():
    rng = np.random.default_rng(5)
    # near-zero vol, pure drift: MC would say 1.0 without the clamp
    r = rng.normal(0.005, 0.0005, 1000)
    p, _, _ = hmm_probability_up(r, horizon=5, seed=2)
    assert p <= 0.99


# --- EM cap: recorded on every row, pinned for reconstruction --------------------
# The cap was 40 until 2026-06-12 (it truncated 97% of fits just short of
# tolerance). Rows carry the cap they were fitted under in their model
# descriptor's `em` token; rows from before the token exist reconstruct under
# the legacy cap — the method that produced them, not today's.

def _history(n: int = 1300, seed: int = 9) -> tuple[pd.DataFrame, date]:
    r = _two_regime_returns(n, seed=seed)
    idx = pd.bdate_range("2020-01-01", periods=n + 1)
    closes = 100 * np.exp(np.concatenate([[0.0], np.cumsum(r)]))
    return pd.DataFrame({"Close": closes}, index=idx), idx[-1].date()


def test_em_cap_descriptor_round_trips_and_legacy_rows_pin_old_cap():
    history, anchor = _history()
    pred, _ = hmm_prediction_with_health("TEST", history, anchor, horizon=5)
    assert f",em{config.HMM_EM_MAX_ITERS})" in pred.model
    assert em_cap_from_model(pred.model) == config.HMM_EM_MAX_ITERS
    # pre-token descriptors (every row logged before 2026-06-12) -> legacy cap
    assert em_cap_from_model("baseline:hmm(K=3,2512d,mc20k)") == config.HMM_EM_LEGACY_ITERS
    assert em_cap_from_model(None) == config.HMM_EM_LEGACY_ITERS


def test_em_iters_reaches_the_fit():
    history, anchor = _history()
    pred, rec = hmm_prediction_with_health("TEST", history, anchor, horizon=5,
                                           em_iters=2)
    assert rec.converged is False
    assert rec.n_iter == 2
    assert ",em2)" in pred.model


def test_legacy_reconstruction_reproduces_the_truncated_fit():
    # A row logged under the old cap must reconstruct under the old cap; a
    # refit at today's cap can land in a different (converged) optimum and
    # would read as a false reconstruction_mismatch.
    history, anchor = _history()
    logged, _ = hmm_prediction_with_health(
        "TEST", history, anchor, horizon=5, em_iters=config.HMM_EM_LEGACY_ITERS)
    legacy_model = "baseline:hmm(K=2,1299d,mc20k)"  # pre-token format
    rebuilt, _ = hmm_prediction_with_health(
        "TEST", history, anchor, horizon=5,
        em_iters=em_cap_from_model(legacy_model))
    assert rebuilt.probability == logged.probability
