"""Offline tests for HMM fit-health monitoring (no network, no key).

The quant bar is the deploy gate, so the harness must catch a broken referee:
a non-converged EM fit, or parameters that swing between consecutive refits,
must taint that day's hmm_filter value — and the taint must survive into the
database and the leaderboard annotation. Policy under test: log + flag (a
tainted value is never dropped, only surfaced).
"""

from __future__ import annotations

import sqlite3
from datetime import date

import numpy as np
import pandas as pd
import pytest

from pythia import config, hmm_baseline, hmm_health, storage
from pythia.hmm_baseline import fit_hmm, hmm_prediction_with_health
from pythia.hmm_health import (
    DURATION_JUMP,
    K_CHANGED,
    MU_JUMP,
    NON_CONVERGED,
    SIGMA_JUMP,
    FitRecord,
    health_check,
    integrity_lines,
    is_tainted,
    parse_flags,
    record_from_row,
    taint_summary,
)
from pythia.storage import Forecast


def _two_regime_returns(n: int = 2000, seed: int = 7) -> np.ndarray:
    """Synthetic returns from a known two-mood process (same as the baseline
    tests): a calm up-drift state and a volatile down-drift state."""
    rng = np.random.default_rng(seed)
    out = np.empty(n)
    state = 0
    for i in range(n):
        if rng.random() < 0.02:
            state = 1 - state
        out[i] = rng.normal(0.0008, 0.008) if state == 0 else rng.normal(-0.002, 0.025)
    return out


def _record(**overrides) -> FitRecord:
    """A healthy two-state fit record; state 0 calm, state 1 volatile."""
    base = dict(
        ticker="SPY", anchor_date="2026-06-10", converged=True, n_iter=12,
        loglik=-100.0, n_states=2, n_obs=2500,
        window_start="2016-06-10", window_end="2026-06-10",
        mu=[0.0008, -0.002], sig=[0.008, 0.025],
        transition=[[0.95, 0.05], [0.10, 0.90]], pi=[0.9, 0.1],
    )
    base.update(overrides)
    return FitRecord(**base)


def _prev(**overrides) -> FitRecord:
    overrides.setdefault("anchor_date", "2026-06-09")
    return _record(**overrides)


# --- convergence telemetry from the fit itself ---------------------------------

def test_fit_reports_convergence_on_identifiable_data():
    # two real regimes -> the 2-state EM has something to find and settles
    fit = fit_hmm(_two_regime_returns(), K=2, seed=1)
    assert fit.converged is True
    assert 0 < fit.n_iter <= 40


def test_forced_non_convergence_is_detected():
    # 2 EM iterations cannot meet tolerance on regime-switching data — the
    # synthetic stand-in for the ~47%-of-windows failure mode on real SPY.
    fit = fit_hmm(_two_regime_returns(), K=2, iters=2, seed=1)
    assert fit.converged is False
    assert fit.n_iter == 2


# --- stability checks between consecutive refits --------------------------------

def test_stable_consecutive_fits_are_clean():
    rec = health_check(_record(), _prev())
    assert rec.flags == []
    assert rec.compared_to == "2026-06-09"
    assert not is_tainted(rec.flags)
    assert "stable" in rec.detail


def test_non_converged_fit_is_flagged_and_tainted():
    rec = health_check(_record(converged=False, n_iter=40), None)
    assert rec.flags == [NON_CONVERGED]
    assert is_tainted(rec.flags)


def test_duration_jump_flagged():
    # volatile regime's expected duration 10d -> 100d (the published failure
    # mode: days flipping to months between refits)
    rec = health_check(
        _record(transition=[[0.95, 0.05], [0.01, 0.99]]), _prev())
    assert rec.flags == [DURATION_JUMP]
    assert is_tainted(rec.flags)


def test_sigma_jump_flagged():
    rec = health_check(_record(sig=[0.008, 0.05]), _prev())  # vol state 2x
    assert rec.flags == [SIGMA_JUMP]


def test_mu_jump_flagged():
    rec = health_check(_record(mu=[0.0008, -0.0035]), _prev())  # 15 bps shift
    assert rec.flags == [MU_JUMP]


def test_small_jitter_within_thresholds_is_clean():
    rec = health_check(
        _record(mu=[0.00085, -0.0016], sig=[0.0088, 0.0275],
                transition=[[0.95, 0.05], [0.05, 0.95]]),  # duration 2x < 3x
        _prev(),
    )
    assert rec.flags == []


def test_states_are_matched_by_sigma_not_label_order():
    # the same fit with state labels permuted must compare clean
    rec = health_check(
        _record(mu=[-0.002, 0.0008], sig=[0.025, 0.008],
                transition=[[0.90, 0.10], [0.05, 0.95]], pi=[0.1, 0.9]),
        _prev(),
    )
    assert rec.flags == []


def test_k_change_is_informational_not_tainting():
    rec = health_check(
        _record(n_states=3, mu=[0.001, 0.0, -0.002], sig=[0.006, 0.012, 0.03],
                transition=[[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]],
                pi=[0.4, 0.4, 0.2]),
        _prev(),
    )
    assert rec.flags == [K_CHANGED]
    assert not is_tainted(rec.flags)


def test_wide_gap_skips_comparison():
    rec = health_check(_record(), _prev(anchor_date="2026-05-01"))
    assert rec.flags == []
    assert rec.compared_to is None
    assert "not compared" in rec.detail


# --- persistence ----------------------------------------------------------------

@pytest.fixture
def conn(tmp_path):
    c = storage.get_connection(tmp_path / "test.db")
    yield c
    c.close()


def test_fit_record_roundtrips_through_db(conn):
    rec = health_check(_record(converged=False), _prev())
    # store the prev first so keys differ
    storage.insert_hmm_fit(conn, _prev())
    storage.insert_hmm_fit(conn, rec)
    rows = storage.fetch_hmm_fits(conn, "SPY")
    back = record_from_row(rows[0])  # newest anchor first
    assert back.anchor_date == "2026-06-10"
    assert back.converged is False
    assert back.flags == rec.flags
    assert back.mu == rec.mu
    assert back.transition == rec.transition
    assert back.compared_to == "2026-06-09"


def test_insert_hmm_fit_is_idempotent(conn):
    assert storage.insert_hmm_fit(conn, _record()) is not None
    assert storage.insert_hmm_fit(conn, _record()) is None
    assert len(storage.fetch_hmm_fits(conn)) == 1


def test_latest_hmm_fit_before_is_strictly_point_in_time(conn):
    for d in ("2026-06-08", "2026-06-09", "2026-06-10"):
        storage.insert_hmm_fit(conn, _record(anchor_date=d))
    prev = storage.latest_hmm_fit_before(conn, "SPY", "2026-06-10")
    assert prev["anchor_date"] == "2026-06-09"  # same-day and future excluded
    assert storage.latest_hmm_fit_before(conn, "SPY", "2026-06-08") is None
    assert storage.latest_hmm_fit_before(conn, "QQQ", "2026-06-10") is None


_OLD_SCHEMA = """
CREATE TABLE forecasts (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    issued_at     TEXT    NOT NULL,
    forecaster    TEXT    NOT NULL,
    ticker        TEXT    NOT NULL,
    claim         TEXT    NOT NULL,
    horizon_days  INTEGER NOT NULL,
    anchor_date   TEXT    NOT NULL,
    anchor_close  REAL    NOT NULL,
    resolves_on   TEXT    NOT NULL,
    probability   REAL    NOT NULL,
    reasoning     TEXT,
    model         TEXT,
    status        TEXT    NOT NULL DEFAULT 'pending',
    outcome       REAL,
    resolved_close REAL,
    resolved_at   TEXT,
    brier         REAL,
    UNIQUE (forecaster, ticker, anchor_date, horizon_days)
);
"""


def test_migration_adds_fit_flags_to_pre_monitoring_db(tmp_path):
    db = tmp_path / "old.db"
    legacy = sqlite3.connect(db)
    legacy.executescript(_OLD_SCHEMA)
    legacy.execute(
        "INSERT INTO forecasts (issued_at, forecaster, ticker, claim, horizon_days,"
        " anchor_date, anchor_close, resolves_on, probability)"
        " VALUES ('2026-06-01T00:00:00+00:00', 'hmm_filter', 'SPY', 'c', 5,"
        " '2026-06-01', 100.0, '2026-06-08', 0.6)"
    )
    legacy.commit()
    legacy.close()

    conn = storage.get_connection(db)
    rows = storage.fetch_all(conn)
    assert rows[0]["fit_flags"] is None  # pre-monitoring rows read as unchecked
    storage.set_fit_flags(conn, rows[0]["id"], "non_converged")
    assert storage.fetch_all(conn)[0]["fit_flags"] == "non_converged"
    conn.close()


# --- end to end: a forced non-converged fit taints the logged forecast ----------

def test_non_converged_fit_marks_the_forecast_row(conn, monkeypatch):
    # Force EM to fail by capping iterations — the synthetic non-convergence case.
    orig = hmm_baseline.fit_hmm
    monkeypatch.setattr(
        hmm_baseline, "fit_hmm",
        lambda r, K, **kw: orig(r, K, iters=2, restarts=kw.get("restarts", 2),
                                seed=kw.get("seed", 0)),
    )

    r = _two_regime_returns(1300, seed=9)
    idx = pd.bdate_range("2020-01-01", periods=1301)
    closes = 100 * np.exp(np.concatenate([[0.0], np.cumsum(r)]))
    history = pd.DataFrame({"Close": closes}, index=idx)
    anchor = idx[-1].date()

    pred, rec = hmm_prediction_with_health("SPY", history, anchor, horizon=5)
    assert rec.converged is False

    health_check(rec, None)
    assert NON_CONVERGED in rec.flags
    storage.insert_hmm_fit(conn, rec)
    storage.insert_forecast(conn, Forecast(
        forecaster=config.HMM_FILTER, ticker="SPY", claim="c", horizon_days=5,
        anchor_date=rec.anchor_date, anchor_close=100.0, resolves_on="2026-06-17",
        probability=pred.probability, reasoning=pred.reasoning, model=pred.model,
        fit_flags=",".join(rec.flags),
    ))

    persisted = storage.fetch_all(conn)[0]
    assert parse_flags(persisted["fit_flags"]) == [NON_CONVERGED]
    summary = taint_summary(storage.fetch_all(conn))
    assert summary.flagged == 1
    lines = integrity_lines(summary)
    assert lines and "non_converged" in lines[0]


# --- leaderboard integrity aggregation ------------------------------------------

def _row(status="resolved", fit_flags=None, forecaster=config.HMM_FILTER):
    return {"forecaster": forecaster, "status": status, "fit_flags": fit_flags}


def test_taint_summary_counts_and_distinguishes_unmonitored():
    rows = [
        _row(fit_flags="non_converged"),                 # tainted, resolved
        _row(fit_flags=""),                              # checked, clean
        _row(status="pending", fit_flags=None),          # predates monitoring
        _row(status="pending", fit_flags="k_changed"),   # informational only
        _row(forecaster="pythia", fit_flags=None),       # other forecaster: ignored
    ]
    s = taint_summary(rows)
    assert (s.total, s.resolved) == (4, 2)
    assert (s.flagged, s.resolved_flagged) == (1, 1)
    assert s.unmonitored == 1
    assert dict(s.by_flag) == {"non_converged": 1}

    lines = integrity_lines(s)
    assert len(lines) == 2  # the flagged-share line + the unmonitored note
    assert "1/4" in lines[0]
    assert "predate" in lines[1]


def test_integrity_lines_silent_when_all_clean():
    rows = [_row(fit_flags=""), _row(status="pending", fit_flags="")]
    assert integrity_lines(taint_summary(rows)) == []
