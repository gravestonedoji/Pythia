"""Offline tests for the isotonic calibration layer (no network, no key)."""

from __future__ import annotations

import numpy as np
import pytest

from pythia import config
from pythia.calibrate import IsotonicCalibrator, fit_isotonic


def test_pav_is_monotone_nondecreasing():
    rng = np.random.default_rng(0)
    x = rng.random(500)
    y = (rng.random(500) < x).astype(float)  # perfectly calibrated data
    ux, fy = fit_isotonic(x, y)
    assert np.all(np.diff(fy) >= -1e-12)


def test_pav_pools_violators():
    # y decreasing in x -> everything pools to the global mean
    x = np.array([0.1, 0.2, 0.3, 0.4])
    y = np.array([1.0, 0.8, 0.4, 0.0])
    _, fy = fit_isotonic(x, y)
    assert np.allclose(fy, y.mean())


def test_pav_recovers_calibrated_identity():
    rng = np.random.default_rng(1)
    x = np.repeat(np.linspace(0.1, 0.9, 9), 400)
    y = (rng.random(len(x)) < x).astype(float)
    ux, fy = fit_isotonic(x, y)
    # the fitted curve should track the diagonal closely
    assert np.max(np.abs(fy - ux)) < 0.06


def _rows(n: int, p: float, hit_rate: float, forecaster: str = config.PYTHIA):
    rng = np.random.default_rng(42)
    return [
        {"forecaster": forecaster, "status": "resolved", "probability": p,
         "outcome": 1.0 if rng.random() < hit_rate else 0.0}
        for _ in range(n)
    ]


def test_calibrator_gates_on_minimum_data():
    rows = _rows(config.ISO_MIN_RESOLVED - 1, 0.7, 0.6)
    assert IsotonicCalibrator.from_resolved_rows(rows, config.PYTHIA) is None


def test_calibrator_ignores_other_forecasters_and_pending():
    rows = _rows(config.ISO_MIN_RESOLVED + 50, 0.7, 0.6, forecaster="hmm_filter")
    assert IsotonicCalibrator.from_resolved_rows(rows, config.PYTHIA) is None


def test_calibrator_deflates_overconfidence():
    # forecaster says 0.85 but is right only 60% of the time -> remap shrinks it
    rows = _rows(100, 0.85, 0.60) + _rows(100, 0.30, 0.45)
    cal = IsotonicCalibrator.from_resolved_rows(rows, config.PYTHIA)
    assert cal is not None and cal.n_fit == 200
    assert cal.apply(0.85) < 0.70
    assert 0.01 <= cal.apply(0.0) <= cal.apply(1.0) <= 0.99  # clamped + monotone


def test_calibrator_records_fit_size():
    rows = _rows(config.ISO_MIN_RESOLVED, 0.6, 0.55)
    cal = IsotonicCalibrator.from_resolved_rows(rows, config.PYTHIA)
    assert cal is not None and cal.n_fit == config.ISO_MIN_RESOLVED
