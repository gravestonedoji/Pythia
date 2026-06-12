"""Isotonic calibration — the statistical half of the Delphi-validated recipe.

Delphi's paired race (Delphi summary.md §8) showed an LLM forecaster is a
confident overclaimer whose value is unlocked by correction layers: prose
lessons (reflect.py) AND a monotonic remapping of its raw probabilities fit on
its own graded record. The two STACK, and the math half cannot oscillate — it
improves monotonically as resolved claims accumulate.

Honesty rules:
- Strictly point-in-time: a corrected probability logged today is produced by
  a calibrator fitted ONLY on claims already RESOLVED today. The fit size is
  recorded in the row's `model` column so any future reader can see exactly
  what the correction knew.
- Each derived column fits on ITS OWN base forecaster's record (pythia_iso on
  pythia's resolved rows, pythia_coached_iso on pythia_coached's) — never on a
  proxy.
- Minimum-data gate: below ISO_MIN_RESOLVED resolved rows the calibrator
  overfits noise (Delphi measured this: LOSO isotonic HURT at ~200-400 training
  points and shone at ~1,700), so the derived row is simply skipped. A missing
  row beats a fudged one.
- Fit target is the realized outcome (0/1) — the only truth real markets give.

Implementation: classic pool-adjacent-violators (PAV), numpy only, ported from
Delphi's calibrate.py. Pure and unit-tested offline (tests/test_calibrate.py).
"""

from __future__ import annotations

import numpy as np

from . import config

PROB_FLOOR, PROB_CEIL = 0.01, 0.99


def fit_isotonic(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """PAV: least-squares nondecreasing fit of y on x.
    Returns (unique_x, fitted_y) — predict via np.interp(p, unique_x, fitted_y)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    order = np.argsort(x)
    xs, ys = x[order], y[order]
    ux, inv = np.unique(xs, return_inverse=True)
    means = np.bincount(inv, weights=ys) / np.bincount(inv)
    weights = np.bincount(inv).astype(float)

    # pool adjacent violators over the unique-x blocks
    val: list[float] = []
    wt: list[float] = []
    cnt: list[int] = []  # how many unique-x points each block spans
    for m, w in zip(means, weights):
        val.append(float(m)); wt.append(float(w)); cnt.append(1)
        while len(val) > 1 and val[-2] > val[-1]:
            merged = (val[-2] * wt[-2] + val[-1] * wt[-1]) / (wt[-2] + wt[-1])
            wt[-2] += wt[-1]; cnt[-2] += cnt[-1]; val[-2] = merged
            del val[-1], wt[-1], cnt[-1]
    fitted = np.repeat(val, cnt)
    return ux, fitted


class IsotonicCalibrator:
    def __init__(self, ux: np.ndarray, fy: np.ndarray, n_fit: int):
        self.ux = np.asarray(ux, float)
        self.fy = np.asarray(fy, float)
        self.n_fit = n_fit  # resolved rows the fit saw (recorded per derived row)

    @classmethod
    def from_resolved_rows(cls, rows, forecaster: str) -> "IsotonicCalibrator | None":
        """Fit on one forecaster's resolved (probability, outcome) record.

        Returns None below the minimum-data gate — the caller skips the derived
        column rather than logging an overfit correction.
        """
        pts = [(r["probability"], r["outcome"]) for r in rows
               if r["forecaster"] == forecaster and r["status"] == "resolved"
               and r["outcome"] is not None]
        if len(pts) < config.ISO_MIN_RESOLVED:
            return None
        x = np.array([p for p, _ in pts])
        y = np.array([o for _, o in pts])
        return cls(*fit_isotonic(x, y), n_fit=len(pts))

    def apply(self, p: float) -> float:
        out = float(np.interp(p, self.ux, self.fy))
        return max(PROB_FLOOR, min(PROB_CEIL, out))
