"""HMM fit-health monitoring — the referee gets a referee.

The quant bar (hmm_baseline.py) is deploy gate #1, so a silently broken or
unstable fit corrupts every leaderboard comparison against it. Published
practitioner work on HMMs over real SPY data found two failure modes: EM
failing to converge on ~47% of rolling windows, and fitted parameters swinging
wildly between refits (e.g. the high-vol regime's expected duration flipping
from days to months) — silently changing what the baseline believes.

This module detects and SURFACES both. It never changes what the HMM emits:
the policy (decided 2026-06-12) is log + flag — a tainted value stays on the
leaderboard, visibly annotated, because dropping it would shrink the record
exactly on the turbulent days where fitting is hard, flattering the bar and
biasing the deploy gate.

Point-in-time discipline: a fit's health is judged against its own data and
the PREVIOUS recorded fit only — nothing dated after the anchor enters any
check. Thresholds live in config.py.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime, timezone

from . import config


def _now_iso() -> str:
    # storage.now_iso duplicated here: storage imports this module for the
    # FitRecord type, so importing back would be a cycle.
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

# Flag identifiers, stored comma-joined on both the hmm_fits row and the
# forecast row's fit_flags column ('' = checked and clean, NULL = unchecked).
NON_CONVERGED = "non_converged"            # EM exhausted iters without meeting tol
DURATION_JUMP = "duration_jump"            # expected regime duration ratio tripped
SIGMA_JUMP = "sigma_jump"                  # per-state vol ratio tripped
MU_JUMP = "mu_jump"                        # per-state mean shift tripped
K_CHANGED = "k_changed"                    # state count differs (informational:
#                                            the 2<->3 switch at the rich-history
#                                            threshold is by design, not a fault)
RECONSTRUCTION_MISMATCH = "reconstruction_mismatch"  # retro-annotation refit did
#                                            not reproduce the logged probability,
#                                            so this health record describes a
#                                            different fit than the logged value

# Flags that taint the day's hmm_filter value on the leaderboard.
TAINTING_FLAGS = (NON_CONVERGED, DURATION_JUMP, SIGMA_JUMP, MU_JUMP,
                  RECONSTRUCTION_MISMATCH)


@dataclass
class FitRecord:
    """Everything worth remembering about one (ticker, anchor_date) fit."""

    ticker: str
    anchor_date: str          # ISO — the fit's as-of date (point-in-time key)
    converged: bool
    n_iter: int
    loglik: float
    n_states: int
    n_obs: int
    window_start: str         # ISO — first close in the fit window
    window_end: str           # ISO — last close in the fit window (<= anchor)
    mu: list[float]           # per-state mean daily log-return (fit order)
    sig: list[float]          # per-state std (fit order)
    transition: list[list[float]]
    pi: list[float]
    flags: list[str] = field(default_factory=list)
    compared_to: str | None = None  # previous fit's anchor_date, when compared
    detail: str | None = None       # human-readable story behind the flags
    fitted_at: str = field(default_factory=_now_iso)

    def params_json(self) -> str:
        return json.dumps({"mu": self.mu, "sig": self.sig,
                           "transition": self.transition, "pi": self.pi})


def record_from_row(row) -> FitRecord:
    """Rebuild a FitRecord from an hmm_fits DB row (sqlite3.Row or mapping)."""
    params = json.loads(row["params"])
    return FitRecord(
        ticker=row["ticker"], anchor_date=row["anchor_date"],
        converged=bool(row["converged"]), n_iter=row["n_iter"],
        loglik=row["loglik"], n_states=row["n_states"], n_obs=row["n_obs"],
        window_start=row["window_start"], window_end=row["window_end"],
        mu=params["mu"], sig=params["sig"],
        transition=params["transition"], pi=params["pi"],
        flags=parse_flags(row["flags"]), compared_to=row["compared_to"],
        detail=row["detail"], fitted_at=row["fitted_at"],
    )


def parse_flags(stored: str | None) -> list[str]:
    """Comma-joined DB flags back to a list ('' and NULL both mean none)."""
    if not stored:
        return []
    return [f for f in stored.split(",") if f]


def is_tainted(flags: list[str]) -> bool:
    """Whether these flags disqualify the fit from being trusted silently."""
    return any(f in TAINTING_FLAGS for f in flags)


def expected_durations(transition: list[list[float]]) -> list[float]:
    """Expected sessions spent in each state per visit: 1 / (1 - T_kk)."""
    return [1.0 / max(1.0 - transition[k][k], 1e-9) for k in range(len(transition))]


def _sigma_order(record: FitRecord) -> list[int]:
    """State indices sorted by vol — labels are arbitrary across refits, so
    regimes are matched by their identifying parameter (calm first)."""
    return sorted(range(record.n_states), key=lambda k: record.sig[k])


def health_check(record: FitRecord, prev: FitRecord | None) -> FitRecord:
    """Fill in flags / compared_to / detail on a fresh record, in place.

    Convergence is judged from the fit itself; stability against `prev`, which
    the caller must guarantee is the most recent fit with a STRICTLY EARLIER
    anchor date (the only past this check is allowed to see).
    """
    parts: list[str] = []

    if not record.converged:
        record.flags.append(NON_CONVERGED)
        parts.append(f"EM used all {record.n_iter} iterations without meeting tolerance")

    if prev is None:
        parts.append("first recorded fit for this ticker — no stability comparison")
    else:
        gap = (date.fromisoformat(record.anchor_date)
               - date.fromisoformat(prev.anchor_date)).days
        if gap > config.HMM_STABILITY_MAX_GAP_DAYS:
            parts.append(
                f"previous fit ({prev.anchor_date}) is {gap}d older than the "
                f"{config.HMM_STABILITY_MAX_GAP_DAYS}d comparison window — not compared"
            )
        elif record.n_states != prev.n_states:
            record.compared_to = prev.anchor_date
            record.flags.append(K_CHANGED)
            parts.append(f"K changed {prev.n_states} -> {record.n_states} "
                         "(by-design history threshold; informational)")
        else:
            record.compared_to = prev.anchor_date
            new_o, old_o = _sigma_order(record), _sigma_order(prev)
            new_d, old_d = expected_durations(record.transition), expected_durations(prev.transition)
            for s, (i, j) in enumerate(zip(new_o, old_o)):
                d_new, d_old = new_d[i], old_d[j]
                ratio = max(d_new, d_old) / max(min(d_new, d_old), 1e-9)
                if ratio > config.HMM_DURATION_JUMP_RATIO:
                    if DURATION_JUMP not in record.flags:
                        record.flags.append(DURATION_JUMP)
                    parts.append(f"state{s} expected duration {d_old:.1f}d -> {d_new:.1f}d "
                                 f"({ratio:.1f}x > {config.HMM_DURATION_JUMP_RATIO:.1f}x)")
                s_new, s_old = record.sig[i], prev.sig[j]
                ratio = max(s_new, s_old) / max(min(s_new, s_old), 1e-12)
                if ratio > config.HMM_SIGMA_JUMP_RATIO:
                    if SIGMA_JUMP not in record.flags:
                        record.flags.append(SIGMA_JUMP)
                    parts.append(f"state{s} sig {s_old * 100:.2f}% -> {s_new * 100:.2f}% "
                                 f"({ratio:.1f}x > {config.HMM_SIGMA_JUMP_RATIO:.1f}x)")
                shift = abs(record.mu[i] - prev.mu[j])
                if shift > config.HMM_MU_JUMP_ABS:
                    if MU_JUMP not in record.flags:
                        record.flags.append(MU_JUMP)
                    parts.append(f"state{s} mu {prev.mu[j] * 100:+.3f}%/d -> "
                                 f"{record.mu[i] * 100:+.3f}%/d "
                                 f"(|shift| > {config.HMM_MU_JUMP_ABS * 100:.2f}%/d)")
            if not record.flags:
                parts.append(f"stable vs {prev.anchor_date}")

    record.detail = "; ".join(parts)
    return record


# --- leaderboard integrity (review report) -------------------------------------

@dataclass
class TaintSummary:
    """How much of the hmm_filter record rests on flagged fits."""

    total: int = 0             # all hmm_filter forecast rows
    resolved: int = 0
    flagged: int = 0           # rows whose fit carries a TAINTING flag
    resolved_flagged: int = 0
    unmonitored: int = 0       # rows that predate health monitoring (NULL flags)
    by_flag: Counter = field(default_factory=Counter)


def taint_summary(rows) -> TaintSummary:
    """Aggregate fit_flags over forecast rows (NULL = unchecked, '' = clean)."""
    s = TaintSummary()
    for row in rows:
        if row["forecaster"] != config.HMM_FILTER:
            continue
        s.total += 1
        resolved = row["status"] == "resolved"
        if resolved:
            s.resolved += 1
        stored = row["fit_flags"]
        if stored is None:
            s.unmonitored += 1
            continue
        flags = parse_flags(stored)
        if is_tainted(flags):
            s.flagged += 1
            if resolved:
                s.resolved_flagged += 1
        for f in flags:
            if f in TAINTING_FLAGS:
                s.by_flag[f] += 1
    return s


def integrity_lines(s: TaintSummary) -> list[str]:
    """Plain-text annotations for the review report (empty list = all clean)."""
    lines: list[str] = []
    if s.flagged:
        breakdown = ", ".join(f"{flag} x{n}" for flag, n in s.by_flag.most_common())
        resolved_part = (
            f"{s.resolved_flagged}/{s.resolved} resolved "
            f"({s.resolved_flagged / s.resolved * 100:.0f}%)"
            if s.resolved else "0 resolved"
        )
        lines.append(
            f"HMM bar: {s.flagged}/{s.total} values "
            f"({s.flagged / s.total * 100:.0f}%) from flagged fits — "
            f"{resolved_part}; {breakdown}. Run `pythia hmm-health` for detail."
        )
    if s.unmonitored:
        lines.append(
            f"{s.unmonitored} hmm_filter rows predate health monitoring — "
            "run `pythia hmm-health --backfill` to annotate them."
        )
    return lines
