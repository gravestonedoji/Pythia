"""High-conviction digest — the alerting half of roadmap v3 (pure, no I/O).

notify.py already emails every forecast batch; this module decides which calls
deserve a human's ATTENTION and builds the digest sections the email carries.
Surfacing-only, wall intact: a flagged call is surfaced for MANUAL review;
nothing here (or anywhere) places, prepares, or suggests the mechanics of an
order.

THE RULE: an LLM arm's own probability crossing |P - 0.5| >= ALERT_CONVICTION_MIN
flags the claim. Gate arms are the live LLM arms only (config.ALERT_GATE_ARMS):
iso arms are excluded because their conviction is a remap artifact at this
record size (the current PAV fit caps every output below 0.5 — Delphi measured
isotonic unreliable at n~200-400), and the bars are excluded because they grade
the product, they don't page the human (the HMM's confident calls graded 38.9%
hit on the early record).

HONESTY RULES:
- Flags are computed from rows READ BACK from the forecasts table, never from
  in-memory results — an alert must never carry a probability that matches no
  logged row.
- Every flag is LOGGED at issue time (digest_alerts, first-write-wins,
  live-only, no backfill) with the threshold and gate arms IN FORCE — the rule
  will drift over the project's life, and "what was actually surfaced" must be
  a point-in-time record, not a retro-derivation under today's config.
- Alert rows are written even when SMTP is unconfigured or --no-notify is
  passed: the record is about the rule; `emailed_at` records whether a human
  was actually notified.
- Alert outcomes are NEVER stored — the scoreboard derives them by joining to
  `forecasts`, so grading has exactly one path (scoring.py) and the alert
  record cannot diverge from the honest one.
- The scoreboard reports per-gating-arm columns alongside the pooled line: a
  3-arm OR-gate is three chances per claim, and a pooled number alone would
  inherit multiple-comparisons flattery.
- Correlation is surfaced inline: when a flagged ticker has a prior alert
  whose window still overlaps, the digest says so — a human reading four USO
  alerts in a row must not count them as four confirmations.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from . import config
from .scoring import is_hit


def flag_high_conviction(
    arm_probs: dict[str, float], *,
    threshold: float | None = None,
    gate_arms: list[str] | None = None,
) -> list[str]:
    """The gate arms whose own probability crosses the conviction line.

    Empty list = no flag. Order follows `gate_arms` (deterministic).
    """
    threshold = config.ALERT_CONVICTION_MIN if threshold is None else threshold
    gate_arms = config.ALERT_GATE_ARMS if gate_arms is None else gate_arms
    return [arm for arm in gate_arms
            if arm in arm_probs and abs(arm_probs[arm] - 0.5) >= threshold]


@dataclass
class FlaggedCall:
    """One flagged claim, assembled for the digest and the alert log."""

    ticker: str
    anchor_date: str
    horizon_days: int
    resolves_on: str
    anchor_close: float
    direction: str              # 'UP' | 'DOWN', from the most extreme gating arm
    probability: float          # that arm's p
    flagged_by: list[str]       # every gate arm that crossed the line
    arm_probs: dict[str, float] = field(default_factory=dict)  # gate arms' ps
    reasoning: str = ""         # the most extreme gating arm's reasoning
    repeat_of: str | None = None  # anchor_date of an overlapping prior alert


def build_flagged_calls(
    batch_rows, prior_alert_rows, *,
    threshold: float | None = None,
    gate_arms: list[str] | None = None,
) -> list[FlaggedCall]:
    """Flags for one batch of LOGGED forecast rows (all arms, several claims).

    `prior_alert_rows` are earlier digest_alerts rows; a prior alert whose
    window still overlaps a flagged claim's anchor sets `repeat_of` (the
    correlation caveat — same bet, not a new confirmation).
    """
    threshold = config.ALERT_CONVICTION_MIN if threshold is None else threshold
    gate_arms = config.ALERT_GATE_ARMS if gate_arms is None else gate_arms

    claims: dict[tuple[str, str, int], dict] = {}
    for r in batch_rows:
        key = (r["ticker"], r["anchor_date"], r["horizon_days"])
        c = claims.setdefault(key, {"rows": {}, "meta": r})
        c["rows"][r["forecaster"]] = r

    out: list[FlaggedCall] = []
    for key in sorted(claims):
        rows = claims[key]["rows"]
        meta = claims[key]["meta"]
        arm_probs = {arm: rows[arm]["probability"]
                     for arm in gate_arms if arm in rows}
        flagged = flag_high_conviction(arm_probs, threshold=threshold,
                                       gate_arms=gate_arms)
        if not flagged:
            continue
        # The headline direction/p comes from the MOST extreme gating arm;
        # ties break by gate-arm order (deterministic).
        lead = max(flagged, key=lambda a: (abs(arm_probs[a] - 0.5),
                                           -gate_arms.index(a)))
        p = arm_probs[lead]
        repeat_of = None
        for a in prior_alert_rows:
            if (a["ticker"] == meta["ticker"]
                    and a["anchor_date"] < meta["anchor_date"]
                    and a["resolves_on"] >= meta["anchor_date"]
                    and (repeat_of is None or a["anchor_date"] > repeat_of)):
                repeat_of = a["anchor_date"]
        out.append(FlaggedCall(
            ticker=meta["ticker"], anchor_date=meta["anchor_date"],
            horizon_days=meta["horizon_days"], resolves_on=meta["resolves_on"],
            anchor_close=meta["anchor_close"],
            direction="UP" if p >= 0.5 else "DOWN", probability=p,
            flagged_by=flagged, arm_probs=arm_probs,
            reasoning=rows[lead]["reasoning"] or "",
            repeat_of=repeat_of,
        ))
    return out


def flagged_from_alert_rows(alert_rows, forecast_rows) -> list[FlaggedCall]:
    """Rebuild FlaggedCalls VERBATIM from logged digest_alerts rows (re-sends).

    Direction, probability, and flagged_by come from the alert row exactly as
    logged — never recomputed under today's config. Forecast rows supply only
    display context (arm probs, reasoning, anchor close).
    """
    by_key: dict[tuple[str, str, str, int], dict] = {}
    for r in forecast_rows:
        by_key[(r["forecaster"], r["ticker"], r["anchor_date"],
                r["horizon_days"])] = r

    out: list[FlaggedCall] = []
    for a in alert_rows:
        flagged_by = a["flagged_by"].split(",")
        gate_arms = a["gate_arms"].split(",")
        arm_probs = {}
        anchor_close = 0.0
        reasoning = ""
        for arm in gate_arms:
            row = by_key.get((arm, a["ticker"], a["anchor_date"], a["horizon_days"]))
            if row is not None:
                arm_probs[arm] = row["probability"]
                anchor_close = row["anchor_close"]
        lead_row = by_key.get((flagged_by[0], a["ticker"], a["anchor_date"],
                               a["horizon_days"]))
        if lead_row is not None:
            reasoning = lead_row["reasoning"] or ""
        repeat_of = None
        for other in alert_rows:
            if (other["ticker"] == a["ticker"]
                    and other["anchor_date"] < a["anchor_date"]
                    and other["resolves_on"] >= a["anchor_date"]
                    and (repeat_of is None or other["anchor_date"] > repeat_of)):
                repeat_of = other["anchor_date"]
        out.append(FlaggedCall(
            ticker=a["ticker"], anchor_date=a["anchor_date"],
            horizon_days=a["horizon_days"], resolves_on=a["resolves_on"],
            anchor_close=anchor_close, direction=a["direction"],
            probability=a["probability"], flagged_by=flagged_by,
            arm_probs=arm_probs, reasoning=reasoning, repeat_of=repeat_of,
        ))
    return out


# --- the alert rule's own gradeable record ---------------------------------------

@dataclass
class AlertStats:
    n_resolved: int = 0
    n_correct: int = 0
    brier_sum: float = 0.0

    @property
    def hit_rate(self) -> float | None:
        return self.n_correct / self.n_resolved if self.n_resolved else None

    @property
    def avg_brier(self) -> float | None:
        return self.brier_sum / self.n_resolved if self.n_resolved else None


def alert_scoreboard(
    alert_rows, forecast_rows
) -> tuple[AlertStats, dict[str, AlertStats]]:
    """(pooled, per-gating-arm) record of every logged alert, derived by join.

    Outcomes come from the forecasts table only — one grading path. The pooled
    line counts each (alert, gating arm) pair, so it mixes up to three rules
    per claim; the per-arm breakdown is the honest denominator.
    """
    by_key: dict[tuple[str, str, str, int], dict] = {}
    for r in forecast_rows:
        by_key[(r["forecaster"], r["ticker"], r["anchor_date"],
                r["horizon_days"])] = r

    pooled = AlertStats()
    per_arm: dict[str, AlertStats] = defaultdict(AlertStats)
    for a in alert_rows:
        for arm in a["flagged_by"].split(","):
            row = by_key.get((arm, a["ticker"], a["anchor_date"],
                              a["horizon_days"]))
            if row is None or row["status"] != "resolved" or row["brier"] is None:
                continue
            for s in (pooled, per_arm[arm]):
                s.n_resolved += 1
                s.brier_sum += row["brier"]
                if is_hit(row["probability"], row["outcome"]):
                    s.n_correct += 1
    return pooled, dict(per_arm)


# --- rendering -------------------------------------------------------------------

_WALL_LINE = ("Surfaced for MANUAL human review only. Pythia places no orders; "
              "nothing in this email is an instruction to trade.")


def _fmt_stats(s: AlertStats) -> str:
    if s.n_resolved == 0:
        return "no resolved alerts yet"
    return (f"{s.n_resolved} resolved / {s.n_correct} correct "
            f"({s.hit_rate * 100:.0f}%) / avg Brier {s.avg_brier:.3f}")


def format_digest_sections(
    flagged: list[FlaggedCall],
    pooled: AlertStats,
    per_arm: dict[str, AlertStats],
    *,
    threshold: float | None = None,
    gate_arms: list[str] | None = None,
    reconstructed: bool = False,
) -> str:
    """The digest block prepended to the batch email body.

    Rendered on EVERY email (a missing section would be ambiguous — "no flags,
    or notify broke?"); on quiet days it says so explicitly. `reconstructed`
    labels a re-send of a pre-feature batch whose flags were recomputed for
    display and never logged.
    """
    threshold = config.ALERT_CONVICTION_MIN if threshold is None else threshold
    gate_arms = config.ALERT_GATE_ARMS if gate_arms is None else gate_arms

    rule = (f"|P(up) - 0.50| >= {threshold:.2f} on "
            f"{', '.join(config.FORECASTER_LABELS.get(a, a) for a in gate_arms)}")
    lines: list[str] = []
    if not flagged:
        lines += [f"No high-conviction calls today ({rule}).", ""]
        if pooled.n_resolved:
            lines += [f"Alert-rule record to date: {_fmt_stats(pooled)} "
                      "(coin = 0.250).", ""]
        return "\n".join(lines)

    header = f"{len(flagged)} HIGH-CONVICTION CALL{'S' if len(flagged) != 1 else ''}"
    if reconstructed:
        header += " (reconstructed for display under the current rule — not logged)"
    lines += [header, _WALL_LINE, f"Rule: {rule}", ""]
    for c in sorted(flagged, key=lambda c: -abs(c.probability - 0.5)):
        others = ", ".join(
            f"{a} {c.arm_probs[a] * 100:.0f}%" for a in c.arm_probs
            if a not in c.flagged_by)
        lines.append(
            f"{c.direction:<4} {c.ticker:<5} P(up) {c.probability * 100:5.1f}%  "
            f"[{', '.join(c.flagged_by)}]" + (f"  (others: {others})" if others else ""))
        lines.append(
            f"     anchor {c.anchor_close:.2f} ({c.anchor_date}) -> "
            f"resolves {c.resolves_on}")
        if c.repeat_of:
            lines.append(
                f"     ! overlapping window — same bet as the {c.repeat_of} "
                "alert, not a new confirmation")
        if c.reasoning:
            lines.append(f"     {c.reasoning}")
        lines.append("")

    lines.append(f"Alert-rule record to date (its out-of-sample test): "
                 f"{_fmt_stats(pooled)} (coin = 0.250).")
    for arm in sorted(per_arm):
        lines.append(f"  {arm}: {_fmt_stats(per_arm[arm])}")
    lines += ["", _WALL_LINE, ""]
    return "\n".join(lines)
