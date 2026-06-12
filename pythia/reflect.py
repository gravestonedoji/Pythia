"""Weekly self-review — the prose half of the Delphi-validated recipe.

Opus reads Pythia's own GRADED record (resolved claims only: probabilities,
outcomes, Brier vs the baselines, the worst misses with their original
reasoning) and distills a short list of lessons. The lessons are stored in
``lessons.txt`` and injected into the COACHED forecast arm's system prompt
(forecaster identity ``pythia_coached``) — the raw ``pythia`` arm never sees
them, so the coaching effect stays measurable forever on identical claims.

Why this design is trusted: Delphi's paired-seed race (Delphi summary.md §8)
showed distilled lessons helped in 10/10 worlds (+0.017 cal err) and that
lessons written by one model coach another — the earlier "lessons oscillate"
scare was path-luck confounding. The race-winning lessons were humility-shaped;
the review prompt below asks for exactly that kind (decision rules, not market
narratives).

Honesty rules:
- The review sees only RESOLVED claims (outcomes already known to everyone);
  it runs AFTER grading, never before, so no information leaks into any open
  forecast.
- Every lessons version is content-hashed; the coached arm logs the hash in its
  `model` column, so any slice of the record can be tied to the exact lessons
  text it used. History accumulates in ``lessons_history.jsonl``.
- No macro, no news: lessons must be about FORECASTING BEHAVIOR (confidence,
  base rates, signal quality), and the prompt says so — the price-only
  ablation stays clean.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from . import config, scoring

_REVIEW_SYSTEM = """\
You are the forecasting coach for Pythia, a short-horizon ETF forecaster that
predicts P(close in N sessions >= today's close) from price action alone.

You will be shown Pythia's GRADED track record: its calibration vs reference
forecasters, and its worst misses with the original reasoning. Your job is to
distill LESSONS that will make its future probabilities better CALIBRATED.

HARD RULES for the lessons you write:
- Lessons must be about forecasting BEHAVIOR: confidence levels, base rates,
  when a price signal deserves conviction and when it does not. They are
  decision rules, not market commentary.
- NO macro, news, rates, or events — Pythia cannot see them, and mentioning
  them would corrupt a controlled experiment.
- No predictions, no ticker-specific rules (the record is too short to support
  them) — only rules that generalize across assets.
- Be concrete and checkable ("when X, cap confidence at Y"), not platitudes.
- At most {max_lessons} lessons, each a single imperative sentence.

Submit them by calling the submit_lessons tool."""

_REVIEW_TOOL = {
    "name": "submit_lessons",
    "description": "Record the distilled forecasting lessons.",
    "input_schema": {
        "type": "object",
        "properties": {
            "diagnosis": {
                "type": "string",
                "description": "2-3 sentences: the main calibration failure you see in the record.",
            },
            "lessons": {
                "type": "array",
                "items": {"type": "string"},
                "description": "The distilled lessons, one imperative sentence each.",
            },
        },
        "required": ["diagnosis", "lessons"],
    },
}


# --- pure context building (offline, unit-tested) ------------------------------

def build_review_context(rows, *, max_misses: int = 10) -> str:
    """Render the graded record into the review prompt's evidence block.

    Pure: takes DB rows, returns text. Only resolved rows are used.
    """
    resolved = [r for r in rows if r["status"] == "resolved" and r["outcome"] is not None]
    mine = sorted((r for r in resolved if r["forecaster"] == config.PYTHIA),
                  key=lambda r: r["brier"], reverse=True)
    if not mine:
        raise RuntimeError("no resolved Pythia forecasts to review")

    stats = scoring.summarize(resolved)
    lines = [f"GRADED RECORD ({len(mine)} resolved Pythia forecasts)", ""]
    lines.append("Leaderboard (avg Brier, lower is better; 0.25 = always saying 0.50):")
    for fc, s in sorted(stats.items(), key=lambda kv: kv[1].avg_brier or 9):
        if s.resolved:
            label = config.FORECASTER_LABELS.get(fc, fc)
            lines.append(f"  {label:<24} n={s.resolved:<4} avg Brier {s.avg_brier:.4f}  "
                         f"hit-rate {s.hit_rate * 100:.0f}%")

    probs = [r["probability"] for r in mine]
    bold = sum(1 for p in probs if abs(p - 0.5) >= 0.15)
    bold_hits = sum(1 for r in mine if abs(r["probability"] - 0.5) >= 0.15
                    and scoring.is_hit(r["probability"], r["outcome"]))
    lines.append("")
    lines.append(f"Confidence profile: mean |P - 0.50| = "
                 f"{sum(abs(p - 0.5) for p in probs) / len(probs):.3f}; "
                 f"{bold} calls at >=15 points from neutral, of which "
                 f"{bold_hits} were directionally right.")

    lines.append("")
    lines.append(f"WORST {min(max_misses, len(mine))} MISSES "
                 "(highest Brier; your reasoning at the time, then what happened):")
    for r in mine[:max_misses]:
        outcome = "TRUE" if r["outcome"] == 1.0 else "FALSE"
        lines.append(f"\n- {r['ticker']} {r['anchor_date']} -> {r['resolves_on']}: "
                     f"said P={r['probability']:.2f}, claim came {outcome} "
                     f"(Brier {r['brier']:.3f})")
        if r["reasoning"]:
            lines.append(f"  reasoning then: {r['reasoning'][:400]}")
    return "\n".join(lines)


def lessons_sha(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()[:10]


def save_lessons(text: str, *, n_resolved: int,
                 path: Path | None = None) -> str:
    """Write lessons.txt (the live version) and append to the history log."""
    path = path or config.LESSONS_PATH
    path.write_text(text, encoding="utf-8")
    sha = lessons_sha(text)
    hist = path.with_name("lessons_history.jsonl")
    with open(hist, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({
            "sha": sha, "saved_at": storage_now(), "n_resolved": n_resolved,
            "lessons": text,
        }) + "\n")
    return sha


def storage_now() -> str:
    from .storage import now_iso
    return now_iso()


def load_lessons(path: Path | None = None) -> tuple[str, str] | None:
    """The live lessons (text, sha), or None if no review has run yet."""
    path = path or config.LESSONS_PATH
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return None
    return text, lessons_sha(text)


# --- the review call ------------------------------------------------------------

def distill_lessons(rows, client=None, *, model: str | None = None) -> tuple[str, str]:
    """Run the self-review on the graded record. Returns (diagnosis, lessons text)."""
    model = model or config.MODEL_REVIEW
    if client is None:
        import anthropic
        client = anthropic.Anthropic()

    context = build_review_context(rows)
    response = client.messages.create(
        model=model,
        max_tokens=2048,
        system=_REVIEW_SYSTEM.format(max_lessons=config.REFLECT_MAX_LESSONS),
        tools=[_REVIEW_TOOL],
        tool_choice={"type": "tool", "name": "submit_lessons"},
        messages=[{"role": "user", "content": context}],
    )
    payload = None
    for block in response.content:
        if getattr(block, "type", None) == "tool_use" and block.name == "submit_lessons":
            payload = block.input
            break
    if payload is None:
        raise RuntimeError("review model did not return a submit_lessons call")
    lessons = [str(s).strip() for s in payload["lessons"] if str(s).strip()]
    lessons = lessons[: config.REFLECT_MAX_LESSONS]
    text = "\n".join(f"- {s}" for s in lessons)
    return str(payload.get("diagnosis", "")).strip(), text
