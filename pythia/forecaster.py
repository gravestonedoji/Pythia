"""The forecaster — Pythia's brain.

Sends recent price action to Claude (claude-opus-4-8) and gets back a
*calibrated* probability plus written reasoning, via a forced tool call so the
output is always structured.

v0 is price-only on purpose. The prompt forbids reasoning about macro (rates,
inflation, the Fed, news) because the model cannot see today's values and stale
pretraining macro is worse than none. This makes price-only Pythia a clean
ablation baseline: when v1 adds real macro (FRED), its Brier can be measured
against this version rather than assumed better.

v1 (macro.py) is now here as a SEPARATE arm, ``pythia_macro``: identical claim,
identical price data, the ONLY difference is a point-in-time macro block in the
prompt and a system prompt that lifts the price-only restriction. The raw
``pythia`` arm stays price-only forever, so the macro effect stays measured on
identical claims for the life of the project.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import date
from typing import Callable

import pandas as pd

from . import config, data

_SYSTEM_TEMPLATE = """\
You are Pythia, a disciplined short-horizon market forecaster. You produce \
*calibrated* probabilities for falsifiable claims about liquid, broad-based \
ETFs, using ONLY the price and volume data provided to you.

HARD RULES — follow them exactly:
- Reason ONLY from the price action, momentum, volume, and price structure in \
the data provided. That data is the entirety of your evidence.
- You must NOT reason about interest rates, inflation, CPI, the Fed, the yield \
curve, earnings, jobs numbers, elections, geopolitics, or any other macro or \
news. You do NOT have today's values for any of these. Any such knowledge from \
your training is STALE, and using it is worse than ignoring it: a confident \
guess about today's macro from old data is a trap. Do not use it or mention it.
- Be calibrated, not dramatic. {drift_prior} Reserve confident probabilities \
for genuinely strong, clear price signals. Avoid false precision.
- "probability" is your probability that the claim is TRUE over the stated \
horizon, as a number between 0 and 1.

Submit your answer by calling the submit_forecast tool. Keep the reasoning \
concise (a few sentences) and grounded only in the provided price action."""


# The v1 macro arm's contract: the price-only restriction is LIFTED and the
# model may reason from the macro block too — but ONLY from the values shown
# (no recalling macro from training, which may be stale), and still no news,
# events, or headlines (which it cannot see). Calibration discipline unchanged.
_SYSTEM_TEMPLATE_MACRO = """\
You are Pythia, a disciplined short-horizon market forecaster. You produce \
*calibrated* probabilities for falsifiable claims about liquid, broad-based \
ETFs, using the PRICE/VOLUME data AND the MACRO DATA BLOCK provided to you.

HARD RULES — follow them exactly:
- Reason from (1) the price action, momentum, volume, and price structure and \
(2) the macro readings in the MACRO DATA BLOCK (Treasury yields, the yield \
curve, breakeven inflation, credit spreads, volatility, and the dollar). These \
two blocks are the entirety of your evidence.
- Use ONLY the macro values shown in the block. Do NOT recall or infer macro \
readings from your training — they may be stale or wrong. The block is your \
single source for macro.
- You must NOT reason about specific news events, earnings, elections, jobs \
reports, central-bank decisions, or geopolitics. You do NOT have today's \
headlines. Any such knowledge from your training is STALE, and using it is \
worse than ignoring it: a confident guess about today's news from old data is \
a trap. Do not use it or mention it.
- Be calibrated, not dramatic. {drift_prior} Reserve confident probabilities \
for genuinely strong, clear signals across the price and macro data. Avoid \
false precision.
- "probability" is your probability that the claim is TRUE over the stated \
horizon, as a number between 0 and 1.

Submit your answer by calling the submit_forecast tool. Keep the reasoning \
concise (a few sentences) and grounded only in the provided price action and \
macro data."""


# The neutral directional prior depends on the asset class. Broad equity ETFs
# have a structural long-run up-drift (a data-free, non-macro fact); commodities,
# bonds, and crypto funds do not, so they must not be nudged upward.
_DRIFT_PRIOR = {
    config.EQUITY: (
        "If the price data shows no real edge, your probability should sit near "
        "this asset's natural drift: broad equity ETFs close up slightly more "
        "often than not, so a touch above 0.50 is the right neutral anchor."
    ),
    config.NON_EQUITY: (
        "If the price data shows no real edge, your probability should sit near "
        "0.50: this asset (a commodity, bond, or crypto fund) has NO reliable "
        "directional drift, so do not assume any upward bias — let the price "
        "action alone move you off 0.50, in either direction."
    ),
}


def build_system_prompt(asset_cls: str, *, macro: bool = False) -> str:
    """System prompt with the directional prior matched to the asset class.

    ``macro=True`` selects the v1 macro-arm prompt, which lifts the price-only
    restriction so the model may reason from the macro block too. The raw arm
    always uses the price-only prompt — the ablation stays clean.
    """
    prior = _DRIFT_PRIOR.get(asset_cls, _DRIFT_PRIOR[config.NON_EQUITY])
    tmpl = _SYSTEM_TEMPLATE_MACRO if macro else _SYSTEM_TEMPLATE
    return tmpl.format(drift_prior=prior)


def _build_tool(*, macro: bool = False) -> dict:
    """The submit_forecast tool, with the reasoning guardrail matched to the arm.

    The base arm's reasoning must stay price-only; the macro arm's may use the
    macro block. Both still forbid news/events/headlines (never visible).
    """
    reason_desc = (
        "Concise reasoning grounded ONLY in the provided macro data and price "
        "action. No news, events, earnings, or headlines."
        if macro else
        "Concise reasoning grounded ONLY in the provided price action. "
        "No macro, rates, inflation, or news."
    )
    return {
        "name": "submit_forecast",
        "description": "Record your calibrated probability and reasoning for the claim.",
        "input_schema": {
            "type": "object",
            "properties": {
                "probability": {
                    "type": "number",
                    "description": "Your P(claim is TRUE), a number between 0 and 1.",
                },
                "reasoning": {
                    "type": "string",
                    "description": reason_desc,
                },
            },
            "required": ["probability", "reasoning"],
        },
    }


# Kept for back-compat with any caller that wants the base tool directly.
_TOOL = _build_tool()


@dataclass
class ForecastResult:
    probability: float
    reasoning: str
    model: str


def _clamp_probability(p: float) -> float:
    """Keep probabilities away from 0/1, which imply impossible certainty."""
    return max(0.01, min(0.99, p))


def build_user_prompt(
    *,
    ticker: str,
    claim: str,
    horizon_days: int,
    anchor_date: date,
    anchor_close: float,
    resolves_on: date,
    price_context: str,
    macro_context: str | None = None,
) -> str:
    macro_block = f"\n\n{macro_context}" if macro_context else ""
    evidence = "price action and macro data" if macro_context else "price action"
    return (
        f"CLAIM TO FORECAST ({ticker}):\n"
        f'"{claim}"\n\n'
        f"Horizon: {horizon_days} trading sessions. Anchor session: "
        f"{anchor_date.isoformat()} (anchor close {anchor_close:.2f}). The claim "
        f"resolves on {resolves_on.isoformat()} using that day's closing price.\n\n"
        f"{price_context}{macro_block}\n\n"
        f"Reason only from the {evidence} above, then call submit_forecast with "
        "your probability that the claim is TRUE."
    )


def forecast(
    ticker: str,
    history: pd.DataFrame,
    *,
    claim: str,
    horizon_days: int,
    anchor_date: date,
    anchor_close: float,
    resolves_on: date,
    client=None,
    model: str | None = None,
    lessons: str | None = None,
    macro_context: str | None = None,
) -> ForecastResult:
    """Produce a calibrated forecast for `claim` from price history.

    ``lessons`` (the coached arm) appends the distilled self-review lessons to
    the system prompt — the only difference between the ``pythia`` and
    ``pythia_coached`` arms, so the coaching effect is cleanly measurable.

    ``macro_context`` (the v1 macro arm) switches to the macro system prompt
    and appends the point-in-time macro block to the user prompt — the only
    difference between ``pythia`` and ``pythia_macro``, so the macro effect is
    cleanly measurable. The two are independent; the cli runs them as separate
    arms and never combines them (an unmeasured combo would muddy the A/B).
    """
    model = model or config.MODEL_FORECAST

    is_macro = macro_context is not None
    user_prompt = build_user_prompt(
        ticker=ticker,
        claim=claim,
        horizon_days=horizon_days,
        anchor_date=anchor_date,
        anchor_close=anchor_close,
        resolves_on=resolves_on,
        price_context=data.format_price_context(history),
        macro_context=macro_context,
    )

    system = build_system_prompt(config.asset_class(ticker), macro=is_macro)
    if lessons:
        system += (
            "\n\nLESSONS FROM YOUR OWN GRADED RECORD — apply them to this "
            f"forecast:\n{lessons}"
        )

    # Subscription transport: same model, same prompts, different auth path —
    # stamped on the row so record slices stay attributable (see config.py).
    if config.transport() == "subscription":
        return forecast_via_subscription(system, user_prompt, model=model)

    if client is None:
        import anthropic

        client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

    response = client.messages.create(
        model=model,
        max_tokens=1024,
        system=system,
        tools=[_build_tool(macro=is_macro)],
        tool_choice={"type": "tool", "name": "submit_forecast"},
        messages=[{"role": "user", "content": user_prompt}],
    )

    payload = _extract_tool_input(response)
    probability = _clamp_probability(float(payload["probability"]))
    reasoning = str(payload["reasoning"]).strip()
    return ForecastResult(probability=probability, reasoning=reasoning, model=model)


def _extract_tool_input(response) -> dict:
    """Pull the submit_forecast tool input out of the SDK response."""
    for block in response.content:
        if getattr(block, "type", None) == "tool_use" and block.name == "submit_forecast":
            return block.input
    raise RuntimeError("Model did not return a submit_forecast tool call")


# --- Subscription transport (headless Claude Code) -------------------------------
# `claude -p` through the user's Claude Code login: the SAME model receives the
# SAME system + user prompts, billed to the plan instead of per token. The API
# transport forces a tool call for structure; the CLI has no tool_choice, so
# this path instructs strict JSON and parses it — refuse (raise) rather than
# fudge when the output doesn't parse. Isolation matters more than usual here:
# no tools, no settings sources (a CLAUDE.md leaking into the subject's prompt
# would silently change the experiment), a neutral cwd, and the metered API key
# stripped from the subprocess env (else the CLI would quietly bill it and the
# whole point of the transport is lost).

_TRANSPORT_TAG = "+via:claude-code"

_JSON_INSTRUCTION = (
    "\n\nOutput format: respond with ONLY a JSON object and no other text, "
    'exactly: {"probability": <number between 0 and 1>, '
    '"reasoning": "<a few concise sentences>"}'
)

# Injectable runner for offline tests: (cmd, env, cwd) -> completed-process-like.
CliRunner = Callable[[list, dict, str], "subprocess.CompletedProcess"]


def _subscription_cmd(system: str, user_prompt: str, model: str) -> list:
    """The exact argv for one headless forecast call (pure; unit-tested)."""
    cli = shutil.which(config.claude_cli())
    if cli is None:
        raise RuntimeError(
            f"Claude Code CLI {config.claude_cli()!r} not found on PATH "
            "(PYTHIA_TRANSPORT=subscription needs it; set PYTHIA_CLAUDE_CLI "
            "to its full path, or switch back to the api transport)")
    head = ["cmd", "/c", cli] if cli.lower().endswith((".cmd", ".bat")) else [cli]
    return [
        *head, "-p",
        "--model", model,
        "--output-format", "json",
        "--system-prompt", system,
        "--tools", "",              # the forecast is a pure text-in/text-out call
        "--setting-sources", "",    # no user/project settings, no CLAUDE.md leakage
        "--no-session-persistence",
        user_prompt + _JSON_INSTRUCTION,
    ]


def _subscription_env() -> dict:
    """Subprocess env with metered credentials stripped (pure; unit-tested).

    With a key present the CLI would silently bill the API instead of the
    subscription; stripping it makes the CLI fall back to the stored login.
    """
    env = dict(os.environ)
    env.pop("ANTHROPIC_API_KEY", None)
    env.pop("ANTHROPIC_AUTH_TOKEN", None)
    return env


def parse_forecast_json(text: str) -> tuple[float, str]:
    """Parse {"probability", "reasoning"} out of model text (pure; unit-tested).

    Accepts surrounding prose/code fences by falling back to the first JSON
    object in the text; raises (never guesses) when nothing valid parses.
    """
    candidates = [text]
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        candidates.append(m.group(0))
    for c in candidates:
        try:
            payload = json.loads(c)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(payload, dict) and "probability" in payload and "reasoning" in payload:
            return (_clamp_probability(float(payload["probability"])),
                    str(payload["reasoning"]).strip())
    raise RuntimeError(f"model output is not the required JSON: {text[:200]!r}")


def forecast_via_subscription(
    system: str, user_prompt: str, *, model: str, run: CliRunner | None = None
) -> ForecastResult:
    """One forecast through the Claude Code login (headless print mode)."""
    cmd = _subscription_cmd(system, user_prompt, model)

    def _default_run(argv, env, cwd):
        return subprocess.run(argv, capture_output=True, text=True,
                              encoding="utf-8", errors="replace",
                              timeout=300, env=env, cwd=cwd)

    run = run or _default_run
    # Neutral cwd: `claude` must not pick up any project's context.
    proc = run(cmd, _subscription_env(), tempfile.gettempdir())
    if proc.returncode != 0:
        raise RuntimeError(
            f"claude -p exited {proc.returncode}: "
            f"{(proc.stderr or proc.stdout or '').strip()[:300]}")
    envelope = json.loads(proc.stdout)
    if envelope.get("is_error"):
        raise RuntimeError(f"claude -p returned an error: "
                           f"{str(envelope.get('result'))[:300]}")
    probability, reasoning = parse_forecast_json(str(envelope.get("result", "")))
    return ForecastResult(probability=probability, reasoning=reasoning,
                          model=f"{model}{_TRANSPORT_TAG}")
