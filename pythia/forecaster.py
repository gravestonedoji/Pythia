"""The forecaster — Pythia's brain.

Sends recent price action to Claude (claude-opus-4-8) and gets back a
*calibrated* probability plus written reasoning, via a forced tool call so the
output is always structured.

v0 is price-only on purpose. The prompt forbids reasoning about macro (rates,
inflation, the Fed, news) because the model cannot see today's values and stale
pretraining macro is worse than none. This makes price-only Pythia a clean
ablation baseline: when v1 adds real macro (FRED), its Brier can be measured
against this version rather than assumed better.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

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


def build_system_prompt(asset_cls: str) -> str:
    """System prompt with the directional prior matched to the asset class."""
    prior = _DRIFT_PRIOR.get(asset_cls, _DRIFT_PRIOR[config.NON_EQUITY])
    return _SYSTEM_TEMPLATE.format(drift_prior=prior)

_TOOL = {
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
                "description": (
                    "Concise reasoning grounded ONLY in the provided price action. "
                    "No macro, rates, inflation, or news."
                ),
            },
        },
        "required": ["probability", "reasoning"],
    },
}


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
) -> str:
    return (
        f"CLAIM TO FORECAST ({ticker}):\n"
        f'"{claim}"\n\n'
        f"Horizon: {horizon_days} trading sessions. Anchor session: "
        f"{anchor_date.isoformat()} (anchor close {anchor_close:.2f}). The claim "
        f"resolves on {resolves_on.isoformat()} using that day's closing price.\n\n"
        f"{price_context}\n\n"
        "Reason only from the price action above, then call submit_forecast with "
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
) -> ForecastResult:
    """Produce a calibrated forecast for `claim` from price history alone.

    ``lessons`` (the coached arm) appends the distilled self-review lessons to
    the system prompt — this is the only difference between the ``pythia`` and
    ``pythia_coached`` arms, so the coaching effect is cleanly measurable.
    """
    model = model or config.MODEL_FORECAST
    if client is None:
        import anthropic

        client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

    user_prompt = build_user_prompt(
        ticker=ticker,
        claim=claim,
        horizon_days=horizon_days,
        anchor_date=anchor_date,
        anchor_close=anchor_close,
        resolves_on=resolves_on,
        price_context=data.format_price_context(history),
    )

    system = build_system_prompt(config.asset_class(ticker))
    if lessons:
        system += (
            "\n\nLESSONS FROM YOUR OWN GRADED RECORD — apply them to this "
            f"forecast:\n{lessons}"
        )

    response = client.messages.create(
        model=model,
        max_tokens=1024,
        system=system,
        tools=[_TOOL],
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
