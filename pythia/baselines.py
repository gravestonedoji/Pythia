"""The baseline ladder — trivial reference forecasters.

Each is scored on the *same* claims as Pythia so the review can answer not just
"did Pythia win?" but "what did it beat, and what did it fail to beat?":

1. coin_flip      — always 0.50. The floor. If Pythia can't beat this, something
                    is broken.
2. drift          — predicts "up" at the asset's historical positive-day ratio
                    (computed from data, never hardcoded). The real bar: broad
                    equities drift up, so beating this means extracting signal
                    beyond the market's natural long bias.
3. naive_momentum — predicts "up" if the last N sessions were net up. The most
                    revealing comparison: if it matches Pythia's Brier, the
                    expensive model is just doing momentum.

Every baseline produces the same claim ("close on resolution day >= anchor
close"), so its probability is P(up over the horizon).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from . import config, data


@dataclass
class BaselinePrediction:
    forecaster: str
    probability: float
    model: str  # rule descriptor stored in the `model` column
    reasoning: str


def coin_flip() -> BaselinePrediction:
    """No information: always 0.50."""
    return BaselinePrediction(
        forecaster="coin_flip",
        probability=0.50,
        model="baseline:coin_flip",
        reasoning="Always predicts 0.50 (no information). Brier is fixed at 0.25.",
    )


def drift(history: pd.DataFrame) -> BaselinePrediction:
    """Predict "up" at the historical positive-day ratio over the lookback."""
    ratio = data.positive_day_ratio(history, config.DRIFT_LOOKBACK_SESSIONS)
    return BaselinePrediction(
        forecaster="drift",
        probability=ratio,
        model=f"baseline:drift({config.DRIFT_LOOKBACK_SESSIONS}d up-ratio)",
        reasoning=(
            f"Historical up-day ratio over the last {config.DRIFT_LOOKBACK_SESSIONS} "
            f"sessions = {ratio:.4f}. Predicts up at that fixed rate, capturing the "
            f"market's natural long bias and nothing more."
        ),
    )


def naive_momentum(history: pd.DataFrame) -> BaselinePrediction:
    """Predict the direction of the last N sessions at a fixed confidence."""
    n = config.MOMENTUM_LOOKBACK_SESSIONS
    up = data.momentum_up(history, n)
    conf = config.MOMENTUM_CONFIDENCE
    probability = conf if up else 1.0 - conf
    direction = "up" if up else "down"
    return BaselinePrediction(
        forecaster="naive_momentum",
        probability=probability,
        model=f"baseline:momentum({n}d, conf={conf:.2f})",
        reasoning=(
            f"Net move over the last {n} sessions was {direction}; predicts {direction} "
            f"at fixed confidence {conf:.2f} (so P(up) = {probability:.2f})."
        ),
    )


def all_baselines(history: pd.DataFrame) -> list[BaselinePrediction]:
    """Run every baseline on the same price history."""
    return [coin_flip(), drift(history), naive_momentum(history)]
