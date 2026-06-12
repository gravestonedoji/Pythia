"""Offline tests for the self-review's pure parts (no network, no key)."""

from __future__ import annotations

import json

import pytest

from pythia import config
from pythia.reflect import build_review_context, lessons_sha, load_lessons, save_lessons


def _row(forecaster, p, outcome, brier, ticker="SPY", reasoning="momentum looked strong"):
    return {
        "forecaster": forecaster, "status": "resolved", "probability": p,
        "outcome": outcome, "brier": brier, "ticker": ticker,
        "anchor_date": "2026-06-01", "resolves_on": "2026-06-08",
        "reasoning": reasoning,
    }


def test_context_includes_leaderboard_and_worst_misses():
    rows = [
        _row(config.PYTHIA, 0.85, 0.0, 0.7225, reasoning="clear breakout"),
        _row(config.PYTHIA, 0.55, 1.0, 0.2025, ticker="QQQ"),
        _row("coin_flip", 0.50, 0.0, 0.25),
    ]
    ctx = build_review_context(rows)
    assert "2 resolved Pythia forecasts" in ctx
    assert "Coin flip" in ctx
    assert "WORST" in ctx
    # the worst miss (highest Brier) comes first, with its original reasoning
    assert ctx.index("said P=0.85") < ctx.index("said P=0.55")
    assert "clear breakout" in ctx


def test_context_uses_resolved_rows_only():
    rows = [
        _row(config.PYTHIA, 0.6, 1.0, 0.16),
        {**_row(config.PYTHIA, 0.9, None, None), "status": "pending"},
    ]
    ctx = build_review_context(rows)
    assert "1 resolved Pythia forecasts" in ctx


def test_context_requires_a_record():
    with pytest.raises(RuntimeError):
        build_review_context([_row("coin_flip", 0.5, 1.0, 0.25)])


def test_lessons_roundtrip_and_history(tmp_path):
    path = tmp_path / "lessons.txt"
    text = "- Cap confidence at 0.65 unless the trend is unbroken.\n- Respect 0.50."
    sha = save_lessons(text, n_resolved=30, path=path)
    assert sha == lessons_sha(text)
    loaded = load_lessons(path)
    assert loaded == (text, sha)
    hist = [json.loads(l) for l in
            (tmp_path / "lessons_history.jsonl").read_text(encoding="utf-8").splitlines()]
    assert hist[0]["sha"] == sha and hist[0]["n_resolved"] == 30


def test_load_lessons_handles_absence(tmp_path):
    assert load_lessons(tmp_path / "missing.txt") is None
    empty = tmp_path / "empty.txt"
    empty.write_text("", encoding="utf-8")
    assert load_lessons(empty) is None
