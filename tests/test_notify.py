"""Offline tests for the email notifier — pure formatting + config, no network.

``send_email`` itself (the SMTP call) isn't exercised here; these cover the
parts that must be correct without a mail server: direction logic, the rendered
body/subject, and that email config is a clean no-op when unset.
"""

from __future__ import annotations

import pytest

from pythia import config, notify


def _preds() -> list[notify.Prediction]:
    return [
        notify.Prediction(ticker="SPY", probability=0.62, resolves_on="2026-06-09",
                           reasoning="Uptrend intact."),
        notify.Prediction(ticker="TLT", probability=0.41, resolves_on="2026-06-09",
                           reasoning="Rolling over."),
        notify.Prediction(ticker="GLD", probability=0.50, resolves_on="2026-06-09",
                           reasoning="No edge."),
    ]


def test_direction_threshold():
    # The claim is "close >= anchor", so exactly 0.50 counts as UP, below is DOWN.
    assert notify._direction(0.50) == "UP"
    assert notify._direction(0.501) == "UP"
    assert notify._direction(0.499) == "DOWN"


def test_subject_counts_up_and_down():
    subject = notify.format_subject(_preds(), issued_on="2026-06-01")
    # 0.62 and 0.50 are UP, 0.41 is DOWN.
    assert subject == "Pythia 2026-06-01: 3 forecasts (2 up / 1 down)"


def test_body_sorted_most_bullish_first_and_includes_fields():
    body = notify.format_body(_preds(), issued_on="2026-06-01", horizon_days=5)
    assert "5 trading sessions" in body
    # Highest probability (SPY) renders before the lowest (TLT).
    assert body.index("SPY") < body.index("GLD") < body.index("TLT")
    # Direction + reasoning both present.
    assert "UP" in body and "DOWN" in body
    assert "Uptrend intact." in body
    # No live-order language leaks in — it's surfacing only.
    assert "no orders placed" in body


def test_email_config_none_when_unset(monkeypatch):
    for var in ("PYTHIA_SMTP_USER", "PYTHIA_SMTP_PASSWORD"):
        monkeypatch.delenv(var, raising=False)
    assert config.email_config() is None


def test_email_config_defaults_sender_and_recipient_to_user(monkeypatch):
    monkeypatch.setenv("PYTHIA_SMTP_USER", "me@example.com")
    monkeypatch.setenv("PYTHIA_SMTP_PASSWORD", "app-pw")
    for var in ("PYTHIA_EMAIL_TO", "PYTHIA_EMAIL_FROM", "PYTHIA_SMTP_HOST", "PYTHIA_SMTP_PORT"):
        monkeypatch.delenv(var, raising=False)
    cfg = config.email_config()
    assert cfg is not None
    assert cfg.sender == "me@example.com"
    assert cfg.recipient == "me@example.com"
    assert cfg.host == config.SMTP_HOST_DEFAULT
    assert cfg.port == config.SMTP_PORT_DEFAULT


def test_notify_predictions_no_send_when_unconfigured(monkeypatch):
    # With no config, notify_predictions must be a no-op (False), not a crash.
    monkeypatch.delenv("PYTHIA_SMTP_USER", raising=False)
    monkeypatch.delenv("PYTHIA_SMTP_PASSWORD", raising=False)
    assert notify.notify_predictions(_preds(), issued_on="2026-06-01", horizon_days=5) is False


def test_notify_predictions_empty_is_false():
    assert notify.notify_predictions([], issued_on="2026-06-01", horizon_days=5) is False
