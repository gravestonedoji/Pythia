"""Email notifications for freshly issued forecasts (roadmap v3, pulled forward).

Surfacing-only: this emails a summary of the predictions Pythia just logged
(direction + P(up) + reasoning) so a human can see what was called. It stays on
the forecasting side of the hard wall — it never places or suggests a live
order beyond what the forecast already records.

Sending uses only the stdlib (``smtplib`` over STARTTLS, Gmail by default) and
is configured entirely through the ``PYTHIA_SMTP_*`` / ``PYTHIA_EMAIL_*``
environment variables (see ``.env.example``). If the config is incomplete,
``send_email`` is a graceful no-op so a forecast run is never blocked by alerting.
"""

from __future__ import annotations

import smtplib
import ssl
from dataclasses import dataclass
from email.message import EmailMessage

from . import config


@dataclass
class Prediction:
    """One Pythia call, reduced to what the email needs to render."""

    ticker: str
    probability: float  # P(up) over the horizon, 0..1
    anchor_date: str    # ISO date of the anchor session (when the call was made)
    anchor_close: float # the price at that time; the level the claim is measured against
    resolves_on: str    # ISO date the claim is graded
    reasoning: str


def _direction(probability: float) -> str:
    """The directional call implied by the probability (claim is 'close >= anchor')."""
    return "UP" if probability >= 0.5 else "DOWN"


def format_subject(predictions: list[Prediction], *, issued_on: str) -> str:
    n = len(predictions)
    ups = sum(1 for p in predictions if p.probability >= 0.5)
    return f"Pythia {issued_on}: {n} forecasts ({ups} up / {n - ups} down)"


def format_body(
    predictions: list[Prediction], *, issued_on: str, horizon_days: int
) -> str:
    """Plain-text summary, most bullish first (mirrors the `why` command order)."""
    lines = [
        f"Pythia predictions issued {issued_on}",
        f"P(up) = chance the ETF closes at or above its anchor price over "
        f"{horizon_days} trading sessions.",
        "'anchor' is the price when the call was made — what the claim is measured against.",
        "",
    ]
    for p in sorted(predictions, key=lambda r: -r.probability):
        lines.append(
            f"{_direction(p.probability):<4} {p.ticker:<5} "
            f"P(up) {p.probability * 100:5.1f}%   "
            f"anchor {p.anchor_close:.2f} ({p.anchor_date})   resolves {p.resolves_on}"
        )
        if p.reasoning:
            lines.append(f"     {p.reasoning}")
        lines.append("")
    lines.append("-- Pythia (forecasts only; no orders placed)")
    return "\n".join(lines)


def send_email(subject: str, body: str, cfg: config.EmailConfig | None = None) -> bool:
    """Send one plain-text email. Returns True if sent, False if not configured."""
    cfg = cfg or config.email_config()
    if cfg is None:
        return False

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = cfg.sender
    msg["To"] = cfg.recipient
    msg.set_content(body)

    context = ssl.create_default_context()
    with smtplib.SMTP(cfg.host, cfg.port) as server:
        server.starttls(context=context)
        server.login(cfg.user, cfg.password)
        server.send_message(msg)
    return True


def notify_predictions(
    predictions: list[Prediction],
    *,
    issued_on: str,
    horizon_days: int,
    cfg: config.EmailConfig | None = None,
) -> bool:
    """Email a batch summary. No-op (returns False) if there's nothing to send
    or email isn't configured."""
    if not predictions:
        return False
    subject = format_subject(predictions, issued_on=issued_on)
    body = format_body(predictions, issued_on=issued_on, horizon_days=horizon_days)
    return send_email(subject, body, cfg=cfg)
