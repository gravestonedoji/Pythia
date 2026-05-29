"""SQLite persistence for the forecast track record.

One table, ``forecasts``, holds every call from Pythia *and* each baseline, so
they can be graded side by side on identical claims. A row is written at issue
time with ``status='pending'`` and updated in place when it resolves.

The ``(forecaster, ticker, anchor_date, horizon_days)`` uniqueness constraint
makes the daily loop idempotent: re-running ``forecast`` the same day is a
no-op rather than a duplicate.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path

from . import config


def now_iso() -> str:
    """Current UTC timestamp, second precision, ISO-8601."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Forecast:
    """A single forecast row (one forecaster, one claim)."""

    forecaster: str
    ticker: str
    claim: str
    horizon_days: int
    anchor_date: str  # ISO date — the reference session
    anchor_close: float  # raw close on anchor_date (the comparison anchor)
    resolves_on: str  # ISO date — when the claim is graded
    probability: float  # P(claim is true), 0..1
    reasoning: str | None
    model: str | None  # LLM id for Pythia; rule descriptor for baselines
    issued_at: str = field(default_factory=now_iso)
    status: str = "pending"
    id: int | None = None
    outcome: float | None = None  # 1.0 if claim true, else 0.0; None until resolved
    resolved_close: float | None = None
    resolved_at: str | None = None
    brier: float | None = None  # (probability - outcome)^2; None until resolved


_SCHEMA = """
CREATE TABLE IF NOT EXISTS forecasts (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    issued_at     TEXT    NOT NULL,
    forecaster    TEXT    NOT NULL,
    ticker        TEXT    NOT NULL,
    claim         TEXT    NOT NULL,
    horizon_days  INTEGER NOT NULL,
    anchor_date   TEXT    NOT NULL,
    anchor_close  REAL    NOT NULL,
    resolves_on   TEXT    NOT NULL,
    probability   REAL    NOT NULL,
    reasoning     TEXT,
    model         TEXT,
    status        TEXT    NOT NULL DEFAULT 'pending',
    outcome       REAL,
    resolved_close REAL,
    resolved_at   TEXT,
    brier         REAL,
    UNIQUE (forecaster, ticker, anchor_date, horizon_days)
);
CREATE INDEX IF NOT EXISTS idx_forecasts_status ON forecasts (status);
CREATE INDEX IF NOT EXISTS idx_forecasts_resolves_on ON forecasts (resolves_on);
"""


def get_connection(path: str | Path | None = None) -> sqlite3.Connection:
    """Open (and initialize) the database, returning a Row-factory connection."""
    db = Path(path) if path is not None else config.db_path()
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    init_db(conn)
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    """Create the schema if it does not already exist."""
    conn.executescript(_SCHEMA)
    conn.commit()


def insert_forecast(conn: sqlite3.Connection, fc: Forecast) -> int | None:
    """Insert a pending forecast. Returns the new row id, or None if a forecast
    for this (forecaster, ticker, anchor_date, horizon) already exists."""
    cur = conn.execute(
        """
        INSERT OR IGNORE INTO forecasts (
            issued_at, forecaster, ticker, claim, horizon_days,
            anchor_date, anchor_close, resolves_on, probability,
            reasoning, model, status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            fc.issued_at, fc.forecaster, fc.ticker, fc.claim, fc.horizon_days,
            fc.anchor_date, fc.anchor_close, fc.resolves_on, fc.probability,
            fc.reasoning, fc.model, fc.status,
        ),
    )
    conn.commit()
    if cur.rowcount == 0:
        return None  # duplicate, ignored
    return cur.lastrowid


def fetch_pending_due(conn: sqlite3.Connection, cutoff: date) -> list[sqlite3.Row]:
    """Pending forecasts whose resolution date is on or before `cutoff`."""
    cur = conn.execute(
        """
        SELECT * FROM forecasts
        WHERE status = 'pending' AND resolves_on <= ?
        ORDER BY resolves_on, ticker, forecaster
        """,
        (cutoff.isoformat(),),
    )
    return cur.fetchall()


def mark_resolved(
    conn: sqlite3.Connection,
    forecast_id: int,
    *,
    outcome: float,
    resolved_close: float,
    brier: float,
    resolved_at: str | None = None,
) -> None:
    """Settle a forecast: record outcome, the resolved close, and the Brier score."""
    conn.execute(
        """
        UPDATE forecasts
        SET status = 'resolved', outcome = ?, resolved_close = ?,
            brier = ?, resolved_at = ?
        WHERE id = ?
        """,
        (outcome, resolved_close, brier, resolved_at or now_iso(), forecast_id),
    )
    conn.commit()


def fetch_all(conn: sqlite3.Connection, status: str | None = None) -> list[sqlite3.Row]:
    """All forecasts (optionally filtered by status), newest issue first."""
    if status is None:
        cur = conn.execute(
            "SELECT * FROM forecasts ORDER BY issued_at DESC, ticker, forecaster"
        )
    else:
        cur = conn.execute(
            "SELECT * FROM forecasts WHERE status = ? "
            "ORDER BY issued_at DESC, ticker, forecaster",
            (status,),
        )
    return cur.fetchall()
