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
    # Fit-health flags, hmm_filter rows only (hmm_health.py): comma-joined flag
    # names, '' = fit checked and clean, NULL = predates health monitoring.
    fit_flags: str | None = None


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
    fit_flags     TEXT,
    UNIQUE (forecaster, ticker, anchor_date, horizon_days)
);
CREATE INDEX IF NOT EXISTS idx_forecasts_status ON forecasts (status);
CREATE INDEX IF NOT EXISTS idx_forecasts_resolves_on ON forecasts (resolves_on);

-- HMM fit-health telemetry (hmm_health.py): one row per (ticker, anchor) fit —
-- convergence, parameters, data window, and any stability flags vs the
-- previous fit. The referee's own audit trail; never used to drop a forecast.
CREATE TABLE IF NOT EXISTS hmm_fits (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker        TEXT    NOT NULL,
    anchor_date   TEXT    NOT NULL,
    fitted_at     TEXT    NOT NULL,
    converged     INTEGER NOT NULL,
    n_iter        INTEGER NOT NULL,
    loglik        REAL    NOT NULL,
    n_states      INTEGER NOT NULL,
    n_obs         INTEGER NOT NULL,
    window_start  TEXT    NOT NULL,
    window_end    TEXT    NOT NULL,
    params        TEXT    NOT NULL,          -- JSON {mu, sig, transition, pi}
    flags         TEXT    NOT NULL DEFAULT '',
    compared_to   TEXT,
    detail        TEXT,
    UNIQUE (ticker, anchor_date)
);
CREATE INDEX IF NOT EXISTS idx_hmm_fits_ticker ON hmm_fits (ticker, anchor_date);

-- Macro snapshots (macro.py): one point-in-time FRED snapshot per anchor date,
-- shared across every ticker anchored that day (macro is per-date, not
-- per-ticker). The series payload is content-hashed so the macro arm's
-- `model` column (+macro:<sha>) ties any forecast row to the exact macro data
-- it saw. First-write-wins like forecasts/hmm_fits — the vintage for a given
-- anchor date never changes, so a re-fetch is an idempotent no-op.
CREATE TABLE IF NOT EXISTS macro_snapshots (
    anchor_date   TEXT PRIMARY KEY,
    fetched_at    TEXT NOT NULL,
    context_sha   TEXT NOT NULL,
    series_json   TEXT NOT NULL,          -- canonical {series_id: [[date, value], ...]}
    source        TEXT NOT NULL DEFAULT 'fred_realtime'
);
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
    """Create the schema if it does not already exist, migrating older DBs."""
    conn.executescript(_SCHEMA)
    # fit_flags arrived 2026-06-12; databases created before then have the
    # forecasts table (so CREATE IF NOT EXISTS skips it) but lack the column.
    cols = {row[1] for row in conn.execute("PRAGMA table_info(forecasts)")}
    if "fit_flags" not in cols:
        conn.execute("ALTER TABLE forecasts ADD COLUMN fit_flags TEXT")
    conn.commit()


def insert_forecast(conn: sqlite3.Connection, fc: Forecast) -> int | None:
    """Insert a pending forecast. Returns the new row id, or None if a forecast
    for this (forecaster, ticker, anchor_date, horizon) already exists."""
    cur = conn.execute(
        """
        INSERT OR IGNORE INTO forecasts (
            issued_at, forecaster, ticker, claim, horizon_days,
            anchor_date, anchor_close, resolves_on, probability,
            reasoning, model, status, fit_flags
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            fc.issued_at, fc.forecaster, fc.ticker, fc.claim, fc.horizon_days,
            fc.anchor_date, fc.anchor_close, fc.resolves_on, fc.probability,
            fc.reasoning, fc.model, fc.status, fc.fit_flags,
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


def set_fit_flags(conn: sqlite3.Connection, forecast_id: int, flags: str) -> None:
    """Mark a forecast row with its fit's health flags ('' = checked, clean)."""
    conn.execute("UPDATE forecasts SET fit_flags = ? WHERE id = ?",
                 (flags, forecast_id))
    conn.commit()


# --- HMM fit-health telemetry (hmm_health.py) ----------------------------------

def insert_hmm_fit(conn: sqlite3.Connection, rec) -> int | None:
    """Persist a checked hmm_health.FitRecord. First write wins per
    (ticker, anchor_date) — like forecasts, re-runs are idempotent no-ops."""
    cur = conn.execute(
        """
        INSERT OR IGNORE INTO hmm_fits (
            ticker, anchor_date, fitted_at, converged, n_iter, loglik,
            n_states, n_obs, window_start, window_end, params, flags,
            compared_to, detail
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            rec.ticker, rec.anchor_date, rec.fitted_at, int(rec.converged),
            rec.n_iter, rec.loglik, rec.n_states, rec.n_obs,
            rec.window_start, rec.window_end, rec.params_json(),
            ",".join(rec.flags), rec.compared_to, rec.detail,
        ),
    )
    conn.commit()
    if cur.rowcount == 0:
        return None  # duplicate, ignored
    return cur.lastrowid


def latest_hmm_fit_before(
    conn: sqlite3.Connection, ticker: str, anchor_date: str
) -> sqlite3.Row | None:
    """The most recent fit STRICTLY before `anchor_date` — the only previous
    fit a point-in-time stability check is allowed to see."""
    cur = conn.execute(
        """
        SELECT * FROM hmm_fits WHERE ticker = ? AND anchor_date < ?
        ORDER BY anchor_date DESC LIMIT 1
        """,
        (ticker, anchor_date),
    )
    return cur.fetchone()


def fetch_hmm_fits(
    conn: sqlite3.Connection, ticker: str | None = None
) -> list[sqlite3.Row]:
    """All fit-health rows (optionally one ticker), newest anchor first."""
    if ticker is None:
        cur = conn.execute(
            "SELECT * FROM hmm_fits ORDER BY anchor_date DESC, ticker")
    else:
        cur = conn.execute(
            "SELECT * FROM hmm_fits WHERE ticker = ? ORDER BY anchor_date DESC",
            (ticker,),
        )
    return cur.fetchall()


# --- Macro snapshots (macro.py) -----------------------------------------------

def insert_macro_snapshot(
    conn: sqlite3.Connection, snap
) -> int | None:
    """Persist a point-in-time macro snapshot. First write wins per anchor_date
    (idempotent) — the vintage for a given anchor date never changes."""
    cur = conn.execute(
        """
        INSERT OR IGNORE INTO macro_snapshots
            (anchor_date, fetched_at, context_sha, series_json, source)
        VALUES (?, ?, ?, ?, ?)
        """,
        (snap.anchor_date, snap.fetched_at, snap.context_sha,
         snap.series_json(), snap.source),
    )
    conn.commit()
    if cur.rowcount == 0:
        return None  # duplicate anchor_date, ignored
    return cur.lastrowid


def fetch_macro_snapshot(
    conn: sqlite3.Connection, anchor_date: str
) -> sqlite3.Row | None:
    """The macro snapshot for one anchor date, or None if none was recorded."""
    cur = conn.execute(
        "SELECT * FROM macro_snapshots WHERE anchor_date = ?",
        (anchor_date,),
    )
    return cur.fetchone()
