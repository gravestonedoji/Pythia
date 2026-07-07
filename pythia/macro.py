"""FRED macro context for the v1 macro-aware arm (``pythia_macro``).

WHY: v0 is price-only on purpose — the prompt forbids macro because stale
pretraining macro is worse than none. v1 lifts that ban for ONE arm only, by
feeding REAL, CURRENT, POINT-IN-TIME macro (Treasury yields, breakeven
inflation, credit spread, VIX, dollar) alongside the same price data the raw
arm sees. The raw ``pythia`` arm stays price-only forever, so the macro effect
is MEASURED on identical claims (same anchor, same price action, same horizon;
the only difference is the macro block in the prompt) rather than assumed.

HONESTY RULES (same spirit as kalshi.py / the rest of Pythia):
- Point-in-time via FRED's realtime API. Every series is fetched with
  ``realtime_start=realtime_end=anchor_date`` AND ``observation_start/end``
  spanning the lookback window, so each returned observation is the value that
  was KNOWN on the anchor day (its real-time period includes the anchor), not
  today's revised value. A backfilled anchor therefore sees the macro the
  forecaster actually had that day — no future-revision leakage. For these
  daily market-implied series revisions are near-nil anyway, but the realtime
  call is free and is the method-faithful choice.
- Macro is fed ONLY to ``pythia_macro``. It never reaches the raw/coached arms
  (the v0 ablation stays clean) and never reaches the isotonic calibrators
  (which fit on the base arms' own records).
- Optional + best-effort. Needs ``FRED_API_KEY``; when unset the macro arm is a
  clean no-op (price-only arms unaffected), exactly like email without SMTP. A
  dead/missing series renders as n/a and never kills the run; if NO series
  returns usable data, ``get_macro_snapshot`` yields None and the arm skips.
- Auditable. One snapshot per anchor date (macro is per-date, shared across all
  30 tickers) is persisted in ``macro_snapshots`` with a content-hashed series
  payload, and the macro arm's ``model`` column carries ``+macro:<sha>`` so any
  slice of the record ties to the exact macro data the arm saw.

No new dependencies: stdlib urllib for HTTP (same pattern as kalshi.py); the
formatter and sha are pure and offline unit-tested (tests/test_macro.py).
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Callable

from . import config


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class MacroSnapshot:
    """The point-in-time macro readings as known on one anchor date.

    ``series`` maps each FRED series id to its observation history (iso-date,
    value) ascending by date, at the anchor-date vintage. Shared across every
    ticker anchored on this date — macro is per-date, not per-ticker.
    """

    anchor_date: str  # ISO — the session whose close is the claim anchor
    series: dict[str, list[tuple[str, float]]]
    context_sha: str
    fetched_at: str
    source: str = "fred_realtime"

    def series_json(self) -> str:
        """Canonical JSON of the series payload (deterministic; matches the sha)."""
        canon = {sid: [[d, v] for d, v in obs]
                 for sid, obs in sorted(self.series.items())}
        return json.dumps(canon, sort_keys=True, separators=(",", ":"))


def compute_context_sha(series: dict[str, list[tuple[str, float]]]) -> str:
    """Short content hash of the series payload — the audit key.

    Two snapshots with the same sha fed the arm identical macro data. Mirrors
    the lessons-sha pattern (reflect.py): any slice of the record ties to the
    exact macro it saw.
    """
    canon = {sid: [[d, v] for d, v in obs] for sid, obs in sorted(series.items())}
    text = json.dumps(canon, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(text.encode()).hexdigest()[:10]


def snapshot_from_row(row) -> MacroSnapshot:
    """Reconstruct a snapshot from a ``macro_snapshots`` DB row."""
    raw = json.loads(row["series_json"])
    series = {sid: [(d, v) for d, v in obs] for sid, obs in raw.items()}
    return MacroSnapshot(
        anchor_date=row["anchor_date"], series=series,
        context_sha=row["context_sha"], fetched_at=row["fetched_at"],
        source=row["source"],
    )


# --- pure context formatting (offline, unit-tested) ---------------------------

def _val_at(obs: list[tuple[str, float]], back: int) -> float | None:
    """Value ``back`` observations before the latest (0 = latest). None if the
    series is too short for that lookback."""
    if len(obs) <= back:
        return None
    return obs[-1 - back][1]


def _fmt_val(v: float | None, fs: config.FredSeries) -> str:
    if v is None:
        return "n/a"
    return f"{v:{fs.fmt}}{fs.unit}"


def _fmt_chg(cur: float | None, prev: float | None, fs: config.FredSeries) -> str:
    """Change in the series' native unit (pp for %-series, points for index)."""
    if cur is None or prev is None:
        return "n/a"
    d = cur - prev
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:{fs.fmt}}{fs.unit}"


def _slope_series(
    dgs10_obs: list[tuple[str, float]],
    dgs2_obs: list[tuple[str, float]],
) -> list[tuple[str, float]]:
    """The 2s10s slope (10y - 2y) on dates present in BOTH series, ascending.

    Derived, not fetched: saves a FRED call and keeps the slope consistent with
    the two yields it's built from (a separately-fetched slope could mismatch
    on vintage)."""
    d2 = dict(dgs2_obs)
    return [(d, v10 - d2[d]) for d, v10 in dgs10_obs if d in d2]


def format_macro_context(snap: MacroSnapshot) -> str:
    """Render the macro block for the ``pythia_macro`` prompt.

    Pure: takes a snapshot, returns text. Shows each series' latest value plus
    the value 5 and 20 observations back (parallel to the price context's
    5d/20d returns), with the change in native units. The 2s10s slope is
    derived from DGS10/DGS2 and injected right after the 2y line.
    """
    slope_fs = config.FredSeries("SLOPE", "2s10s slope (derived)", "%", ".2f")
    lines = [
        f"MACRO CONTEXT - point-in-time values as known on the anchor date "
        f"{snap.anchor_date} (latest first; 5d/20d = value 5 and 20 sessions "
        "prior; delta = change in native units):",
    ]
    rendered_any = False
    for fs in config.FRED_SERIES:
        obs = snap.series.get(fs.series_id, [])
        if not obs:
            lines.append(f"  {fs.label + ':':<26} n/a (no data as of anchor date)")
            continue
        rendered_any = True
        cur, v5, v20 = _val_at(obs, 0), _val_at(obs, 5), _val_at(obs, 20)
        lines.append(
            f"  {fs.label + ':':<26} {_fmt_val(cur, fs):>9}   "
            f"(5d: {_fmt_val(v5, fs)}, 20d: {_fmt_val(v20, fs)})   "
            f"d5d {_fmt_chg(cur, v5, fs)}, d20d {_fmt_chg(cur, v20, fs)}"
        )
        if fs.series_id == "DGS2":
            slope = _slope_series(snap.series.get("DGS10", []), obs)
            if slope:
                sc, s5, s20 = _val_at(slope, 0), _val_at(slope, 5), _val_at(slope, 20)
                lines.append(
                    f"  {'2s10s slope (derived):':<26} {_fmt_val(sc, slope_fs):>9}   "
                    f"(5d: {_fmt_val(s5, slope_fs)}, 20d: {_fmt_val(s20, slope_fs)})   "
                    f"d5d {_fmt_chg(sc, s5, slope_fs)}, d20d {_fmt_chg(sc, s20, slope_fs)}"
                )
    if not rendered_any:
        lines.append("  (no macro series available as of the anchor date)")
    return "\n".join(lines)


# --- network (read-only FRED realtime API, throttled) -------------------------

_LAST_CALL = [0.0]  # throttle clock, shared across calls in this process


def _get(url: str) -> dict:
    wait = config.FRED_THROTTLE_S - (time.monotonic() - _LAST_CALL[0])
    if wait > 0:
        time.sleep(wait)
    req = urllib.request.Request(
        url, headers={"User-Agent": "pythia/0.1 (macro arm; read-only FRED)"})
    with urllib.request.urlopen(req, timeout=20) as r:
        payload = json.loads(r.read().decode())
    _LAST_CALL[0] = time.monotonic()
    return payload


def _fred_get(
    series_id: str, anchor_date: date, api_key: str,
) -> list[tuple[str, float]]:
    """Fetch one series' recent history at the anchor-date vintage.

    ``realtime_start=realtime_end=anchor_date`` selects observations whose
    real-time period includes the anchor day (the value KNOWN that day);
    ``observation_start/end`` bounds the window to the lookback. FRED encodes
    missing readings as the string ``"."`` — skipped.
    """
    start = (anchor_date - timedelta(days=config.FRED_LOOKBACK_DAYS)).isoformat()
    end = anchor_date.isoformat()
    params = urllib.parse.urlencode({
        "series_id": series_id,
        "realtime_start": end,
        "realtime_end": end,
        "observation_start": start,
        "observation_end": end,
        "api_key": api_key,
        "file_type": "json",
    })
    payload = _get(f"{config.FRED_API_BASE}?{params}")
    out: list[tuple[str, float]] = []
    for o in payload.get("observations", []):
        v = o.get("value")
        if v is None or v == ".":
            continue
        try:
            out.append((o["date"], float(v)))
        except (TypeError, ValueError, KeyError):
            continue
    out.sort(key=lambda x: x[0])
    return out


# A series fetcher: (series_id, anchor_date, api_key) -> observation history.
# Injectable so tests verify the point-in-time request params without network.
SeriesFetcher = Callable[[str, date, str], list[tuple[str, float]]]


def get_macro_snapshot(
    anchor_date: date,
    *,
    api_key: str | None = None,
    fetcher: SeriesFetcher | None = None,
) -> MacroSnapshot | None:
    """Build the point-in-time macro snapshot for one anchor date.

    Returns None (macro arm no-ops) when no API key is set or no series returns
    usable data. A dead/missing series is noted as n/a in the context, never
    fatal — a missing macro row beats a fudged one (same rule as Kalshi).
    """
    key = api_key if api_key is not None else config.fred_api_key()
    if not key:
        return None
    fetch = fetcher or _fred_get
    series: dict[str, list[tuple[str, float]]] = {}
    for fs in config.FRED_SERIES:
        try:
            obs = fetch(fs.series_id, anchor_date, key)
        except Exception:  # noqa: BLE001 - one dead series must not kill the rest
            obs = []
        if obs:
            series[fs.series_id] = obs
    if not series:
        return None
    return MacroSnapshot(
        anchor_date=anchor_date.isoformat(), series=series,
        context_sha=compute_context_sha(series), fetched_at=_now_iso(),
        source="fred_realtime",
    )
