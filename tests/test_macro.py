"""Offline tests for the v1 macro arm (no network, no key).

Covers the pure core (formatter, sha, snapshot round-trip) and the point-in-time
contract of the FRED fetch (the methodological crux: every series is read at the
anchor-date vintage via the realtime API, so a backfilled anchor sees the macro
the forecaster actually had, not today's revisions).
"""

from __future__ import annotations

import urllib.parse
from datetime import date, timedelta

import pytest

from pythia import config, forecaster, macro, storage


# --- helpers -------------------------------------------------------------------

def _series_obs(latest: float, v5: float, v20: float, n: int = 25,
                start: str = "2026-04-01") -> list[tuple[str, float]]:
    """An ascending obs list with known latest / 5-back / 20-back values.

    The formatter reads index -1 (latest), -6 (5 back), -21 (20 back); we place
    the chosen values there and fill the rest with `latest`."""
    base = date.fromisoformat(start)
    vals = [latest] * n
    vals[n - 6] = v5
    vals[n - 21] = v20
    return [((base + timedelta(days=i)).isoformat(), v) for i, v in enumerate(vals)]


def _full_series() -> dict[str, list[tuple[str, float]]]:
    return {
        "DGS10": _series_obs(4.30, 4.40, 4.25),
        "DGS2": _series_obs(4.80, 4.85, 4.70),
        "T10YIE": _series_obs(2.20, 2.22, 2.18),
        "BAMLH0A0HYM2": _series_obs(3.21, 3.15, 3.05),
        "VIXCLS": _series_obs(14.2, 12.8, 13.5),
        "DTWEXBGS": _series_obs(108.45, 108.20, 107.90),
    }


def _snap(series=None, anchor="2026-06-18") -> macro.MacroSnapshot:
    s = series if series is not None else _full_series()
    return macro.MacroSnapshot(
        anchor_date=anchor, series=s,
        context_sha=macro.compute_context_sha(s),
        fetched_at="2026-06-18T17:00:00+00:00",
    )


# --- format_macro_context ------------------------------------------------------

def test_format_renders_every_series_and_derived_slope():
    out = macro.format_macro_context(_snap())
    assert "MACRO CONTEXT" in out
    assert "2026-06-18" in out  # the anchor date is stated
    # every configured series is labelled
    for fs in config.FRED_SERIES:
        assert fs.label in out
    # latest values render with units
    assert "4.30%" in out   # DGS10 latest
    assert "4.80%" in out   # DGS2 latest
    assert "14.2" in out    # VIX (no % unit)
    assert "108.45" in out  # dollar
    # the 2s10s slope is derived (10y - 2y = 4.30 - 4.80 = -0.50)
    assert "2s10s slope (derived)" in out
    assert "-0.50%" in out
    # changes show in native units
    assert "-0.10%" in out  # DGS10 d5d (4.30 - 4.40)


def test_slope_line_sits_right_after_the_2y_line():
    out = macro.format_macro_context(_snap())
    lines = out.splitlines()
    i_2y = next(i for i, ln in enumerate(lines) if "2y Treasury yield" in ln)
    i_slope = next(i for i, ln in enumerate(lines) if "2s10s slope" in ln)
    i_10y = next(i for i, ln in enumerate(lines) if "10y Treasury yield" in ln)
    assert i_10y < i_2y < i_slope  # 10y, then 2y, then the derived slope
    # and nothing is wedged between 2y and slope
    assert i_slope == i_2y + 1


def test_missing_series_renders_na_but_keeps_the_slope():
    s = _full_series()
    del s["VIXCLS"]  # VIX unavailable as of the anchor date
    out = macro.format_macro_context(_snap(s))
    assert "VIX" in out and "n/a" in out
    # slope still derived from DGS10 + DGS2
    assert "2s10s slope (derived)" in out
    assert "-0.50%" in out


def test_no_slope_when_either_yield_is_missing():
    for missing in ("DGS10", "DGS2"):
        s = _full_series()
        del s[missing]
        out = macro.format_macro_context(_snap(s))
        assert "2s10s slope (derived)" not in out


def test_empty_snapshot_says_no_data():
    out = macro.format_macro_context(_snap(series={}))
    assert "no macro series available" in out


def test_short_series_shows_na_for_the_lookback_it_cant_reach():
    # only 3 observations: latest is fine, 5d and 20d are not
    s = {"VIXCLS": [("2026-06-16", 14.0), ("2026-06-17", 14.1), ("2026-06-18", 14.2)]}
    out = macro.format_macro_context(_snap(s))
    vix_line = next(ln for ln in out.splitlines() if "VIX" in ln)
    assert "14.2" in vix_line        # latest
    assert "n/a" in vix_line         # 5d / 20d unreachable


# --- compute_context_sha -------------------------------------------------------

def test_sha_is_deterministic():
    s = _full_series()
    assert macro.compute_context_sha(s) == macro.compute_context_sha(s)


def test_sha_changes_when_a_value_changes():
    s = _full_series()
    sha0 = macro.compute_context_sha(s)
    s["DGS10"][-1] = (s["DGS10"][-1][0], 9.99)
    assert macro.compute_context_sha(s) != sha0


def test_sha_is_independent_of_dict_insertion_order():
    s1 = _full_series()
    s2 = {k: s1[k] for k in reversed(list(s1))}
    assert macro.compute_context_sha(s1) == macro.compute_context_sha(s2)


def test_snapshot_series_json_round_trips_through_sorted_canonical_form():
    snap = _snap()
    import json
    parsed = json.loads(snap.series_json())
    # keys are the six series, sorted; values are [date, value] pairs
    assert list(parsed.keys()) == sorted(parsed.keys())
    assert parsed["DGS10"][0] == [_full_series()["DGS10"][0][0],
                                  _full_series()["DGS10"][0][1]]


# --- point-in-time FRED fetch --------------------------------------------------

def test_fred_get_uses_realtime_params_point_in_time(monkeypatch):
    """The methodological crux: every call pins realtime_start=realtime_end=
    anchor_date so only the anchor-day vintage is returned (no future revisions)."""
    captured: list[str] = []

    def fake_get(url: str) -> dict:
        captured.append(url)
        return {"observations": []}

    monkeypatch.setattr(macro, "_get", fake_get)
    macro._fred_get("DGS10", date(2026, 6, 18), "KEY")

    assert len(captured) == 1
    q = urllib.parse.parse_qs(urllib.parse.urlparse(captured[0]).query)
    assert q["series_id"] == ["DGS10"]
    assert q["api_key"] == ["KEY"]
    assert q["file_type"] == ["json"]
    # the vintage is pinned to the anchor date on BOTH ends
    assert q["realtime_start"] == ["2026-06-18"]
    assert q["realtime_end"] == ["2026-06-18"]
    # observation window is the lookback up to and including the anchor
    expected_start = (date(2026, 6, 18) - timedelta(days=config.FRED_LOOKBACK_DAYS)).isoformat()
    assert q["observation_start"] == [expected_start]
    assert q["observation_end"] == ["2026-06-18"]


def test_fred_get_skips_missing_values(monkeypatch):
    # FRED encodes missing readings as "."
    def fake_get(url: str) -> dict:
        return {"observations": [
            {"date": "2026-06-17", "value": "4.40"},
            {"date": "2026-06-18", "value": "."},      # holiday / no print
            {"date": "2026-06-19", "value": "4.32"},
            {"date": "2026-06-20", "value": ""},        # treated as missing too
        ]}

    monkeypatch.setattr(macro, "_get", fake_get)
    obs = macro._fred_get("DGS10", date(2026, 6, 19), "KEY")
    assert obs == [("2026-06-17", 4.40), ("2026-06-19", 4.32)]


# --- get_macro_snapshot orchestration ------------------------------------------

def _recording_fetcher(captured: list):
    def fetch(series_id, anchor_date, api_key):
        captured.append((series_id, anchor_date, api_key))
        return _series_obs(1.0, 1.0, 1.0)
    return fetch


def test_get_macro_snapshot_fetches_all_series_and_records_sha():
    captured: list = []
    snap = macro.get_macro_snapshot(
        date(2026, 6, 18), api_key="KEY", fetcher=_recording_fetcher(captured))
    assert snap is not None
    assert snap.anchor_date == "2026-06-18"
    assert set(snap.series) == {fs.series_id for fs in config.FRED_SERIES}
    assert snap.context_sha == macro.compute_context_sha(snap.series)
    # every series was requested, all with the same anchor date and key
    assert {c[0] for c in captured} == {fs.series_id for fs in config.FRED_SERIES}
    assert {c[1] for c in captured} == {date(2026, 6, 18)}
    assert {c[2] for c in captured} == {"KEY"}


def test_get_macro_snapshot_no_key_returns_none(monkeypatch):
    # api_key=None falls back to the environment; strip it so this test is
    # hermetic even on the live machine (where .env sets a real key — without
    # this, the call would hit FRED for real and break the offline contract).
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    assert macro.get_macro_snapshot(date(2026, 6, 18), api_key=None) is None


def test_get_macro_snapshot_all_empty_returns_none():
    snap = macro.get_macro_snapshot(
        date(2026, 6, 18), api_key="KEY",
        fetcher=lambda sid, ad, k: [])
    assert snap is None


def test_get_macro_snapshot_skips_a_dead_series_not_the_whole_run():
    def fetch(series_id, anchor_date, api_key):
        if series_id == "VIXCLS":
            raise RuntimeError("FRED 500")
        return _series_obs(1.0, 1.0, 1.0)

    snap = macro.get_macro_snapshot(
        date(2026, 6, 18), api_key="KEY", fetcher=fetch)
    assert snap is not None
    assert "VIXCLS" not in snap.series
    assert len(snap.series) == len(config.FRED_SERIES) - 1


# --- persistence ---------------------------------------------------------------

@pytest.fixture
def conn(tmp_path):
    c = storage.get_connection(tmp_path / "test.db")
    yield c
    c.close()


def test_macro_snapshot_roundtrips_through_db(conn):
    snap = _snap()
    assert storage.insert_macro_snapshot(conn, snap) is not None
    row = storage.fetch_macro_snapshot(conn, snap.anchor_date)
    assert row is not None
    back = macro.snapshot_from_row(row)
    assert back.anchor_date == snap.anchor_date
    assert back.context_sha == snap.context_sha
    assert back.source == snap.source
    assert back.series == snap.series


def test_insert_macro_snapshot_is_idempotent(conn):
    snap = _snap()
    assert storage.insert_macro_snapshot(conn, snap) is not None
    # second insert for the same anchor date is a no-op (vintage never changes)
    assert storage.insert_macro_snapshot(conn, snap) is None
    assert len(conn.execute("SELECT 1 FROM macro_snapshots").fetchall()) == 1


def test_macro_snapshots_table_exists_on_a_fresh_db(conn):
    # init_db must create the table for older DBs migrating up
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='macro_snapshots'"
    ).fetchall()
    assert len(rows) == 1


# --- forecaster prompt variants ------------------------------------------------

def test_base_system_prompt_forbids_macro():
    p = forecaster.build_system_prompt(config.EQUITY)
    assert "interest rates" in p            # the ban is stated
    assert "ONLY the price" in p


def test_macro_system_prompt_lifts_price_only_but_keeps_news_ban():
    p = forecaster.build_system_prompt(config.EQUITY, macro=True)
    assert "MACRO DATA BLOCK" in p          # macro is now allowed
    assert "news" in p                      # news is still banned
    # and the stale-training guardrail survives
    assert "STALE" in p


def test_base_tool_forbids_macro_in_reasoning():
    base = forecaster._build_tool(macro=False)
    assert "No macro" in base["input_schema"]["properties"]["reasoning"]["description"]


def test_macro_tool_allows_macro_in_reasoning():
    t = forecaster._build_tool(macro=True)
    desc = t["input_schema"]["properties"]["reasoning"]["description"]
    assert "macro data" in desc
    assert "No macro" not in desc


def test_user_prompt_appends_macro_block_and_evidence_wording():
    base = forecaster.build_user_prompt(
        ticker="SPY", claim="c", horizon_days=5,
        anchor_date=date(2026, 6, 18), anchor_close=100.0,
        resolves_on=date(2026, 6, 25), price_context="PRICE")
    assert "Reason only from the price action above" in base
    assert "MACRO CONTEXT" not in base

    with_macro = forecaster.build_user_prompt(
        ticker="SPY", claim="c", horizon_days=5,
        anchor_date=date(2026, 6, 18), anchor_close=100.0,
        resolves_on=date(2026, 6, 25), price_context="PRICE",
        macro_context="MACRO CONTEXT\n  10y: 4.30%")
    assert "MACRO CONTEXT" in with_macro
    assert "Reason only from the price action and macro data above" in with_macro
