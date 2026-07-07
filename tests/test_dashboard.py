"""Dashboard (dashboard.py + `pythia publish`): binning, honesty, leak guards.

Offline and deterministic — rows are dicts; the CLI test uses a temp DB via
PYTHIA_DB_PATH and typer's CliRunner.
"""

from __future__ import annotations

import json
from datetime import date

import pytest

from pythia import config, dashboard, hmm_health, storage

FROZEN = "2026-07-07T12:00:00+00:00"


def _row(fc="pythia", p=0.55, outcome=1.0, resolves="2026-06-10",
         status="resolved", fit_flags=None, ticker="SPY",
         anchor="2026-06-03"):
    brier = (p - outcome) ** 2 if status == "resolved" else None
    return {
        "forecaster": fc, "probability": p, "outcome": outcome if status == "resolved" else None,
        "brier": brier, "status": status, "resolves_on": resolves,
        "fit_flags": fit_flags, "ticker": ticker, "anchor_date": anchor,
        "horizon_days": 5,
        # Fields the dashboard must NEVER read or render (leak-guard bait):
        "claim": "LEAK_CLAIM_SENTINEL",
        "reasoning": "LEAK_REASONING_SENTINEL",
    }


def _record(n=120, fc="pythia", hit_rate=0.55, start=0):
    """n resolved rows with a mix of probabilities and outcomes."""
    rows = []
    for i in range(n):
        day = date(2026, 3, 2) + __import__("datetime").timedelta(days=(start + i) % 90)
        rows.append(_row(
            fc=fc, p=0.45 + (i % 30) / 100.0,
            outcome=1.0 if (i * 7) % 100 < hit_rate * 100 else 0.0,
            resolves=day.isoformat(), ticker=f"T{i % 10}",
            anchor=(day - __import__("datetime").timedelta(days=7)).isoformat()))
    return rows


def _build(rows, today=date(2026, 7, 7), **kw):
    taint = hmm_health.taint_summary(rows)
    return dashboard.build_dashboard_data(
        rows, taint, today=today, generated_at=FROZEN, **kw)


# --- Wilson ---------------------------------------------------------------------

def test_wilson_pinned_values():
    lo, hi = dashboard.wilson_interval(5, 10)
    assert lo == pytest.approx(0.2366, abs=1e-3)
    assert hi == pytest.approx(0.7634, abs=1e-3)


def test_wilson_edges_and_containment():
    lo, hi = dashboard.wilson_interval(0, 25)
    assert lo == 0.0 and 0.0 <= hi < 0.25
    lo, hi = dashboard.wilson_interval(25, 25)
    assert hi == 1.0 and lo > 0.75
    assert dashboard.wilson_interval(0, 0) == (0.0, 1.0)
    for k, n in ((3, 25), (12, 25), (24, 25)):
        lo, hi = dashboard.wilson_interval(k, n)
        assert lo <= k / n <= hi


# --- binning --------------------------------------------------------------------

def test_bin_count_formula_and_floors():
    pts410 = [(0.4 + (i % 40) / 100, float(i % 2)) for i in range(410)]
    bins = dashboard.calibration_bins(pts410)
    assert len(bins) == config.DASH_CAL_MAX_BINS  # 410//25=16, capped at 8
    assert sum(b.n for b in bins) == 410
    assert all(b.n >= config.DASH_CAL_MIN_BIN for b in bins)

    pts150 = pts410[:150]
    assert len(dashboard.calibration_bins(pts150)) == 6  # 150 // 25

    assert dashboard.calibration_bins(pts410[:24]) == []  # below one bin


def test_bins_monotone_in_predicted_p_even_when_clustered():
    # Everything clustered in 0.55-0.65 — fixed-width deciles would be mostly
    # empty; quantile bins must all be populated and ordered.
    pts = [(0.55 + (i % 11) / 100.0, float(i % 3 == 0)) for i in range(200)]
    bins = dashboard.calibration_bins(pts)
    assert len(bins) == 8
    means = [b.p_mean for b in bins]
    assert means == sorted(means)
    assert all(b.n >= 25 for b in bins)


# --- honesty rendering -------------------------------------------------------------

def test_embargo_banner_inside_month_three_then_demoted():
    rows = _record(120)
    early = _build(rows, today=date(2026, 7, 7))  # first resolve ~2026-03-02
    # 90+ days passed in this record -> no banner
    assert early.embargo is None
    young = _build(_record(120), today=date(2026, 4, 1))
    assert young.embargo is not None and "do not rank" in young.embargo
    html = dashboard.render_html(young)
    assert "do not rank" in html


def test_curve_gate_and_skip_placeholder():
    rows = _record(120, fc="pythia") + _record(40, fc="pythia_coached")
    d = _build(rows)
    by_fc = {c.forecaster: c for c in d.curves}
    assert by_fc["pythia"].bins and by_fc["pythia"].skip_reason is None
    coached = by_fc["pythia_coached"]
    assert coached.skip_reason is not None and "currently 40" in coached.skip_reason
    assert coached.bins == []
    html = dashboard.render_html(d)
    assert "curve appears at" in html


def test_taint_line_appears_iff_flagged_rows_exist():
    clean = _record(30) + [_row(fc="hmm_filter", fit_flags="")]
    assert _build(clean).taint_lines == []
    tainted = _record(30) + [_row(fc="hmm_filter", fit_flags="non_converged")]
    d = _build(tainted)
    assert d.taint_lines and "HMM bar" in d.taint_lines[0]
    assert "HMM bar" in dashboard.render_html(d)


def test_paper_and_alert_sections_render_from_aggregates():
    from pythia import digest
    paper_rows = [{
        "forecaster": "pythia", "probability": 0.7, "anchor_date": "2026-07-01",
        "expiry_date": "2026-07-08", "entry_mid": 1.0, "entry_ask": 1.1,
        "intrinsic": 2.0, "status": "settled", "id": 1,
    }]
    stats = digest.AlertStats(n_resolved=3, n_correct=3, brier_sum=0.3)
    d = _build(_record(30), paper_rows=paper_rows,
               alert_stats=(stats, {"pythia": stats}))
    html = dashboard.render_html(d)
    assert "Paper book" in html and "1 simulated positions (1 settled)" in html
    assert "Alert rule" in html and "3 resolved / 3 correct" in html


# --- determinism / change detection ---------------------------------------------------

def test_same_inputs_render_byte_identical_html():
    rows = _record(120)
    assert dashboard.render_html(_build(rows)) == dashboard.render_html(_build(rows))


def test_content_sha_excludes_timestamp_but_sees_new_rows():
    rows = _record(120)
    taint = hmm_health.taint_summary(rows)
    a = dashboard.build_dashboard_data(rows, taint, today=date(2026, 7, 7),
                                       generated_at=FROZEN)
    b = dashboard.build_dashboard_data(rows, taint, today=date(2026, 7, 7),
                                       generated_at="2026-07-08T12:00:00+00:00")
    assert a.content_sha == b.content_sha  # timestamp excluded
    grown = rows + [_row(resolves="2026-07-01", ticker="NEW")]
    c = _build(grown)
    assert c.content_sha != a.content_sha


# --- leak guards (the important ones) ---------------------------------------------------

def test_no_claim_text_reasoning_or_scripts_reach_the_page():
    rows = _record(120) + [_row(status="pending", p=0.987654)]
    d = _build(rows)
    html = dashboard.render_html(d)
    js = dashboard.render_json(d)
    for out in (html, js):
        assert "LEAK_CLAIM_SENTINEL" not in out
        assert "LEAK_REASONING_SENTINEL" not in out
    assert "<script" not in html and "<link" not in html
    # The single external reference is the repo link.
    assert html.count("http") == 1 and "github.com/gravestonedoji/Pythia" in html
    # A pending call's probability must not be published (counts only).
    assert "0.987654" not in html and "0.987654" not in js
    assert "98.8%" not in html


def test_json_payload_is_aggregates_only():
    payload = json.loads(dashboard.render_json(_build(_record(120))))
    assert "board" in payload and "curves" in payload

    def keys_of(node):
        if isinstance(node, dict):
            for k, v in node.items():
                yield k
                yield from keys_of(v)
        elif isinstance(node, list):
            for v in node:
                yield from keys_of(v)

    # No row-level fields anywhere in the structure — aggregates only.
    assert {"reasoning", "claim", "anchor_close", "probability"}.isdisjoint(
        set(keys_of(payload)))


# --- publish CLI --------------------------------------------------------------------------

@pytest.fixture()
def cli_env(tmp_path, monkeypatch):
    db = tmp_path / "test.db"
    monkeypatch.setenv("PYTHIA_DB_PATH", str(db))
    conn = storage.get_connection(db)
    from pythia.storage import Forecast
    for i in range(3):
        rid = storage.insert_forecast(conn, Forecast(
            forecaster="pythia", ticker=f"T{i}", claim="c", horizon_days=5,
            anchor_date="2026-06-01", anchor_close=100.0,
            resolves_on="2026-06-08", probability=0.6,
            reasoning="r", model="m"))
        storage.mark_resolved(conn, rid, outcome=1.0, resolved_close=101.0,
                              brier=0.16)
    conn.close()
    return tmp_path


def test_publish_writes_once_and_skips_unchanged(cli_env, tmp_path):
    from typer.testing import CliRunner
    from pythia.cli import app

    out = tmp_path / "site"
    runner = CliRunner()
    r1 = runner.invoke(app, ["publish", "--out", str(out)])
    assert r1.exit_code == 0, r1.output
    assert (out / "index.html").exists() and (out / "data.json").exists()
    assert (out / ".nojekyll").exists()

    conn = storage.get_connection()
    assert storage.latest_publish(conn) is not None
    first_id = storage.latest_publish(conn)["id"]

    r2 = runner.invoke(app, ["publish", "--out", str(out)])
    assert r2.exit_code == 0
    assert "unchanged" in r2.output
    assert storage.latest_publish(conn)["id"] == first_id  # no new audit row

    r3 = runner.invoke(app, ["publish", "--out", str(out), "--force"])
    assert r3.exit_code == 0
    assert storage.latest_publish(conn)["id"] != first_id
    conn.close()


def test_publish_stdout_prints_html_and_writes_nothing(cli_env, tmp_path):
    from typer.testing import CliRunner
    from pythia.cli import app

    out = tmp_path / "site2"
    runner = CliRunner()
    r = runner.invoke(app, ["publish", "--stdout"])
    assert r.exit_code == 0
    assert "<title>" in r.output
    assert not out.exists()
    conn = storage.get_connection()
    assert storage.latest_publish(conn) is None  # stdout does not log a publish
    conn.close()
