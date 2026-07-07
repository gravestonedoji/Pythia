"""High-conviction digest (digest.py + the alert log): gating, rendering, grading.

Offline like the rest of the suite — forecast/alert rows are plain dicts (or a
temp DB where storage semantics are the thing under test).
"""

from __future__ import annotations

import pytest

from pythia import config, digest, storage


def frow(fc, p, ticker="SPY", anchor="2026-07-07", horizon=5,
         resolves="2026-07-14", anchor_close=620.0, status="pending",
         outcome=None, brier=None, reasoning="momentum strong"):
    return {
        "forecaster": fc, "ticker": ticker, "anchor_date": anchor,
        "horizon_days": horizon, "resolves_on": resolves,
        "anchor_close": anchor_close, "probability": p, "status": status,
        "outcome": outcome, "brier": brier, "reasoning": reasoning,
    }


def arow(ticker="USO", anchor="2026-07-02", horizon=5, resolves="2026-07-09",
         direction="DOWN", p=0.33, flagged_by="pythia",
         gate_arms=None, emailed_at=None):
    return {
        "ticker": ticker, "anchor_date": anchor, "horizon_days": horizon,
        "resolves_on": resolves, "direction": direction, "probability": p,
        "flagged_by": flagged_by,
        "gate_arms": gate_arms or ",".join(config.ALERT_GATE_ARMS),
        "threshold": config.ALERT_CONVICTION_MIN, "emailed_at": emailed_at,
    }


# --- the gate -------------------------------------------------------------------

def test_threshold_boundary_inclusive():
    assert digest.flag_high_conviction({"pythia": 0.65}) == ["pythia"]
    assert digest.flag_high_conviction({"pythia": 0.649}) == []
    assert digest.flag_high_conviction({"pythia": 0.35}) == ["pythia"]
    assert digest.flag_high_conviction({"pythia": 0.351}) == []


def test_only_gate_arms_can_trip_the_flag():
    # An iso arm at 0.19 and the HMM at 0.71 are both past the line — neither
    # is a gate arm, so neither flags.
    assert digest.flag_high_conviction(
        {"pythia_iso": 0.19, "hmm_filter": 0.71, "pythia": 0.55}) == []
    flagged = digest.flag_high_conviction(
        {"pythia_coached": 0.68, "pythia": 0.55})
    assert flagged == ["pythia_coached"]


def test_multi_arm_flag_lists_every_crossing_arm_in_gate_order():
    flagged = digest.flag_high_conviction(
        {"pythia_macro": 0.70, "pythia": 0.66, "pythia_coached": 0.55})
    assert flagged == ["pythia", "pythia_macro"]


# --- building flagged calls --------------------------------------------------------

def test_build_flagged_calls_headline_from_most_extreme_arm():
    rows = [frow("pythia", 0.66), frow("pythia_coached", 0.30),
            frow("pythia_iso", 0.19), frow("hmm_filter", 0.71)]
    calls = digest.build_flagged_calls(rows, [])
    assert len(calls) == 1
    c = calls[0]
    assert c.flagged_by == ["pythia", "pythia_coached"]
    assert c.direction == "DOWN" and c.probability == 0.30  # |0.30-0.5| > |0.66-0.5|
    assert set(c.arm_probs) == {"pythia", "pythia_coached"}  # gate arms only


def test_build_flagged_calls_quiet_batch_is_empty():
    rows = [frow("pythia", 0.55), frow("pythia_coached", 0.48)]
    assert digest.build_flagged_calls(rows, []) == []


def test_repeat_of_marks_overlapping_prior_alert_only():
    rows = [frow("pythia", 0.70, ticker="USO", anchor="2026-07-07")]
    overlapping = arow(ticker="USO", anchor="2026-07-02", resolves="2026-07-09")
    closed = arow(ticker="USO", anchor="2026-06-20", resolves="2026-06-29")
    other_ticker = arow(ticker="GLD", anchor="2026-07-06", resolves="2026-07-13")
    calls = digest.build_flagged_calls(rows, [overlapping, closed, other_ticker])
    assert calls[0].repeat_of == "2026-07-02"

    calls = digest.build_flagged_calls(rows, [closed])
    assert calls[0].repeat_of is None  # that window closed before this anchor


# --- the scoreboard (derived by join, outcomes never stored) --------------------------

def test_alert_scoreboard_pooled_and_per_arm():
    alerts = [
        arow(ticker="SPY", anchor="2026-06-23", resolves="2026-06-30",
             direction="UP", p=0.70, flagged_by="pythia,pythia_coached"),
        arow(ticker="USO", anchor="2026-06-24", resolves="2026-07-01",
             direction="DOWN", p=0.30, flagged_by="pythia"),
    ]
    forecasts = [
        frow("pythia", 0.70, ticker="SPY", anchor="2026-06-23",
             status="resolved", outcome=1.0, brier=0.09),
        frow("pythia_coached", 0.68, ticker="SPY", anchor="2026-06-23",
             status="resolved", outcome=1.0, brier=0.1024),
        frow("pythia", 0.30, ticker="USO", anchor="2026-06-24",
             status="resolved", outcome=1.0, brier=0.49),  # called DOWN, went UP
    ]
    pooled, per_arm = digest.alert_scoreboard(alerts, forecasts)
    assert pooled.n_resolved == 3 and pooled.n_correct == 2
    assert pooled.avg_brier == pytest.approx((0.09 + 0.1024 + 0.49) / 3)
    assert per_arm["pythia"].n_resolved == 2 and per_arm["pythia"].n_correct == 1
    assert per_arm["pythia_coached"].n_resolved == 1
    assert per_arm["pythia_coached"].hit_rate == 1.0


def test_alert_scoreboard_excludes_pending():
    alerts = [arow(ticker="SPY", anchor="2026-07-07", flagged_by="pythia")]
    forecasts = [frow("pythia", 0.70, anchor="2026-07-07", status="pending")]
    pooled, per_arm = digest.alert_scoreboard(alerts, forecasts)
    assert pooled.n_resolved == 0 and per_arm == {}


# --- rendering --------------------------------------------------------------------------

def _one_call(**kw):
    rows = [frow("pythia", kw.pop("p", 0.70))]
    return digest.build_flagged_calls(rows, [])


def test_digest_carries_the_wall_language_and_rule():
    text = digest.format_digest_sections(
        _one_call(), digest.AlertStats(), {})
    assert "MANUAL human review" in text
    assert "no orders" in text
    assert "instruction to trade" in text
    assert f">= {config.ALERT_CONVICTION_MIN:.2f}" in text


def test_digest_quiet_day_says_so_explicitly():
    text = digest.format_digest_sections([], digest.AlertStats(), {})
    assert "No high-conviction calls today" in text


def test_digest_renders_repeat_caveat_and_scoreboard():
    rows = [frow("pythia", 0.70, ticker="USO", anchor="2026-07-07")]
    calls = digest.build_flagged_calls(
        rows, [arow(ticker="USO", anchor="2026-07-02", resolves="2026-07-09")])
    pooled = digest.AlertStats(n_resolved=12, n_correct=11, brier_sum=1.44)
    text = digest.format_digest_sections(calls, pooled, {"pythia": pooled})
    assert "same bet as the 2026-07-02 alert" in text
    assert "12 resolved / 11 correct" in text
    assert "out-of-sample" in text


def test_digest_reconstructed_label():
    text = digest.format_digest_sections(
        _one_call(), digest.AlertStats(), {}, reconstructed=True)
    assert "reconstructed" in text and "not logged" in text


def test_flagged_from_alert_rows_is_verbatim_not_recomputed():
    # The stored alert says DOWN at 0.30 by pythia; today's forecast row for
    # that claim shows 0.55 (as if the arm were re-run) — the render must keep
    # the LOGGED values.
    alerts = [arow(ticker="SPY", anchor="2026-06-23", resolves="2026-06-30",
                   direction="DOWN", p=0.30, flagged_by="pythia")]
    forecasts = [frow("pythia", 0.55, ticker="SPY", anchor="2026-06-23",
                      reasoning="original reasoning")]
    calls = digest.flagged_from_alert_rows(alerts, forecasts)
    assert calls[0].direction == "DOWN"
    assert calls[0].probability == 0.30
    assert calls[0].reasoning == "original reasoning"


# --- storage semantics ---------------------------------------------------------------------

@pytest.fixture()
def conn(tmp_path):
    c = storage.get_connection(tmp_path / "test.db")
    yield c
    c.close()


def _insert(conn, **kw):
    defaults = dict(ticker="SPY", anchor_date="2026-07-07", horizon_days=5,
                    resolves_on="2026-07-14", direction="UP", probability=0.70,
                    flagged_by="pythia", threshold=0.15,
                    gate_arms="pythia,pythia_coached,pythia_macro")
    defaults.update(kw)
    return storage.insert_digest_alert(conn, **defaults)


def test_fresh_db_has_digest_alerts_and_inserts_are_first_write_wins(conn):
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    assert "digest_alerts" in tables
    rid = _insert(conn)
    assert rid is not None
    assert _insert(conn, probability=0.99) is None  # duplicate ignored
    rows = storage.fetch_digest_alerts(conn)
    assert len(rows) == 1 and rows[0]["probability"] == 0.70
    # The config in force was stamped on the row.
    assert rows[0]["threshold"] == 0.15
    assert rows[0]["gate_arms"] == "pythia,pythia_coached,pythia_macro"


def test_mark_alert_emailed_sets_timestamp_once(conn):
    rid = _insert(conn)
    assert storage.fetch_digest_alerts(conn)[0]["emailed_at"] is None
    storage.mark_alert_emailed(conn, rid)
    first = storage.fetch_digest_alerts(conn)[0]["emailed_at"]
    assert first is not None
    storage.mark_alert_emailed(conn, rid)  # no-op: delivery time is a fact
    assert storage.fetch_digest_alerts(conn)[0]["emailed_at"] == first


def test_pre_digest_db_gains_the_table(tmp_path):
    import sqlite3
    db = tmp_path / "old.db"
    raw = sqlite3.connect(db)
    raw.execute(
        "CREATE TABLE forecasts (id INTEGER PRIMARY KEY, forecaster TEXT, "
        "status TEXT, resolves_on TEXT, fit_flags TEXT)")
    raw.commit()
    raw.close()
    c = storage.get_connection(db)
    tables = {r[0] for r in c.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    assert "digest_alerts" in tables
    c.close()
