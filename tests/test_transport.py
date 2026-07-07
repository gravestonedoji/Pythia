"""Subscription transport (forecaster.py): argv/env builders, JSON parse, routing.

Offline: the CLI runner is injected; shutil.which is monkeypatched so no test
depends on Claude Code being installed.
"""

from __future__ import annotations

import json
import subprocess

import pytest

from pythia import config, forecaster


@pytest.fixture(autouse=True)
def _fake_cli(monkeypatch):
    monkeypatch.setattr("pythia.forecaster.shutil.which",
                        lambda name: r"C:\tools\claude.EXE")


# --- config.transport() -----------------------------------------------------------

def test_transport_defaults_to_api(monkeypatch):
    monkeypatch.delenv("PYTHIA_TRANSPORT", raising=False)
    assert config.transport() == "api"
    monkeypatch.setenv("PYTHIA_TRANSPORT", "nonsense")
    assert config.transport() == "api"  # unknown values fail safe to metered


def test_transport_subscription(monkeypatch):
    monkeypatch.setenv("PYTHIA_TRANSPORT", "Subscription")
    assert config.transport() == "subscription"


# --- argv / env builders ------------------------------------------------------------

def test_subscription_cmd_isolates_the_call():
    cmd = forecaster._subscription_cmd("SYS", "USER", "claude-opus-4-8")
    joined = " ".join(cmd)
    assert cmd[0].endswith("claude.EXE") and "-p" in cmd
    assert "--model claude-opus-4-8" in joined
    # Isolation: no tools, no settings/CLAUDE.md, no session files.
    assert "--tools " in joined and "--setting-sources " in joined
    assert "--no-session-persistence" in joined
    assert cmd[cmd.index("--system-prompt") + 1] == "SYS"
    # The user prompt is last and carries the strict-JSON instruction.
    assert cmd[-1].startswith("USER") and '"probability"' in cmd[-1]


def test_subscription_cmd_wraps_cmd_shims(monkeypatch):
    monkeypatch.setattr("pythia.forecaster.shutil.which",
                        lambda name: r"C:\npm\claude.CMD")
    cmd = forecaster._subscription_cmd("s", "u", "m")
    assert cmd[:2] == ["cmd", "/c"]  # .cmd shims need the shell on Windows


def test_subscription_cmd_missing_cli_is_a_readable_error(monkeypatch):
    monkeypatch.setattr("pythia.forecaster.shutil.which", lambda name: None)
    with pytest.raises(RuntimeError, match="not found on PATH"):
        forecaster._subscription_cmd("s", "u", "m")


def test_subscription_env_strips_metered_credentials(monkeypatch):
    # With a key in the env the CLI would silently bill the API — the whole
    # point of the transport is that it must not.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "tok")
    monkeypatch.setenv("SOMETHING_ELSE", "kept")
    env = forecaster._subscription_env()
    assert "ANTHROPIC_API_KEY" not in env
    assert "ANTHROPIC_AUTH_TOKEN" not in env
    assert env["SOMETHING_ELSE"] == "kept"


# --- JSON parsing ---------------------------------------------------------------------

def test_parse_forecast_json_strict():
    p, r = forecaster.parse_forecast_json(
        '{"probability": 0.62, "reasoning": "Uptrend intact."}')
    assert p == 0.62 and r == "Uptrend intact."


def test_parse_forecast_json_tolerates_fences_and_prose():
    text = 'Here is my forecast:\n```json\n{"probability": 0.41, "reasoning": "x"}\n```'
    p, r = forecaster.parse_forecast_json(text)
    assert p == 0.41 and r == "x"


def test_parse_forecast_json_clamps_certainty():
    p, _ = forecaster.parse_forecast_json('{"probability": 1.0, "reasoning": "sure"}')
    assert p == 0.99  # impossible certainty clamped, same as the API path


def test_parse_forecast_json_refuses_garbage():
    with pytest.raises(RuntimeError, match="not the required JSON"):
        forecaster.parse_forecast_json("I think it goes up.")
    with pytest.raises(RuntimeError, match="not the required JSON"):
        forecaster.parse_forecast_json('{"probability": 0.5}')  # missing reasoning


# --- end-to-end through an injected runner -----------------------------------------------

def _proc(stdout: str, code: int = 0, stderr: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=code,
                                       stdout=stdout, stderr=stderr)


def test_forecast_via_subscription_stamps_the_transport():
    seen = {}

    def run(cmd, env, cwd):
        seen["cmd"], seen["env"], seen["cwd"] = cmd, env, cwd
        return _proc(json.dumps({
            "type": "result", "is_error": False,
            "result": '{"probability": 0.58, "reasoning": "Momentum holds."}',
        }))

    res = forecaster.forecast_via_subscription(
        "SYS", "USER", model="claude-opus-4-8", run=run)
    assert res.probability == 0.58
    assert res.reasoning == "Momentum holds."
    # The row's model descriptor records the auth path (changeover token).
    assert res.model == "claude-opus-4-8+via:claude-code"
    assert "ANTHROPIC_API_KEY" not in seen["env"]


def test_forecast_via_subscription_surfaces_cli_failures():
    with pytest.raises(RuntimeError, match="exited 1"):
        forecaster.forecast_via_subscription(
            "s", "u", model="m",
            run=lambda c, e, w: _proc("", code=1, stderr="limit reached"))
    with pytest.raises(RuntimeError, match="returned an error"):
        forecaster.forecast_via_subscription(
            "s", "u", model="m",
            run=lambda c, e, w: _proc(json.dumps(
                {"is_error": True, "result": "usage limit reached"})))
