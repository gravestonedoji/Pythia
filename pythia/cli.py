"""Pythia command-line interface.

Three commands make up the v0 loop, all runnable by hand:

    pythia forecast   # predict + log for the watchlist (Pythia + baselines)
    pythia resolve    # settle matured forecasts and score them
    pythia review     # print the track record, side by side with the baselines
"""

from __future__ import annotations

import os
import re
import sys
from datetime import date
from typing import List, Optional

# Make output robust on legacy Windows consoles: model reasoning (and our own
# formatting) may contain non-ASCII, which would otherwise crash on a cp1252
# stdout. Reconfiguring to UTF-8 with replacement avoids that.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except (AttributeError, ValueError):
        pass

import typer
from rich import box
from rich.console import Console
from rich.table import Table

from . import (
    baselines, calibrate, config, data, forecaster, hmm_baseline, hmm_health,
    kalshi, macro, reflect, scoring, storage,
)
from . import notify as notifier  # aliased: the `notify` command below would shadow it
from .storage import Forecast

app = typer.Typer(
    add_completion=False,
    help="Pythia - a self-grading market forecaster (ETFs only, no real-money execution).",
)
console = Console()


def build_claim(ticker: str, anchor_date: date, anchor_close: float, resolves_on: date) -> str:
    """The canonical, falsifiable claim text (matches scoring.compute_outcome)."""
    return (
        f"{ticker} closing price on {resolves_on.isoformat()} will be greater than or "
        f"equal to its {anchor_date.isoformat()} close of {anchor_close:.2f}."
    )


def _macro_snapshot_for(
    anchor_date_iso: str, cache: dict, conn
) -> "macro.MacroSnapshot | None":
    """Get the point-in-time macro snapshot for one anchor date, fetching once.

    Macro is per-date (shared across every ticker anchored that day), so the
    first ticker on a run pays the FRED fetch and persists the snapshot; the
    rest reuse it from the cache (or the DB on a later run). Returns None when
    FRED is unconfigured or returned nothing usable — the macro arm then no-ops.
    """
    if anchor_date_iso in cache:
        return cache[anchor_date_iso]
    snap = None
    row = storage.fetch_macro_snapshot(conn, anchor_date_iso)
    if row is not None:
        snap = macro.snapshot_from_row(row)
    else:
        try:
            snap = macro.get_macro_snapshot(date.fromisoformat(anchor_date_iso))
        except Exception as exc:  # noqa: BLE001 - FRED down shouldn't kill the run
            console.print(f"  [yellow]macro fetch failed: {exc}[/yellow]")
            snap = None
        if snap is not None:
            storage.insert_macro_snapshot(conn, snap)
    cache[anchor_date_iso] = snap
    return snap


def _fmt_pct(x: Optional[float]) -> str:
    return f"{x * 100:.1f}%" if x is not None else "-"


def _fmt_brier(x: Optional[float]) -> str:
    return f"{x:.4f}" if x is not None else "-"


# --- forecast ----------------------------------------------------------------

@app.command()
def forecast(
    ticker: Optional[List[str]] = typer.Option(
        None, "--ticker", "-t", help="Ticker(s) to forecast (default: the watchlist)."
    ),
    horizon: int = typer.Option(
        config.DEFAULT_HORIZON_DAYS, "--horizon", "-h", help="Horizon in trading sessions."
    ),
    notify: bool = typer.Option(
        True, "--notify/--no-notify",
        help="Email a summary of the predictions made (if SMTP is configured in .env).",
    ),
) -> None:
    """Form and log one forecast per ticker, for Pythia and every baseline."""
    tickers = ticker or config.WATCHLIST
    notifications: list[notifier.Prediction] = []

    if not os.environ.get("ANTHROPIC_API_KEY"):
        console.print(
            "[red]ANTHROPIC_API_KEY is not set.[/red] Copy .env.example to .env and add "
            "your key (the forecast step needs it to call the model)."
        )
        raise typer.Exit(code=1)

    import anthropic

    client = anthropic.Anthropic()
    conn = storage.get_connection()

    # The correction stack, loaded once per run (point-in-time by construction):
    # lessons for the coached arm; per-arm isotonic calibrators fitted on each
    # base arm's OWN resolved record (None until ISO_MIN_RESOLVED rows exist).
    lessons = reflect.load_lessons()
    if lessons:
        console.print(f"[dim]coached arm on (lessons {lessons[1]})[/dim]")
    else:
        console.print("[dim]no lessons.txt yet — run `pythia reflect` once the "
                      "record has matured; only the raw arm will be logged.[/dim]")
    _resolved_rows = storage.fetch_all(conn, status="resolved")
    calibrators = {
        base: calibrate.IsotonicCalibrator.from_resolved_rows(_resolved_rows, base)
        for base in (config.PYTHIA, config.PYTHIA_COACHED)
    }

    # v1 macro arm: identical claim + price data plus a point-in-time FRED
    # macro block. The ONLY difference from raw pythia, so the macro effect is
    # measured on identical claims. No-ops cleanly when FRED_API_KEY is unset
    # (price-only arms unaffected). The snapshot is per-anchor-date and shared
    # across tickers; fetched once, persisted for audit.
    macro_enabled = config.fred_api_key() is not None
    macro_cache: dict[str, object] = {}
    if macro_enabled:
        console.print("[dim]macro arm on (FRED point-in-time)[/dim]")
    else:
        console.print("[dim]FRED_API_KEY not set - pythia_macro arm off "
                      "(price-only arms unaffected).[/dim]")

    issued = 0
    skipped = 0
    for tk in tickers:
        try:
            history = data.get_price_history(tk)
            anchor_date, anchor_close = data.anchor_from_history(history)
            resolves_on = data.trading_days_forward(anchor_date, horizon)
            claim = build_claim(tk, anchor_date, anchor_close, resolves_on)

            console.print(
                f"\n[bold]{tk}[/bold]  anchor {anchor_date} close {anchor_close:.2f}  "
                f"->  resolves {resolves_on} (+{horizon} sessions)"
            )
            console.print(f"  [dim]{claim}[/dim]")

            console.print("  [dim]calling the model...[/dim]")
            pres = forecaster.forecast(
                tk, history,
                claim=claim, horizon_days=horizon,
                anchor_date=anchor_date, anchor_close=anchor_close,
                resolves_on=resolves_on, client=client,
            )

            rows = [Forecast(
                forecaster=config.PYTHIA, ticker=tk, claim=claim, horizon_days=horizon,
                anchor_date=anchor_date.isoformat(), anchor_close=anchor_close,
                resolves_on=resolves_on.isoformat(), probability=pres.probability,
                reasoning=pres.reasoning, model=pres.model,
            )]
            # Coached arm: same claim, same data, lessons appended to the system
            # prompt — the ONLY difference, so coaching stays measurable.
            arm_probs = {config.PYTHIA: pres.probability}
            if lessons:
                cres = forecaster.forecast(
                    tk, history,
                    claim=claim, horizon_days=horizon,
                    anchor_date=anchor_date, anchor_close=anchor_close,
                    resolves_on=resolves_on, client=client,
                    lessons=lessons[0],
                )
                arm_probs[config.PYTHIA_COACHED] = cres.probability
                rows.append(Forecast(
                    forecaster=config.PYTHIA_COACHED, ticker=tk, claim=claim,
                    horizon_days=horizon, anchor_date=anchor_date.isoformat(),
                    anchor_close=anchor_close, resolves_on=resolves_on.isoformat(),
                    probability=cres.probability, reasoning=cres.reasoning,
                    model=f"{cres.model}+lessons:{lessons[1]}",
                ))
            # Macro arm (v1): same claim, same price data, plus a point-in-time
            # FRED macro block in the prompt and the macro system prompt that
            # lifts the price-only restriction. The ONLY difference from raw
            # pythia, so the macro effect is measured on identical claims. The
            # macro sha is recorded in `model` so any record slice ties to the
            # exact macro data the arm saw.
            if macro_enabled:
                snap = _macro_snapshot_for(anchor_date.isoformat(), macro_cache, conn)
                if snap is not None:
                    mres = forecaster.forecast(
                        tk, history,
                        claim=claim, horizon_days=horizon,
                        anchor_date=anchor_date, anchor_close=anchor_close,
                        resolves_on=resolves_on, client=client,
                        macro_context=macro.format_macro_context(snap),
                    )
                    rows.append(Forecast(
                        forecaster=config.PYTHIA_MACRO, ticker=tk, claim=claim,
                        horizon_days=horizon, anchor_date=anchor_date.isoformat(),
                        anchor_close=anchor_close, resolves_on=resolves_on.isoformat(),
                        probability=mres.probability, reasoning=mres.reasoning,
                        model=f"{mres.model}+macro:{snap.context_sha}",
                    ))
            # Derived isotonic arms: free (no model call), strictly point-in-time
            # remaps of the base arms' probabilities. Skipped until the base
            # arm's resolved record clears ISO_MIN_RESOLVED.
            for base, iso_name in ((config.PYTHIA, config.PYTHIA_ISO),
                                   (config.PYTHIA_COACHED, config.PYTHIA_COACHED_ISO)):
                cal = calibrators.get(base)
                if cal is None or base not in arm_probs:
                    continue
                p_iso = cal.apply(arm_probs[base])
                rows.append(Forecast(
                    forecaster=iso_name, ticker=tk, claim=claim, horizon_days=horizon,
                    anchor_date=anchor_date.isoformat(), anchor_close=anchor_close,
                    resolves_on=resolves_on.isoformat(), probability=p_iso,
                    reasoning=(f"Isotonic remap of {base}'s {arm_probs[base]:.2f} "
                               f"-> {p_iso:.2f}, fitted on its {cal.n_fit} resolved "
                               "claims as of issue time."),
                    model=f"derived:isotonic(base={base},n={cal.n_fit})",
                ))
            for b in baselines.all_baselines(history):
                rows.append(Forecast(
                    forecaster=b.forecaster, ticker=tk, claim=claim, horizon_days=horizon,
                    anchor_date=anchor_date.isoformat(), anchor_close=anchor_close,
                    resolves_on=resolves_on.isoformat(), probability=b.probability,
                    reasoning=b.reasoning, model=b.model,
                ))
            # The HMM quant bar needs years of history, so it fetches its own.
            # Best-effort: a young fund (not enough sessions) or a fetch error
            # skips this baseline without touching the others. Every fit's
            # health (convergence + stability vs the previous fit) is recorded
            # and stamped onto the row — a flagged fit still logs its value
            # (dropping it would bias the bar), it just can't hide.
            try:
                deep = data.get_price_history(tk, lookback_days=config.HMM_LOOKBACK_DAYS)
                hb, fit_rec = hmm_baseline.hmm_prediction_with_health(
                    tk, deep, anchor_date, horizon)
                prev_row = storage.latest_hmm_fit_before(conn, tk, anchor_date.isoformat())
                hmm_health.health_check(
                    fit_rec, hmm_health.record_from_row(prev_row) if prev_row else None)
                storage.insert_hmm_fit(conn, fit_rec)
                if hmm_health.is_tainted(fit_rec.flags):
                    console.print(f"  [yellow]hmm_filter fit flagged: {fit_rec.detail}[/yellow]")
                rows.append(Forecast(
                    forecaster=hb.forecaster, ticker=tk, claim=claim, horizon_days=horizon,
                    anchor_date=anchor_date.isoformat(), anchor_close=anchor_close,
                    resolves_on=resolves_on.isoformat(), probability=hb.probability,
                    reasoning=hb.reasoning, model=hb.model,
                    fit_flags=",".join(fit_rec.flags),
                ))
            except Exception as exc:  # noqa: BLE001 - quant bar is best-effort
                console.print(f"  [yellow]hmm_filter skipped: {exc}[/yellow]")
            # Kalshi market odds: read-only, live-book-only, mapped tickers only.
            # Sparse by design — most claims find no contract within the settle
            # tolerance and skip; a missing row beats a fudged one (kalshi.py).
            if tk.upper() in config.KALSHI_SERIES:
                try:
                    kb = kalshi.kalshi_prediction(tk, anchor_date, anchor_close, resolves_on)
                    rows.append(Forecast(
                        forecaster=kb.forecaster, ticker=tk, claim=claim,
                        horizon_days=horizon, anchor_date=anchor_date.isoformat(),
                        anchor_close=anchor_close, resolves_on=resolves_on.isoformat(),
                        probability=kb.probability, reasoning=kb.reasoning, model=kb.model,
                    ))
                except Exception as exc:  # noqa: BLE001 - odds column is best-effort
                    console.print(f"  [yellow]kalshi skipped: {exc}[/yellow]")

            table = Table(box=None, pad_edge=False)
            table.add_column("forecaster")
            table.add_column("P(up)", justify="right")
            table.add_column("status")
            for fc in rows:
                rid = storage.insert_forecast(conn, fc)
                if rid is None:
                    skipped += 1
                    state = "[yellow]already logged[/yellow]"
                else:
                    issued += 1
                    state = "[green]logged[/green]"
                    if fc.forecaster == config.PYTHIA:
                        notifications.append(notifier.Prediction(
                            ticker=fc.ticker, probability=fc.probability,
                            anchor_date=fc.anchor_date, anchor_close=fc.anchor_close,
                            resolves_on=fc.resolves_on, reasoning=fc.reasoning or "",
                        ))
                label = config.FORECASTER_LABELS.get(fc.forecaster, fc.forecaster)
                table.add_row(label, f"{fc.probability * 100:.1f}%", state)
            console.print(table)
            console.print(f"  [dim]Pythia: {pres.reasoning}[/dim]")
        except Exception as exc:  # noqa: BLE001 - one bad ticker shouldn't kill the run
            console.print(f"  [red]{tk}: {exc}[/red]")

    console.print(f"\n[bold green]Done.[/bold green] {issued} logged, {skipped} already present.")

    # Email is best-effort: a logged run must never fail because alerting did.
    if notify and notifications:
        try:
            sent = notifier.notify_predictions(
                notifications, issued_on=date.today().isoformat(), horizon_days=horizon,
            )
            if sent:
                console.print(f"[green]Emailed {len(notifications)} prediction(s).[/green]")
            else:
                console.print(
                    "[dim]Email not configured — set PYTHIA_SMTP_USER and "
                    "PYTHIA_SMTP_PASSWORD in .env to receive prediction alerts.[/dim]"
                )
        except Exception as exc:  # noqa: BLE001 - alerting must not break a logged run
            console.print(f"[yellow]Forecasts logged, but the email failed: {exc}[/yellow]")


# --- reflect -------------------------------------------------------------------

@app.command(name="reflect")
def reflect_command(
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Run the review and print the lessons without saving."
    ),
) -> None:
    """Weekly self-review: distill lessons from the graded record (coached arm).

    Reads only RESOLVED claims, asks the review model for behavior-level lessons,
    and writes lessons.txt — which switches the pythia_coached arm on for every
    subsequent `pythia forecast`. The raw arm is never touched.
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        console.print("[red]ANTHROPIC_API_KEY is not set.[/red]")
        raise typer.Exit(code=1)

    conn = storage.get_connection()
    rows = storage.fetch_all(conn)
    n_resolved = sum(1 for r in rows
                     if r["forecaster"] == config.PYTHIA and r["status"] == "resolved")
    if n_resolved < config.REFLECT_MIN_RESOLVED:
        console.print(
            f"[yellow]Only {n_resolved} resolved Pythia claims; need "
            f">= {config.REFLECT_MIN_RESOLVED} for lessons that generalize. "
            "Let the record mature first.[/yellow]"
        )
        raise typer.Exit(code=1)

    console.print(f"Reviewing {n_resolved} resolved claims with {config.MODEL_REVIEW}...")
    diagnosis, lessons_text = reflect.distill_lessons(rows)
    console.print(f"\n[bold]Diagnosis:[/bold] {diagnosis}\n")
    console.print("[bold]Lessons:[/bold]")
    console.print(lessons_text, markup=False)

    if dry_run:
        console.print("\n[dim]--dry-run: nothing saved.[/dim]")
        return
    sha = reflect.save_lessons(lessons_text, n_resolved=n_resolved)
    console.print(
        f"\n[green]Saved lessons {sha}[/green] -> {config.LESSONS_PATH.name}. "
        "The pythia_coached arm uses them from the next forecast run."
    )


# --- backfill-hmm --------------------------------------------------------------

@app.command(name="backfill-hmm")
def backfill_hmm(
    limit: Optional[int] = typer.Option(None, "--limit", help="Backfill at most N claims."),
) -> None:
    """Add the hmm_filter baseline to existing claims, strictly point-in-time.

    Legitimate reconstruction, not a backtest trick: the HMM forecast for an old
    anchor date uses only raw closes dated <= that anchor — exactly the data that
    was available on the day. Inserted rows are pending; the normal `resolve`
    pass grades any whose resolution date has already arrived.
    """
    conn = storage.get_connection()
    all_rows = storage.fetch_all(conn)

    claims = {}
    for r in all_rows:
        if r["forecaster"] == config.PYTHIA:
            claims[(r["ticker"], r["anchor_date"], r["horizon_days"])] = r
    have = {
        (r["ticker"], r["anchor_date"], r["horizon_days"])
        for r in all_rows if r["forecaster"] == config.HMM_FILTER
    }
    todo = [claims[k] for k in sorted(claims) if k not in have]
    if limit:
        todo = todo[:limit]
    if not todo:
        console.print("[dim]Nothing to backfill — every claim already has an hmm_filter row.[/dim]")
        return

    console.print(f"Backfilling [bold]{len(todo)}[/bold] claims (one deep fetch per ticker, "
                  "one point-in-time fit per claim)...")
    history_cache: dict[str, object] = {}
    done = 0
    failed = 0
    for r in todo:
        tk = r["ticker"]
        anchor = date.fromisoformat(r["anchor_date"])
        try:
            if tk not in history_cache:
                history_cache[tk] = data.get_price_history(
                    tk, lookback_days=config.HMM_LOOKBACK_DAYS)
            hb, fit_rec = hmm_baseline.hmm_prediction_with_health(
                tk, history_cache[tk], anchor, r["horizon_days"])
            prev_row = storage.latest_hmm_fit_before(conn, tk, r["anchor_date"])
            hmm_health.health_check(
                fit_rec, hmm_health.record_from_row(prev_row) if prev_row else None)
            storage.insert_hmm_fit(conn, fit_rec)
            rid = storage.insert_forecast(conn, Forecast(
                forecaster=hb.forecaster, ticker=tk, claim=r["claim"],
                horizon_days=r["horizon_days"], anchor_date=r["anchor_date"],
                anchor_close=r["anchor_close"], resolves_on=r["resolves_on"],
                probability=hb.probability, reasoning=hb.reasoning, model=hb.model,
                fit_flags=",".join(fit_rec.flags),
            ))
            done += 1
            state = "logged" if rid is not None else "already present"
            console.print(f"  {tk} {r['anchor_date']}  P(up)={hb.probability * 100:.1f}%  [dim]{state}[/dim]")
        except Exception as exc:  # noqa: BLE001 - keep going; report at the end
            failed += 1
            console.print(f"  [yellow]{tk} {r['anchor_date']}: {exc}[/yellow]")

    console.print(f"\n[bold green]{done} backfilled[/bold green], {failed} skipped. "
                  "Run `pythia resolve` to grade any that have already matured.")


# --- hmm-health ----------------------------------------------------------------

def _backfill_hmm_health(conn) -> None:
    """Retro-annotate hmm_filter rows that predate health monitoring.

    Fits are deterministic — seeded from (ticker, anchor) over raw closes, on
    the same window (trimmed to the n_obs recorded in the row's model field)
    and under the same EM cap (the row's `em` descriptor token; rows from
    before the cap was raised pin the legacy 40) — so refitting point-in-time
    reproduces the fit behind each logged value up to source-data drift
    (HMM_RECONSTRUCTION_TOL). A reconstruction that lands
    further from the logged probability than that is flagged
    reconstruction_mismatch rather than trusted. Nothing is dropped or altered
    beyond the health mark.
    """
    all_rows = storage.fetch_all(conn)
    fits_by_key = {(f["ticker"], f["anchor_date"]): f
                   for f in storage.fetch_hmm_fits(conn)}
    todo = sorted(
        (r for r in all_rows
         if r["forecaster"] == config.HMM_FILTER and r["fit_flags"] is None),
        key=lambda r: (r["ticker"], r["anchor_date"]),
    )
    if not todo:
        console.print("[dim]Nothing to backfill — every hmm_filter row is health-checked.[/dim]")
        return

    console.print(f"Health-checking [bold]{len(todo)}[/bold] hmm_filter rows "
                  "(point-in-time reconstruction, one deep fetch per ticker)...")
    history_cache: dict[str, object] = {}
    done = 0
    failed = 0
    for r in todo:
        tk = r["ticker"]
        key = (tk, r["anchor_date"])
        try:
            # A health record may already exist (e.g. an interrupted earlier
            # pass): mark the row from it instead of refitting.
            existing = fits_by_key.get(key)
            if existing is not None:
                storage.set_fit_flags(conn, r["id"], existing["flags"])
                done += 1
                continue
            if tk not in history_cache:
                history_cache[tk] = data.get_price_history(
                    tk, lookback_days=config.HMM_LOOKBACK_DAYS)
            anchor = date.fromisoformat(r["anchor_date"])
            hist = history_cache[tk]
            # Reconstruct on the SAME window the live fit saw. Today's deep
            # fetch reaches HMM_LOOKBACK_DAYS back from TODAY, so its window
            # STARTS later than the original's did — a few missing leading
            # sessions makes EM land elsewhere and every row would read as a
            # false mismatch. The original's session count is recorded in its
            # model descriptor; trim the anchor slice to exactly that.
            m = re.search(r"K=\d+,(\d+)d,", r["model"] or "")
            if m is not None:
                closes_idx = [d for d in hist["Close"].dropna().index
                              if d.date() <= anchor]
                hist = hist.loc[closes_idx[-(int(m.group(1)) + 1):]]
            # ... and under the SAME EM cap: rows logged before the cap was
            # raised (2026-06-12) reconstruct with the legacy 40, or a
            # converged refit would read as a false reconstruction_mismatch.
            hb, fit_rec = hmm_baseline.hmm_prediction_with_health(
                tk, hist, anchor, r["horizon_days"],
                em_iters=hmm_baseline.em_cap_from_model(r["model"]))
            prev_row = storage.latest_hmm_fit_before(conn, tk, r["anchor_date"])
            hmm_health.health_check(
                fit_rec, hmm_health.record_from_row(prev_row) if prev_row else None)
            if abs(hb.probability - r["probability"]) > config.HMM_RECONSTRUCTION_TOL:
                fit_rec.flags.append(hmm_health.RECONSTRUCTION_MISMATCH)
                fit_rec.detail = (
                    f"{fit_rec.detail}; reconstruction gave P(up)={hb.probability:.4f} "
                    f"but the logged value is {r['probability']:.4f} — this health "
                    "record describes a different fit than the logged one"
                )
            storage.insert_hmm_fit(conn, fit_rec)
            storage.set_fit_flags(conn, r["id"], ",".join(fit_rec.flags))
            done += 1
            note = (f"[yellow]{', '.join(fit_rec.flags)}[/yellow]"
                    if hmm_health.is_tainted(fit_rec.flags) else "[dim]clean[/dim]")
            console.print(f"  {tk} {r['anchor_date']}  {note}")
        except Exception as exc:  # noqa: BLE001 - keep going; report at the end
            failed += 1
            console.print(f"  [yellow]{tk} {r['anchor_date']}: {exc}[/yellow]")
    console.print(f"\n[bold green]{done} health-checked[/bold green], {failed} skipped.")


@app.command(name="hmm-health")
def hmm_health_command(
    ticker: Optional[str] = typer.Option(None, "--ticker", "-t", help="Only show this ticker."),
    flagged_only: bool = typer.Option(False, "--flagged", help="Only show flagged fits."),
    limit: int = typer.Option(40, "--limit", "-n", help="Max fit rows to show."),
    backfill: bool = typer.Option(
        False, "--backfill",
        help="First reconstruct health records for hmm_filter rows that predate "
             "monitoring (deterministic point-in-time refits; needs network).",
    ),
) -> None:
    """Audit the quant bar: convergence and stability of every recorded HMM fit.

    The hmm_filter baseline is deploy gate #1, so its own fits are monitored:
    EM that fails to converge, or parameters that jump beyond the thresholds in
    config.py between consecutive refits, taint that day's value. Tainted
    values stay on the leaderboard (policy: log + flag, never drop) — this
    command shows exactly which fits are trusted and why.
    """
    conn = storage.get_connection()
    if backfill:
        _backfill_hmm_health(conn)
        console.print()

    fits = storage.fetch_hmm_fits(conn, ticker.upper() if ticker else None)
    if not fits:
        console.print("[dim]No fit-health records yet. Run `pythia forecast` or "
                      "`pythia hmm-health --backfill`.[/dim]")
        return

    n_nonconv = sum(1 for f in fits if not f["converged"])
    n_tainted = sum(1 for f in fits if hmm_health.is_tainted(hmm_health.parse_flags(f["flags"])))
    console.print(
        f"[bold]{len(fits)}[/bold] fits recorded — "
        f"{n_nonconv} non-converged, {n_tainted} tainted, {len(fits) - n_tainted} clean."
    )
    for line in hmm_health.integrity_lines(hmm_health.taint_summary(storage.fetch_all(conn))):
        console.print(f"[yellow]{line}[/yellow]")

    shown = fits
    if flagged_only:
        shown = [f for f in fits if f["flags"]]

    table = Table(title=f"HMM fit health (showing up to {limit})", box=box.ASCII)
    table.add_column("anchor", no_wrap=True)
    table.add_column("ticker", no_wrap=True)
    table.add_column("K", justify="right", no_wrap=True)
    table.add_column("n_obs", justify="right", no_wrap=True)
    table.add_column("iters", justify="right", no_wrap=True)
    table.add_column("converged", no_wrap=True)
    table.add_column("flags")
    table.add_column("detail")
    for f in shown[:limit]:
        conv = "[green]yes[/green]" if f["converged"] else "[red]NO[/red]"
        flags = hmm_health.parse_flags(f["flags"])
        flag_text = (f"[yellow]{', '.join(flags)}[/yellow]" if flags else "[dim]-[/dim]")
        table.add_row(
            f["anchor_date"], f["ticker"], str(f["n_states"]), str(f["n_obs"]),
            str(f["n_iter"]), conv, flag_text, f["detail"] or "-",
        )
    console.print(table)


# --- resolve -----------------------------------------------------------------

@app.command()
def resolve(
    on: Optional[str] = typer.Option(
        None, "--date", help="Resolve as of this ISO date (default: today). For backfilling."
    ),
) -> None:
    """Settle every matured forecast against the real close and score it."""
    today = date.fromisoformat(on) if on else date.today()
    conn = storage.get_connection()
    results = scoring.resolve_due(conn, today=today)

    if not results:
        console.print("[dim]Nothing due to resolve.[/dim]")
        return

    table = Table(title=f"Resolution as of {today.isoformat()}", box=box.ASCII)
    table.add_column("ticker", no_wrap=True)
    table.add_column("forecaster")
    table.add_column("resolves_on", no_wrap=True)
    table.add_column("outcome", no_wrap=True)
    table.add_column("Brier", justify="right", no_wrap=True)
    table.add_column("detail")

    resolved = 0
    skipped = 0
    for r in results:
        if r.status == "resolved":
            resolved += 1
            outcome = "[green]TRUE[/green]" if r.outcome == 1.0 else "[red]FALSE[/red]"
            table.add_row(
                r.ticker, config.FORECASTER_LABELS.get(r.forecaster, r.forecaster),
                r.resolves_on, outcome, _fmt_brier(r.brier), r.detail,
            )
        else:
            skipped += 1
            table.add_row(
                r.ticker, config.FORECASTER_LABELS.get(r.forecaster, r.forecaster),
                r.resolves_on, "[yellow]skipped[/yellow]", "-", r.detail,
            )
    console.print(table)
    console.print(f"\n[bold green]{resolved} resolved[/bold green], {skipped} skipped.")


# --- review ------------------------------------------------------------------

@app.command()
def review(
    ticker: Optional[str] = typer.Option(None, "--ticker", "-t", help="Filter the call log by ticker."),
    forecaster_filter: Optional[str] = typer.Option(
        None, "--forecaster", "-f", help="Filter the call log by forecaster."
    ),
    status: Optional[str] = typer.Option(None, "--status", help="Filter the call log: pending | resolved."),
    limit: int = typer.Option(40, "--limit", "-n", help="Max rows in the call log."),
    log: bool = typer.Option(True, "--log/--no-log", help="Show the per-call log."),
) -> None:
    """Print the track record: leaderboard + every call, side by side with baselines."""
    conn = storage.get_connection()
    all_rows = storage.fetch_all(conn)

    if not all_rows:
        console.print("[dim]No forecasts logged yet. Run `pythia forecast` first.[/dim]")
        return

    stats = scoring.summarize(all_rows)

    # Leaderboard, best (lowest avg Brier) first; unscored forecasters last.
    board = Table(title="Track record - lower Brier is better (0.25 = always 0.50)", box=box.ASCII)
    board.add_column("forecaster")
    board.add_column("resolved", justify="right")
    board.add_column("pending", justify="right")
    board.add_column("hit-rate", justify="right")
    board.add_column("avg Brier", justify="right")

    def sort_key(s: scoring.ForecasterStats):
        return (0, s.avg_brier) if s.avg_brier is not None else (1, 0.0)

    # Leaderboard integrity: the quant bar is only a fair gate if its fits are
    # trusted — values from non-converged or unstable fits stay on the board
    # (dropping them would bias it) but must be visible (hmm_health.py).
    taint = hmm_health.taint_summary(all_rows)

    for s in sorted(stats.values(), key=sort_key):
        if s.resolved == 0 and s.pending == 0:
            continue
        label = config.FORECASTER_LABELS.get(s.forecaster, s.forecaster)
        if s.forecaster == config.PYTHIA:
            label = f"[bold cyan]{label}[/bold cyan]"
        if s.forecaster == config.HMM_FILTER and taint.flagged:
            label = f"{label} [yellow]*[/yellow]"
        board.add_row(
            label, str(s.resolved), str(s.pending),
            _fmt_pct(s.hit_rate), _fmt_brier(s.avg_brier),
        )
    console.print(board)
    for line in hmm_health.integrity_lines(taint):
        console.print(f"[yellow]* {line}[/yellow]")

    if not log:
        return

    rows = all_rows
    if ticker:
        rows = [r for r in rows if r["ticker"] == ticker.upper()]
    if forecaster_filter:
        rows = [r for r in rows if r["forecaster"] == forecaster_filter]
    if status:
        rows = [r for r in rows if r["status"] == status]

    call_log = Table(title=f"Call log (showing up to {limit})", box=box.ASCII)
    call_log.add_column("issued", no_wrap=True)
    call_log.add_column("ticker", no_wrap=True)
    call_log.add_column("forecaster")
    call_log.add_column("P(up)", justify="right", no_wrap=True)
    call_log.add_column("resolves_on", no_wrap=True)
    call_log.add_column("outcome", no_wrap=True)
    call_log.add_column("Brier", justify="right", no_wrap=True)

    for r in rows[:limit]:
        if r["status"] == "resolved":
            outcome = "[green]TRUE[/green]" if r["outcome"] == 1.0 else "[red]FALSE[/red]"
        else:
            outcome = "[dim]pending[/dim]"
        call_log.add_row(
            r["issued_at"][:10], r["ticker"],
            config.FORECASTER_LABELS.get(r["forecaster"], r["forecaster"]),
            f"{r['probability'] * 100:.1f}%", r["resolves_on"],
            outcome, _fmt_brier(r["brier"]),
        )
    console.print(call_log)


# --- why ---------------------------------------------------------------------

@app.command()
def why(
    ticker: Optional[str] = typer.Option(None, "--ticker", "-t", help="Only show this ticker."),
    on: Optional[str] = typer.Option(
        None, "--date", help="Show the batch issued on this date (default: the latest)."
    ),
    all_batches: bool = typer.Option(
        False, "--all", help="Show every batch, not just the most recent one."
    ),
) -> None:
    """Show Pythia's predictions with the reasoning behind each one."""
    conn = storage.get_connection()
    rows = [r for r in storage.fetch_all(conn) if r["forecaster"] == config.PYTHIA]
    if ticker:
        rows = [r for r in rows if r["ticker"] == ticker.upper()]
    if not rows:
        console.print("[dim]No Pythia forecasts logged yet. Run a forecast first.[/dim]")
        return

    issue_dates = sorted({r["issued_at"][:10] for r in rows}, reverse=True)
    if on:
        keep = {on}
    elif all_batches:
        keep = set(issue_dates)
    else:
        keep = {issue_dates[0]}

    shown = [r for r in rows if r["issued_at"][:10] in keep]
    if not shown:
        console.print(f"[dim]No Pythia forecasts issued on {on}.[/dim]")
        return

    # Newest batch first; within a batch, most bullish (highest P) first.
    for d in sorted({r["issued_at"][:10] for r in shown}, reverse=True):
        batch = sorted(
            (r for r in shown if r["issued_at"][:10] == d),
            key=lambda r: -r["probability"],
        )
        console.print(
            f"\n[bold]Pythia predictions issued {d}[/bold]  "
            "[dim](P(up) = chance the ETF closes at or above its anchor price)[/dim]"
        )
        for r in batch:
            verdict = ""
            if r["status"] == "resolved":
                if r["outcome"] == 1.0:
                    verdict = "   [green]came TRUE[/green]"
                else:
                    verdict = "   [red]came FALSE[/red]"
            console.print(
                f"\n[bold cyan]{r['ticker']}[/bold cyan]  "
                f"P(up) = {r['probability'] * 100:.1f}%   "
                f"[dim]resolves {r['resolves_on']}[/dim]{verdict}"
            )
            # markup=False so any brackets in the model's text are shown literally.
            console.print(f"  {r['reasoning']}", style="dim", markup=False)


# --- notify ------------------------------------------------------------------

@app.command()
def notify(
    on: Optional[str] = typer.Option(
        None, "--date", help="Email the Pythia batch issued on this date (default: the latest)."
    ),
    test: bool = typer.Option(
        False, "--test", help="Send a tiny test email to confirm SMTP works, then exit."
    ),
) -> None:
    """Email a summary of a Pythia forecast batch (re-send an existing batch, or --test SMTP)."""
    if config.email_config() is None:
        console.print(
            "[red]Email not configured.[/red] Set PYTHIA_SMTP_USER and "
            "PYTHIA_SMTP_PASSWORD (optionally PYTHIA_EMAIL_TO) in .env. See .env.example."
        )
        raise typer.Exit(code=1)

    if test:
        notifier.send_email(
            "Pythia - test email",
            "If you can read this, Pythia's email alerts are configured correctly.",
        )
        console.print("[green]Test email sent.[/green]")
        return

    conn = storage.get_connection()
    rows = [r for r in storage.fetch_all(conn) if r["forecaster"] == config.PYTHIA]
    if not rows:
        console.print("[dim]No Pythia forecasts logged yet. Run `pythia forecast` first.[/dim]")
        raise typer.Exit(code=1)

    issue_dates = sorted({r["issued_at"][:10] for r in rows}, reverse=True)
    target = on or issue_dates[0]
    batch = [r for r in rows if r["issued_at"][:10] == target]
    if not batch:
        console.print(f"[dim]No Pythia forecasts issued on {target}.[/dim]")
        raise typer.Exit(code=1)

    preds = [
        notifier.Prediction(
            ticker=r["ticker"], probability=r["probability"],
            anchor_date=r["anchor_date"], anchor_close=r["anchor_close"],
            resolves_on=r["resolves_on"], reasoning=r["reasoning"] or "",
        )
        for r in batch
    ]
    notifier.notify_predictions(
        preds, issued_on=target, horizon_days=batch[0]["horizon_days"],
    )
    console.print(f"[green]Emailed {len(preds)} prediction(s) from {target}.[/green]")


if __name__ == "__main__":
    app()
