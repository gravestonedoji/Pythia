"""Pythia command-line interface.

Three commands make up the v0 loop, all runnable by hand:

    pythia forecast   # predict + log for the watchlist (Pythia + baselines)
    pythia resolve    # settle matured forecasts and score them
    pythia review     # print the track record, side by side with the baselines
"""

from __future__ import annotations

import os
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

from . import baselines, config, data, forecaster, scoring, storage
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
            for b in baselines.all_baselines(history):
                rows.append(Forecast(
                    forecaster=b.forecaster, ticker=tk, claim=claim, horizon_days=horizon,
                    anchor_date=anchor_date.isoformat(), anchor_close=anchor_close,
                    resolves_on=resolves_on.isoformat(), probability=b.probability,
                    reasoning=b.reasoning, model=b.model,
                ))

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

    for s in sorted(stats.values(), key=sort_key):
        if s.resolved == 0 and s.pending == 0:
            continue
        label = config.FORECASTER_LABELS.get(s.forecaster, s.forecaster)
        if s.forecaster == config.PYTHIA:
            label = f"[bold cyan]{label}[/bold cyan]"
        board.add_row(
            label, str(s.resolved), str(s.pending),
            _fmt_pct(s.hit_rate), _fmt_brier(s.avg_brier),
        )
    console.print(board)

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
