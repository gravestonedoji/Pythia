"""The public dashboard (roadmap v4) — pure generation, `scoring.py`-style.

Two pure stages: ``build_dashboard_data(rows, taint, ...)`` reduces the record
to AGGREGATES, and ``render_html(data)`` / ``render_json(data)`` turn those
into a single self-contained page (inline CSS, hand-rolled inline SVG, no
scripts, no CDN, no webfonts — it works from file:// and GitHub Pages alike).
All I/O lives in the CLI command.

AGGREGATES ONLY — the page must never leak the DB by another door. No claim
text, no per-claim probabilities, no LLM reasoning, no pending calls (only
pending COUNTS; publishing live probabilities before resolution is an owner-
level disclosure decision, deliberately not taken here). The leak-guard tests
plant sentinel strings in rows and assert they never reach the output.

HONESTY FURNITURE IS FIRST-CLASS CONTENT, rendered from data, not hardcoded:
- the month-3 embargo banner (record age < DASH_RANK_EMBARGO_DAYS), which
  auto-demotes to the standing correlated-samples note once passed;
- Wilson 95% whiskers on every calibration bin — fat whiskers on 25-count
  bins are the honest visual statement of how thin the record is;
- equal-count (quantile) bins with a hard per-bin floor, and NO curve at all
  below DASH_CAL_MIN_RESOLVED (LLM probabilities cluster around 0.45-0.70, so
  fixed-width deciles would render six empty bins that LOOK like data; and a
  missing curve beats a fudged one — the kalshi/iso rule);
- the HMM taint annotation (verbatim reuse of hmm_health's lines), the kalshi
  sparse-coverage note, and a fixed research/no-advice/no-live-trading
  disclaimer.

CHANGE DETECTION: content_sha hashes the canonical aggregate payload EXCLUDING
the generated-at stamp, so a weekend run with nothing new resolved hashes
identically and `pythia publish` skips the no-op write (the `publishes` table
is the audit trail). Publishing the docs/ folder anywhere (git, Pages) is a
separate HUMAN decision — nothing in this module or the publish command
touches git.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from datetime import date
from html import escape

from . import config, scoring
from .hmm_health import integrity_lines


# --- pure statistics ---------------------------------------------------------------

def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Closed-form Wilson 95% interval for k successes in n trials (no scipy).

    NOTE the caveat rendered next to every use: Wilson assumes independent
    trials, and overlapping 5-session claims are not independent — the real
    intervals are wider than these.
    """
    if n <= 0:
        return 0.0, 1.0
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, centre - half), min(1.0, centre + half)


@dataclass
class CalBin:
    p_mean: float   # mean predicted probability in the bin
    freq: float     # realized frequency
    n: int
    lo: float       # Wilson 95%
    hi: float


@dataclass
class CalCurve:
    forecaster: str
    label: str
    n_fit: int
    bins: list[CalBin] = field(default_factory=list)
    skip_reason: str | None = None


def calibration_bins(pts: list[tuple[float, float]], *,
                     min_bin: int | None = None,
                     max_bins: int | None = None) -> list[CalBin]:
    """Equal-count bins over (predicted p, outcome) points.

    Quantile bins keep every bin populated where the probabilities cluster;
    the per-bin floor keeps the whiskers meaningful. Bin count degrades
    gracefully as the record thins: min(max_bins, n // min_bin).
    """
    min_bin = config.DASH_CAL_MIN_BIN if min_bin is None else min_bin
    max_bins = config.DASH_CAL_MAX_BINS if max_bins is None else max_bins
    pts = sorted(pts, key=lambda t: t[0])
    n = len(pts)
    n_bins = min(max_bins, n // min_bin)
    if n_bins < 1:
        return []
    bounds = [round(i * n / n_bins) for i in range(n_bins + 1)]
    out: list[CalBin] = []
    for i in range(n_bins):
        chunk = pts[bounds[i]:bounds[i + 1]]
        if not chunk:
            continue
        k = sum(o for _, o in chunk)
        lo, hi = wilson_interval(int(k), len(chunk))
        out.append(CalBin(
            p_mean=sum(p for p, _ in chunk) / len(chunk),
            freq=k / len(chunk), n=len(chunk), lo=lo, hi=hi,
        ))
    return out


# --- aggregation ---------------------------------------------------------------------

@dataclass
class BoardRow:
    forecaster: str
    label: str
    resolved: int
    pending: int
    hit_rate: float | None
    avg_brier: float | None
    tainted: bool = False


@dataclass
class DashboardData:
    generated_at: str
    n_rows: int
    n_resolved: int          # resolved rows across all columns
    n_claims_resolved: int   # resolved claims (raw-arm rows — the control arm is on every claim)
    pending_claims: int
    board: list[BoardRow]
    curves: list[CalCurve]
    brier_series: dict[str, list[tuple[str, float]]]  # cumulative avg Brier by resolves_on
    accum_series: list[tuple[str, int]]               # resolved claims over time
    taint_lines: list[str]
    kalshi_note: str
    embargo: str | None
    paper_lines: list[str]
    alert_lines: list[str]
    content_sha: str = ""


# Curves are drawn for the arms and the quant bar (its calibration is part of
# deploy gate #1). coin_flip/drift/momentum are excluded: constant-probability
# forecasters have degenerate single-point curves.
_CURVE_FORECASTERS = [*config.PYTHIA_ARMS, config.HMM_FILTER]

# Time-series lines stay legible by requiring a minimum record per arm;
# coin_flip is included as the flat 0.25 reference.
_SERIES_MIN_RESOLVED = 25


def build_dashboard_data(
    rows, taint, *,
    today: date,
    generated_at: str,
    paper_rows=None,
    alert_stats=None,
) -> DashboardData:
    """Reduce the forecast record (plus optional v2/v3 records) to aggregates.

    `taint` is hmm_health.taint_summary(rows); `paper_rows` are
    paper_positions rows; `alert_stats` is digest.alert_scoreboard's
    (pooled, per_arm) tuple. Everything derived, nothing configured.
    """
    stats = scoring.summarize(rows)

    def sort_key(s):
        return (0, s.avg_brier) if s.avg_brier is not None else (1, 0.0)

    board = [
        BoardRow(
            forecaster=s.forecaster,
            label=config.FORECASTER_LABELS.get(s.forecaster, s.forecaster),
            resolved=s.resolved, pending=s.pending,
            hit_rate=round(s.hit_rate, 4) if s.hit_rate is not None else None,
            avg_brier=round(s.avg_brier, 4) if s.avg_brier is not None else None,
            tainted=(s.forecaster == config.HMM_FILTER and taint.flagged > 0),
        )
        for s in sorted(stats.values(), key=sort_key)
        if s.resolved or s.pending
    ]

    resolved_rows = [r for r in rows
                     if r["status"] == "resolved" and r["outcome"] is not None]

    curves: list[CalCurve] = []
    for fc in _CURVE_FORECASTERS:
        pts = [(r["probability"], r["outcome"]) for r in resolved_rows
               if r["forecaster"] == fc]
        if not pts and not (stats.get(fc) and stats[fc].pending):
            continue  # arm not live at all — nothing to say yet
        label = config.FORECASTER_LABELS.get(fc, fc)
        if len(pts) < config.DASH_CAL_MIN_RESOLVED:
            curves.append(CalCurve(
                forecaster=fc, label=label, n_fit=len(pts),
                skip_reason=(f"curve appears at {config.DASH_CAL_MIN_RESOLVED} "
                             f"resolved (currently {len(pts)})")))
            continue
        curves.append(CalCurve(forecaster=fc, label=label, n_fit=len(pts),
                               bins=calibration_bins(pts)))

    brier_series: dict[str, list[tuple[str, float]]] = {}
    for fc in [*config.PYTHIA_ARMS, config.HMM_FILTER, "coin_flip"]:
        fc_rows = sorted((r for r in resolved_rows if r["forecaster"] == fc
                          and r["brier"] is not None),
                         key=lambda r: r["resolves_on"])
        if len(fc_rows) < _SERIES_MIN_RESOLVED:
            continue
        pts: list[tuple[str, float]] = []
        total = 0.0
        for i, r in enumerate(fc_rows, start=1):
            total += r["brier"]
            point = (r["resolves_on"], round(total / i, 4))
            if pts and pts[-1][0] == point[0]:
                pts[-1] = point  # one point per resolution date (the last)
            else:
                pts.append(point)
        brier_series[fc] = pts

    # Record accumulation counts CLAIMS (raw-arm rows: the control arm exists
    # on every claim, so its count is the claim count).
    raw_resolved = sorted((r for r in resolved_rows
                           if r["forecaster"] == config.PYTHIA),
                          key=lambda r: r["resolves_on"])
    accum: list[tuple[str, int]] = []
    for i, r in enumerate(raw_resolved, start=1):
        if accum and accum[-1][0] == r["resolves_on"]:
            accum[-1] = (r["resolves_on"], i)
        else:
            accum.append((r["resolves_on"], i))

    embargo = None
    if raw_resolved:
        first = date.fromisoformat(raw_resolved[0]["resolves_on"])
        age = (today - first).days
        if age < config.DASH_RANK_EMBARGO_DAYS:
            embargo = (
                f"EARLY RECORD — do not rank yet. The first claim resolved "
                f"{age} days ago; the first honest leaderboard read is month 3 "
                f"({config.DASH_RANK_EMBARGO_DAYS} days). Claims within a day and "
                "across overlapping windows are correlated, so the effective "
                "sample is far smaller than the row counts."
            )

    k_res = sum(1 for r in resolved_rows if r["forecaster"] == config.KALSHI)
    k_pend = stats[config.KALSHI].pending if config.KALSHI in stats else 0
    kalshi_note = (
        f"The market bar is sparse by design: {k_res} resolved / {k_pend} pending "
        f"rows, mapped tickers only ({', '.join(sorted(config.KALSHI_UNDERLYING))}), "
        "matched contracts may settle up to 2 sessions off the claim (the gap is "
        "recorded per row and slightly eases the \"beat the market\" gate)."
    )

    paper_lines: list[str] = []
    if paper_rows:
        n_pos = len(paper_rows)
        n_settled = sum(1 for r in paper_rows if r["status"] == "settled")
        anchors = {r["anchor_date"] for r in paper_rows}
        paper_lines.append(
            f"{n_pos} simulated positions ({n_settled} settled) across "
            f"{len(anchors)} anchor date(s); entries at the logged after-close "
            "mid, settled at intrinsic off the official close. No orders, ever."
        )
        if n_settled:
            from . import pnl as paper_pnl
            books = paper_pnl.ladder(paper_rows, policies=("fixed",))
            best = sorted(
                ((fc, b) for (fc, _), b in books.items() if b.trades),
                key=lambda t: -t[1].pnl)
            parts = [
                f"{config.FORECASTER_LABELS.get(fc, fc)} ${b.pnl:+,.0f} on "
                f"${b.staked:,.0f} staked" for fc, b in best]
            paper_lines.append("Fixed-stake books (settled trades only): "
                               + "; ".join(parts) + ".")

    alert_lines: list[str] = []
    if alert_stats is not None:
        pooled, per_arm = alert_stats
        if pooled.n_resolved or per_arm:
            alert_lines.append(
                f"Alert rule (|P-0.5| >= {config.ALERT_CONVICTION_MIN}, live LLM "
                f"arms): pooled {pooled.n_resolved} resolved / "
                f"{pooled.n_correct} correct"
                + (f" / avg Brier {pooled.avg_brier:.3f}" if pooled.n_resolved else "")
                + " (coin = 0.250). Threshold chosen post-hoc; this record is "
                  "its out-of-sample test.")
            for arm in sorted(per_arm):
                s = per_arm[arm]
                alert_lines.append(
                    f"{arm}: {s.n_resolved} resolved / {s.n_correct} correct"
                    + (f" / avg Brier {s.avg_brier:.3f}" if s.n_resolved else ""))

    data = DashboardData(
        generated_at=generated_at,
        n_rows=len(rows), n_resolved=len(resolved_rows),
        n_claims_resolved=len(raw_resolved),
        pending_claims=stats[config.PYTHIA].pending if config.PYTHIA in stats else 0,
        board=board, curves=curves, brier_series=brier_series,
        accum_series=accum, taint_lines=list(integrity_lines(taint)),
        kalshi_note=kalshi_note, embargo=embargo,
        paper_lines=paper_lines, alert_lines=alert_lines,
    )
    data.content_sha = hashlib.sha256(
        json.dumps(_payload(data), sort_keys=True).encode()).hexdigest()[:16]
    return data


def _payload(data: DashboardData) -> dict:
    """The canonical aggregate payload — EXCLUDES generated_at so an unchanged
    record hashes identically day to day (macro context_sha precedent)."""
    return {
        "generator": config.DASH_GENERATOR,
        "n_rows": data.n_rows,
        "n_resolved": data.n_resolved,
        "n_claims_resolved": data.n_claims_resolved,
        "pending_claims": data.pending_claims,
        "board": [{
            "forecaster": b.forecaster, "label": b.label, "resolved": b.resolved,
            "pending": b.pending, "hit_rate": b.hit_rate,
            "avg_brier": b.avg_brier, "tainted": b.tainted,
        } for b in data.board],
        "curves": [{
            "forecaster": c.forecaster, "n_fit": c.n_fit,
            "skip_reason": c.skip_reason,
            "bins": [{"p_mean": round(b.p_mean, 4), "freq": round(b.freq, 4),
                      "n": b.n, "lo": round(b.lo, 4), "hi": round(b.hi, 4)}
                     for b in c.bins],
        } for c in data.curves],
        "brier_series": data.brier_series,
        "accum_series": data.accum_series,
        "taint_lines": data.taint_lines,
        "kalshi_note": data.kalshi_note,
        # Boolean, not the text: the banner embeds the record's age in days,
        # which would change the sha every calendar day and defeat the
        # unchanged-skip during the entire embargo window.
        "embargo_active": data.embargo is not None,
        "paper_lines": data.paper_lines,
        "alert_lines": data.alert_lines,
    }


def render_json(data: DashboardData) -> str:
    """The same aggregates machine-readably — keeps 'aggregates only' auditable."""
    return json.dumps(
        {**_payload(data), "generated_at": data.generated_at,
         "content_sha": data.content_sha},
        sort_keys=True, indent=1)


# --- SVG (hand-rolled: text, deterministic, zero deps) --------------------------------

_COLORS = {
    "pythia": "#0e7490", "pythia_coached": "#7c3aed", "pythia_macro": "#b45309",
    "pythia_iso": "#0f766e", "pythia_coached_iso": "#4d7c0f",
    "hmm_filter": "#b91c1c", "kalshi": "#be185d", "coin_flip": "#6b7280",
    "drift": "#57534e", "naive_momentum": "#78716c",
}


def _svg_calibration(curve: CalCurve, size: int = 240) -> str:
    pad, plot = 34, size - 2 * 34
    x = lambda p: pad + p * plot          # noqa: E731
    y = lambda f: size - pad - f * plot   # noqa: E731
    color = _COLORS.get(curve.forecaster, "#111")
    s = [f'<svg viewBox="0 0 {size} {size}" role="img" '
         f'aria-label="calibration: {escape(curve.label)}">']
    s.append(f'<rect x="{pad}" y="{pad}" width="{plot}" height="{plot}" '
             'fill="none" stroke="#d4d4d4"/>')
    s.append(f'<line x1="{x(0)}" y1="{y(0)}" x2="{x(1)}" y2="{y(1)}" '
             'stroke="#d4d4d4" stroke-dasharray="4 3"/>')
    for t in (0.0, 0.5, 1.0):
        s.append(f'<text x="{x(t):.0f}" y="{size - pad + 14}" font-size="9" '
                 f'text-anchor="middle" fill="#666">{t:.1f}</text>')
        s.append(f'<text x="{pad - 6}" y="{y(t) + 3:.0f}" font-size="9" '
                 f'text-anchor="end" fill="#666">{t:.1f}</text>')
    for b in curve.bins:
        bx = x(b.p_mean)
        s.append(f'<line x1="{bx:.1f}" y1="{y(b.lo):.1f}" x2="{bx:.1f}" '
                 f'y2="{y(b.hi):.1f}" stroke="{color}" stroke-width="1.5" '
                 'opacity="0.55"/>')
        s.append(f'<circle cx="{bx:.1f}" cy="{y(b.freq):.1f}" r="3.2" '
                 f'fill="{color}"><title>predicted {b.p_mean:.2f}, realized '
                 f'{b.freq:.2f}, n={b.n}</title></circle>')
    s.append(f'<text x="{size / 2:.0f}" y="12" font-size="10" text-anchor="middle" '
             f'fill="#333">{escape(curve.label)} (n={curve.n_fit})</text>')
    s.append(f'<text x="{size / 2:.0f}" y="{size - 4}" font-size="9" '
             'text-anchor="middle" fill="#888">predicted P(up)</text>')
    s.append("</svg>")
    return "".join(s)


def _svg_lines(series: dict[str, list[tuple[str, float]]], *,
               width: int = 740, height: int = 240,
               y_label: str, y_max: float | None = None,
               y_ref: float | None = None) -> str:
    pad_l, pad_r, pad_t, pad_b = 44, 10, 16, 30
    plot_w, plot_h = width - pad_l - pad_r, height - pad_t - pad_b
    dates = sorted({d for pts in series.values() for d, _ in pts})
    if not dates or not series:
        return ""
    dx = {d: i for i, d in enumerate(dates)}
    span = max(1, len(dates) - 1)
    vmax = y_max if y_max is not None else max(
        v for pts in series.values() for _, v in pts) * 1.1 or 1.0
    x = lambda d: pad_l + dx[d] / span * plot_w            # noqa: E731
    y = lambda v: pad_t + (1 - min(v, vmax) / vmax) * plot_h  # noqa: E731
    s = [f'<svg viewBox="0 0 {width} {height}" role="img" '
         f'aria-label="{escape(y_label)}">']
    s.append(f'<rect x="{pad_l}" y="{pad_t}" width="{plot_w}" height="{plot_h}" '
             'fill="none" stroke="#d4d4d4"/>')
    for frac in (0.0, 0.5, 1.0):
        v = vmax * frac
        s.append(f'<text x="{pad_l - 6}" y="{y(v) + 3:.0f}" font-size="9" '
                 f'text-anchor="end" fill="#666">{v:.2f}</text>')
    if y_ref is not None and y_ref <= vmax:
        s.append(f'<line x1="{pad_l}" y1="{y(y_ref):.1f}" '
                 f'x2="{pad_l + plot_w}" y2="{y(y_ref):.1f}" stroke="#bbb" '
                 'stroke-dasharray="4 3"/>')
    for d in (dates[0], dates[-1]):
        s.append(f'<text x="{x(d):.0f}" y="{height - 12}" font-size="9" '
                 f'text-anchor="middle" fill="#666">{escape(d)}</text>')
    lx = pad_l + 8
    for fc, pts in series.items():
        color = _COLORS.get(fc, "#111")
        path = " ".join(f"{x(d):.1f},{y(v):.1f}" for d, v in pts)
        s.append(f'<polyline points="{path}" fill="none" stroke="{color}" '
                 'stroke-width="1.6"/>')
        label = escape(config.FORECASTER_LABELS.get(fc, fc))
        s.append(f'<rect x="{lx}" y="{pad_t + 4}" width="8" height="8" fill="{color}"/>')
        s.append(f'<text x="{lx + 11}" y="{pad_t + 12}" font-size="9" '
                 f'fill="#333">{label}</text>')
        lx += 11 + 6.2 * len(label) + 14
    s.append(f'<text x="12" y="{pad_t + 10}" font-size="9" fill="#888" '
             f'transform="rotate(-90 12 {pad_t + 10})" text-anchor="end">'
             f'{escape(y_label)}</text>')
    s.append("</svg>")
    return "".join(s)


# --- HTML ------------------------------------------------------------------------------

_CSS = """
body { font: 14px/1.5 -apple-system, "Segoe UI", system-ui, sans-serif;
       color: #1c1917; background: #fafaf9; margin: 0; }
main { max-width: 860px; margin: 0 auto; padding: 16px 20px 48px; }
h1 { font-size: 22px; margin: 8px 0 2px; }
h2 { font-size: 16px; margin: 28px 0 6px; }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th, td { text-align: left; padding: 4px 10px; border-bottom: 1px solid #e7e5e4; }
td.r, th.r { text-align: right; }
.strip { background: #1c1917; color: #fafaf9; font-size: 12px;
         padding: 6px 20px; }
.banner { background: #fef3c7; border: 1px solid #f59e0b; padding: 8px 12px;
          font-size: 13px; margin: 12px 0; }
.note { color: #57534e; font-size: 12px; }
.taint { color: #92400e; font-size: 12px; }
.grid { display: flex; flex-wrap: wrap; gap: 8px; }
.grid svg { flex: 0 1 240px; max-width: 100%; height: auto; }
.chart { overflow-x: auto; }
.chart svg { min-width: 640px; width: 100%; height: auto; }
footer { margin-top: 36px; font-size: 12px; color: #78716c;
         border-top: 1px solid #e7e5e4; padding-top: 10px; }
.skip { border: 1px dashed #d6d3d1; color: #78716c; font-size: 12px;
        flex: 0 1 240px; display: flex; align-items: center;
        justify-content: center; text-align: center; padding: 12px;
        min-height: 120px; }
"""

_DISCLAIMER = ("A research project measuring LLM forecast calibration. "
               "Simulated forecasting record only — no investment advice, no "
               "recommendations, no live trading; every probability is logged "
               "before the outcome is known and never edited.")


def render_html(data: DashboardData) -> str:
    """The whole page as one self-contained HTML string (deterministic for a
    given DashboardData — the determinism tests depend on it)."""
    h: list[str] = []
    # Without an explicit charset, file:// and some servers default to
    # windows-1252 and the page's em-dashes/≥ render as mojibake.
    h.append('<meta charset="utf-8">')
    h.append('<meta name="viewport" content="width=device-width, initial-scale=1">')
    h.append(f"<title>{escape(config.DASH_TITLE)}</title>")
    h.append(f"<style>{_CSS}</style>")
    h.append(f'<div class="strip">{escape(_DISCLAIMER)}</div>')
    h.append("<main>")
    h.append(f"<h1>{escape(config.DASH_TITLE)}</h1>")
    h.append(f'<p class="note">{data.n_claims_resolved} claims resolved, '
             f'{data.pending_claims} pending, across '
             f'{len(config.WATCHLIST)} ETFs; every forecaster column is graded '
             'on identical claims (raw closes, trading-calendar math, '
             'first-write-wins).</p>')
    if data.embargo:
        h.append(f'<div class="banner">{escape(data.embargo)}</div>')
    else:
        h.append('<p class="note">Claims within a day and across overlapping '
                 'windows are correlated — effective sample size is well below '
                 'the row counts.</p>')

    h.append("<h2>Track record</h2>")
    h.append('<p class="note">Lower Brier is better; 0.25 = always saying '
             '50%. Hit-rate is the coarse directional view only.</p>')
    h.append("<table><tr><th>forecaster</th><th class=r>resolved</th>"
             "<th class=r>pending</th><th class=r>hit-rate</th>"
             "<th class=r>avg Brier</th></tr>")
    for b in data.board:
        label = escape(b.label) + (" *" if b.tainted else "")
        hr = f"{b.hit_rate * 100:.1f}%" if b.hit_rate is not None else "-"
        br = f"{b.avg_brier:.4f}" if b.avg_brier is not None else "-"
        h.append(f"<tr><td>{label}</td><td class=r>{b.resolved}</td>"
                 f"<td class=r>{b.pending}</td><td class=r>{hr}</td>"
                 f"<td class=r>{br}</td></tr>")
    h.append("</table>")
    for line in data.taint_lines:
        h.append(f'<p class="taint">* {escape(line)}</p>')
    h.append(f'<p class="note">{escape(data.kalshi_note)}</p>')

    h.append("<h2>Calibration</h2>")
    h.append('<p class="note">Equal-count bins; whiskers are Wilson 95% '
             'intervals, which assume independent trials — overlapping '
             '5-session claims are not independent, so the true intervals are '
             'wider than shown.</p>')
    h.append('<div class="grid">')
    for c in data.curves:
        if c.skip_reason:
            h.append(f'<div class="skip">{escape(c.label)}:<br>'
                     f'{escape(c.skip_reason)}</div>')
        else:
            h.append(_svg_calibration(c))
    h.append("</div>")

    if data.brier_series:
        h.append("<h2>Brier over time (running average)</h2>")
        h.append('<div class="chart">' + _svg_lines(
            data.brier_series, y_label="avg Brier", y_max=0.4, y_ref=0.25)
            + "</div>")
    if data.accum_series:
        h.append("<h2>Record accumulation</h2>")
        h.append('<div class="chart">' + _svg_lines(
            {"pythia": [(d, float(v)) for d, v in data.accum_series]},
            y_label="claims resolved") + "</div>")

    if data.paper_lines:
        h.append("<h2>Paper book (simulated options)</h2>")
        for line in data.paper_lines:
            h.append(f'<p class="note">{escape(line)}</p>')
    if data.alert_lines:
        h.append("<h2>High-conviction alert rule</h2>")
        for line in data.alert_lines:
            h.append(f'<p class="note">{escape(line)}</p>')

    h.append("<h2>Method</h2>")
    h.append('<p class="note">One falsifiable claim per ETF per trading day '
             '("raw close in 5 sessions &gt;= today\'s raw close") with a '
             'probability logged before the outcome; graded against raw '
             'closes via exchange-calendar math; reference forecasters score '
             'the same claims (coin flip, drift, momentum, a point-in-time '
             'Gaussian-HMM quant bar, and live Kalshi odds where a contract '
             'matches). Correction arms (lessons, macro, isotonic) differ '
             'from the raw arm by exactly one ingredient each, so every layer '
             'is measured, never assumed. No autonomous real-money execution, '
             'ever.</p>')
    h.append("<footer>")
    h.append(f"generated {escape(data.generated_at)} · "
             f"{escape(config.DASH_GENERATOR)} · content {data.content_sha} · "
             f'<a href="https://github.com/gravestonedoji/Pythia">source</a>')
    h.append("</footer></main>")
    return "\n".join(h)
