"""Audit: does the quant bar's EM converge at the production cap (iters=40)?

Backstory: Delphi found its own Baum-Welch cap (iters=100) silently truncated
the fit mid-climb; running to tolerance moved its quant bar from ~56% to ~87%
signal capture (Delphi summary.md sections 17-18). Pythia's hmm-health pass of
2026-06-12 already found 130/134 recorded fits non-converged at iters=40 —
this script measures what that truncation costs and what a converged bar
would have said instead, on the SAME data and seeds the live loop uses.

MEASUREMENT ONLY: no DB writes, no logged row touched, no defaults changed by
running this. One deep fetch per watchlist ticker (network, same as the daily
loop), then per ticker:
  - the production fit  (iters=40, restarts=2, live (ticker, anchor) seed)
  - the converged fit   (same everything, cap high enough to meet tolerance)
and the actual bar value P(close >= anchor in 5 sessions) under each, with the
live MC seeds — so |dP| is exactly how much the logged value would have moved.

Run:  .venv\\Scripts\\python.exe audit_em_convergence.py     (results also
saved to audit_em_convergence.json)
"""
from __future__ import annotations

import json
import time

import numpy as np

from pythia import config, data
from pythia.hmm_baseline import (_seed_for, filter_belief, fit_hmm,
                                 prob_up_over_horizon)

PROD_CAP = 40       # today's fit_hmm default
HIGH_CAP = 8000     # generous; the loop breaks at tolerance long before this
TRAJECTORY_TICKER = "SPY"
TRAJECTORY_CAPS = [40, 100, 300, 1000, 3000]

rows: list[dict] = []
t_all = time.time()
for tk in config.WATCHLIST:
    try:
        hist = data.get_price_history(tk, lookback_days=config.HMM_LOOKBACK_DAYS)
        anchor, _ = data.anchor_from_history(hist)
        past = hist[[d.date() <= anchor for d in hist.index]]
        closes = past["Close"].dropna()
        r = np.diff(np.log(closes.values))
        n = len(r)
        if n < config.HMM_MIN_SESSIONS:
            print(f"{tk:>5}: only {n} sessions — skipped (young fund, same as live)")
            continue
        k = config.HMM_STATES if n >= config.HMM_RICH_HISTORY_SESSIONS else 2
        seed = _seed_for(tk, anchor)

        t0 = time.time()
        f40 = fit_hmm(r, K=k, iters=PROD_CAP, seed=seed)
        t40 = time.time() - t0
        t0 = time.time()
        fhi = fit_hmm(r, K=k, iters=HIGH_CAP, seed=seed)
        thi = time.time() - t0

        def p_up(fit):
            b = filter_belief(r, fit)
            p = prob_up_over_horizon(fit, b, config.DEFAULT_HORIZON_DAYS,
                                     n_paths=config.HMM_MC_PATHS, seed=seed + 1)
            return float(np.clip(p, 0.01, 0.99))

        p40, phi = p_up(f40), p_up(fhi)
        rows.append(dict(
            ticker=tk, anchor=anchor.isoformat(), n_obs=n, K=k,
            conv40=f40.converged, n_iter40=f40.n_iter, ll40=f40.loglik,
            conv_hi=fhi.converged, n_iter_hi=fhi.n_iter, ll_hi=fhi.loglik,
            dll=fhi.loglik - f40.loglik, p40=p40, p_hi=phi, dp=abs(phi - p40),
            t40=round(t40, 2), t_hi=round(thi, 2),
        ))
        print(f"{tk:>5} n={n:>5} K={k}  conv@40={str(f40.converged):<5} "
              f"tol met at {fhi.n_iter:>5} iters ({'yes' if fhi.converged else 'STILL NO'})  "
              f"dloglik={fhi.loglik - f40.loglik:>9.2f}  "
              f"P(up) {p40:.3f}->{phi:.3f} |dP|={abs(phi - p40):.3f}  "
              f"fit {t40:.1f}s->{thi:.1f}s", flush=True)
    except Exception as exc:  # noqa: BLE001 - audit every ticker we can
        print(f"{tk:>5}: {exc}", flush=True)

# --- loglik trajectory on one representative ticker -------------------------
traj = []
spy = next((x for x in rows if x["ticker"] == TRAJECTORY_TICKER), None)
if spy is not None:
    hist = data.get_price_history(TRAJECTORY_TICKER, lookback_days=config.HMM_LOOKBACK_DAYS)
    anchor, _ = data.anchor_from_history(hist)
    past = hist[[d.date() <= anchor for d in hist.index]]
    r = np.diff(np.log(past["Close"].dropna().values))
    seed = _seed_for(TRAJECTORY_TICKER, anchor)
    print(f"\n{TRAJECTORY_TICKER} loglik trajectory (cap -> loglik, winning restart):")
    for cap in TRAJECTORY_CAPS:
        f = fit_hmm(r, K=config.HMM_STATES, iters=cap, seed=seed)
        traj.append(dict(cap=cap, loglik=f.loglik, converged=f.converged, n_iter=f.n_iter))
        print(f"  cap {cap:>5}: loglik {f.loglik:.2f}  converged={f.converged} "
              f"(n_iter={f.n_iter})", flush=True)

# --- summary -----------------------------------------------------------------
n = len(rows)
conv40 = sum(1 for x in rows if x["conv40"])
conv_hi = sum(1 for x in rows if x["conv_hi"])
need = sorted(x["n_iter_hi"] for x in rows if x["conv_hi"])
dps = sorted(x["dp"] for x in rows)
print("\n" + "=" * 78)
print(f"AUDIT SUMMARY ({n} tickers fitted, {time.time() - t_all:.0f}s total)")
print(f"  converged at the production cap (40): {conv40}/{n}")
print(f"  converged at cap {HIGH_CAP}: {conv_hi}/{n}"
      + (f"  (iters needed: median {need[len(need) // 2]}, max {need[-1]})" if need else ""))
print(f"  loglik left on the table at 40: median "
      f"{sorted(x['dll'] for x in rows)[n // 2]:.1f}, max {max(x['dll'] for x in rows):.1f}")
print(f"  bar-value drift |dP(up)|: median {dps[n // 2]:.3f}, max {dps[-1]:.3f}, "
      f">=0.01 on {sum(1 for d in dps if d >= 0.01)}/{n}, "
      f">=0.05 on {sum(1 for d in dps if d >= 0.05)}/{n}")
print(f"  fit time per ticker: prod median "
      f"{sorted(x['t40'] for x in rows)[n // 2]:.1f}s -> converged median "
      f"{sorted(x['t_hi'] for x in rows)[n // 2]:.1f}s")

with open("audit_em_convergence.json", "w", encoding="utf-8") as fh:
    json.dump({"rows": rows, "trajectory": traj,
               "prod_cap": PROD_CAP, "high_cap": HIGH_CAP}, fh, indent=2)
print("saved -> audit_em_convergence.json")
