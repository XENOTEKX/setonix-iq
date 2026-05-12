#!/usr/bin/env python3
"""
Amdahl's law scaling model: fit T(n) = T1 * (f + (1-f)/n)
to the benchmark run data and evaluate prediction quality.

Usage:
    python3.11 tools/scaling_model_analysis.py
Outputs:
    tools/scaling_model_analysis.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "logs" / "runs"
OUT_PNG  = ROOT / "tools" / "scaling_model_analysis.png"

# ── Datasets to analyse ──────────────────────────────────────────────────
GROUPS = {
    ("xlarge_mf.fa",        "gadi"):    "xlarge_mf — Gadi",
    ("xlarge_mf.fa",        "setonix"): "xlarge_mf — Setonix",
    ("large_modelfinder.fa","gadi"):    "large_mf — Gadi",
    ("large_modelfinder.fa","setonix"): "large_mf — Setonix",
}
COLORS = {
    ("xlarge_mf.fa",        "gadi"):    "#1f77b4",
    ("xlarge_mf.fa",        "setonix"): "#ff7f0e",
    ("large_modelfinder.fa","gadi"):    "#2ca02c",
    ("large_modelfinder.fa","setonix"): "#d62728",
}

# ── Load runs ────────────────────────────────────────────────────────────
def load_runs():
    rows = []
    for f in sorted(RUNS_DIR.glob("*.json")):
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        prof   = d.get("profile") or {}
        ds     = prof.get("dataset")
        plat   = d.get("platform")
        thr    = prof.get("threads")
        mpi    = prof.get("mpi_ranks") or 1
        summ   = d.get("summary") or {}
        timing = d.get("timing") or [{}]
        wall   = summ.get("total_time") or timing[0].get("time_s")
        if not (ds and plat and thr and wall):
            continue
        if mpi > 1:          # skip MPI multi-rank runs
            continue
        rows.append({"ds": ds, "platform": plat, "threads": thr,
                     "wall_s": wall, "nc": bool(d.get("non_canonical"))})
    return rows


# ── Amdahl's law ─────────────────────────────────────────────────────────
def amdahl(n: np.ndarray, T1: float, f: float) -> np.ndarray:
    """T(n) = T1 * (f + (1-f)/n)"""
    return T1 * (f + (1 - f) / n)


def fit_group(ns: np.ndarray, ts: np.ndarray):
    """Fit Amdahl's law; return (T1, f, pcov)."""
    p0 = [ts[ns == 1].mean() if (ns == 1).any() else ts.max(), 0.1]
    try:
        popt, pcov = curve_fit(amdahl, ns, ts, p0=p0,
                               bounds=([0, 0], [np.inf, 1]),
                               maxfev=10_000)
    except Exception:
        popt = p0
        pcov = None
    return popt, pcov


# ── Main ─────────────────────────────────────────────────────────────────
def main():
    rows = load_runs()

    # Collect per-group best (min wall) per thread count
    best: dict[tuple, dict[int, float]] = {k: {} for k in GROUPS}
    for r in rows:
        key = (r["ds"], r["platform"])
        if key not in GROUPS:
            continue
        t = r["threads"]
        w = r["wall_s"]
        if t not in best[key] or w < best[key][t]:
            best[key][t] = w

    # ── Figure layout ────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.38)

    all_actual    = []
    all_predicted = []

    axes = {}
    for idx, (key, title) in enumerate(GROUPS.items()):
        ax = fig.add_subplot(gs[idx // 2, idx % 2])
        axes[key] = ax

        data = best[key]
        if len(data) < 3:
            ax.set_title(title + "\n(insufficient data)")
            continue

        ns = np.array(sorted(data.keys()), dtype=float)
        ts = np.array([data[int(n)] for n in ns], dtype=float)

        # Fit
        popt, _ = fit_group(ns, ts)
        T1_fit, f_fit = popt

        # Smooth curve
        n_plot = np.linspace(1, ns.max() * 1.05, 300)
        t_plot = amdahl(n_plot, T1_fit, f_fit)

        # Predicted at each measured thread count
        t_pred = amdahl(ns, T1_fit, f_fit)

        # Residuals for annotation
        resid_pct = 100 * (ts - t_pred) / t_pred

        # Pearson r (log-scale is more natural for wall times)
        r_val, p_val = pearsonr(np.log(t_pred), np.log(ts))

        # Accumulate for global correlation
        all_actual.extend(ts.tolist())
        all_predicted.extend(t_pred.tolist())

        col = COLORS[key]

        # Fitted curve
        ax.plot(n_plot, t_plot / 3600, "-", color=col, lw=2, alpha=0.8,
                label=f"Amdahl fit  T₁={T1_fit/3600:.2f}h  f={f_fit:.3f}")

        # Actual data points
        ax.scatter(ns, ts / 3600, color=col, s=80, zorder=5,
                   edgecolors="white", linewidths=0.8, label="Actual")

        # Error bars (% residual)
        for n_i, t_i, pred_i, res_i in zip(ns, ts, t_pred, resid_pct):
            ax.annotate(f"{res_i:+.0f}%",
                        xy=(n_i, t_i / 3600),
                        xytext=(6, 4), textcoords="offset points",
                        fontsize=7, color="dimgray")

        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("OMP threads", fontsize=10)
        ax.set_ylabel("Wall time (h)", fontsize=10)
        ax.set_title(f"{title}\n"
                     f"Pearson r={r_val:.4f}  (log space)  "
                     f"n={len(ns)} points",
                     fontsize=10)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.6)
        ax.set_xticks(ns)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    # ── Global predicted vs actual scatter ───────────────────────────────
    ax_global = fig.add_axes([0.12, -0.25, 0.76, 0.22])
    all_actual    = np.array(all_actual)
    all_predicted = np.array(all_predicted)

    for idx, (key, title) in enumerate(GROUPS.items()):
        data = best[key]
        if len(data) < 3:
            continue
        ns = np.array(sorted(data.keys()), dtype=float)
        ts = np.array([data[int(n)] for n in ns], dtype=float)
        popt, _ = fit_group(ns, ts)
        t_pred = amdahl(ns, *popt)
        ax_global.scatter(t_pred / 3600, ts / 3600,
                          color=COLORS[key], s=70,
                          edgecolors="white", linewidths=0.6,
                          label=title, zorder=4)

    lo = min(all_actual.min(), all_predicted.min()) / 3600 * 0.8
    hi = max(all_actual.max(), all_predicted.max()) / 3600 * 1.2
    ax_global.plot([lo, hi], [lo, hi], "k--", lw=1.2, alpha=0.6, label="Perfect prediction")
    ax_global.set_xscale("log")
    ax_global.set_yscale("log")
    ax_global.set_xlabel("Predicted wall time (h) — Amdahl fit", fontsize=10)
    ax_global.set_ylabel("Actual wall time (h)", fontsize=10)

    r_global, p_global = pearsonr(np.log(all_predicted), np.log(all_actual))
    n_pts = len(all_actual)

    # RMSE in log space
    rmse_log = np.sqrt(np.mean((np.log(all_actual) - np.log(all_predicted)) ** 2))
    # Mean absolute % error
    mape = np.mean(np.abs((all_actual - all_predicted) / all_predicted)) * 100

    ax_global.set_title(
        f"Global predicted vs actual — all groups combined\n"
        f"Pearson r = {r_global:.4f}  (n={n_pts})   "
        f"MAPE = {mape:.1f}%   RMSE(log) = {rmse_log:.3f}",
        fontsize=11, fontweight="bold")
    ax_global.legend(fontsize=8, ncol=2, loc="upper left")
    ax_global.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.6)

    fig.suptitle("Amdahl's Law Scaling Model: Predicted vs Actual Wall Time",
                 fontsize=13, fontweight="bold", y=1.01)

    plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    print(f"Saved → {OUT_PNG}")
    print(f"\nGlobal summary (log-space Pearson):")
    print(f"  r = {r_global:.4f}  (n={n_pts} data points)")
    print(f"  MAPE = {mape:.1f}%")
    print(f"  RMSE(log) = {rmse_log:.3f}")
    print()
    print("Per-group:")
    for key, title in GROUPS.items():
        data = best[key]
        if len(data) < 3:
            print(f"  {title}: insufficient data")
            continue
        ns = np.array(sorted(data.keys()), dtype=float)
        ts = np.array([data[int(n)] for n in ns], dtype=float)
        popt, _ = fit_group(ns, ts)
        T1_fit, f_fit = popt
        t_pred = amdahl(ns, T1_fit, f_fit)
        r_val, _ = pearsonr(np.log(t_pred), np.log(ts))
        mape_g = np.mean(np.abs((ts - t_pred) / t_pred)) * 100
        print(f"  {title:35s}  r={r_val:.4f}  T1={T1_fit/3600:.2f}h  "
              f"f={f_fit:.3f}  MAPE={mape_g:.1f}%  n={len(ns)}")


if __name__ == "__main__":
    main()
