r"""Publication figures for the PINN volatility study.

Reads the outputs of one run (default ``runs/main``) and writes, to ``reports/paper_figures/``:

* ``figN_<name>.pdf`` (vector, fonts embedded) and ``figN_<name>.png`` (300 dpi) for each figure;
* ``figN_<name>.csv``: the exact numbers plotted, so every figure can be re-drawn or tabulated;
* ``table_main_results.csv`` / ``.tex``: headline RMSE with 95% bootstrap confidence intervals;
* ``CAPTIONS.md``: a ready-to-edit caption for every figure.

Confidence intervals use a moving-block bootstrap over test *days* (block length 10, 2,000
replications): option errors on the same day are strongly correlated, and days are autocorrelated,
so resampling single options would understate the uncertainty.

Run from the repository root:  python reports/paper_figures.py [--run runs/main] [--reps 2000]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "paper_figures"

# --------------------------------------------------------------------------------------------- style
# Categorical colours follow the volatility input everywhere (validated palette, fixed order).
COLOR = {"garch": "#2a78d6", "lstm": "#eb6834", "iv_atm": "#1baf7a", "iv_surface": "#eda100",
         "har": "#e87ba4", "hv21": "#008300", "iv_atm_same_day": "#4a3aa7"}
LABEL = {"garch": "GARCH(1,1)", "lstm": "LSTM", "iv_atm": "IV ATM", "iv_surface": "IV surface",
         "har": "HAR-RV", "hv21": "HV 21d", "iv_atm_same_day": "IV ATM (same day)",
         "iv_contract_same_day": "Contract IV (oracle)"}
ENGINE_LABEL = {"bs": "Black–Scholes", "pinn_p": "PINN (physics only)", "pinn_h": "PINN hybrid (λ = 0.1)",
                "ffnn": "FFNN (no physics)", "cn": "Crank–Nicolson"}
CORE = ["garch", "lstm", "iv_atm", "iv_surface"]
RANKED = ["iv_surface", "iv_atm_same_day", "iv_atm", "garch", "har", "lstm", "hv21"]
INK, INK2, GRID, SHADE = "#0b0b0b", "#52514e", "#e4e3df", "#efeee9"
REGIMES = {"COVID crash": ("2020-02-19", "2020-04-30"), "2022 bear market": ("2022-01-03", "2022-10-12")}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8, "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "legend.frameon": False, "savefig.dpi": 300, "savefig.bbox": "tight",
    "pdf.fonttype": 42, "ps.fonttype": 42, "lines.linewidth": 1.6,
})
FULL, HALF = 6.5, 3.25  # inches: full text width / one column


def save(fig: plt.Figure, name: str, data: pd.DataFrame | None = None) -> None:
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)
    if data is not None:
        data.to_csv(OUT / f"{name}.csv", index=False, float_format="%.6g")
    print(f"  {name}")


def shade_regimes(ax: plt.Axes, label: bool = True) -> None:
    for name, (a, b) in REGIMES.items():
        ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color=SHADE, zorder=0, lw=0)
        if label:
            ax.text(pd.Timestamp(a), 1.0, f" {name}", transform=ax.get_xaxis_transform(), va="top", ha="left",
                    fontsize=7, color=INK2)


# --------------------------------------------------------------------------------------------- bootstrap
def block_indices(n_days: int, reps: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """Moving-block bootstrap day indices, shape (reps, n_days)."""
    n_blocks = int(np.ceil(n_days / block))
    starts = rng.integers(0, n_days - block + 1, size=(reps, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(reps, -1)[:, :n_days]
    return idx


def rmse_ci(sse: np.ndarray, cnt: np.ndarray, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Point RMSE and 95% percentile CI from per-day sums. sse, cnt: (n_days, k)."""
    point = np.sqrt(sse.sum(0) / cnt.sum(0))
    boot = np.sqrt(sse[idx].sum(1) / np.maximum(cnt[idx].sum(1), 1))  # (reps, k)
    lo, hi = np.nanpercentile(boot, [2.5, 97.5], axis=0)
    return point, lo, hi


# --------------------------------------------------------------------------------------------- data
def load(run: Path) -> tuple[pd.DataFrame, dict]:
    print("loading predictions (≈ 1.4 GB) ...")
    p = pd.read_pickle(run / "predictions.pkl")
    keep = ["quote_date", "mid", "tau", "fwd_moneyness", "dte"] + \
           [c for c in p.columns if c.startswith("sigma_")] + [c for c in p.columns if c.startswith("V_")]
    p = p[keep]
    # Mode H: average the 5 seeds (the ensemble forecast), keep per-seed RMSE from the metrics file.
    for t in CORE:
        cols = [c for c in p.columns if c.startswith(f"V_pinn_h0.1_{t}_s")]
        if cols:
            p[f"V_pinn_h_{t}"] = p[cols].mean(axis=1)
            p = p.drop(columns=cols)
    p["day"] = pd.factorize(p["quote_date"], sort=True)[0]
    meta = {"days": np.sort(p["quote_date"].unique())}
    return p, meta


def per_day(p: pd.DataFrame, cols: list[str], group: pd.Series | None = None) -> tuple:
    """Per-day (and per-group) sums of squared errors vs mid, and counts."""
    err2 = pd.DataFrame({c: (p[c] - p["mid"]) ** 2 for c in cols})
    err2["day"] = p["day"].to_numpy()
    if group is not None:
        err2["g"] = group.to_numpy()
    g = err2.groupby(["day"] if group is None else ["day", "g"], observed=True)
    sse, cnt = g[cols].sum(), g[cols].count()
    return sse, cnt


# --------------------------------------------------------------------------------------------- figures
def fig1_rmse_ci(p: pd.DataFrame, idx: np.ndarray) -> pd.DataFrame:
    rows = []
    for eng, prefix in [("bs", "V_bs_"), ("pinn_p", "V_pinn_p_"), ("pinn_h", "V_pinn_h_"), ("ffnn", "V_ffnn_")]:
        ts = [t for t in RANKED if f"{prefix}{t}" in p]
        sse, cnt = per_day(p, [f"{prefix}{t}" for t in ts])
        pt, lo, hi = rmse_ci(sse.to_numpy(), cnt.to_numpy(), idx)
        rows += [dict(engine=eng, treatment=t, rmse=a, ci_low=b, ci_high=c) for t, a, b, c in zip(ts, pt, lo, hi)]
    d = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(FULL, 3.1))
    y = {t: i for i, t in enumerate(RANKED[::-1])}
    style = {"bs": dict(marker="o", mfc="white", mec=INK, color=INK, dy=0.22),
             "pinn_p": dict(marker="s", mfc=INK2, mec=INK2, color=INK2, dy=0.0),
             "pinn_h": dict(marker="^", mfc="#8a8984", mec="#8a8984", color="#8a8984", dy=-0.22)}
    for eng, st in style.items():
        s = d[d.engine == eng]
        yy = [y[t] + st["dy"] for t in s.treatment]
        ax.errorbar(s.rmse, yy, xerr=[s.rmse - s.ci_low, s.ci_high - s.rmse], fmt="none", ecolor=st["color"],
                    elinewidth=1.2, capsize=2.5)
        ax.plot(s.rmse, yy, ls="none", marker=st["marker"], ms=5, mfc=st["mfc"], mec=st["mec"], mew=1.1,
                label=ENGINE_LABEL[eng])
    ax.set_yticks(list(y.values()), [LABEL[t] for t in y])
    ax.set_xlim(0, d[d.engine != "ffnn"].ci_high.max() * 1.08)
    ax.set_xlabel("RMSE against market mid price (index points), 95% block-bootstrap CI")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower left", ncol=3, bbox_to_anchor=(0, 1.0))
    save(fig, "fig1_rmse_by_input_ci", d)
    return d


def fig2_error_distribution(p: pd.DataFrame) -> None:
    stats, rows = [], []
    for t in RANKED:
        e = (p[f"V_pinn_p_{t}"] - p["mid"]).to_numpy()
        q = np.nanpercentile(e, [5, 25, 50, 75, 95])
        stats.append(dict(label=LABEL[t], whislo=q[0], q1=q[1], med=q[2], q3=q[3], whishi=q[4], mean=np.nanmean(e),
                          fliers=[]))
        rows.append(dict(treatment=t, p5=q[0], p25=q[1], median=q[2], p75=q[3], p95=q[4], mean=np.nanmean(e)))
    fig, ax = plt.subplots(figsize=(FULL, 2.9))
    bp = ax.bxp(stats, showmeans=True, showfliers=False, patch_artist=True, widths=0.55,
                meanprops=dict(marker="D", ms=4, mfc="white", mec=INK),
                medianprops=dict(color=INK, lw=1.4), whiskerprops=dict(color=INK2), capprops=dict(color=INK2))
    for patch, t in zip(bp["boxes"], RANKED):
        patch.set_facecolor(COLOR[t])
        patch.set_alpha(0.85)
        patch.set_edgecolor(INK2)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_ylabel("Pricing error, model − market (index points)")
    ax.set_xticks(range(1, len(RANKED) + 1), [LABEL[t].replace(" (", "\n(") for t in RANKED], rotation=0)
    ax.grid(axis="x", visible=False)
    ax.legend(handles=[Line2D([], [], marker="D", ls="none", mfc="white", mec=INK, label="mean"),
                       Line2D([], [], color=INK, lw=1.4, label="median")], loc="lower left", ncol=2,
              bbox_to_anchor=(0, 1.0))
    ax.text(1.0, 1.03, "boxes: 25–75%, whiskers: 5–95%; physics-only PINN", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=7, color=INK2)
    save(fig, "fig2_error_distribution", pd.DataFrame(rows))


def _bucket_fig(p: pd.DataFrame, idx: np.ndarray, col: str, edges: list[float], labels: list[str], name: str,
                xlabel: str) -> None:
    b = pd.cut(p[col], edges, labels=labels, include_lowest=True)
    cols = [f"V_pinn_p_{t}" for t in CORE]
    sse, cnt = per_day(p, cols, b)
    rows = []
    days = sse.index.get_level_values(0).unique()
    for lab in labels:
        s = sse.xs(lab, level="g").reindex(days, fill_value=0.0).to_numpy()
        c = cnt.xs(lab, level="g").reindex(days, fill_value=0).to_numpy()
        pt, lo, hi = rmse_ci(s, c, idx)
        rows += [dict(bucket=lab, treatment=t, rmse=a, ci_low=x, ci_high=z, n=int(c[:, j].sum()))
                 for j, (t, a, x, z) in enumerate(zip(CORE, pt, lo, hi))]
    d = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(FULL, 2.8))
    x = np.arange(len(labels))
    ends = []
    for t in CORE:
        s = d[d.treatment == t]
        ax.fill_between(x, s.ci_low, s.ci_high, color=COLOR[t], alpha=0.18, lw=0)
        ax.plot(x, s.rmse, color=COLOR[t], marker="o", ms=4.5, label=LABEL[t])
        ends.append([float(s.rmse.iloc[-1]), t])
    gap = 0.06 * d.ci_high.max()  # keep end labels apart
    ends.sort()
    for k in range(1, len(ends)):
        ends[k][0] = max(ends[k][0], ends[k - 1][0] + gap)
    for yv, t in ends:
        ax.text(x[-1] + 0.08, yv, LABEL[t], color=INK, fontsize=7.5, va="center")
    ax.set_xticks(x, labels)
    ax.set_xlim(-0.2, len(labels) - 0.4)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("RMSE (index points)")
    ax.set_ylim(0, None)
    ax.legend(loc="upper left", ncol=4, bbox_to_anchor=(0, 1.13))
    save(fig, name, d)


def fig3_moneyness(p: pd.DataFrame, idx: np.ndarray) -> None:
    _bucket_fig(p, idx, "fwd_moneyness", [0.80, 0.90, 0.97, 1.03, 1.10, 1.2001],
                ["0.80–0.90", "0.90–0.97", "0.97–1.03", "1.03–1.10", "1.10–1.20"], "fig3_rmse_by_moneyness",
                "Forward moneyness K/F")


def fig4_maturity(p: pd.DataFrame, idx: np.ndarray) -> None:
    _bucket_fig(p, idx, "dte", [7, 30, 60, 120, 180.5], ["7–30", "30–60", "60–120", "120–180"],
                "fig4_rmse_by_maturity", "Days to expiry")


def fig5_time_series(p: pd.DataFrame) -> None:
    cols = [f"V_pinn_p_{t}" for t in CORE]
    sse, cnt = per_day(p, cols)
    daily = np.sqrt(sse / cnt)
    daily.index = pd.DatetimeIndex(np.sort(p["quote_date"].unique()))
    roll = np.sqrt((sse.rolling(21, min_periods=10).sum() / cnt.rolling(21, min_periods=10).sum()).to_numpy())
    roll = pd.DataFrame(roll, index=daily.index, columns=CORE)
    fig, ax = plt.subplots(figsize=(FULL, 2.8))
    shade_regimes(ax)
    for t in CORE:
        ax.plot(roll.index, roll[t], color=COLOR[t], lw=1.4, label=LABEL[t])
    ax.set_yscale("log")
    ax.set_ylabel("RMSE, 21-day rolling (log scale)")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.legend(loc="upper right", ncol=4, bbox_to_anchor=(1, 1.14))
    out = roll.copy()
    out.columns = [f"rmse21_{c}" for c in out.columns]
    for t in CORE:
        out[f"rmse_daily_{t}"] = daily[f"V_pinn_p_{t}"]
    save(fig, "fig5_rmse_over_time", out.reset_index(names="date"))


def fig6_vol_forecasts(p: pd.DataFrame, data_dir: Path) -> None:
    """Median input volatility for ~1-month options vs realised volatility over the next 21 days."""
    m = p[(p.dte >= 21) & (p.dte <= 42) & (p.fwd_moneyness.between(0.97, 1.03))]
    med = m.groupby("quote_date")[[f"sigma_{t}" for t in ["garch", "lstm", "iv_atm"]]].median()
    und = pd.read_csv(data_dir / "spx_underlying_daily.csv", parse_dates=["date"]).set_index("date")
    r2 = und["log_return"] ** 2
    fwd = np.sqrt(252 / 21 * r2[::-1].rolling(21).sum()[::-1].shift(-1))
    med["realised_21d_forward"] = fwd.reindex(med.index)
    fig, ax = plt.subplots(figsize=(FULL, 2.8))
    shade_regimes(ax)
    ax.plot(med.index, med.realised_21d_forward * 100, color=INK, lw=1.0, ls="-", alpha=0.75,
            label="Realised (next 21 days)")
    for t in ["garch", "lstm", "iv_atm"]:
        ax.plot(med.index, med[f"sigma_{t}"] * 100, color=COLOR[t], lw=1.3, label=LABEL[t])
    ax.set_ylabel("Annualised volatility (%)")
    ax.set_ylim(0, None)
    ax.legend(loc="upper right", ncol=4, bbox_to_anchor=(1, 1.14))
    save(fig, "fig6_vol_forecasts_vs_realised", med.reset_index().rename(columns={"quote_date": "date"}))


def fig7_dm_heatmap(run: Path) -> None:
    dm = pd.read_csv(run / "dm_tests.csv")
    dm = dm[(dm.engine == "pinn_p") & ~dm.treatment_a.str.contains("contract") & ~dm.treatment_b.str.contains(
        "contract")].copy()
    # Holm–Bonferroni adjustment within the engine
    order = np.argsort(dm.p_value.to_numpy())
    m = len(dm)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * dm.p_value.to_numpy()[i]))
        adj[i] = running
    dm["p_holm"] = adj
    ts = RANKED
    M = pd.DataFrame(np.nan, index=ts, columns=ts)
    P = pd.DataFrame(np.nan, index=ts, columns=ts)
    for _, r in dm.iterrows():
        M.loc[r.treatment_a, r.treatment_b], M.loc[r.treatment_b, r.treatment_a] = r.dm, -r.dm
        P.loc[r.treatment_a, r.treatment_b] = P.loc[r.treatment_b, r.treatment_a] = r.p_holm
    cmap = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f2f1ec", "#e34948"])
    lim = np.nanmax(np.abs(M.to_numpy()))
    fig, ax = plt.subplots(figsize=(FULL * 0.78, FULL * 0.62))
    im = ax.imshow(M.to_numpy(), cmap=cmap, norm=TwoSlopeNorm(0, -lim, lim))
    for i in range(len(ts)):
        for j in range(len(ts)):
            if i == j:
                continue
            v, pv = M.iloc[i, j], P.iloc[i, j]
            star = "***" if pv < 0.001 else "**" if pv < 0.01 else "*" if pv < 0.05 else ""
            ax.text(j, i, f"{v:.1f}\n{star}", ha="center", va="center", fontsize=6.8,
                    color="white" if abs(v) > lim * 0.6 else INK)
    ax.set_xticks(range(len(ts)), [LABEL[t] for t in ts], rotation=35, ha="right")
    ax.set_yticks(range(len(ts)), [LABEL[t] for t in ts])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("DM statistic (row vs column)\nnegative = row more accurate", fontsize=7.5)
    ax.set_title("Diebold–Mariano tests, PINN (physics only); stars: Holm-adjusted p < .05 / .01 / .001",
                 fontsize=8, loc="left")
    save(fig, "fig7_dm_test_matrix", dm[["treatment_a", "treatment_b", "dm", "p_value", "p_holm", "days"]])


def fig8_solver_accuracy(p: pd.DataFrame, run: Path) -> None:
    vbs = pd.read_csv(run / "metrics_vs_bs.csv")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(FULL, 2.9), gridspec_kw=dict(width_ratios=[1, 1.15]),
                                 layout="constrained")
    t = "iv_surface"
    hb = a1.hexbin(p[f"V_bs_{t}"], p[f"V_pinn_p_{t}"], gridsize=70, bins="log", mincnt=1,
                   cmap=LinearSegmentedColormap.from_list("seq", ["#d6e6f8", "#2a78d6", "#0d3a73"]), lw=0)
    hi = float(np.nanpercentile(p[f"V_bs_{t}"], 99.9))
    a1.plot([0, hi], [0, hi], color=INK, lw=0.8, ls="--")
    a1.set_xlim(0, hi)
    a1.set_ylim(0, hi)
    a1.set_xlabel("Black–Scholes price")
    a1.set_ylabel("PINN price")
    a1.set_title("(a) PINN vs analytical price\n(IV-surface input)", loc="left", fontsize=8)
    fig.colorbar(hb, ax=a1, fraction=0.05, pad=0.02).set_label("options (log)", fontsize=7)
    order = ["pinn_p", "pinn_h0.1", "ffnn"]
    sub = vbs[vbs.engine.isin(order) & vbs.treatment.isin(CORE)]
    w = 0.2
    for j, t2 in enumerate(CORE):
        vals = [float(sub[(sub.engine == e) & (sub.treatment == t2)].rmse.iloc[0]) for e in order]
        a2.bar(np.arange(3) + (j - 1.5) * (w + 0.01), vals, w, color=COLOR[t2], label=LABEL[t2])
    a2.set_yscale("log")
    a2.set_xticks(range(3), ["PINN\nphysics only", "PINN hybrid\nλ = 0.1", "FFNN\nno physics"])
    a2.set_ylabel("RMSE vs analytical BS (log)")
    a2.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    cn = vbs[vbs.engine == "cn"].rmse.max()
    a2.axhline(cn, color=INK2, lw=0.8, ls=":")
    a2.text(2.45, cn * 1.15, "Crank–Nicolson", fontsize=6.8, color=INK2, ha="right")
    a2.grid(axis="x", visible=False)
    a2.set_title("(b) Solver error by engine\n", loc="left", fontsize=8)
    a2.set_ylim(top=a2.get_ylim()[1] * 4)
    a2.legend(loc="upper left", fontsize=7, ncol=2)
    save(fig, "fig8_solver_accuracy", vbs)


def fig9_mode_h_seeds(run: Path) -> None:
    seeds = pd.read_csv(run / "metrics_mode_h_seeds.csv")
    mk = pd.read_csv(run / "metrics_market.csv")
    fig, ax = plt.subplots(figsize=(FULL, 2.4))
    for i, t in enumerate(CORE):
        s = seeds[seeds.treatment == t].rmse
        ax.scatter(s, np.full(len(s), i) + np.linspace(-0.12, 0.12, len(s)), color=COLOR[t], s=22, zorder=3,
                   edgecolor="white", linewidth=0.6)
        pp = float(mk[(mk.engine == "pinn_p") & (mk.treatment == t)].rmse.iloc[0])
        ax.plot([pp], [i], marker="|", ms=14, mew=2, color=INK, zorder=4)
    ax.set_yticks(range(len(CORE)), [LABEL[t] for t in CORE])
    ax.set_xlabel("RMSE against market (index points)")
    ax.grid(axis="y", visible=False)
    ax.legend(handles=[Line2D([], [], marker="o", ls="none", color=INK2, label="hybrid PINN, one seed each"),
                       Line2D([], [], marker="|", ls="none", ms=12, mew=2, color=INK, label="physics-only PINN")],
              loc="upper right")
    save(fig, "fig9_hybrid_seed_spread", seeds)


def fig10_forecast_accuracy(run: Path) -> None:
    v = pd.read_csv(run / "vol_forecast_metrics.csv")
    v = v[v.model.isin(["garch", "lstm", "iv_atm", "har"])]
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.6), sharey=True)
    for ax, split, ttl in [(axes[0], "validation", "(a) Validation 2019"), (axes[1], "test", "(b) Test 2020–2023")]:
        for t in ["garch", "lstm", "iv_atm", "har"]:
            s = v[(v.split == split) & (v.model == t)].sort_values("horizon")
            ax.plot(s.horizon, s.qlike, color=COLOR[t], marker="o", ms=3.5, label=LABEL[t])
        ax.set_xscale("log")
        ax.set_xticks([5, 10, 21, 42, 63, 126], ["5", "10", "21", "42", "63", "126"])
        ax.minorticks_off()
        ax.set_xlabel("Forecast horizon (trading days)")
        ax.set_title(ttl, loc="left", fontsize=8)
    axes[0].set_ylabel("QLIKE loss (lower is better)")
    axes[1].legend(loc="upper left", fontsize=7)
    save(fig, "fig10_vol_forecast_qlike", v)


def main_table(d1: pd.DataFrame) -> None:
    cell = {(r.treatment, r.engine): f"{r.rmse:.2f} [{r.ci_low:.2f}, {r.ci_high:.2f}]" for r in d1.itertuples()}
    rows = [{"Volatility input": LABEL[tr], **{ENGINE_LABEL[e]: cell.get((tr, e), "—")
                                               for e in ["bs", "pinn_p", "pinn_h", "ffnn"]}} for tr in RANKED]
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "table_main_results.csv", index=False)
    cols = list(tab.columns)
    tex = ["\\begin{table}[t]", "\\centering", "\\small",
           "\\caption{Pricing RMSE against the market mid price (index points) with 95\\% moving-block bootstrap "
           "confidence intervals over test days. SPX monthly options, 2020--2023, $n = 1{,}993{,}333$.}",
           "\\label{tab:main-results}", "\\begin{tabular}{l" + "c" * (len(cols) - 1) + "}", "\\toprule",
           " & ".join(cols) + " \\\\", "\\midrule"]
    tex += [" & ".join(str(v).replace("—", "--") for v in r) + " \\\\" for r in tab.itertuples(index=False)]
    tex += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    latex = "\n".join(tex).replace("λ", "$\\lambda$").replace("–", "--")  # pdfLaTeX-safe
    (OUT / "table_main_results.tex").write_text(latex, encoding="utf-8")
    print("  table_main_results (.csv, .tex)")


CAPTIONS = """# Figure captions (edit freely)

All figures: SPX monthly (AM-settled) European options, test period 2020-01-02 to 2023-12-29,
n = 1,993,333 options on 992 trading days. RMSE is against the market mid price in index points.
95% confidence intervals: moving-block bootstrap over trading days (block 10, {reps} replications).
Each figure exists as vector PDF (for LaTeX/Word) and 300-dpi PNG; the plotted numbers are in the CSV
with the same name.

**Figure 1 — Pricing accuracy by volatility input.** RMSE against the market mid price for each
volatility input under three pricing engines (analytical Black–Scholes, physics-only PINN and the
hybrid PINN with λ = 0.1; hybrid values average five seeds). Implied-volatility inputs are markedly
more accurate than forecast-based inputs, and the three engines are nearly indistinguishable for
any given input. (`fig1_rmse_by_input_ci`)

**Figure 2 — Distribution of pricing errors.** Model price minus market mid for the physics-only
PINN. Boxes span the 25th–75th percentiles, whiskers the 5th–95th; the diamond marks the mean.
Forecast-based inputs (GARCH, HAR, LSTM, HV) systematically underprice options, consistent with a
volatility risk premium. (`fig2_error_distribution`)

**Figure 3 — Error across moneyness.** RMSE by forward-moneyness bucket with 95% CIs (shaded).
Single-volatility inputs fail most for low strikes (out-of-the-money puts), where the implied
volatility skew is steepest; the IV surface removes most of this error. (`fig3_rmse_by_moneyness`)

**Figure 4 — Error across maturity.** RMSE by days-to-expiry bucket with 95% CIs. Errors grow with
maturity because option vega increases with time to expiry. (`fig4_rmse_by_maturity`)

**Figure 5 — Error over time.** 21-day rolling RMSE (log scale) of the physics-only PINN for each
volatility input. Shaded: COVID crash (19 Feb–30 Apr 2020) and the 2022 bear market (3 Jan–12 Oct
2022). (`fig5_rmse_over_time`)

**Figure 6 — Volatility inputs versus realised volatility.** Median input volatility for near-the-money
options with 21–42 days to expiry, against SPX realised volatility over the following 21 trading
days. (`fig6_vol_forecasts_vs_realised`)

**Figure 7 — Pairwise forecast comparisons.** Diebold–Mariano statistics on daily squared pricing
errors (physics-only PINN); negative values mean the row input is more accurate than the column
input. Stars give Holm–Bonferroni-adjusted significance across all pairs. The oracle
(contract's own same-day IV) is excluded. (`fig7_dm_test_matrix`)

**Figure 8 — PINN solver accuracy.** (a) PINN versus analytical Black–Scholes prices for identical
inputs (IV-surface treatment). (b) RMSE against analytical Black–Scholes by engine and input; the
dotted line is the Crank–Nicolson benchmark. The physics-only PINN reproduces Black–Scholes to
about 0.3 index points, far below the volatility-induced error. (`fig8_solver_accuracy`)

**Figure 9 — Seed sensitivity of the hybrid PINN.** RMSE of each of the five hybrid-PINN seeds
(dots) against the physics-only PINN (bar). (`fig9_hybrid_seed_spread`)

**Figure 10 — Volatility forecast accuracy.** QLIKE loss of variance forecasts against realised
variance by horizon, (a) validation year 2019 and (b) test period 2020–2023. The LSTM ranks best in
validation but worst in the test period. (`fig10_vol_forecast_qlike`)

**Table — Main results with confidence intervals.** `table_main_results.csv` and
`table_main_results.tex` (LaTeX, needs `\\usepackage{{booktabs}}`).
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/main")
    ap.add_argument("--data-dir", default="../data-clean")
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--block", type=int, default=10)
    a = ap.parse_args()
    run = (ROOT / a.run).resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    p, meta = load(run)
    idx = block_indices(len(meta["days"]), a.reps, a.block, np.random.default_rng(0))
    print("figures:")
    d1 = fig1_rmse_ci(p, idx)
    fig2_error_distribution(p)
    fig3_moneyness(p, idx)
    fig4_maturity(p, idx)
    fig5_time_series(p)
    fig6_vol_forecasts(p, (ROOT / a.data_dir).resolve())
    fig7_dm_heatmap(run)
    fig8_solver_accuracy(p, run)
    fig9_mode_h_seeds(run)
    fig10_forecast_accuracy(run)
    main_table(d1)
    (OUT / "CAPTIONS.md").write_text(CAPTIONS.format(reps=a.reps), encoding="utf-8")
    print(f"done: {OUT}")


if __name__ == "__main__":
    main()
