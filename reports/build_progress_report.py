"""Build the project progress & improvement report (PDF) from the outputs in code/runs/main.

Run from the repository root after `evaluate`:  python reports/build_progress_report.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "main"
OUT = ROOT / "reports"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
YEARLY_CSV = OUT / "panel_by_year.csv"

# ---------------------------------------------------------------- palette (reference categorical, light mode)
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SERIES = {"garch": BLUE, "lstm": ORANGE, "iv_atm": AQUA, "iv_surface": YELLOW}
LABEL = {"garch": "GARCH(1,1)", "lstm": "LSTM", "iv_atm": "IV (ATM, prev. day)", "iv_surface": "IV surface (prev. day)",
         "har": "HAR-RV", "hv21": "Hist. vol 21d", "iv_atm_same_day": "IV ATM (same day)",
         "iv_contract_same_day": "Contract IV (oracle)"}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "legend.frameon": False, "savefig.dpi": 200, "savefig.bbox": "tight"})

mk = pd.read_csv(RUN / "metrics_market.csv")
vbs = pd.read_csv(RUN / "metrics_vs_bs.csv")
strat = pd.read_csv(RUN / "metrics_stratified.csv")
vfm = pd.read_csv(RUN / "vol_forecast_metrics.csv")
seeds = pd.read_csv(RUN / "metrics_mode_h_seeds.csv")
if (RUN / "panel.pkl").exists():
    # Panel size per quote year, after the monthly fix and under the old Friday-only flag.
    _p = pd.read_pickle(RUN / "panel.pkl")[["quote_date", "expire_date"]]
    _e = _p.expire_date
    _old = (_e.dt.weekday == 4) & _e.dt.day.between(15, 21)
    yearly = pd.DataFrame({"after": _p.groupby(_p.quote_date.dt.year).size(),
                           "before": _p[_old].groupby(_p[_old].quote_date.dt.year).size()}).fillna(0).astype(int)
    yearly.index.name = "quote_date"
    yearly.to_csv(YEARLY_CSV)
    del _p
else:
    yearly = pd.read_csv(YEARLY_CSV, index_col=0)


def rm(engine: str, t: str, col: str = "rmse") -> float:
    return float(mk[(mk.engine == engine) & (mk.treatment == t)][col].iloc[0])


# ---------------------------------------------------------------- figures
def fig_panel_by_year() -> Path:
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    x = range(len(yearly))
    w = 0.38
    ax.bar([i - w / 2 - 0.01 for i in x], yearly["before"] / 1e3, w, color=ORANGE, label="Before fix (Friday-only flag)")
    ax.bar([i + w / 2 + 0.01 for i in x], yearly["after"] / 1e3, w, color=BLUE, label="After fix (commit 7dcead9)")
    ax.set_xticks(list(x), [str(y) for y in yearly.index])
    ax.set_ylabel("Options in panel (thousands)")
    ax.axvspan(-0.5, 4.5, color="#f3f2ee", zorder=0)
    ax.text(2.0, ax.get_ylim()[1] * 0.92, "Thursday-dated monthlies\n(2012 – Aug 2016)", ha="center", va="top",
            color=INK2, fontsize=8)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.18), ncol=2)
    p = FIG / "fig1_panel_by_year.png"
    fig.savefig(p)
    plt.close(fig)
    return p


def fig_rmse_by_treatment() -> Path:
    order = ["iv_surface", "iv_atm_same_day", "iv_atm", "garch", "har", "lstm", "hv21"]
    vals = [rm("pinn_p", t) for t in order]
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    y = list(range(len(order)))[::-1]
    ax.barh(y, vals, height=0.62, color=BLUE)
    for yi, v in zip(y, vals):
        ax.text(v + 0.6, yi, f"{v:.1f}", va="center", color=INK, fontsize=8)
    ax.set_yticks(y, [LABEL[t] for t in order])
    ax.set_xlabel("RMSE vs market mid (index points) — PINN Mode P, test 2020–2023")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(vals) * 1.12)
    p = FIG / "fig2_rmse_by_treatment.png"
    fig.savefig(p)
    plt.close(fig)
    return p


def _grouped(dim: str, buckets: list[str], labels: list[str], fname: str, xlabel: str) -> Path:
    d = strat[(strat.dimension == dim) & (strat.engine == "pinn_p")]
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    ts = ["garch", "lstm", "iv_atm", "iv_surface"]
    w = 0.2
    for j, t in enumerate(ts):
        v = [float(d[(d.bucket == b) & (d.treatment == t)].rmse.iloc[0]) for b in buckets]
        xs = [i + (j - 1.5) * (w + 0.012) for i in range(len(buckets))]
        ax.bar(xs, v, w, color=SERIES[t], label=LABEL[t])
    ax.set_xticks(range(len(buckets)), labels)
    ax.set_ylabel("RMSE (index points)")
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.16), ncol=4)
    p = FIG / fname
    fig.savefig(p)
    plt.close(fig)
    return p


def fig_moneyness() -> Path:
    b = ["K/F 0.80-0.90", "K/F 0.90-0.97", "K/F 0.97-1.03", "K/F 1.03-1.10", "K/F 1.10-1.20"]
    return _grouped("moneyness_bucket", b, ["0.80–0.90", "0.90–0.97", "0.97–1.03", "1.03–1.10", "1.10–1.20"],
                    "fig3_moneyness.png", "Forward moneyness K/F (low = OTM puts / ITM calls)")


def fig_regime() -> Path:
    b = ["calm", "covid_crash", "bear_2022", "other"]
    return _grouped("regime", b, ["Calm", "COVID crash 2020", "Bear market 2022", "Other"], "fig4_regime.png",
                    "Market regime (test period)")


def fig_vol_bias() -> Path:
    """Mean forecast vs mean realised vol on the test split, by horizon."""
    d = vfm[(vfm.split == "test") & vfm.model.isin(["garch", "lstm", "iv_atm"])]
    fig, ax = plt.subplots(figsize=(7.2, 2.7))
    for t in ["garch", "lstm", "iv_atm"]:
        s = d[d.model == t].sort_values("horizon")
        ax.plot(s.horizon, s.mean_forecast_vol * 100, color=SERIES[t], lw=2, marker="o", ms=4, label=LABEL[t])
    s = d[d.model == "garch"].sort_values("horizon")
    ax.plot(s.horizon, s.mean_realized_vol * 100, color=INK2, lw=2, ls="--", label="Realised vol (truth)")
    ax.set_xlabel("Forecast horizon (trading days)")
    ax.set_ylabel("Mean annualised vol (%)")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.17), ncol=4)
    p = FIG / "fig5_vol_level.png"
    fig.savefig(p)
    plt.close(fig)
    return p


figs = {"panel": fig_panel_by_year(), "treat": fig_rmse_by_treatment(), "mny": fig_moneyness(),
        "regime": fig_regime(), "vol": fig_vol_bias()}

# ---------------------------------------------------------------- document styles
FONTS = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
pdfmetrics.registerFont(TTFont("DejaVu", str(FONTS / "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("DejaVu-Bold", str(FONTS / "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFont(TTFont("DejaVu-Oblique", str(FONTS / "DejaVuSans-Oblique.ttf")))
pdfmetrics.registerFont(TTFont("DejaVuMono", str(FONTS / "DejaVuSansMono.ttf")))
from reportlab.lib.fonts import addMapping  # noqa: E402

addMapping("DejaVu", 0, 0, "DejaVu")
addMapping("DejaVu", 1, 0, "DejaVu-Bold")
addMapping("DejaVu", 0, 1, "DejaVu-Oblique")
addMapping("DejaVu", 1, 1, "DejaVu-Bold")

ss = getSampleStyleSheet()
ACCENT = colors.HexColor(BLUE)
body = ParagraphStyle("body", parent=ss["Normal"], fontName="DejaVu", fontSize=9.6, leading=13.6, spaceAfter=5,
                      textColor=colors.HexColor(INK))
small = ParagraphStyle("small", parent=body, fontSize=8.2, leading=11, textColor=colors.HexColor(INK2))
cap = ParagraphStyle("cap", parent=small, alignment=TA_CENTER, spaceBefore=2, spaceAfter=10)
h1 = ParagraphStyle("h1", parent=body, fontName="DejaVu-Bold", fontSize=16, leading=20, spaceBefore=4, spaceAfter=8,
                    textColor=ACCENT)
h2 = ParagraphStyle("h2", parent=body, fontName="DejaVu-Bold", fontSize=12, leading=16, spaceBefore=10, spaceAfter=4)
h3 = ParagraphStyle("h3", parent=body, fontName="DejaVu-Bold", fontSize=10.2, leading=14, spaceBefore=6, spaceAfter=2)
qlab = ParagraphStyle("q", parent=body, fontName="DejaVu-Bold", fontSize=9.4, textColor=ACCENT, spaceAfter=1,
                      spaceBefore=3)
bullet = ParagraphStyle("bullet", parent=body, leftIndent=12, bulletIndent=2, spaceAfter=2)
title = ParagraphStyle("title", parent=h1, fontSize=22, leading=27, textColor=colors.HexColor(INK), spaceAfter=6)
sub = ParagraphStyle("sub", parent=body, fontSize=11, leading=15, textColor=colors.HexColor(INK2))
cell = ParagraphStyle("cell", parent=body, fontSize=8.2, leading=11.8, spaceAfter=0)
cellb = ParagraphStyle("cellb", parent=cell, fontName="DejaVu-Bold")

story: list = []
P = lambda t, s=body: story.append(Paragraph(t, s))  # noqa: E731


def bullets(items: list[str]) -> None:
    for it in items:
        story.append(Paragraph(it, bullet, bulletText="•"))


def qa(pairs: list[tuple[str, str]]) -> None:
    for q, a in pairs:
        story.append(KeepTogether([Paragraph(q, qlab), Paragraph(a, body)]))


def table(rows: list[list], widths: list[float], header: bool = True, zebra: bool = True) -> Table:
    data = [[Paragraph(str(c), cellb if (header and i == 0) else cell) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=[w * cm for w in widths], repeatRows=1 if header else 0)
    st = [("VALIGN", (0, 0), (-1, -1), "TOP"), ("LINEBELOW", (0, 0), (-1, 0), 0.8, colors.HexColor(INK2)),
          ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
          ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4)]
    if zebra:
        for i in range(1, len(rows)):
            if i % 2 == 0:
                st.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#f5f4f0")))
    t.setStyle(TableStyle(st))
    return t


def figure(path: Path, caption: str, width_cm: float = 16.5, lead: str | None = None) -> None:
    img = Image(str(path))
    ratio = img.imageHeight / img.imageWidth
    img.drawWidth, img.drawHeight = width_cm * cm, width_cm * cm * ratio
    parts = ([Paragraph(lead, h2)] if lead else []) + [img, Paragraph(caption, cap)]
    story.append(KeepTogether(parts))


def callout(text: str) -> None:
    t = Table([[Paragraph(text, body)]], colWidths=[16.5 * cm])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#eef4fc")),
                           ("LINEBEFORE", (0, 0), (0, -1), 3, ACCENT), ("LEFTPADDING", (0, 0), (-1, -1), 9),
                           ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
    story.append(t)
    story.append(Spacer(1, 6))


# ---------------------------------------------------------------- numbers used in text
g_bs, l_bs, a_bs, s_bs = (rm("bs", t) for t in ["garch", "lstm", "iv_atm", "iv_surface"])
g_p, l_p, a_p, s_p = (rm("pinn_p", t) for t in ["garch", "lstm", "iv_atm", "iv_surface"])
g_h, l_h, a_h, s_h = (rm("pinn_h0.1", t) for t in ["garch", "lstm", "iv_atm", "iv_surface"])
g_bias, l_bias = rm("pinn_p", "garch", "bias"), rm("pinn_p", "lstm", "bias")
pv = vbs[vbs.engine == "pinn_p"]
solver_lo, solver_hi = pv.rmse.min(), pv.rmse.max()
cn_hi = vbs[vbs.engine == "cn"].rmse.max()
hv = vbs[vbs.engine == "pinn_h0.1"].set_index("treatment").rmse
reg = strat[(strat.dimension == "regime") & (strat.engine == "pinn_p")].set_index(["bucket", "treatment"]).rmse
mny = strat[(strat.dimension == "moneyness_bucket") & (strat.engine == "pinn_p")].set_index(["bucket", "treatment"])
lstm_mz = vfm[(vfm.model == "lstm") & (vfm.split == "test") & (vfm.horizon == 21)].iloc[0]
lstm_val = vfm[(vfm.model == "lstm") & (vfm.split == "validation") & (vfm.horizon == 21)].iloc[0]

# ================================================================= COVER
story += [Spacer(1, 2.2 * cm)]
P("Research progress report", sub)
P("Effect of the Volatility Forecasting Method (GARCH, LSTM, Implied Volatility) on the Pricing Accuracy of a "
  "Physics-Informed Neural Network for European Options", title)
P("What happened, why it happened, what the first full run on real SPX data shows, and what we should improve "
  "next — explained step by step.", sub)
story.append(Spacer(1, 0.6 * cm))
story.append(table([
    ["Item", "Value"],
    ["Data", "OptionsDX SPX end-of-day quotes 2012–2023 (144 monthly files, 15.07 M raw rows)"],
    ["Test sample", "1,993,333 European SPX monthly options on 992 trading days (2020-01-02 → 2023-12-29)"],
    ["Code", "pinnvol pipeline, branch claude/pinn-research-pipeline-4u075w, commit 7dcead9"],
    ["Hardware", "Laptop, NVIDIA RTX 3050 Ti (4 GB), 7 GB RAM, WSL2"],
    ["Run completed", "4 October 2026, 21:15 (laptop time, UTC+7) — total compute ≈ 6 h"],
], [3.2, 13.3]))
story.append(Spacer(1, 0.8 * cm))
P("<b>How to read this report.</b> Section 1 is a one-page summary. Section 2 explains the study design in plain "
  "words. Section 3 tells the story of the run as a sequence of questions (<i>What happened? Why? What did we do? "
  "What does it change?</i>). Section 4 presents the results. Section 5 lists the improvements, each with the "
  "evidence behind it, the reason it matters, the concrete action and what we expect to happen. Section 6 is the "
  "roadmap.", body)
story.append(PageBreak())

# ================================================================= 1. SUMMARY
P("1. Executive summary", h1)
callout(f"<b>Main finding.</b> The choice of volatility input dominates pricing accuracy; the PINN itself does not. "
        f"With the same PINN, the error against market prices ranges from <b>{s_p:.1f}</b> index points (IV surface) "
        f"to <b>{g_p:.1f}</b> (GARCH) and <b>{l_p:.1f}</b> (LSTM). The PINN reproduces the analytical Black-Scholes "
        f"price to within {solver_lo:.2f}–{solver_hi:.2f} points for every input, i.e. its own error is only "
        f"about 1–5% of the volatility-induced error.")
P("<b>What we learned</b>", h3)
bullets([
    f"<b>Ranking (all engines agree):</b> IV surface ({s_p:.2f}) &lt; IV ATM ({a_p:.2f}) &lt; GARCH ({g_p:.2f}) &lt; "
    f"HAR ({rm('pinn_p', 'har'):.2f}) &lt; LSTM ({l_p:.2f}) &lt; 21-day historical vol ({rm('pinn_p', 'hv21'):.2f}). "
    "Diebold-Mariano tests: IV beats GARCH and LSTM with p &lt; 1e-9; GARCH beats LSTM only narrowly (p ≈ 0.03).",
    f"<b>Forecast-based inputs underprice options:</b> mean pricing bias is {g_bias:.1f} points for GARCH and "
    f"{l_bias:.1f} for LSTM. Options embed a volatility risk premium and a volatility smile that a single "
    "historical-vol number cannot carry.",
    f"<b>The LSTM was the best volatility forecaster in validation (2019) but the worst of the three in the test "
    f"period</b> — especially in the COVID crash, where its pricing RMSE was {reg[('covid_crash', 'lstm')]:.1f} "
    f"points versus {reg[('covid_crash', 'garch')]:.1f} for GARCH.",
    f"<b>The hybrid PINN (Mode H, λ = 0.1)</b> moves slightly towards market prices (GARCH {g_p:.1f} → {g_h:.1f}) "
    "but the gain is small; only one λ was tried.",
])
P("<b>What went wrong along the way (and is now fixed)</b>", h3)
bullets([
    "A <b>data-dating issue</b> silently removed every monthly option before September 2016: the vendor dates "
    "monthly expiries on Thursday in that period. Found on 4 Oct, fixed in commit 7dcead9; the panel grew from "
    "2.75 M to 3.28 M options and now starts in January 2012.",
    "An <b>environment failure</b> (broken shell during installation) and a <b>logging problem</b> (buffered output, "
    "≈ 50 min of compute lost) delayed the run. Neither affects the results.",
])
P("<b>Top three improvements (Section 5)</b>", h3)
bullets([
    "<b>Separate the volatility <i>level</i> from the smile</b>: give GARCH and LSTM the market's smile shape so the "
    "comparison measures forecasting skill, not the absence of a smile.",
    "<b>Correct the volatility risk premium</b> for historical forecasts and <b>strengthen the LSTM</b> "
    "(monthly refits, longer history, implied-vol features).",
    "<b>Make the statistics and pipeline robust</b>: multiple-testing control, per-regime tests, timestamped logs "
    "and checkpointing of the 5-hour pricing stage.",
])
story.append(PageBreak())

# ================================================================= 2. DESIGN
P("2. What the study does (plain-language design)", h1)
P("<b>The research question.</b> A PINN prices a European option by learning a function V(S, τ; σ, r, q) that "
  "satisfies the Black-Scholes PDE. The PINN needs one number it cannot observe: the volatility σ. We ask: "
  "<i>how much does the way we forecast σ (GARCH, LSTM or implied volatility) change the accuracy of the PINN's "
  "prices?</i>", body)
P("The Black-Scholes PDE in time-to-maturity τ, with terminal and boundary conditions, is", body)
P("∂V/∂τ = ½ σ² S² ∂²V/∂S² + (r − q) S ∂V/∂S − r V,   V(S, 0) = max(S − K, 0) (call) or max(K − S, 0) (put),",
  ParagraphStyle("eq", parent=body, fontName="DejaVu-Oblique", alignment=TA_CENTER, spaceBefore=4, spaceAfter=8))
P("and the PINN is trained on the loss L = w<sub>f</sub>·‖PDE residual‖² + w<sub>T</sub>·‖terminal error‖² + "
  "w<sub>B</sub>·‖boundary error‖² (+ λ·w<sub>f</sub>·‖market-price error‖² in the hybrid Mode H). Current weights: "
  "w<sub>f</sub> = 1, w<sub>T</sub> = 10, w<sub>B</sub> = 1.", body)
P("Pipeline — five stages", h2)
story.append(table([
    ["Stage", "What it does", "Output / runtime"],
    ["prepare", "Loads cleaned quotes, keeps SPX monthly (AM-settled) options with 7–180 days to expiry, forward "
     "moneyness 0.8–1.2, spread ≤ 20%, no-arbitrage; attaches Treasury rates and parity-implied dividend yields.",
     "panel.pkl — 3,284,324 options; 56 s"],
    ["vol", "Builds every volatility input walk-forward (no look-ahead): GARCH(1,1) and HAR-RV refit monthly, "
     "LSTM refit quarterly (5 seeds), historical vol (21/63 d), implied-vol ATM term structure and smile surface "
     "from the previous day.", "panel_vol.pkl, forecast metrics; 161 s"],
    ["train-pinn", "Trains one physics-only PINN (Mode P) over the whole domain σ ∈ [0.05, 0.90], τ ∈ [0, 0.55] "
     "(10k Adam + L-BFGS steps). One network serves every volatility input.", "pinn_mode_p.pt; 384 s"],
    ["price", "Prices every test option with Black-Scholes, Mode P PINN, Crank-Nicolson (2,000-option check), a "
     "plain neural network (FFNN control) and the hybrid PINN (Mode H, refit each quarter on the previous 12 "
     "months, 5 seeds).", "predictions.pkl (1.3 GB); 4 h 53 min"],
    ["evaluate", "RMSE / MAE / MAPE vs market mid and vs analytical BS, stratified by maturity, moneyness, regime; "
     "Diebold-Mariano tests.", "report.md + CSVs; 4 min"],
], [2.2, 10.0, 4.3]))
P("Why two kinds of error?", h2)
bullets([
    "<b>Error vs the market mid-price</b> answers the research question: how close is the model price to what "
    "the option actually traded at? It mixes the volatility error and the solver error.",
    "<b>Error vs analytical Black-Scholes with the same σ</b> isolates the solver: if the PINN were perfect this "
    "would be zero. It tells us whether differences between treatments come from σ or from the network.",
])
P("Data split: training 2012–2018, validation 2019, test 2020–2023 (includes the COVID crash and the 2022 bear "
  "market). Volatility forecasts start on 2019-01-01 and use only information available on the day before.", body)
story.append(PageBreak())

# ================================================================= 3. WHAT HAPPENED
P("3. What happened, step by step", h1)
P("The timeline below covers the run on the real data. Each step is told as four questions so the reasoning is "
  "visible, not only the outcome.", body)
story.append(table([
    ["When (UTC+7)", "Event"],
    ["28 Sep", "Raw OptionsDX files profiled and cleaned into data-clean/ (14.7 M rows). Pipeline code written "
     "and tested on synthetic data. A first attempt to run on the laptop stopped when the shell broke during "
     "package installation."],
    ["4 Oct 13:35", "Restart: Python environment rebuilt, torch 2.14.1 + CUDA confirmed on the RTX 3050 Ti; "
     "19/19 tests pass; Treasury rates downloaded."],
    ["4 Oct ~15:00", "prepare #1: panel of 2.75 M options — but it starts on 2015-06-22. vol #1 and train-pinn run."],
    ["4 Oct 15:03", "Question raised: why do the monthlies start in 2015? Diagnosis → vendor dating convention."],
    ["4 Oct 15:21", "Pricing stopped; fix pulled (7dcead9); prepare and vol re-run → 3.28 M options from 2012."],
    ["4 Oct 16:20", "price restarted with unbuffered logs after ≈ 50 min of silent running; projected 5.1 h."],
    ["4 Oct 21:15", "price (4 h 53 min) and evaluate finished. Report generated."],
], [2.8, 13.7]))

P("Step 1 — Rebuilding the environment", h2)
qa([
    ("What happened?", "An earlier session stopped part-way through installing packages: the terminal began "
     "returning file-system errors and the Python environment was left half-built (no activation script)."),
    ("Why?", "The shell itself broke while a long installation was running in the foreground — an infrastructure "
     "fault, not a code fault."),
    ("What did we do?", "Recreated the environment from scratch, installed the project with its test "
     "dependencies, and from then on ran every long command in the background with its own log file, checking "
     "it with short commands."),
    ("What does it change?", "Nothing in the results. The PyTorch build (2.14.1, CUDA 13) detected the GPU "
     "directly, so no special CUDA install was needed."),
])

P("Step 2 — The 2015 mystery: monthly options were missing", h2)
qa([
    ("What happened?", "The first panel contained 2,745,719 options but started on <b>22 June 2015</b>, although "
     "the data begin in January 2012. Between mid-2015 and mid-2016 there were only one to three expiries per "
     "day; before that, none."),
    ("Why?", "The study uses <i>monthly</i> SPX options (AM-settled, expiring on the third Friday). The cleaning "
     "step marked an option as monthly if its expiry date was a third Friday. But OptionsDX records the monthly "
     "expiry on its <b>last trading day — the Thursday before</b> — until August 2016, and on the Friday only "
     "afterwards (plus Good-Friday months, when the exchange itself moves expiry to Thursday). So every monthly "
     "before September 2016 failed the test. The panel started on 22 June 2015 simply because that is 180 days "
     "before 18 December 2015, the one expiry of that period the vendor dated on a Friday."),
    ("What did we do?", "Measured which weekday each month's expiry fell on across all 144 files, confirmed the "
     "Thursday-dated series are the monthlies (largest open series, quarterly cycle), and proposed a rule. The fix "
     "(commit 7dcead9) treats a Thursday expiry as the monthly when the next day is the third Friday and that "
     "Friday is either Good Friday or not quoted; it also adds one day to time-to-expiry for those rows, because "
     "settlement is at Friday's open. The raw and cleaned data were not modified."),
    ("What does it change?", "The panel grew to <b>3,284,324 options over all 2,992 trading days</b> "
     "(2012-01-03 → 2023-12-29). The test period gained the April 2019 and April 2022 Good-Friday months. "
     "Because the volatility models are fitted on the index series (not on option quotes), and the test starts "
     "in 2020, the headline comparison was only mildly affected — but the training-period option data, needed "
     "for several improvements below, now exists."),
])
figure(figs["panel"], "Figure 1 — Options in the panel per year, before and after the monthly-dating fix. Before the "
                      "fix, 2012–2014 were empty and 2015–2016 were nearly empty.")

P("Step 3 — Training the PINN", h2)
qa([
    ("What happened?", "Mode P trained in 384 s: 10,000 Adam steps then L-BFGS; final loss 1.7e-06. Against the "
     "closed-form Black-Scholes price on 20,000 random points, the mean absolute error was 1.06e-04 in normalised "
     "units (0.43 index points at K = 4000)."),
    ("Why was it not re-trained after the data fix?", "Mode P never sees market data — it learns the PDE over a "
     "box of (S/K, τ, σ, r, q). The data fix cannot change it, so we kept it, saving time."),
])

P("Step 4 — Pricing, and the silent log", h2)
qa([
    ("What happened?", "The pricing stage ran for 50 minutes with an empty log, so its total runtime could not "
     "be projected (the plan said: if it exceeds 8 h, reduce Mode H seeds from 5 to 2)."),
    ("Why?", "Python buffers printed output when it is written to a file rather than a screen; progress lines "
     "only appear when the buffer fills or the program ends. Attaching a debugger to look inside required "
     "administrator rights."),
    ("What did we do?", "Stopped it and restarted with unbuffered output. The first quarterly block then took "
     "≈ 18 min; the projection was 5.1 h, so all 5 seeds were kept. Actual: 4 h 53 min."),
    ("What does it change?", "≈ 50 min of compute lost; no effect on results. It motivates improvement E1 "
     "(timestamped logs, checkpointing)."),
])
story.append(PageBreak())

# ================================================================= 4. RESULTS
P("4. What the results show", h1)
P("4.1 Headline: pricing error against the market", h2)
rows = [["Volatility input", "Black-Scholes", "PINN Mode P", "PINN Mode H (λ=0.1)", "FFNN control"]]
for t in ["iv_surface", "iv_atm", "garch", "lstm"]:
    f = mk[(mk.engine == "ffnn") & (mk.treatment == t)]
    rows.append([LABEL[t], f"{rm('bs', t):.2f}", f"{rm('pinn_p', t):.2f}", f"{rm('pinn_h0.1', t):.2f}",
                 f"{float(f.rmse.iloc[0]):.2f}" if len(f) else "—"])
for t in ["har", "hv21", "iv_atm_same_day"]:
    rows.append([LABEL[t], f"{rm('bs', t):.2f}", f"{rm('pinn_p', t):.2f}", "—", "—"])
story.append(table(rows, [4.6, 2.8, 2.8, 3.5, 2.8]))
P("RMSE in index points, 1,993,333 test options (2020–2023). The contract's own same-day IV (an oracle that uses "
  "the option's price to price itself) gives 0.19 (BS) and 0.37 (PINN) and is excluded from the ranking.", small)
figure(figs["treat"], "Figure 2 — Pricing error of the physics-only PINN by volatility input. Implied-volatility inputs "
                      "are 1.4–5.8× more accurate than forecast-based inputs.")
P("<b>Reading it.</b> Moving from GARCH to the IV surface cuts the error by about 78%. The three engines "
  "(BS, PINN P, PINN H) give almost the same numbers for each input, which is the key methodological result: "
  "the PINN is a faithful Black-Scholes solver, so <i>the volatility input explains almost all of the pricing "
  "error</i>. The FFNN control (a network with no physics) is much worse with the IV surface (20.2 vs 7.3), "
  "showing the value of the PDE constraint.", body)

figure(figs["mny"], "Figure 3 — Error by moneyness. GARCH and LSTM fail most on low strikes (out-of-the-money puts), "
                    "where the market smile is steepest; the IV surface is flat across moneyness.",
       lead="4.2 Where the errors come from")
P(f"For K/F 0.80–0.90 the GARCH error is {mny.loc[('K/F 0.80-0.90', 'garch')].rmse:.1f} points with a bias of "
  f"{mny.loc[('K/F 0.80-0.90', 'garch')].bias:.1f}, versus {mny.loc[('K/F 0.80-0.90', 'iv_surface')].rmse:.1f} for "
  "the IV surface. A single σ cannot reproduce the volatility skew: deep out-of-the-money puts trade at much "
  "higher implied volatility than at-the-money options. Errors also grow with maturity (GARCH 10.8 points at "
  "7–30 days, 48.3 at 120–180 days), because a constant σ error is multiplied by vega, which rises with τ.", body)
figure(figs["regime"], "Figure 4 — Error by market regime. The LSTM breaks down in the COVID crash.")
P(f"In the 2020 COVID crash the LSTM error reached {reg[('covid_crash', 'lstm')]:.1f} points, about "
  f"twice GARCH ({reg[('covid_crash', 'garch')]:.1f}). In calm markets the LSTM is competitive "
  f"({reg[('calm', 'lstm')]:.1f} vs {reg[('calm', 'garch')]:.1f}).", body)

figure(figs["vol"], "Figure 5 — Mean forecast vs mean realised volatility on the test period. The LSTM forecasts are "
                    "systematically too low; implied vol sits above realised vol (the volatility risk premium).",
       lead="4.3 How good were the volatility forecasts themselves?")
story.append(table([
    ["21-day horizon", "Validation 2019 QLIKE", "Test 2020–23 QLIKE", "Test Mincer-Zarnowitz slope (ideal 1)"],
    *[[LABEL[m], f"{vfm[(vfm.model == m) & (vfm.split == 'validation') & (vfm.horizon == 21)].qlike.iloc[0]:.3f}",
       f"{vfm[(vfm.model == m) & (vfm.split == 'test') & (vfm.horizon == 21)].qlike.iloc[0]:.3f}",
       f"{vfm[(vfm.model == m) & (vfm.split == 'test') & (vfm.horizon == 21)].mz_b.iloc[0]:.2f}"]
      for m in ["garch", "lstm", "iv_atm", "har"]],
], [4.0, 4.0, 3.8, 4.7]))
P(f"QLIKE is a standard loss for variance forecasts (lower is better). The LSTM was best in 2019 "
  f"({lstm_val.qlike:.3f}) and worst in the test ({lstm_mz.qlike:.3f}): it learned a calm market and did not "
  "adapt quickly to 2020 and 2022.", small)

P("4.4 Solver accuracy and the hybrid PINN", h2)
story.append(table([
    ["Engine", "RMSE vs analytical BS (same σ)", "Meaning"],
    ["Crank-Nicolson (2,000 options)", f"≤ {cn_hi:.3f}", "Reference numerical solver — essentially exact"],
    ["PINN Mode P", f"{solver_lo:.2f} – {solver_hi:.2f}", "≈ 1–5% of the volatility-induced error; adequate"],
    ["PINN Mode H, λ = 0.1", f"{hv.min():.2f} – {hv.max():.2f}",
     "Deliberately departs from BS to fit market prices"],
    ["FFNN control", "19 – 52", "No physics: not a BS solver at all"],
], [5.0, 4.5, 7.0]))
P(f"Mode H's seed-to-seed spread is small (GARCH RMSE {seeds[seeds.treatment == 'garch'].rmse.min():.2f}–"
  f"{seeds[seeds.treatment == 'garch'].rmse.max():.2f}), so five seeds are more than enough; two would do.", body)
story.append(PageBreak())

# ================================================================= 5. IMPROVEMENTS
P("5. What we can improve — and why", h1)
P("Each item gives the <b>evidence</b> from this run, <b>why</b> it matters for the research question, the "
  "concrete <b>action</b>, and what we <b>expect</b> to happen. Expectations are hypotheses to be tested, not "
  "results. Priority: <b>High</b> = changes the conclusions; <b>Medium</b> = strengthens them; <b>Low</b> = "
  "engineering/quality.", body)


def improvement(code: str, name: str, prio: str, evidence: str, why: str, action: str, expect: str,
                cost: str) -> None:
    head = Paragraph(f"{code}. {name} <font color='{INK2}' size='8.5'>— priority: {prio}</font>", h2)
    t = table([["", ""], ["Evidence", evidence], ["Why it matters", why], ["Action", action],
               ["What we expect", expect], ["Cost", cost]], [3.1, 13.4], header=False, zebra=False)
    t.setStyle(TableStyle([("BACKGROUND", (0, 1), (0, -1), colors.HexColor("#f5f4f0")),
                           ("LINEBELOW", (0, 1), (-1, -2), 0.3, colors.HexColor(GRID)),
                           ("FONTNAME", (0, 1), (0, -1), "DejaVu-Bold")]))
    story.append(KeepTogether([head, t]))
    story.append(Spacer(1, 4))


P("A. Research design", h2)
improvement(
    "A1", "Separate the volatility level from the smile", "High",
    f"Most of the GARCH/LSTM error sits in the wings: at K/F 0.80–0.90 GARCH RMSE is "
    f"{mny.loc[('K/F 0.80-0.90', 'garch')].rmse:.1f} vs {mny.loc[('K/F 0.80-0.90', 'iv_surface')].rmse:.1f} "
    "for the IV surface. GARCH, LSTM and IV ATM each give one σ per day and maturity; the IV surface also gives a "
    "smile.",
    "The research question is about <i>forecasting methods</i>. Today the comparison partly measures 'has a smile "
    "vs has no smile'. An IV-surface win is therefore partly structural, not a forecasting win. Examiners will "
    "ask this.",
    "Add 'smile-adjusted' treatments: σ<sub>i</sub> = σ<sub>forecast</sub>(T) × IV<sub>surface</sub>(K,T) / "
    "IV<sub>ATM</sub>(T), using the previous day's smile shape. Then GARCH, LSTM and IV ATM compete on the "
    "<i>level</i> with an identical smile.",
    "The gap between the methods narrows, and the remaining difference measures forecasting skill only. Wing "
    "errors for GARCH and LSTM should fall sharply.",
    "Small code change in vol; re-run vol + price for the new treatments (Mode P only ≈ 30 min; with Mode H ≈ +2 h).")
improvement(
    "A2", "Correct for the volatility risk premium", "High",
    f"All historical forecasts underprice: bias {g_bias:.1f} (GARCH), {l_bias:.1f} (LSTM). Figure 5: implied vol "
    "sits above realised vol on average.",
    "Option prices embed a premium for bearing volatility risk, so even a perfect forecast of <i>realised</i> "
    "volatility will underprice options. Without this correction the study partly compares 'physical' with "
    "'risk-neutral' volatility.",
    "Estimate a rolling premium k<sub>t</sub> = mean(IV<sub>ATM</sub> / σ<sub>forecast</sub>) over the previous "
    "12 months (no look-ahead) and price with k<sub>t</sub>·σ<sub>forecast</sub>. Report both raw and "
    "premium-adjusted results.",
    "The bias for GARCH/LSTM should shrink towards zero and their RMSE fall, isolating the dynamic forecasting "
    "skill from the level shift.",
    "Small: one function in vol; price re-run ≈ 30 min for Mode P.")
improvement(
    "A3", "Strengthen the LSTM", "High",
    f"Best in validation (QLIKE {lstm_val.qlike:.3f}), worst in test ({lstm_mz.qlike:.3f}); Mincer-Zarnowitz slope "
    f"{lstm_mz.mz_b:.2f} (forecasts too flat); mean 21-day forecast 17.0% vs 19.3% realised; COVID RMSE "
    f"{reg[('covid_crash', 'lstm')]:.1f}.",
    "A weak LSTM makes 'LSTM loses' a statement about this configuration, not about deep learning. The model is "
    "small (1 layer, 32 units), sees only past returns, is refit quarterly, and trains on ≈ 1,700–2,900 daily "
    "samples since 2012.",
    "(1) Refit monthly like GARCH. (2) Add a longer history via <i>paths.extra_prices_csv</i> (SPX closes from "
    "2000, already supported). (3) Add implied-vol/VIX features (an 'LSTM-IV' hybrid). (4) Predict log realised "
    "variance at the option's own horizon. (5) Choose hyper-parameters on a validation window that includes a "
    "stress period.",
    "More reactive forecasts in crises and a slope closer to 1. The LSTM should at least match GARCH; with IV "
    "features it may approach IV ATM.",
    "Medium: vol stage grows from minutes to perhaps 30–60 min; price as above.")
improvement(
    "A4", "Explore the hybrid weight λ", "Medium",
    f"Only λ = 0.1 was run. Gains are small (GARCH {g_p:.1f} → {g_h:.1f}; LSTM {l_p:.1f} → {l_h:.1f}) and the "
    f"network drifts from BS by {hv.min():.1f}–{hv.max():.1f} points.",
    "λ controls how much the PINN trusts market prices versus physics. One value cannot show the trade-off, "
    "which is the most interesting PINN-specific result of the project.",
    "Run λ ∈ {0.01, 0.1, 1, 10} with 2 seeds (seed spread is tiny); plot market error and BS error against λ.",
    "A U-shaped or monotone curve showing where data help and where they cause over-fitting.",
    "≈ 2 h of GPU time per λ value with 2 seeds.")
improvement(
    "A5", "Treat the same-day contract IV as an oracle", "Medium",
    "It gives 0.19 RMSE because it uses the option's own price to price itself.",
    "If presented alongside the forecasting methods, it can be misread as a winning method.",
    "Label it 'upper bound / oracle' in every table and exclude it from rankings and DM tests (or report it "
    "separately).", "Clearer tables; no change to conclusions.", "Trivial.")

P("B. Statistics and evaluation", h2)
improvement(
    "B1", "Control for multiple comparisons; test per regime", "Medium",
    "The report runs 28 pairwise Diebold-Mariano tests per engine (≈ 70 in total) at nominal p-values.",
    "With many tests some 'significant' results are expected by chance; the GARCH-vs-LSTM p ≈ 0.03 is exactly the "
    "kind of result that can disappear after correction.",
    "Apply Holm-Bonferroni or report a Model Confidence Set (Hansen et al., 2011); repeat DM tests within each "
    "regime and maturity bucket.",
    "IV > GARCH/LSTM will survive any correction (p &lt; 1e-9); GARCH vs LSTM may not. That is an honest, "
    "stronger statement.", "Evaluate-only: minutes, no re-pricing.")
improvement(
    "B2", "Report errors in volatility units and relative terms", "Low",
    "RMSE in index points is dominated by long-dated, high-premium options (120–180 d errors are ~5× the 7–30 d "
    "errors). MAPE is noisy for cheap options.",
    "Readers compare across maturities and with the literature, which often uses IV-RMSE or percentage errors.",
    "Lead with IV-RMSE (already computed) and vega-weighted errors next to price RMSE.",
    "Same ranking, fairer weighting across maturities.", "Evaluate-only.")

P("C. Data", h2)
improvement(
    "C1", "Document and propagate the monthly-dating fix", "Medium",
    "The fix lives in the loader; the is_third_friday column in data-clean/ is still wrong for 2012–2016.",
    "Anyone reading the cleaned files directly (e.g. for a chart in the thesis) will repeat the mistake.",
    "Add a note to data-clean/README and the data profile; optionally rename the column to "
    "'is_friday_dated_monthly' in a future rebuild of the cleaned data.",
    "No silent data loss in future analyses.", "Trivial.")
improvement(
    "C2", "Use the 2012–2018 option data and run the robustness sample", "Medium",
    "After the fix, 2012–2018 options exist (≈ 0.97 M rows) but no model uses them yet: volatility models use the "
    "index, the test starts in 2020, Mode H looks back 12 months.",
    "These years can calibrate the risk premium (A2), give the LSTM-IV more history (A3), and give Mode H longer "
    "windows. The weekly (SPXW) sample tests whether conclusions hold beyond monthlies.",
    "Use the training-period panel for A2/A3; run once with sample.contract_family: all.",
    "Better-calibrated inputs; a robustness table for the thesis.", "One extra full run (≈ 6 h).")

P("D. PINN quality", h2)
improvement(
    "D1", "Tighten the physics-only PINN", "Low",
    f"PINN error vs BS is {solver_lo:.2f}–{solver_hi:.2f} points, ≈ 50× Crank-Nicolson; with the oracle IV only "
    "84% of PINN prices fall inside the bid-ask spread vs 99.99% for BS.",
    "Small relative to volatility error, so it does not change the ranking — but a reviewer may ask why the "
    "PINN is not as accurate as a classical solver.",
    "Try learning-rate-annealing loss weights (weighting: lra, already implemented), more L-BFGS steps and a "
    "wider network; add a no-arbitrage (convexity) penalty.",
    "Solver error below 0.1 points, strengthening the claim that the PINN is a faithful BS solver.",
    "≈ 10–20 min per training run.")

P("E. Engineering and reproducibility", h2)
improvement(
    "E1", "Timestamped, unbuffered logs and checkpointing", "Low",
    "≈ 50 min lost to a silent, buffered log; the 5-hour price stage keeps everything in memory and writes "
    "only at the end.",
    "A crash at hour 4 would lose all pricing work; runtime decisions need visible progress.",
    "Flush logs with timestamps per block; save each quarterly block's predictions and resume from the last "
    "finished block; store predictions as Parquet/float32 (1.3 GB → ≈ 0.4 GB).",
    "Safe long runs and less disk/RAM pressure on a 7 GB laptop.", "Small code change.")
improvement(
    "E2", "Freeze the environment", "Low",
    "torch 2.14.1 + CUDA 13 and other versions were installed fresh during the run.",
    "Exact reproducibility of numbers for the thesis appendix.",
    "Commit a lock file (uv pip freeze) and record git commit + config in each run folder (config_used.yaml is "
    "already saved).", "Results reproducible bit-for-bit on this laptop.", "Trivial.")
story.append(PageBreak())

# ================================================================= 6. ROADMAP
P("6. What will happen next — roadmap", h1)
P("The order puts cheap, conclusion-changing work first. Compute times are for this laptop.", body)
story.append(table([
    ["Phase", "Work", "Compute", "Answers the question"],
    ["1", "B1, B2, A5, C1, E1, E2 — statistics, labelling, docs, logging, checkpointing", "Minutes (evaluate "
     "only)", "Which differences are real after multiple-testing control?"],
    ["2", "A1 smile-adjusted and A2 premium-adjusted treatments (Mode P + BS)", "≈ 1 h",
     "Is IV better because it forecasts better, or because it carries the smile and premium?"],
    ["3", "A3 stronger LSTM (monthly refit, longer history, IV features)", "≈ 1–2 h",
     "Can a well-specified deep-learning forecaster match GARCH / IV?"],
    ["4", "A4 λ grid for the hybrid PINN (2 seeds)", "≈ 8 h (overnight)",
     "How much should a PINN trust market data vs physics?"],
    ["5", "C2 robustness (SPXW weeklies) and D1 PINN tightening", "≈ 6–8 h",
     "Do conclusions hold on other contracts and with a more accurate PINN?"],
    ["6", "Write-up: final tables and figures, regenerate this report", "—", "—"],
], [1.3, 7.2, 2.8, 5.2]))
P("What we expect overall", h2)
bullets([
    "The <b>qualitative ranking</b> (IV ≥ GARCH ≥ LSTM in raw form) is very likely to stay, since it is large and "
    "highly significant.",
    "After A1 + A2, the <b>gap between forecast-based and implied inputs should shrink</b> substantially, and the "
    "remaining gap is the true contribution of the forecasting method — the cleanest answer to the research "
    "question.",
    "The <b>PINN-specific contribution</b> will come mainly from A4 (data–physics trade-off) and D1 (solver "
    "accuracy), since the physics-only PINN currently behaves like Black-Scholes.",
])
P("Appendix — key settings of this run", h2)
story.append(table([
    ["Setting", "Value"],
    ["Sample filters", "monthly AM-settled SPX; 7–180 days; forward moneyness 0.80–1.20; relative spread ≤ 20%"],
    ["Splits", "train 2012–2018 · validation 2019 · test 2020–2023"],
    ["GARCH / HAR", "refit monthly, ≥ 750 observations"],
    ["LSTM", "sequence 60 d, 1 layer × 32 units, 60 epochs, 5 seeds, refit quarterly"],
    ["PINN", "5 layers × 64 tanh; domain log(S/K) ∈ [−2.5, 2.5], τ ∈ [0, 0.55], σ ∈ [0.05, 0.90]; "
             "weights PDE 1 / terminal 10 / boundary 1; 10k Adam + 1.5k L-BFGS"],
    ["Mode H", "λ = 0.1; 12-month trailing window; quarterly blocks; 1,500 steps; 5 seeds"],
    ["Rates / dividends", "FRED DGS1MO/3MO/6MO/1 Treasury yields matched by maturity; dividend yield implied from put-call parity"],
], [3.5, 13.0]))


def on_page(canv, doc):  # noqa: ANN001
    canv.saveState()
    canv.setFont("DejaVu", 7.5)
    canv.setFillColor(colors.HexColor(INK2))
    canv.drawString(2 * cm, 1.2 * cm, "PINN volatility study — progress report, 4 October 2026")
    canv.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"{doc.page}")
    canv.restoreState()


out = OUT / "PINN_volatility_progress_report.pdf"
doc = SimpleDocTemplate(str(out), pagesize=A4, leftMargin=2.2 * cm, rightMargin=2.2 * cm, topMargin=1.8 * cm,
                        bottomMargin=1.9 * cm, title="PINN volatility study — progress and improvement report",
                        author="Asia")
doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
print(out)
