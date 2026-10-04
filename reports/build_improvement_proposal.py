"""Build the improvement proposal (Markdown + PDF) from one source.

Every item has an ID (A1, B1, ...). To start work, name the ID(s): "do A2", "do Phase 2",
"do A1 and A2 without Mode H", "do A4 with lambdas 0.05 and 0.5".

Run from the repository root:  python reports/build_improvement_proposal.py
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib
from reportlab.lib import colors
from reportlab.lib.fonts import addMapping
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

OUT = Path(__file__).resolve().parent
DATE = "4 October 2026"
BASELINE = "runs/main (commit 7dcead9): IV surface 7.29, IV ATM 23.37, GARCH 33.44, LSTM 36.21 RMSE (PINN Mode P)"

# --------------------------------------------------------------------------------------------- content
# Inline markup: **bold**, `code`. Kept simple so the same strings render in Markdown and PDF.

HOW_TO = [
    "Pick one or more item IDs below and tell me, e.g. **\"do A2\"**, **\"do Phase 2\"**, **\"do A1 and A2 without "
    "Mode H\"**, or **\"do A4 with λ = 0.05 and 0.5\"**. You can change any default in the item when you ask "
    "(\"do A3 but keep quarterly refits\").",
    "For each item I will: implement the change on branch `claude/pinn-research-pipeline-4u075w`, add tests, run "
    "`pytest`, run the experiment on the laptop in the background, and send you the key numbers against the "
    "baseline plus anything that went wrong.",
    "Every experiment writes to its **own folder** `runs/<ID>/` with its own config `configs/<ID>.yaml`, so "
    "`runs/main` stays as the untouched baseline and results are always comparable.",
    "I commit locally after each item; I **push only when you say so**. `dataset/` and `data-clean/` are never "
    "modified.",
    "Where an item needs a decision from you (marked **Decision needed**), I ask before starting.",
]

RULES = [
    ("Baseline", BASELINE),
    ("Test sample", "unchanged: 1,993,333 monthly SPX options, 2020-01-02 → 2023-12-29, unless the item says "
                    "otherwise"),
    ("Metrics", "RMSE / MAE / MAPE vs market mid and vs analytical Black-Scholes; IV-RMSE; DM tests — always for "
                "GARCH, LSTM and IV side by side"),
    ("No look-ahead", "every new input uses only information available at the previous close; a test checks it"),
    ("Runtime guard", "long stages run with nohup + unbuffered logs; if a run projects beyond 8 h I report "
                      "and reduce seeds/subsample before continuing"),
]

PHASES = [
    ("Phase 1", "Make experiments cheap and statistics sound", ["E1", "E3", "E2", "B1", "B2", "A5", "C1"],
     "≈ 1 day of coding; evaluate-only runs (minutes)"),
    ("Phase 2", "Fix the comparison itself", ["A1", "A2"], "≈ 1 h compute (Mode P + BS only)"),
    ("Phase 3", "Give the LSTM a fair chance", ["A3"], "≈ 1–2 h compute"),
    ("Phase 4", "The PINN-specific question: data vs physics", ["A4"], "≈ 8 h (overnight)"),
    ("Phase 5", "Robustness and solver quality", ["C2", "D1"], "≈ 6–8 h"),
    ("Phase 6", "Write-up", ["W1"], "minutes"),
]

# Each item: id, title, priority, phase, goal, why, change (list), config (list), run (list), done (list),
# compute, depends, decision (or None), report
ITEMS = [
    dict(id="E1", title="Timestamped logs and resumable pricing", priority="Low (enabler)", phase="Phase 1",
         goal="Never lose a long run again and always know how far it is.",
         why="≈ 50 min were lost to a silent, buffered log; `price` (≈ 5 h) holds everything in memory and writes "
             "only at the end, so a crash at hour 4 loses all of it.",
         change=["`cli.py`: force line-buffered stdout; prefix every log line with a timestamp and elapsed time.",
                 "`pipeline/pricing.py`: save each quarterly block to `runs/<ID>/blocks/<quarter>.parquet` and skip "
                 "blocks already on disk (resume).",
                 "Store `predictions` as Parquet with float32 prices (≈ 1.3 GB → ≈ 0.4 GB)."],
         config=["`price.resume: true` (default)"],
         run=["`python -m pinnvol price -c configs/default.yaml` — kill it after one block, restart, confirm it "
              "resumes."],
         done=["Log lines carry timestamps; a restarted run resumes from the last finished block.",
               "`evaluate` on the Parquet output reproduces `runs/main/report.md` numbers exactly."],
         compute="coding ≈ 2 h; verification ≈ 30 min", depends="—", decision=None,
         report="Resume test result; file-size change; confirmation that numbers are identical."),
    dict(id="E3", title="Re-price only what changed", priority="High (enabler)", phase="Phase 1",
         goal="Let new treatments or engines be priced in minutes instead of re-running the full 5-hour stage.",
         why="Most improvements add one volatility input or change one engine. Today `price` recomputes every "
             "engine × treatment, including Mode H (the 5-hour part).",
         change=["`pipeline/pricing.py`: `price.only_engines` / `price.only_treatments` filters; merge new columns "
                 "into an existing predictions file from a `price.base_run` folder.",
                 "`pinn.mode_h.treatments` already exists — document it."],
         config=["`price.base_run: runs/main`", "`price.only_engines: [bs, pinn_p]`",
                 "`price.only_treatments: [garch_smile, ...]`"],
         run=["Re-price one existing treatment with the filter and compare with `runs/main`."],
         done=["Re-pricing a single treatment with BS + Mode P takes < 15 min and matches `runs/main` to "
               "floating-point precision."],
         compute="coding ≈ 2 h", depends="E1", decision=None,
         report="Timing of a single-treatment re-price; equality check."),
    dict(id="E2", title="Freeze the environment", priority="Low", phase="Phase 1",
         goal="Make every number in the thesis reproducible.",
         why="torch 2.14.1 + CUDA 13 and all other packages were installed fresh during the run; versions are "
             "not recorded.",
         change=["Add `requirements-lock.txt` (from `uv pip freeze`).",
                 "Write `git_commit`, package versions and GPU name into each run folder (`run_info.json`)."],
         config=[], run=["Any stage; check `run_info.json`."],
         done=["Each `runs/<ID>/` contains `config_used.yaml` and `run_info.json`."],
         compute="≈ 30 min", depends="—", decision=None, report="The lock file and an example `run_info.json`."),
    dict(id="B1", title="Multiple-testing control and per-regime tests", priority="Medium", phase="Phase 1",
         goal="Report only differences that survive correction for running ≈ 70 tests.",
         why="DM tests are reported at nominal p-values. GARCH vs LSTM (p ≈ 0.03) may not survive correction.",
         change=["`evaluation/stats.py`: Holm–Bonferroni adjusted p-values per engine; Model Confidence Set "
                 "(Hansen, Lunde & Nason 2011) with a stationary block bootstrap (numpy only).",
                 "`pipeline/report.py`: DM tests also per regime (calm / COVID crash / bear 2022) and per maturity "
                 "bucket; new columns `p_holm`, table `mcs.csv`."],
         config=["`evaluation.mcs: {alpha: 0.10, block: 10, reps: 2000}`"],
         run=["`python -m pinnvol evaluate -c configs/default.yaml` (no re-pricing)."],
         done=["`dm_tests.csv` has `p_holm`; `mcs.csv` lists the surviving set per engine; report.md updated.",
               "Unit test: MCS keeps the true best model on simulated losses."],
         compute="evaluate-only, ≈ 5–10 min", depends="—", decision=None,
         report="Which comparisons stay significant after correction; the MCS per engine and regime."),
    dict(id="B2", title="Volatility-unit and vega-weighted errors", priority="Low", phase="Phase 1",
         goal="Compare fairly across maturities and with the literature.",
         why="Price RMSE is dominated by long-dated options (120–180 d errors ≈ 5× the 7–30 d errors).",
         change=["`evaluation/metrics.py`: vega-weighted RMSE; IV-RMSE moved to the headline table; RMSE by "
                 "maturity bucket in report.md."],
         config=[], run=["`evaluate` only."], done=["Headline table shows price RMSE, IV-RMSE and vega-weighted "
                                                    "RMSE side by side."],
         compute="evaluate-only", depends="—", decision=None, report="The new headline table."),
    dict(id="A5", title="Label the same-day contract IV as an oracle", priority="Medium", phase="Phase 1",
         goal="Stop the oracle being read as a competing method.",
         why="`iv_contract_same_day` prices an option with its own implied volatility (RMSE 0.19).",
         change=["`pipeline/report.py`: move oracle rows into a separate 'reference bounds' table; exclude them "
                 "from DM tests and rankings."],
         config=[], run=["`evaluate` only."], done=["No oracle rows in rankings or DM tables."],
         compute="evaluate-only", depends="—", decision=None, report="Updated report.md."),
    dict(id="C1", title="Finish documenting the monthly-dating issue", priority="Low", phase="Phase 1",
         goal="Nobody repeats the 2015 mistake when using the cleaned files directly.",
         why="The repo README now warns about it (commit 5d72bab), but `data-profile/` does not, and "
             "`data-clean/` must not be edited.",
         change=["Add the finding (weekday table, dates) to `data-profile/spx_data_profile.md`."],
         config=[], run=["—"], done=["The data profile has a 'Monthly expiry dating' section."],
         compute="minutes", depends="—", decision=None, report="Link to the new section."),
    dict(id="A1", title="Smile-adjusted forecast treatments", priority="High", phase="Phase 2",
         goal="Compare forecasting methods on the volatility level with an identical smile.",
         why="GARCH/LSTM errors concentrate in the wings (K/F 0.80–0.90: GARCH 37.3 vs IV surface 5.3). Today "
             "the comparison partly measures 'smile vs no smile', not forecasting skill.",
         change=["`vol/implied.py`: from the previous day's fitted smile, the shape factor "
                 "s(K,T) = IV_surf(K,T) / IV_ATM(T).",
                 "`pipeline/volatility.py`: new columns `sigma_<m>_smile = sigma_<m> × s(K,T)` for "
                 "m ∈ {garch, lstm, har, iv_atm}. (`iv_atm_smile` should reproduce `iv_surface` — a built-in "
                 "sanity check.)",
                 "Test: no same-day information enters `s(K,T)`."],
         config=["`treatments: [..., garch_smile, lstm_smile, har_smile, iv_atm_smile]`",
                 "`price.only_engines: [bs, pinn_p]` (Mode H optional)"],
         run=["`vol` → `price` (filtered, needs E3) → `evaluate`, output `runs/A1/`."],
         done=["`iv_atm_smile` RMSE within 0.5 of `iv_surface`.",
               "Table: raw vs smile-adjusted RMSE for GARCH, LSTM, HAR, IV ATM; moneyness chart redone."],
         compute="≈ 30–60 min (without Mode H); +≈ 2 h with Mode H at 2 seeds", depends="E3 (recommended)",
         decision=None, report="How much of the GARCH/LSTM vs IV gap remains once all share the smile."),
    dict(id="A2", title="Volatility-risk-premium adjustment", priority="High", phase="Phase 2",
         goal="Separate the level shift between physical and risk-neutral volatility from forecasting skill.",
         why="All forecast-based inputs underprice: mean bias −19.6 (GARCH), −21.1 (LSTM) index points; implied "
             "vol sits above realised vol on average.",
         change=["`vol/`: rolling variance ratio "
                 "k_t = (252-day mean of IV_ATM,30d²) ÷ (252-day mean of σ̂_m²), using data up to t−1 only.",
                 "New columns `sigma_<m>_vrp = sqrt(k_t) × sigma_<m>`, and the combination `<m>_vrp_smile` with A1."],
         config=["`volatility.vrp: {window: 252, horizon: 30}`",
                 "`treatments: [..., garch_vrp, lstm_vrp, garch_vrp_smile, lstm_vrp_smile]`"],
         run=["`vol` → filtered `price` → `evaluate`, output `runs/A2/`."],
         done=["Bias of GARCH/LSTM inputs reported before and after; no-look-ahead test passes."],
         compute="≈ 30–60 min", depends="E3 (recommended); A1 for the combined treatment",
         decision="Whether adjusted treatments become the headline results or a separate robustness table.",
         report="Bias and RMSE before/after; the remaining gap to IV = the forecasting contribution."),
    dict(id="A3", title="A stronger LSTM", priority="High", phase="Phase 3",
         goal="Test whether a well-specified deep-learning forecaster can match GARCH or IV.",
         why="Best in validation 2019 (QLIKE 0.231), worst in test (0.932); forecasts too low (17.0% vs 19.3% "
             "realised at 21 d); COVID-crash pricing RMSE 74.7 vs 37.7 for GARCH. The model is small, return-only "
             "and refit quarterly.",
         change=["A3a — monthly refit (config only: `volatility.lstm.refit: M`).",
                 "A3b — longer history: SPX closes from 2000 through `paths.extra_prices_csv` (already supported).",
                 "A3c — implied-vol features: VIX from FRED (`VIXCLS`, free, same downloader as the rates) and the "
                 "lagged ATM IV term structure.",
                 "A3d — target log realised variance at each option's own horizon.",
                 "A3e — validation window that includes a stress period (e.g. 2018 Q4)."],
         config=["`volatility.lstm: {refit: M, features: [returns, vix, iv_atm], target: log_rv}`"],
         run=["Ablation: A3a, then +A3b, +A3c, +A3d, each as `runs/A3x/`, compared on QLIKE first, pricing second."],
         done=["Test QLIKE (21 d) of the best variant ≤ GARCH's 0.657, or a documented reason why not.",
               "COVID-crash pricing RMSE reported for every variant."],
         compute="≈ 15–30 min per variant for `vol`; pricing ≈ 30 min each with E3",
         depends="E3 (recommended)",
         decision="Source for SPX closes before 2012 (A3b): download from a public source, or you provide a file.",
         report="An ablation table: which change helped and by how much."),
    dict(id="A4", title="Hybrid PINN: λ grid", priority="Medium", phase="Phase 4",
         goal="Show how much a PINN should trust market data versus physics.",
         why="Only λ = 0.1 was run: small gains (GARCH 33.4 → 32.4) and drift from BS of 0.9–2.7 points.",
         change=["Config only (`pinn.mode_h.lambdas` is already a list); a new figure: market RMSE and BS-gap "
                 "against λ for each volatility input."],
         config=["`pinn.mode_h: {lambdas: [0.01, 0.1, 1, 10], seeds: [0, 1]}`"],
         run=["`price` with `only_engines: [pinn_h]` (needs E3), output `runs/A4/`."],
         done=["λ curve per treatment; recommended λ with justification."],
         compute="≈ 2 h per λ at 2 seeds ≈ 8 h total (overnight)", depends="E1, E3",
         decision="The λ values (default 0.01, 0.1, 1, 10).",
         report="The λ trade-off figure and table."),
    dict(id="C2", title="Use 2012–2018 options and run the weekly robustness sample", priority="Medium",
         phase="Phase 5",
         goal="Check that conclusions hold beyond monthly SPX options.",
         why="After the fix, ≈ 0.97 M training-period options exist but no model uses them; only monthlies were "
             "tested.",
         change=["Use the 2012–2018 panel for VRP calibration (A2) and LSTM-IV history (A3c).",
                 "A robustness run with `sample.contract_family: all` (adds SPXW weeklies, PM-settled)."],
         config=["`configs/C2_spxw.yaml`: `sample.contract_family: all`, `sample.max_options_per_day` if needed"],
         run=["Full pipeline into `runs/C2/` (subsampled if the projection exceeds 8 h)."],
         done=["Robustness table: same ranking (or documented differences) on the wider sample."],
         compute="≈ 6 h (one full run)", depends="E1",
         decision="Subsample size if the SPXW run is too slow.",
         report="Side-by-side headline tables, monthly vs all contracts."),
    dict(id="D1", title="Tighten the physics-only PINN", priority="Low", phase="Phase 5",
         goal="Bring the PINN's error vs Black-Scholes below 0.1 index points.",
         why="Solver error is 0.31–0.37 points (≈ 50× Crank-Nicolson); with the oracle IV only 84% of PINN "
             "prices fall inside the bid-ask spread vs 99.99% for BS.",
         change=["Try, in order: `weighting: lra`; L-BFGS 1,500 → 3,000 steps; width 64 → 128; a convexity "
                 "(no-arbitrage) penalty on ∂²V/∂K² ≥ 0."],
         config=["`pinn.mode_p: {weighting: lra, lbfgs_steps: 3000}`, `pinn.hidden: 128`"],
         run=["`train-pinn` per variant into `runs/D1x/`; re-price BS + Mode P only (E3)."],
         done=["Validation MAE vs closed form and test solver RMSE per variant; best one < 0.1 points."],
         compute="≈ 10–20 min training per variant + ≈ 20 min pricing", depends="E3",
         decision=None, report="Variant table; whether the headline results change (expected: no)."),
    dict(id="W1", title="Final write-up", priority="—", phase="Phase 6",
         goal="Turn the results into thesis-ready tables and figures.",
         why="Every phase changes numbers; the report should be regenerated from the final runs.",
         change=["Extend `reports/build_progress_report.py` to compare baseline vs improved runs; export LaTeX "
                 "tables."],
         config=[], run=["`python reports/build_progress_report.py`"],
         done=["Updated PDF and LaTeX tables from the final runs."],
         compute="minutes", depends="the phases you chose", decision=None, report="The new PDF."),
]

BY_ID = {it["id"]: it for it in ITEMS}

# --------------------------------------------------------------------------------------------- markdown


def to_markdown() -> str:
    L = [f"# Improvement proposal — PINN volatility study", "",
         f"_{DATE} · baseline: {BASELINE}_", "",
         "Each improvement has an **ID**. Tell me the ID(s) to start work. Evidence for every item is in "
         "`reports/PINN_volatility_progress_report.pdf`.", "", "## How to ask", ""]
    L += [f"- {h}" for h in HOW_TO]
    L += ["", "## Ground rules for every item", "", "| Rule | Value |", "|---|---|"]
    L += [f"| {a} | {b} |" for a, b in RULES]
    L += ["", "## Overview", "", "| Phase | Aim | Items | Compute |", "|---|---|---|---|"]
    for ph, aim, ids, comp in PHASES:
        L.append(f"| {ph} | {aim} | {', '.join(ids)} | {comp} |")
    L += ["", "| ID | Title | Priority | Depends on | Decision needed |", "|---|---|---|---|---|"]
    for it in ITEMS:
        L.append(f"| **{it['id']}** | {it['title']} | {it['priority']} | {it['depends']} | "
                 f"{'yes' if it['decision'] else '—'} |")
    for ph, aim, ids, _ in PHASES:
        L += ["", f"## {ph} — {aim}"]
        for i in ids:
            it = BY_ID[i]
            L += ["", f"### {it['id']} · {it['title']}  ", f"Priority: {it['priority']} · Compute: {it['compute']} · "
                                                           f"Depends on: {it['depends']}", "",
                  f"**Goal.** {it['goal']}", "", f"**Why.** {it['why']}", "", "**Change.**"]
            L += [f"- {c}" for c in it["change"]]
            if it["config"]:
                L += ["", "**Config.**"] + [f"- {c}" for c in it["config"]]
            L += ["", "**Run.**"] + [f"- {c}" for c in it["run"]]
            L += ["", "**Done when.**"] + [f"- {c}" for c in it["done"]]
            if it["decision"]:
                L += ["", f"**Decision needed.** {it['decision']}"]
            L += ["", f"**I report back.** {it['report']}"]
    L += ["", "## Example requests", "",
          "- \"do Phase 1\" — E1, E3, E2, B1, B2, A5, C1 in that order, one report at the end.",
          "- \"do A1 and A2\" — both treatments in one `vol` + filtered `price` run (`runs/A1A2/`).",
          "- \"do A3a and A3c only\" — monthly refit plus VIX features, no longer history.",
          "- \"do A4 with λ = 0.05, 0.5, 5 and 3 seeds\" — overrides the defaults.",
          "- \"do D1 but only LRA\" — one variant.", ""]
    return "\n".join(L)


# --------------------------------------------------------------------------------------------- PDF
FONTS = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
for name, f in [("DejaVu", "DejaVuSans.ttf"), ("DejaVu-Bold", "DejaVuSans-Bold.ttf"),
                ("DejaVu-Oblique", "DejaVuSans-Oblique.ttf"), ("DejaVuMono", "DejaVuSansMono.ttf")]:
    pdfmetrics.registerFont(TTFont(name, str(FONTS / f)))
addMapping("DejaVu", 0, 0, "DejaVu")
addMapping("DejaVu", 1, 0, "DejaVu-Bold")
addMapping("DejaVu", 0, 1, "DejaVu-Oblique")
addMapping("DejaVu", 1, 1, "DejaVu-Bold")

INK, INK2, BLUE, SOFT = "#0b0b0b", "#52514e", "#2a78d6", "#f5f4f0"
ss = getSampleStyleSheet()
body = ParagraphStyle("b", parent=ss["Normal"], fontName="DejaVu", fontSize=9.4, leading=13.2, spaceAfter=4,
                      textColor=colors.HexColor(INK))
small = ParagraphStyle("s", parent=body, fontSize=8.2, leading=11.6, spaceAfter=0)
smallb = ParagraphStyle("sb", parent=small, fontName="DejaVu-Bold")
h1 = ParagraphStyle("h1", parent=body, fontName="DejaVu-Bold", fontSize=16, leading=20, spaceAfter=8,
                    textColor=colors.HexColor(BLUE))
h2 = ParagraphStyle("h2", parent=body, fontName="DejaVu-Bold", fontSize=12.5, leading=16, spaceBefore=8, spaceAfter=4)
h3 = ParagraphStyle("h3", parent=body, fontName="DejaVu-Bold", fontSize=11, leading=14, spaceBefore=4, spaceAfter=2)
meta = ParagraphStyle("m", parent=small, textColor=colors.HexColor(INK2), spaceAfter=4)
title = ParagraphStyle("t", parent=h1, fontSize=21, leading=26, textColor=colors.HexColor(INK))
bullet = ParagraphStyle("bl", parent=body, leftIndent=12, bulletIndent=2, spaceAfter=2)


def rl(s: str) -> str:
    """Markdown-ish inline markup → ReportLab markup."""
    s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"`(.+?)`", r"<font name='DejaVuMono' size='8.3'>\1</font>", s)
    return s


def grid(rows: list[list[str]], widths: list[float], header: bool = True) -> Table:
    data = [[Paragraph(rl(c), smallb if header and i == 0 else small) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=[w * cm for w in widths], repeatRows=1 if header else 0)
    st = [("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 3),
          ("BOTTOMPADDING", (0, 0), (-1, -1), 3), ("LEFTPADDING", (0, 0), (-1, -1), 4)]
    if header:
        st.append(("LINEBELOW", (0, 0), (-1, 0), 0.8, colors.HexColor(INK2)))
    for i in range(1 if header else 0, len(rows)):
        if i % 2 == 0:
            st.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor(SOFT)))
    t.setStyle(TableStyle(st))
    return t


def item_block(it: dict) -> list:
    head = [Paragraph(f"<font color='{BLUE}'>{it['id']}</font> · {rl(it['title'])}", h3),
            Paragraph(rl(f"Priority: {it['priority']} · Compute: {it['compute']} · Depends on: {it['depends']}"),
                      meta)]
    rows = [["Goal", it["goal"]], ["Why (evidence)", it["why"]],
            ["Change", "<br/>".join("• " + c for c in it["change"])]]
    if it["config"]:
        rows.append(["Config", "<br/>".join("• " + c for c in it["config"])])
    rows += [["Run", "<br/>".join("• " + c for c in it["run"])],
             ["Done when", "<br/>".join("• " + c for c in it["done"])]]
    if it["decision"]:
        rows.append(["Decision needed", it["decision"]])
    rows.append(["I report back", it["report"]])
    data = [[Paragraph(f"<b>{a}</b>", small), Paragraph(rl(b).replace("&lt;br/&gt;", "<br/>"), small)]
            for a, b in rows]
    t = Table(data, colWidths=[3.0 * cm, 13.6 * cm])
    st = [("VALIGN", (0, 0), (-1, -1), "TOP"), ("BACKGROUND", (0, 0), (0, -1), colors.HexColor(SOFT)),
          ("LINEBELOW", (0, 0), (-1, -2), 0.3, colors.HexColor("#e4e3df")),
          ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#d6d5d0")),
          ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if it["decision"]:
        i = [r[0] for r in rows].index("Decision needed")
        st.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#fff4dc")))
    t.setStyle(TableStyle(st))
    return [KeepTogether(head + [t]), Spacer(1, 8)]


def build_pdf(path: Path) -> None:
    S: list = [Spacer(1, 0.6 * cm), Paragraph("Improvement proposal", meta),
               Paragraph("PINN volatility study — what to improve next, item by item", title),
               Paragraph(rl(f"{DATE} · Baseline: {BASELINE}"), meta), Spacer(1, 6),
               Paragraph("How to ask", h2)]
    S += [Paragraph(rl(h), bullet, bulletText="•") for h in HOW_TO]
    S += [Paragraph("Ground rules for every item", h2), grid([["Rule", "Value"], *[[a, b] for a, b in RULES]],
                                                             [3.2, 13.4])]
    S += [Paragraph("Overview", h2),
          grid([["Phase", "Aim", "Items", "Compute"], *[[p, a, ", ".join(i), c] for p, a, i, c in PHASES]],
               [2.0, 6.6, 4.4, 3.6]), Spacer(1, 8),
          grid([["ID", "Title", "Priority", "Depends on", "Decision"],
                *[[f"**{it['id']}**", it["title"], it["priority"], it["depends"], "yes" if it["decision"] else "—"]
                  for it in ITEMS]], [1.1, 6.6, 2.6, 4.6, 1.7])]
    for ph, aim, ids, comp in PHASES:
        S += [PageBreak() if ph in ("Phase 1", "Phase 3", "Phase 5") else Spacer(1, 4),
              Paragraph(f"{ph} — {aim}", h1), Paragraph(rl(f"Compute: {comp}"), meta)]
        for i in ids:
            S += item_block(BY_ID[i])
    S += [Paragraph("Example requests", h2)]
    S += [Paragraph(rl(x), bullet, bulletText="•") for x in [
        "\"do Phase 1\" — E1, E3, E2, B1, B2, A5, C1 in that order, one report at the end.",
        "\"do A1 and A2\" — both treatments in one `vol` + filtered `price` run (`runs/A1A2/`).",
        "\"do A3a and A3c only\" — monthly refit plus VIX features, no longer history.",
        "\"do A4 with λ = 0.05, 0.5, 5 and 3 seeds\" — overrides the defaults.",
        "\"do D1 but only LRA\" — one variant."]]

    def on_page(c, d):  # noqa: ANN001
        c.saveState()
        c.setFont("DejaVu", 7.5)
        c.setFillColor(colors.HexColor(INK2))
        c.drawString(2 * cm, 1.2 * cm, "PINN volatility study — improvement proposal")
        c.drawRightString(A4[0] - 2 * cm, 1.2 * cm, str(d.page))
        c.restoreState()

    doc = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=2.2 * cm, rightMargin=2.2 * cm, topMargin=1.8 * cm,
                            bottomMargin=1.9 * cm, title="Improvement proposal — PINN volatility study",
                            author="Asia")
    doc.build(S, onFirstPage=on_page, onLaterPages=on_page)


if __name__ == "__main__":
    (OUT / "IMPROVEMENT_PROPOSAL.md").write_text(to_markdown(), encoding="utf-8")
    build_pdf(OUT / "PINN_improvement_proposal.pdf")
    print(OUT / "IMPROVEMENT_PROPOSAL.md")
    print(OUT / "PINN_improvement_proposal.pdf")
