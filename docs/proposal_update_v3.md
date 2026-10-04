# Research Proposal Update (v3)

**PINN option pricing under GARCH, LSTM and implied volatility (SPX)**
Amends the revised proposal (v2) after the first full run on the real data. Last updated 4 October 2026.

---

## How to use this file

Every proposed improvement has a short code (A1, A2, ... E2). To ask for work, reply in the project thread with codes:

| You write | What happens |
|---|---|
| `do A1 and A2` | Build those items, rerun the stages they need, report the new results |
| `do phase 2` | Run one bundle from [Section 6](#6-phases) |
| `do all High` | Every item marked High priority |
| `plan A3` | Explain the change in detail first, run nothing |
| `do A4 with λ 0.01, 1, 10` | Override a default from [Section 7](#7-open-choices-and-defaults) |

You can also edit this file: change a default in Section 7, cross out an item, or add a note under an item, then send the file (or say "read the proposal update in the repo"). The **Status** column in Section 4 is updated as items are done.

---

## 1. What changed since proposal v2

The v2 proposal described the planned study. Building and running it changed these details; they are already in the code and belong in the methodology chapter.

| Proposal v2 (section) | As implemented | Why |
|---|---|---|
| Data: OptionMetrics or Cboe DataShop (§6) | OptionsDX SPX end-of-day files, 2012–2023 | Affordable; DataShop needs a licence of about $1k/month |
| Sample 2012–2025, test 2020–2025 (§7) | 2012–2023; training 2012–2018, validation 2019, test 2020–2023 | OptionsDX coverage ends in 2023 |
| Standard monthlies, third-Friday expiry (§5.2) | Date rule: third Friday, or the Thursday before it when the vendor dates by last trading day (until Aug 2016) or Good Friday moves expiry | Without it, every monthly from 2012 to mid-2016 was lost |
| Risk-free rate (§6.3) | FRED Treasury yields (1, 3, 6, 12 months), interpolated to each maturity, continuous compounding | Free and standard |
| Dividend yield (§6.4) | Implied from put-call parity per date and expiry | No dividend data in the files |
| Implied volatility | Recomputed from the mid with the same $r$ and $q$; vendor IV and Greeks not used | Vendor values inconsistent |
| PINN inputs (§11.1) | Network sees forward moneyness $y = x + (r-q)\tau$ and $v = \sigma\sqrt\tau$; PDE still enforced in $(x, \tau)$ by automatic differentiation | Five raw inputs stalled at about 16 points of error |
| PINN loss (§11.5, §16.6) | Residual down-weighted near expiry by $\min(1, v/0.05)$; errors scaled to option size; terminal weight 10; domain $x \in [-2.5, 2.5]$ | Error down to about 0.3–0.4 points |
| Mode H λ grid (§11.5) | Only $\lambda = 0.1$ so far; 5 seeds, 12-month window, quarterly refits | Time; the grid is item A4 |
| Benchmarks (§12) | Black-Scholes, Crank-Nicolson (2,000-option subsample), data-only FFNN, HAR, 21-day historical vol | As planned, plus HAR |

### 1.1 First full run (4 October 2026)

Test 2020–2023: 1,993,333 options on 992 days. RMSE against the market mid, index points.

| Input | Black-Scholes | PINN (physics) | PINN (hybrid, λ 0.1) |
|---|---|---|---|
| IV surface, previous day (C2) | 7.28 | 7.29 | 7.33 |
| IV ATM, previous day (C1) | 23.42 | 23.37 | 22.97 |
| GARCH (A) | 33.50 | 33.44 | 32.35 |
| LSTM (B) | 36.27 | 36.21 | 35.07 |

- The PINN is within 0.31–0.37 points of exact Black-Scholes for every input, so the volatility input drives the result.
- GARCH and LSTM underprice on average (bias −19.6 and −21.1 points), most of all for out-of-the-money puts (K/F 0.80–0.90: GARCH 37.3 against 5.3 for the IV surface).
- COVID crash: LSTM 74.7 against GARCH 37.7. Calm markets: 22.2 against 22.9.
- The LSTM was the best forecaster in the 2019 validation year (QLIKE 0.231) and the worst in the test (0.932).

Full numbers: `reports/PINN_volatility_progress_report.pdf` and `runs/main/` on the team laptop.

---

## 2. Where the hypotheses stand

Numbering as in proposal v2 (with its revised H3 and new H5, H6). Evidence is from the first run only.

| | Hypothesis | Evidence so far | Settled by |
|---|---|---|---|
| H1 | The volatility input significantly affects PINN pricing error | Supported: gaps of 7–43 points; DM p < 1e-9 for IV against GARCH and LSTM | B1 |
| H2 | Relative performance differs across moneyness and maturity | Supported for moneyness (largest gaps in the put wing); maturity not yet summarised | B1, B2 |
| H3 (rev.) | Lagged ATM IV beats GARCH and LSTM, most at short maturities | First part supported (23.4 vs 33.5 and 36.3); maturity part untested | B1 |
| H4 | LSTM may help in changing regimes; GARCH stays competitive | Opposite in the COVID crash; close in calm markets | A3 |
| H5 | Forecast ranking differs from pricing ranking because of the risk premium | Indicated: forecast-based inputs underprice by about 20 points | A2, B3 |
| H6 | Hybrid PINN: gap between inputs shrinks as λ grows | Untested (one λ only) | A4 |

---

## 3. Proposed amendments to the research design

"IV beats GARCH and LSTM" currently mixes three effects: forecasting skill, the smile (only the IV surface has one) and the volatility risk premium (only implied volatility contains it). v3 separates them.

**New sub-question SQ8.** How much of the pricing advantage of implied volatility over GARCH and LSTM remains after the time-series forecasts are given the same smile shape and a risk-premium adjustment estimated only from past data?

**New hypothesis H7.** After smile and risk-premium adjustment (A1 + A2), the RMSE gap between forecast-based inputs and lagged ATM IV falls by at least half, and the remaining difference is not significant after multiple-testing control.

**New treatments** (added to §10 and §15; every input is known at the close of day $t-1$ or earlier):

| Treatment | Definition |
|---|---|
| GARCH-S, LSTM-S, IV-ATM-S (A1) | $\sigma_i = \sigma_{forecast}(T)\cdot IV_{surface}(K,T)\,/\,IV_{ATM}(T)$, previous day's smile shape |
| GARCH-P, LSTM-P (A2) | $\sigma_i = k_t\,\sigma_{forecast}(T)$, $k_t$ = average ratio of ATM IV to the forecast over the previous 12 months |
| GARCH-SP, LSTM-SP | Both adjustments |
| LSTM-IV (A3, optional) | LSTM with lagged VIX or ATM IV as extra inputs, reported separately so the pure LSTM (B) stays as defined |

Treatments A, B, C1 and C2 stay unchanged: the first-run comparison remains primary and the new treatments explain it.

---

## 4. Improvement menu

Laptop times are estimates for the RTX 3050 Ti laptop, scaled from the first run.

| Code | Improvement | Priority | Reruns | Laptop time | Status |
|---|---|---|---|---|---|
| A1 | Separate the volatility level from the smile | High | vol, price, evaluate | ≈ 30 min (PINN-P); +2 h with hybrid | Not started |
| A2 | Correct for the volatility risk premium | High | vol, price, evaluate | ≈ 30 min (PINN-P) | Not started |
| A3 | Strengthen the LSTM | High | vol, price, evaluate | ≈ 1–2 h | Not started |
| A4 | Explore the hybrid weight λ | Medium | price, evaluate | ≈ 2 h per λ (2 seeds) | Not started |
| A5 | Label the same-day contract IV as an oracle | Medium | evaluate | Minutes | Not started |
| B1 | Multiple-testing control; DM tests per regime and bucket | Medium | evaluate | Minutes | Not started |
| B2 | Errors in volatility units and relative terms | Low | evaluate | Minutes | Not started |
| B3 | Volatility forecast accuracy table in the paper | Medium | none | None | Not started |
| C1 | Document the monthly-dating fix in the data | Medium | none | None | Partly done (README) |
| C2 | Use 2012–2018 options; weekly-options robustness sample | Medium | full run | ≈ 6 h | Not started |
| D1 | Tighten the physics-only PINN | Low | train-pinn, price, evaluate | ≈ 20 min + rerun | Not started |
| E1 | Timestamped logs and checkpointing in price | Low | none | None | Not started |
| E2 | Freeze the software environment | Low | none | None | Not started |

Codes match the progress report in `reports/` (B3 is new here).

---

## 5. Improvement details

### A. Research design

**A1. Separate the volatility level from the smile** · High · proposal §10, §15
- *What changes:* smile-adjusted treatments (Section 3). GARCH, LSTM and ATM IV keep their own level but borrow the previous day's smile shape.
- *Why:* the comparison is partly "has a smile vs has no smile". Most GARCH and LSTM error is in the out-of-the-money puts, where the smile is steepest.
- *Expected:* wing errors for GARCH and LSTM fall sharply; the remaining gap measures forecasting skill.

**A2. Correct for the volatility risk premium** · High · §10, §16.5
- *What changes:* multiply each GARCH and LSTM forecast by a rolling 12-month ratio of ATM IV to the forecast. Report raw and adjusted.
- *Why:* option prices include a premium for volatility risk, so even a perfect forecast of realised volatility underprices. The −20-point bias shows this.
- *Expected:* bias moves towards zero and RMSE falls; what remains is dynamic forecasting skill.

**A3. Strengthen the LSTM** · High · §7, §10.2
- *What changes:* refit monthly like GARCH; train on SPX closes from 2000 (`paths.extra_prices_csv`, already supported); choose settings on a validation window that includes a stressed period; optionally the LSTM-IV hybrid.
- *Why:* trained on only about 1,700–2,900 days, it learned a calm market and broke down in the COVID crash. A weak setup makes "LSTM loses" a statement about this configuration, not about deep learning.
- *Expected:* more reactive in crises; at least matches GARCH. With IV inputs it may approach ATM IV.

**A4. Explore the hybrid weight λ** · Medium · §11.5
- *What changes:* run the hybrid PINN at several λ with 2 seeds (the 5-seed spread was tiny); plot error against the market and against Black-Scholes versus λ.
- *Why:* only λ = 0.1 was run. The data-physics trade-off is the most PINN-specific result, and H6 needs it.
- *Expected:* market error falls and Black-Scholes error rises with λ; λ is then chosen on 2019.

**A5. Label the same-day contract IV as an oracle** · Medium · §16.2
- *What changes:* mark it as an upper bound in all tables; leave it out of rankings and DM tests.
- *Why:* it prices each option with its own implied volatility (RMSE 0.19), so it can be misread as a winning method.
- *Expected:* clearer tables, same conclusions.

### B. Statistics and evaluation

**B1. Multiple-testing control; tests per regime and bucket** · Medium · §14
- *What changes:* Holm-Bonferroni on the pairwise Diebold-Mariano tests (Model Confidence Set as a check); repeat within each regime, maturity and moneyness bucket.
- *Why:* about 70 tests at nominal p-values; GARCH vs LSTM (p = 0.03) may not survive.
- *Expected:* IV vs GARCH/LSTM survives any correction; GARCH vs LSTM may not.

**B2. Errors in volatility units and relative terms** · Low · §13.3
- *What changes:* lead with IV-RMSE and vega-weighted errors next to price RMSE.
- *Why:* price RMSE is dominated by long-dated options, MAPE by cheap ones; the literature often uses IV-RMSE.
- *Expected:* same ranking, fairer weighting across maturities.

**B3. Volatility forecast accuracy table** · Medium · §13.1
- *What changes:* report QLIKE, variance MSE and Mincer-Zarnowitz regressions per model and split (already computed in `vol_forecast_metrics.csv`).
- *Why:* answers sub-question 1 and, with A2, H5.
- *Expected:* shows the LSTM's validation-to-test reversal and implied volatility's upward bias side by side.

### C. Data

**C1. Document the monthly-dating fix** · Medium · §9
- *What changes:* note in the cleaned-data README and the data profile; rename the flag in any future rebuild.
- *Why:* `is_third_friday` in the cleaned files is still wrong for 2012–2016.
- *Expected:* no silent data loss in later analyses.

**C2. Use 2012–2018 options; weekly robustness sample** · Medium · §5.2, §7
- *What changes:* use training-period options to calibrate A2 and give the hybrid PINN longer windows; one rerun with all expiries including weeklies.
- *Why:* about 0.97 million 2012–2018 options are now available but unused; reviewers expect results beyond monthlies.
- *Expected:* better-calibrated inputs and a robustness table.

### D. PINN quality

**D1. Tighten the physics-only PINN** · Low · §11, §16.6
- *What changes:* adaptive loss weights, more L-BFGS steps, a wider network, a convexity penalty.
- *Why:* PINN error (0.31–0.37 points) is about 50 times Crank-Nicolson's; with the oracle IV only 84% of PINN prices are inside the bid-ask spread vs 99.99% for Black-Scholes.
- *Expected:* error below 0.1 points.

### E. Engineering

**E1. Timestamped logs and checkpointing** · Low
- *What changes:* timestamped progress per quarter; save each quarter and resume from the last finished one; smaller prediction files.
- *Why:* a buffered log cost 50 minutes; a crash late in the 5-hour pricing stage would lose everything.
- *Expected:* safe long runs; about 0.4 GB instead of 1.3 GB of predictions.

**E2. Freeze the software environment** · Low · §21
- *What changes:* commit a lock file; record commit and config in each run folder.
- *Why:* exact reproducibility for the thesis appendix.

---

## 6. Phases

| Phase | Items | Laptop time | Question it answers |
|---|---|---|---|
| 1 | A5, B1, B2, B3, C1, E1, E2 | Minutes | Which differences are real after multiple-testing control? |
| 2 | A1, A2 | ≈ 1 h | Is IV better because it forecasts better, or because it carries the smile and premium? (SQ8, H7) |
| 3 | A3 | ≈ 1–2 h | Can a well-specified deep-learning forecaster match GARCH or IV? |
| 4 | A4 | ≈ 8 h (overnight) | How much should a PINN trust market data against physics? (H6) |
| 5 | C2, D1 | ≈ 6–8 h | Do conclusions hold on other contracts and with a more accurate PINN? |
| 6 | Final tables, figures, regenerated report | – | – |

**Recommended start: phases 1 and 2.** They are cheap, and phase 2 decides how the main result is framed.

---

## 7. Open choices and defaults

Edit the right-hand column to change a default.

| Item | Choice | Default |
|---|---|---|
| A1 | Smile source | Previous day's fitted quadratic smile (same as C2) |
| A2 | Premium window | Trailing 12 months; ratio averaged over ATM options at the 21-day horizon |
| A3 | LSTM-IV hybrid | Separate treatment; pure LSTM kept as Treatment B |
| A3 | Longer history | SPX daily closes from 2000 |
| A4 | λ values | 0.01, 0.1, 1, 10 with seeds 0 and 1 |
| B1 | Correction method | Holm-Bonferroni within each engine; Model Confidence Set as a check |
| C2 | Robustness sample | All expiries (monthly and weekly), same filters |

---

## 8. Updated work plan

| Step | Work | Who |
|---|---|---|
| 1 | Phases 1 and 2: code in the repo, run on the laptop, results in the thread | Claude, laptop |
| 2 | Phase 3 (LSTM) and phase 4 (λ sweep, overnight) | Claude, laptop |
| 3 | Phase 5 robustness, if time allows | Claude, laptop |
| 4 | Freeze results; write Section 1 changes into the methodology chapter | Team |
| 5 | Results and discussion chapters, figures, presentation | Team |

---

## Change log

| Date | Change |
|---|---|
| 2026-10-04 | v3 created after the first full run |
