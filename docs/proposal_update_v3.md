# Research Proposal Update (v3)

**PINN option pricing under GARCH, LSTM and implied volatility (SPX)**
Amends the revised proposal (v2) after the first full run on the real data. Last updated 4 October 2026.

---

## How to use this file

This file amends the research proposal: what changed in the method, where the hypotheses stand, and the new question the improvements answer. The **list of improvements you can request** (codes A1 to W1, phases 1 to 6, acceptance criteria and defaults) lives in one place only: [`IMPROVEMENT_PROPOSAL.md`](../IMPROVEMENT_PROPOSAL.md). Ask for work by those codes, for example "do Phase 1" or "do A1 and A2".

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
| H5 | Forecast ranking differs from pricing ranking because of the risk premium | Indicated: forecast-based inputs underprice by about 20 points | A2, forecast-accuracy table |
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

## 4. Improvements

See [`IMPROVEMENT_PROPOSAL.md`](../IMPROVEMENT_PROPOSAL.md). Items A1 and A2 test SQ8 and H7, A3 tests H4, A4 tests H6, and B1 settles H1-H3.

---

## Change log

| Date | Change |
|---|---|
| 2026-10-04 | v3 created after the first full run |
| 2026-10-04 | Improvement menu moved to IMPROVEMENT_PROPOSAL.md (repo root) so there is one list |
