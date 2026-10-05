# Figure captions (edit freely)

All figures: SPX monthly (AM-settled) European options, test period 2020-01-02 to 2023-12-29,
n = 1,993,333 options on 992 trading days. RMSE is against the market mid price in index points.
95% confidence intervals: moving-block bootstrap over trading days (block 10, 2000 replications).
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
`table_main_results.tex` (LaTeX, needs `\usepackage{booktabs}`).
