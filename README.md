# PINN option pricing under GARCH, LSTM and implied volatility (SPX)

Code for the senior research project *Effect of the Volatility Forecasting Method (GARCH, LSTM,
and Implied Volatility) on the Pricing Accuracy of a Physics-Informed Neural Network (PINN) for
European Options* (SIIT, Thammasat University).

The pipeline takes the cleaned OptionsDX SPX end-of-day files (2012-2023), builds
walk-forward volatility forecasts, feeds each one to the same PINN, and compares prices with
analytical Black-Scholes, Crank-Nicolson, a data-only neural network and the market mid.
Section numbers below refer to the revised research proposal.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # numpy, pandas, scipy, torch, pyyaml, pytest
pytest                            # about 1 minute on a laptop CPU
```

## Data

The option data is licensed and never goes in this repository (`.gitignore` excludes it).
Point `paths.data_dir` at the folder written by `scripts/clean_spx.py` (standard library only; run it
from the folder that holds the raw OptionsDX `dataset/`: `python3 code/scripts/clean_spx.py`):

```
data-clean/
  options/spx_clean_YYYYMM.csv.gz   one row per option: quote_date, expire_date, option_type,
                                    S, K, dte, tau, bid, ask, mid, rel_spread, iv_vendor,
                                    is_third_friday, ...
  spx_underlying_daily.csv          date, S, log_return
```

Risk-free rates come from FRED (DGS1MO, DGS3MO, DGS6MO, DGS1); `pinnvol fetch-rates` downloads
them (no API key). Dividend yields are backed out from put-call parity, so no dividend data is needed.

OptionsDX dates the AM-settled monthly by its last trading day (the Thursday before the third
Friday) until August 2016, so the `is_third_friday` column in the cleaned files misses those
monthlies. `load_options` re-derives the flag (`flag_am_monthlies`); do not filter on the raw column.

## Run the full study

From the repository folder, with `data-clean/` one level up (edit `configs/default.yaml` otherwise):

```bash
python -m pinnvol fetch-rates -c configs/default.yaml   # once; writes data-external/fred/
python -m pinnvol prepare     -c configs/default.yaml   # option panel with r, q, F, market IV
python -m pinnvol vol         -c configs/default.yaml   # GARCH, HAR, LSTM, lagged IV
python -m pinnvol train-pinn  -c configs/default.yaml   # Mode P PINN + validation vs Black-Scholes
python -m pinnvol price       -c configs/default.yaml   # all engines x all volatility inputs
python -m pinnvol evaluate    -c configs/default.yaml   # tables, DM tests, report.md
# or all five at once:
python -m pinnvol all -c configs/default.yaml
```

Each stage saves its output in `runs/main/` and the next stage reads it back, so stages can be
re-run on their own. On a CPU, the slow steps are the LSTM (5 seeds x quarterly refits), the
Mode P PINN (about 30-60 minutes; a CUDA GPU is used automatically when present), and the Mode H
fine-tuning (treatments x seeds x 16 test quarters). `sample.max_options_per_day` subsamples
the test set if pricing takes too long; `pinn.mode_h.enabled: false` skips Mode H.

### Try it without the real data

```bash
python -m pinnvol synth -c configs/synthetic.yaml   # synthetic files in the cleaned schema
python -m pinnvol all   -c configs/synthetic.yaml   # small sizes, about 15 minutes on CPU
```

The synthetic market simulates GJR-GARCH returns and prices options off a smile around a
risk-neutral ATM level that carries a variance risk premium, so the expected ranking
(lagged IV beats GARCH/LSTM beats historical volatility) is known in advance.

## What each stage does

| Stage | Module | Proposal |
|---|---|---|
| `prepare` | `pipeline/prepare.py` | Sections 5.2, 6.3, 6.4, 9 |
| `vol` | `vol/garch.py`, `vol/lstm.py`, `vol/har.py`, `vol/implied.py`, `vol/realized.py` | Sections 10, 13.1 |
| `train-pinn` | `pinn/model.py`, `pinn/loss.py`, `pinn/sampling.py`, `pinn/trainer.py` | Section 11 |
| `price` | `pipeline/pricing.py`, `pricing/` | Sections 12, 15 |
| `evaluate` | `evaluation/`, `pipeline/report.py` | Sections 13.2-13.4, 14 |

**Sample (`prepare`).** Third-Friday AM-settled monthlies by default (`sample.contract_family: all`
adds weeklies for robustness). Time to maturity runs to the 09:30 settlement,
$\tau = (\text{DTE} - 6.5/24)/365$. $r(t,\tau)$ is interpolated from the FRED curve. For each
date and expiry, $F_{t,T} = K^* + e^{r\tau}(C - P)$ at the strikes where call and put are closest,
and $q = r - \ln(F/S)/\tau$. Filters: $7 \le \text{DTE} \le 180$, $0.8 \le K/F \le 1.2$,
relative spread $\le 20\%$, mid inside the static no-arbitrage bounds. Market IV is
recomputed from the mid with the same $r$ and $q$ (vendor Greeks and IV are not used).

**Volatility inputs (`vol`).** All forecasts made at the close of day $t$ use returns through $t$
only; models are refitted on an expanding window (GARCH and HAR monthly, LSTM quarterly).
Options with DTE calendar days get horizon $H = \text{round}(252\,\text{DTE}/365)$ trading days.

| Column | Input |
|---|---|
| `sigma_garch` | Treatment A: GARCH(1,1) QML, $\hat\sigma_{t,H} = \sqrt{\tfrac{252}{H}\sum_{h=1}^{H}\sigma^2_{t+h\mid t}}$ in closed form |
| `sigma_lstm` | Treatment B: LSTM on $(r, \lvert r\rvert, r^2, RV^{21})$, direct outputs for $H \in \{5,10,21,42,63,126\}$, QLIKE loss, 5 seeds averaged, total-variance interpolation |
| `sigma_iv_atm` | Treatment C1 (primary): previous day's ATM IV term structure at $\tau_i$ |
| `sigma_iv_surface` | Treatment C2 (reference): previous day's quadratic smile at $(\ln K/F, \tau_i)$ |
| `sigma_hv21`, `sigma_har`, `sigma_gjr` | Baselines: 21-day historical vol, HAR-RV, GJR-GARCH (add `gjr` to `treatments`) |
| `sigma_iv_atm_same_day`, `sigma_iv_contract_same_day` | Reference bounds only, not treatments (Section 16.2) |

Forecast accuracy against forward realised variance (MSE on variance, QLIKE, Mincer-Zarnowitz)
is written to `vol_forecast_metrics.csv`.

**PINN (`train-pinn`).** In log-moneyness $x = \ln(S/K)$ and $V = K u$, the network solves

$$
\frac{\partial u}{\partial \tau} = \tfrac12\sigma^2 u_{xx} + \left(r - q - \tfrac12\sigma^2\right)u_x - r u,
\qquad u(x, 0) = \max(e^x - 1, 0),
$$

for calls over $(x, \tau, r, q, \sigma)$; puts come from put-call parity, so it holds exactly.
**Mode P** ($w_D = 0$) is trained once on Sobol collocation points and shared by every
treatment. **Mode H** fine-tunes it per treatment, per test quarter, per seed on the previous
12 months of quotes with $\lambda = w_D/w_f$ (`pinn.mode_h.lambdas`, a list, so the Section 11.5
sweep is one config line).

**Engines (`price`).** Every input is priced on the same contracts by `bs`, `pinn_p`,
`pinn_h<lambda>` (seed average; per-seed columns kept), `ffnn` and `cn` (Crank-Nicolson on a
2,000-option subsample).

**Outputs (`evaluate`).**

| File | Contents |
|---|---|
| `metrics_market.csv` | RMSE, MAE, MAPE (mid $\ge$ 0.50), $R^2$, bias, in-spread rate, IVRMSE vs. the market mid |
| `metrics_vs_bs.csv` | Solver error of every engine vs. analytical Black-Scholes with the same inputs |
| `metrics_stratified.csv` | The same by maturity, $K/F$ bucket, call/put and regime (COVID crash, 2022 bear market, calm) |
| `metrics_mode_h_seeds.csv` | Mode H metrics per seed |
| `dm_tests.csv` | Diebold-Mariano tests on daily-averaged loss differentials, Newey-West variance, HLN correction |
| `vol_forecast_metrics.csv`, `treatment_inputs.csv`, `pinn_mode_p.json`, `prepare_log.csv` | Diagnostics |
| `report.md` | Headline tables |

## Design choices that go beyond the proposal

These are decisions the team should confirm or change; each is a config switch.

1. **Network inputs (`pinn.features`).** With the five raw inputs of Section 11.1 (`raw`), the
   PINN plateaued at about 0.4% of the strike (around 16 index points at $K = 4000$) against the
   closed form. The default `canonical` feeds the network the forward log-moneyness
   $y = x + (r-q)\tau$ and the total standard deviation $v = \sigma\sqrt\tau$, and writes
   $u = e^{-r\tau}(1 + e^y)\,N_\theta$. It is a change of variables of the same PDE (the residual is
   still taken in $x$ and $\tau$ by automatic differentiation, and no closed-form price is used
   in training), but it reduces a five-dimensional problem to two.
2. **Residual weighting (`pinn.mode_p.residual_v0`).** Near expiry $u_{xx} \sim 1/(\sigma\sqrt\tau)$,
   so unweighted residuals there are unbounded and wreck training. Each residual is multiplied by
   $\min(1, \sigma\sqrt\tau/0.05)$, and all errors by $1/(e^{-r\tau}(1 + e^y))$ so deep
   in-the-money points do not dominate.
3. **Far-field domain.** The Dirichlet conditions of Section 11.4 are only accurate several
   standard deviations from the money; at $\sigma\sqrt\tau = 0.6$ the proposal's
   $x \in [\ln 0.5, \ln 1.5]$ is too narrow, so the default is $x \in [-2.5, 2.5]$.
4. **Terminal weight 10** (`pinn.mode_p.weights.terminal`): with equal weights the payoff fit
   limited accuracy. **Learning-rate annealing** is implemented (`weighting: lra`) but off by default: with the
   unbounded near-expiry residuals it pushed the boundary weights above $10^5$ and made results worse.
5. **Inputs outside the PINN domain** ($\tau$, $r$, $q$, $\sigma$) are clipped for every engine, so
   solver error compares identical inputs; `treatment_inputs.csv` reports the clipped share.
6. **Crank-Nicolson** uses two fully implicit start-up steps (Rannacher) to damp the payoff kink.

## Known limitations

- **PINN solver error.** With the default settings (10,000 Adam + 1,500 L-BFGS steps, terminal
  weight 10), Mode P reached a mean absolute error of 7.5e-5 of the strike against the closed form
  on held-out points with $0.8 \le K/S \le 1.2$, 7-180 DTE and $\sigma \in [0.05, 0.9]$: about
  **0.3 index points at $K = 4000$** (max 1.6 points), after about 25 minutes on 2 CPU threads.
  With terminal weight 1 the same run gave 4.5 points. `pinn_mode_p.json` records the validation
  of every run; check it before reading `pinn_p` results. Short runs (as in
  `configs/synthetic.yaml`) are far less accurate.
- The realised-variance proxy is squared close-to-close returns; a range-based estimator needs
  daily OHLC, which the cleaned files do not carry.
- Third-Friday expiries in the OptionsDX files may mix AM-settled SPX with PM-settled SPXW
  quotes; there is no root column to separate them.
- Mode H and FFNN settings in `configs/synthetic.yaml` are tiny for speed; do not read results from them.

## Layout

```
src/pinnvol/
  config.py           defaults; YAML configs are merged over them
  cli.py              python -m pinnvol <stage>
  data/               loader, FRED rates, implied forwards, synthetic data
  pricing/            Black-Scholes, implied vol, Crank-Nicolson
  vol/                GARCH/GJR, HAR, LSTM, realised vol, lagged IV surfaces, horizon rules
  pinn/               model, losses, collocation sampling, Mode P / Mode H / FFNN training
  evaluation/         metrics, stratification, Diebold-Mariano
  pipeline/           the five stages
configs/              default.yaml (real data), synthetic.yaml (smoke test)
scripts/clean_spx.py  OptionsDX raw files -> data-clean/ (filters, stale-day removal, cleaning log)
reports/             progress report: build_progress_report.py (reads runs/main) -> PDF + figures
tests/                closed-form checks, parity, no-look-ahead checks, PINN residual
```
