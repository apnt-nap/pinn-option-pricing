"""Experiment configuration.

Every run is driven by one YAML file (see ``configs/``). The YAML is merged over the
defaults below, so a config only needs to list what it changes. Relative paths in
``paths`` are resolved against the directory the command is run from.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULTS: dict[str, Any] = {
    "seed": 0,
    "device": "auto",  # "auto" | "cpu" | "cuda" | "mps"
    "paths": {
        # Folder produced by clean_spx.py: options/spx_clean_YYYYMM.csv.gz + spx_underlying_daily.csv
        "data_dir": "data-clean",
        # FRED CSVs (DGS1MO.csv, ...), written by `pinnvol fetch-rates`
        "rates_dir": "data-external/fred",
        # Optional longer SPX close history (columns: date, S) for GARCH/LSTM estimation
        "extra_prices_csv": None,
        "output_dir": "runs/default",
    },
    "period": {
        "start": "2012-01-01",
        "train_end": "2018-12-31",
        "valid_end": "2019-12-31",
        "test_end": "2023-12-31",
    },
    "sample": {
        "dte_min": 7,
        "dte_max": 180,
        "fwd_moneyness_min": 0.80,
        "fwd_moneyness_max": 1.20,
        "max_rel_spread": 0.20,
        # "monthly": third-Friday AM-settled expiries only (primary spec, Section 5.2);
        # "all": every expiry in the files (robustness).
        "contract_family": "monthly",
        # AM settlement: tau runs from the 16:00 quote to the 09:30 open on expiry day.
        "am_settlement_hours": 6.5,
        # Optional cap on test-set options per quote date (random, seeded) to speed up runs.
        "max_options_per_day": None,
    },
    "rates": {
        "series": {"DGS1MO": 1 / 12, "DGS3MO": 0.25, "DGS6MO": 0.5, "DGS1": 1.0},
        # Used only when no FRED files are found; None makes missing rates an error.
        "fallback_rate": None,
    },
    "forwards": {
        "n_strikes": 3,  # strikes with the smallest |C - P| used for the parity forward
        "default_q": 0.015,
        "q_clip": [-0.02, 0.06],
    },
    "volatility": {
        "forecast_start": "2019-01-01",  # first date that needs forecasts (validation start)
        "horizons": [5, 10, 21, 42, 63, 126],  # trading-day grid for LSTM/HAR/evaluation
        "hv_windows": [21, 63],
        "garch": {"kind": "garch", "refit": "M", "min_obs": 750},
        "har": {"refit": "M", "min_obs": 750},
        "lstm": {
            "seq_len": 60,
            "hidden": 32,
            "layers": 1,
            "dropout": 0.0,
            "epochs": 60,
            "batch_size": 128,
            "lr": 1.0e-3,
            "weight_decay": 1.0e-5,
            "patience": 10,
            "val_frac": 0.15,
            "seeds": [0, 1, 2, 3, 4],
            "refit": "Q",
            "min_obs": 750,
        },
        "iv": {"atm_band": 0.10, "smile_band": 0.25, "min_smile_points": 5},
    },
    # Volatility inputs supplied to the pricing engines. Treatments A, B, C1, C2,
    # the naive baselines, and the two same-day reference bounds.
    "treatments": ["garch", "lstm", "iv_atm", "iv_surface", "hv21", "har", "iv_atm_same_day", "iv_contract_same_day"],
    "pinn": {
        "domain": {
            # ln(S/K): wide enough that the far-field condition is accurate for the pricing
            # region |ln(K/F)| <= 0.22 even at high sigma * sqrt(tau)
            "x": [-2.5, 2.5],
            "tau": [0.0, 0.55],
            "sigma": [0.05, 0.90],
            "r": [0.0, 0.07],
            "q": [-0.02, 0.06],
        },
        "hidden": 64,
        "layers": 5,
        "activation": "tanh",
        # How (x, tau, r, q, sigma) enter the network (see pinn/model.py):
        #   "canonical": u = e^{-r tau}(1 + e^y) N(y, sigma sqrt(tau)), y = x + (r - q) tau
        #   "similarity": the five raw inputs plus sigma sqrt(tau) and y / (sigma sqrt(tau))
        #   "raw": the five raw inputs only (Section 11.1 as written)
        "features": "canonical",
        "mode_p": {
            "adam_steps": 10000,
            "adam_lr": 2.0e-3,
            "lbfgs_steps": 1500,
            "n_collocation": 8192,
            "n_terminal": 2048,
            "n_boundary": 1024,
            "kink_fraction": 0.2,
            "resample_every": 250,
            "residual_v0": 0.05,  # residual weight min(1, sigma sqrt(tau) / v0); null = unweighted
            "boundary": "dirichlet",  # "dirichlet" | "linear" (V_SS = 0)
            # terminal weight 10: with 1 the payoff fit was the bottleneck (about 4.5 index points
            # MAE at K = 4000); with 10 validation MAE fell to about 0.3 points
            "weights": {"pde": 1.0, "terminal": 10.0, "boundary": 1.0},
            "weighting": "fixed",  # "fixed" | "lra" (learning-rate annealing, Wang et al. 2021)
            "lra_every": 100,
            "lra_alpha": 0.9,
            "n_validation": 20000,
        },
        "mode_h": {
            "enabled": True,
            "lambdas": [0.1],  # lambda = w_D / w_f
            "seeds": [0, 1, 2, 3, 4],
            "window_months": 12,
            "block": "Q",
            "steps": 1500,
            "lr": 5.0e-4,
            "n_data": 20000,
            "n_collocation": 4096,
        },
    },
    "benchmarks": {
        "crank_nicolson": {"enabled": True, "n_sample": 2000, "nx": 400, "nt": 200},
        "ffnn": {"enabled": True, "steps": 3000, "lr": 1.0e-3, "n_data": 50000},
    },
    "evaluation": {
        "mape_min_price": 0.5,
        "maturity_buckets": [7, 30, 60, 120, 180],
        "moneyness_buckets": [0.80, 0.90, 0.97, 1.03, 1.10, 1.20],  # K/F
        "regimes": {
            "covid_crash": ["2020-02-19", "2020-04-30"],
            "bear_2022": ["2022-01-03", "2022-10-12"],
        },
        "calm_atm_iv": 0.15,  # "calm" regime: same-day 30-day ATM IV below this
        "dm_loss": "squared",  # "squared" | "absolute"
    },
}


def _merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict) and k != "series":
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


class AttrDict(dict):
    """Dict with attribute access, recursively (``cfg.pinn.mode_p.adam_steps``)."""

    def __getattr__(self, name: str) -> Any:
        try:
            v = self[name]
        except KeyError as e:
            raise AttributeError(name) from e
        return AttrDict(v) if isinstance(v, dict) and not isinstance(v, AttrDict) else v


@dataclass
class Config:
    raw: dict[str, Any] = field(default_factory=lambda: copy.deepcopy(DEFAULTS))

    def __getattr__(self, name: str) -> Any:
        if name == "raw":
            raise AttributeError(name)
        v = self.raw[name]
        return AttrDict(v) if isinstance(v, dict) else v

    @property
    def output_dir(self) -> Path:
        p = Path(self.raw["paths"]["output_dir"])
        p.mkdir(parents=True, exist_ok=True)
        return p

    def dump(self, path: Path) -> None:
        path.write_text(yaml.safe_dump(self.raw, sort_keys=False))


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> Config:
    """Load a YAML config merged over :data:`DEFAULTS`."""
    user: dict[str, Any] = {}
    if path is not None:
        user = yaml.safe_load(Path(path).read_text()) or {}
    raw = _merge(DEFAULTS, user)
    if overrides:
        raw = _merge(raw, overrides)
    return Config(raw)
