r"""Stages 3-4: train the physics-only PINN, then price the test sample.

For every volatility input $\sigma^{(m)}$ the same contracts are priced by

* ``bs``: analytical Black-Scholes-Merton with the same $(S, K, \tau, r, q, \sigma^{(m)})$;
* ``pinn_p``: the Mode P PINN (one network shared by all inputs);
* ``cn``: Crank-Nicolson on a fixed random subsample;
* ``pinn_h{lambda}``: Mode H hybrid PINNs, walk-forward: for each test block (quarter)
  the Mode P network is fine-tuned on quotes from the preceding ``window_months`` with
  that input, once per seed; the prediction is the seed average;
* ``ffnn``: a data-only network trained on the same windows (no PDE).

$(\tau, r, q, \sigma)$ are clipped to the PINN training domain for **all** engines so that
solver error $V_{PINN} - V_{BS}$ compares identical inputs; the share of clipped inputs
is reported.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch

from ..config import Config
from ..pinn.model import BlackScholesPINN
from ..pinn.trainer import finetune_mode_h, resolve_device, train_ffnn, train_mode_p
from ..pricing.black_scholes import bs_price
from ..pricing.crank_nicolson import cn_price


def train_pinn(cfg: Config) -> BlackScholesPINN:
    """Train Mode P and save ``pinn_mode_p.pt`` and ``pinn_mode_p.json`` (validation + history)."""
    device = resolve_device(cfg.device)
    model, hist, val = train_mode_p(cfg.raw["pinn"], seed=cfg.seed, device=device)
    torch.save(model.checkpoint(), cfg.output_dir / "pinn_mode_p.pt")
    (cfg.output_dir / "pinn_mode_p.json").write_text(json.dumps({"validation": val, "history": vars(hist)}, indent=1))
    return model


def load_pinn(cfg: Config) -> tuple[BlackScholesPINN, dict]:
    device = resolve_device(cfg.device)
    ckpt = torch.load(cfg.output_dir / "pinn_mode_p.pt", map_location=device, weights_only=False)
    info = json.loads((cfg.output_dir / "pinn_mode_p.json").read_text())
    return BlackScholesPINN.from_checkpoint(ckpt, device), info


def test_sample(cfg: Config, panel: pd.DataFrame) -> pd.DataFrame:
    """Test-period options with every treatment available (a common, paired sample)."""
    test = panel[(panel["quote_date"] > pd.Timestamp(cfg.period.valid_end)) &
                 (panel["quote_date"] <= pd.Timestamp(cfg.period.test_end))]
    cols = [f"sigma_{t}" for t in cfg.treatments]
    missing = [c for c in cols if c not in test]
    if missing:
        raise KeyError(f"Treatments without a volatility column: {missing}")
    test = test[np.isfinite(test[cols].to_numpy()).all(axis=1)]
    cap = cfg.sample.max_options_per_day
    if cap:
        test = test.sample(frac=1.0, random_state=cfg.seed).groupby("quote_date").head(int(cap))
        test = test.sort_values(["quote_date", "expire_date", "is_call", "K"])
    return test.reset_index(drop=True)


def _inputs(model: BlackScholesPINN, df: pd.DataFrame, sigma_col: str) -> dict[str, np.ndarray]:
    tau, r, q, s = model.clip_inputs(df["tau"].to_numpy(), df["r"].to_numpy(), df["q"].to_numpy(),
                                     df[sigma_col].to_numpy())
    return {"S": df["S"].to_numpy(), "K": df["K"].to_numpy(), "tau": tau, "r": r, "q": q, "sigma": s,
            "is_call": df["is_call"].to_numpy()}


def _data_dict(model: BlackScholesPINN, df: pd.DataFrame, sigma_col: str) -> dict[str, np.ndarray]:
    a = _inputs(model, df, sigma_col)
    return {"x": np.log(a["S"] / a["K"]), "tau": a["tau"], "r": a["r"], "q": a["q"], "sigma": a["sigma"],
            "is_call": a["is_call"], "u_mkt": df["mid"].to_numpy() / a["K"]}


def price_test(cfg: Config, panel: pd.DataFrame | None = None, model: BlackScholesPINN | None = None) -> pd.DataFrame:
    """Price the test sample with every engine and treatment; saves ``predictions.pkl``."""
    out = cfg.output_dir
    if panel is None:
        panel = pd.read_pickle(out / "panel_vol.pkl")
    if model is None:
        model, _ = load_pinn(cfg)
    device = resolve_device(cfg.device)
    test = test_sample(cfg, panel)
    print(f"[price] test sample: {len(test):,} options on {test['quote_date'].nunique()} days")
    clip_rows = []
    for t in cfg.treatments:
        a = _inputs(model, test, f"sigma_{t}")
        test[f"V_bs_{t}"] = bs_price(a["S"], a["K"], a["tau"], a["r"], a["q"], a["sigma"], a["is_call"])
        test[f"V_pinn_p_{t}"] = model.price(**a)
        raw = test[f"sigma_{t}"].to_numpy()
        clip_rows.append({"treatment": t, "sigma_clipped_share": float(np.mean(raw != a["sigma"])),
                          "sigma_mean": float(np.mean(raw)), "sigma_median": float(np.median(raw))})
    pd.DataFrame(clip_rows).to_csv(out / "treatment_inputs.csv", index=False)

    cn = cfg.benchmarks.crank_nicolson
    if cn.enabled:
        rng = np.random.default_rng(cfg.seed)
        idx = np.sort(rng.choice(len(test), size=min(cn.n_sample, len(test)), replace=False))
        print(f"[price] Crank-Nicolson on {len(idx)} options x {len(cfg.treatments)} inputs")
        sub = test.iloc[idx]
        for t in cfg.treatments:
            a = _inputs(model, sub, f"sigma_{t}")
            col = np.full(len(test), np.nan)
            col[idx] = cn_price(a["S"], a["K"], a["tau"], a["r"], a["q"], a["sigma"], a["is_call"], cn.nx, cn.nt)
            test[f"V_cn_{t}"] = col

    mh = cfg.pinn.mode_h
    ff = cfg.benchmarks.ffnn
    if mh.enabled or ff.enabled:
        _walk_forward_data_models(cfg, panel, test, model, device)

    test.to_pickle(out / "predictions.pkl")
    return test


def _walk_forward_data_models(cfg: Config, panel: pd.DataFrame, test: pd.DataFrame, model: BlackScholesPINN,
                              device: torch.device) -> None:
    """Mode H PINNs and the FFNN control, refit per test block on a trailing window."""
    mh, ff = cfg.pinn.mode_h, cfg.benchmarks.ffnn
    info = json.loads((cfg.output_dir / "pinn_mode_p.json").read_text())
    weights = info["validation"]["final_weights"]
    treatments = list(mh.get("treatments", ["garch", "lstm", "iv_atm", "iv_surface"]))
    treatments = [t for t in treatments if t in cfg.treatments]
    blocks = test["quote_date"].dt.to_period(mh.block)
    for t in treatments:
        for lam in mh.lambdas if mh.enabled else []:
            for s in mh.seeds:
                test[f"V_pinn_h{lam}_{t}_s{s}"] = np.nan
        if ff.enabled:
            test[f"V_ffnn_{t}"] = np.nan
    for b in blocks.unique():
        rows = np.flatnonzero((blocks == b).to_numpy())
        b0 = b.start_time
        w0 = b0 - pd.DateOffset(months=int(mh.window_months))
        win = panel[(panel["quote_date"] >= w0) & (panel["quote_date"] < b0)]
        blk = test.iloc[rows]
        for t in treatments:
            col = f"sigma_{t}"
            wt = win[np.isfinite(win[col].to_numpy())]
            if len(wt) < 100:
                print(f"[price] block {b}: only {len(wt)} training quotes for {t}, skipped")
                continue
            wt = wt.sample(min(len(wt), int(mh.n_data)), random_state=cfg.seed)
            data = _data_dict(model, wt, col)
            a = _inputs(model, blk, col)
            if mh.enabled:
                for lam in mh.lambdas:
                    for s in mh.seeds:
                        m = finetune_mode_h(model, data, float(lam), weights, cfg.raw["pinn"], int(s), device)
                        test.iloc[rows, test.columns.get_loc(f"V_pinn_h{lam}_{t}_s{s}")] = m.price(**a)
            if ff.enabled:
                wf = win[np.isfinite(win[col].to_numpy())]
                wf = wf.sample(min(len(wf), int(ff.n_data)), random_state=cfg.seed)
                f = train_ffnn(_data_dict(model, wf, col), model.domain, cfg.pinn.hidden, cfg.pinn.layers,
                               int(ff.steps), float(ff.lr), cfg.seed, device)
                test.iloc[rows, test.columns.get_loc(f"V_ffnn_{t}")] = f.price(**a)
        print(f"[price] block {b}: {len(rows)} test options, window {w0.date()}..{b0.date()} ({len(win)} quotes)")
    for t in treatments:
        for lam in mh.lambdas if mh.enabled else []:
            cols = [f"V_pinn_h{lam}_{t}_s{s}" for s in mh.seeds]
            test[f"V_pinn_h{lam}_{t}"] = test[cols].mean(axis=1)
