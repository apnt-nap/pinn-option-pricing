"""Command-line entry point: ``python -m pinnvol <command> -c <config.yaml>``.

Commands (run in this order, or ``all``):

* ``synth``        write a synthetic dataset in the cleaned-data schema (for testing)
* ``fetch-rates``  download FRED Treasury yields into ``paths.rates_dir``
* ``prepare``      build the filtered option panel with r, q, F and market IV
* ``vol``          walk-forward GARCH / HAR / LSTM forecasts and lagged IV; attach to options
* ``train-pinn``   train the physics-only (Mode P) PINN and validate it against Black-Scholes
* ``price``        price the test sample with every engine and volatility input
* ``evaluate``     write the metric tables, Diebold-Mariano tests and report.md
* ``all``          prepare, vol, train-pinn, price, evaluate
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from .config import load_config


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="pinnvol", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["synth", "fetch-rates", "prepare", "vol", "train-pinn", "price", "evaluate", "all"])
    ap.add_argument("-c", "--config", default=None, help="YAML config (merged over the defaults)")
    ap.add_argument("--threads", type=int, default=None, help="torch CPU threads")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    if a.threads:
        torch.set_num_threads(a.threads)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    cfg.dump(cfg.output_dir / "config_used.yaml")
    t0 = time.time()

    if a.command == "synth":
        from .data.synthetic import make_synthetic_dataset
        make_synthetic_dataset(cfg.paths.data_dir, cfg.paths.rates_dir, **cfg.raw.get("synthetic", {}))
    elif a.command == "fetch-rates":
        from .data.rates import download_fred
        download_fred(list(cfg.rates.series), cfg.paths.rates_dir)
    else:
        from .pipeline import build_volatility, evaluate, prepare_panel, price_test, train_pinn
        steps = ["prepare", "vol", "train-pinn", "price", "evaluate"] if a.command == "all" else [a.command]
        for s in steps:
            print(f"==> {s}")
            {"prepare": prepare_panel, "vol": build_volatility, "train-pinn": train_pinn,
             "price": price_test, "evaluate": evaluate}[s](cfg)
    print(f"done in {time.time() - t0:.0f}s; outputs in {cfg.output_dir}")


if __name__ == "__main__":
    main()
