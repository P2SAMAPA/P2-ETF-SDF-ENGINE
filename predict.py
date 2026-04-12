#!/usr/bin/env python3
"""
predict.py - Daily signal generation for SDF Engine.

Workflow:
  1. List all results/fold*.json files in HF dataset
  2. Pick the best config per universe (highest Sharpe)
  3. Re-run the full SDF pipeline on the LATEST data window
  4. Write latest_signals.json to HF — the only file app.py ever reads

Usage:
    python predict.py
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from io import BytesIO

from huggingface_hub import HfApi, hf_hub_download, list_repo_files
from pandas.tseries.offsets import CustomBusinessDay
from pandas.tseries.holiday import USFederalHolidayCalendar

from configs import CONFIG
from preprocessor import Preprocessor
from pca_extractor import PCAExtractor
from sparse_rotation import SparseRotation
from var_forecast import VARForecast
from return_reconstruction import ReturnReconstructor
from cross_sectional_score import CrossSectionalScorer
from backtest_engine import BacktestEngine

warnings.filterwarnings("ignore")

RESULTS_REPO = "P2SAMAPA/p2-etf-sdf-engine-results"
SIGNALS_FILE = "latest_signals.json"


def get_next_trading_date() -> str:
    us_cal = USFederalHolidayCalendar()
    nyse   = CustomBusinessDay(calendar=us_cal)
    return (datetime.now().date() + nyse).strftime("%Y-%m-%d")


def load_all_results(token: str) -> list[dict]:
    """
    Download every results/fold*.json from HF and return as a list of dicts.
    Each dict has top-level keys: fold, learning_rate, model_type,
    equity: {...metrics...}, fi_commodity: {...metrics...}
    """
    print("[predict] Scanning result files...")
    try:
        all_files = list(list_repo_files(RESULTS_REPO, repo_type="dataset", token=token))
    except Exception as e:
        print(f"  [ERROR] Could not list repo files: {e}")
        return []

    result_files = [f for f in all_files if f.startswith("results/fold") and f.endswith(".json")]
    print(f"  Found {len(result_files)} result file(s)")

    records = []
    for fname in result_files:
        try:
            path = hf_hub_download(
                repo_id=RESULTS_REPO,
                filename=fname,
                repo_type="dataset",
                token=token,
                force_download=True,
            )
            with open(path) as f:
                records.append(json.load(f))
        except Exception as e:
            print(f"  [WARN] Could not load {fname}: {e}")

    print(f"  Loaded {len(records)} result(s) successfully")
    return records


def pick_best_config(records: list[dict], universe_key: str) -> dict | None:
    """
    From all job results, pick the one with the highest Sharpe for the
    given universe_key ('equity' or 'fi_commodity').
    Returns the full record dict (includes fold/lr/model_type + universe block).
    """
    valid = [r for r in records if universe_key in r
             and np.isfinite(r[universe_key].get('sharpe_ratio', float('nan')))]

    if not valid:
        return None

    best = max(valid, key=lambda r: r[universe_key]['sharpe_ratio'])
    u    = best[universe_key]
    print(f"  Best config [{universe_key}]: fold={best['fold']}  "
          f"lr={best['learning_rate']}  model={best['model_type']}  "
          f"Sharpe={u['sharpe_ratio']:.4f}  AnnRet={u['annual_return']*100:.2f}%")
    return best


def run_signal(universe_key: str, assets: list, benchmark: str,
               best_record: dict, token: str) -> dict:
    """
    Re-run the SDF pipeline on the latest data window using the best config's
    hyperparameters. Returns the signal dict for this universe.
    """
    print(f"\n[predict] Generating signal for {universe_key}...")

    lr                   = float(best_record['learning_rate'])
    residual_penalty     = lr
    factor_exposure_weight = lr * 3

    window_size = CONFIG['backtest']['window_strategies']['rolling']['window_size']
    top_n       = CONFIG['backtest']['rebalance']['top_n']

    # Load latest data
    engine = BacktestEngine(assets, benchmark, hf_token=token)
    returns, macro, _ = engine.prepare_data(
        start_date=CONFIG['backtest']['start_date'],
        end_date=CONFIG['backtest']['end_date'],
    )

    if len(returns) < window_size:
        raise ValueError(f"Not enough data: {len(returns)} rows < window {window_size}")

    train_returns = returns.iloc[-window_size:]
    train_macro   = macro.loc[train_returns.index]

    preprocessor = Preprocessor()
    tr_filled    = preprocessor.fill_missing_values(train_returns)
    tm_filled    = preprocessor.fill_missing_values(train_macro)

    # PCA
    pca = PCAExtractor(
        min_factors=CONFIG['sdf_model']['pca']['min_factors'],
        max_factors=CONFIG['sdf_model']['pca']['max_factors'],
        standardize=True,
    )
    pca.fit(tr_filled)
    factors = pca.get_factors(tr_filled.index)

    # Sparse rotation
    rotator = SparseRotation(max_iter=CONFIG['sdf_model']['rotation']['max_iter'])
    rotator.fit(pca.loadings_)
    sparse_loadings = SparseRotation.create_sparse_mask(rotator.rotated_loadings_, top_n=3)

    # VAR forecast
    forecaster = VARForecast(
        lag_order=CONFIG['sdf_model']['var']['lag_order'],
        use_kalman=CONFIG['sdf_model']['var']['use_kalman'],
    )
    forecasted_factors = forecaster.predict_factors(factors, tm_filled, horizon=1)

    # Reconstruct returns
    reconstructor = ReturnReconstructor(
        residual_penalty=residual_penalty,
        top_loadings_per_factor=3,
    )
    reconstructor.fit(tr_filled, factors, sparse_loadings)
    forecasted_returns, _ = reconstructor.reconstruct(forecasted_factors)

    # Scale to decimal
    def scale(v):
        v = float(v)
        if not np.isfinite(v): return 0.0
        if abs(v) > 1.0: v = v / 100.0
        return float(np.clip(v, -0.5, 0.5))

    forecasted_dict = {asset: scale(ret) for asset, ret in zip(assets, forecasted_returns)}

    # Score & rank
    scorer = CrossSectionalScorer(
        factor_exposure_weight=factor_exposure_weight,
        residual_vol_penalty=residual_penalty,
    )
    scorer.fit(assets, factors.columns.tolist(), sparse_loadings, reconstructor.residual_std_)
    scores_df   = scorer.compute_scores(forecasted_returns, forecasted_factors)
    selected    = scorer.select_top_n(scores_df, top_n)
    scores_dict = {row['asset']: float(row['composite_score']) for _, row in scores_df.iterrows()}

    sorted_etfs  = sorted(forecasted_dict.items(), key=lambda x: x[1], reverse=True)
    top_etf, top_ret = sorted_etfs[0]

    # Pull metrics from the best training record (already computed correctly)
    u = best_record[universe_key]

    return {
        "universe":           universe_key,
        "benchmark":          benchmark,
        "signal_date":        get_next_trading_date(),
        "generated_at":       datetime.utcnow().isoformat(),
        "last_data_date":     str(returns.index[-1].date()),
        "top_etf":            top_etf,
        "top_return":         round(top_ret, 6),
        "top_etfs":           selected['asset'].tolist(),
        "forecasted_returns": {k: round(v, 6) for k, v in forecasted_dict.items()},
        "scores":             {k: round(v, 6) for k, v in scores_dict.items()},
        # Backtest metrics from best training run
        "sharpe_ratio":       round(float(u.get('sharpe_ratio',  0)), 4),
        "annual_return":      round(float(u.get('annual_return', 0)), 6),
        "max_drawdown":       round(float(u.get('max_drawdown',  0)), 6),
        "volatility":         round(float(u.get('volatility',    0)), 6),
        "win_rate":           round(float(u.get('win_rate',      0)), 4),
        # Config provenance
        "best_fold":          int(best_record['fold']),
        "best_lr":            float(best_record['learning_rate']),
        "best_model":         str(best_record['model_type']),
    }


def upload_signals(payload: dict, token: str) -> None:
    api  = HfApi(token=token)
    data = json.dumps(payload, indent=2).encode("utf-8")
    api.upload_file(
        path_or_fileobj=BytesIO(data),
        path_in_repo=SIGNALS_FILE,
        repo_id=RESULTS_REPO,
        repo_type="dataset",
        commit_message=f"Update signals ({payload['generated_at']})",
    )
    print(f"\n[predict] Uploaded {SIGNALS_FILE} to {RESULTS_REPO}")


def main():
    token = os.getenv("HF_TOKEN")
    if not token:
        print("[ERROR] HF_TOKEN not set")
        sys.exit(1)

    print("=" * 60)
    print(f"SDF ENGINE — Daily Signal Generation")
    print(f"Run at: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 60)

    records = load_all_results(token)
    if not records:
        print("[ERROR] No result files found — run training first")
        sys.exit(1)

    # ── Equity ────────────────────────────────────────────────────────────────
    eq_best = pick_best_config(records, "equity")
    if eq_best is None:
        print("[ERROR] No valid equity results found")
        sys.exit(1)

    eq_signal = run_signal(
        universe_key="equity",
        assets=CONFIG['universes']['equity']['assets'],
        benchmark=CONFIG['universes']['equity']['benchmark'],
        best_record=eq_best,
        token=token,
    )

    # ── FI / Commodities ──────────────────────────────────────────────────────
    fi_best = pick_best_config(records, "fi_commodity")
    if fi_best is None:
        print("[ERROR] No valid FI results found")
        sys.exit(1)

    fi_signal = run_signal(
        universe_key="fi_commodity",
        assets=CONFIG['universes']['fi_commodities']['assets'],
        benchmark=CONFIG['universes']['fi_commodities']['benchmark'],
        best_record=fi_best,
        token=token,
    )

    # ── Upload ────────────────────────────────────────────────────────────────
    payload = {
        "generated_at": datetime.utcnow().isoformat(),
        "equity":       eq_signal,
        "fi_commodity": fi_signal,
    }
    upload_signals(payload, token)

    print("\n=== Signal generation complete ===")
    print(f"  Equity  top={eq_signal['top_etf']} ({eq_signal['top_return']*100:+.3f}%)  "
          f"Sharpe={eq_signal['sharpe_ratio']:.2f}  AnnRet={eq_signal['annual_return']*100:.1f}%")
    print(f"  FI      top={fi_signal['top_etf']} ({fi_signal['top_return']*100:+.3f}%)  "
          f"Sharpe={fi_signal['sharpe_ratio']:.2f}  AnnRet={fi_signal['annual_return']*100:.1f}%")


if __name__ == "__main__":
    main()
