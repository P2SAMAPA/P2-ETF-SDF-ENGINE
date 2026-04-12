#!/usr/bin/env python3
"""
predict.py - Daily signal generation for SDF Engine.

Workflow:
  1. Read equity_results.parquet and fi_results.parquet from HF
  2. Pick the best config (highest Sharpe) from each
  3. Re-run the full SDF pipeline on the LATEST data window using that config
  4. Write a single latest_signals.json to HF dataset

This runs AFTER all training jobs complete (triggered by daily_predict.yml).
It is the sole source of truth for app.py — the parquet files are training
logs only and are never read by the dashboard.

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

from huggingface_hub import HfApi, hf_hub_download
from pandas.tseries.offsets import CustomBusinessDay
from pandas.tseries.holiday import USFederalHolidayCalendar

from configs import CONFIG
from backtest_engine import BacktestEngine

warnings.filterwarnings("ignore")

RESULTS_REPO = "P2SAMAPA/p2-etf-sdf-engine-results"
SIGNALS_FILE = "latest_signals.json"


def get_next_trading_date() -> str:
    us_cal = USFederalHolidayCalendar()
    nyse = CustomBusinessDay(calendar=us_cal)
    return (datetime.now().date() + nyse).strftime("%Y-%m-%d")


def load_best_config(parquet_filename: str, token: str) -> dict | None:
    """
    Download the results parquet and return the row with the highest Sharpe.
    This is the config that performed best across all training folds/lrs.
    """
    try:
        path = hf_hub_download(
            repo_id=RESULTS_REPO,
            filename=parquet_filename,
            repo_type="dataset",
            token=token,
            force_download=True,
        )
        df = pd.read_parquet(path)
        if len(df) == 0:
            print(f"  [WARN] {parquet_filename} is empty")
            return None

        best_idx = df["sharpe_ratio"].idxmax()
        row = df.loc[best_idx].to_dict()
        print(f"  Best config from {parquet_filename}: "
              f"fold={row.get('fold')}, lr={row.get('learning_rate')}, "
              f"model={row.get('model_type')}, Sharpe={row.get('sharpe_ratio', 0):.4f}, "
              f"AnnRet={row.get('annual_return', 0)*100:.2f}%")
        return row
    except Exception as e:
        print(f"  [ERROR] Could not load {parquet_filename}: {e}")
        return None


def run_signal(universe: str, assets: list, benchmark: str,
               best_config: dict, token: str) -> dict:
    """
    Re-run the SDF pipeline on the latest data window using the best config's
    hyperparameters (residual_penalty / factor_exposure_weight come from lr).
    Returns a signal dict ready to be written to latest_signals.json.
    """
    print(f"\n[{universe}] Generating signal with best config...")

    lr = float(best_config.get("learning_rate", 0.01))
    # Mirror override_config() from train.py
    residual_penalty = lr
    factor_exposure_weight = lr * 3

    end_date = CONFIG["backtest"]["end_date"]
    window_size = CONFIG["backtest"]["window_strategies"]["rolling"]["window_size"]
    top_n = CONFIG["backtest"]["rebalance"]["top_n"]

    engine = BacktestEngine(assets, benchmark, hf_token=token)
    returns, macro, _ = engine.prepare_data(
        start_date=CONFIG["backtest"]["start_date"],
        end_date=end_date,
    )

    if len(returns) < window_size:
        raise ValueError(f"Not enough data: {len(returns)} rows < window {window_size}")

    # Use the very last available window
    train_returns = returns.iloc[-window_size:]
    train_macro = macro.loc[train_returns.index]

    from backtest_engine import BacktestConfig
    from preprocessor import Preprocessor
    from pca_extractor import PCAExtractor
    from sparse_rotation import SparseRotation
    from var_forecast import VARForecast
    from return_reconstruction import ReturnReconstructor
    from cross_sectional_score import CrossSectionalScorer

    preprocessor = Preprocessor()
    train_ret_filled = preprocessor.fill_missing_values(train_returns)
    train_mac_filled = preprocessor.fill_missing_values(train_macro)

    # PCA
    pca = PCAExtractor(
        min_factors=CONFIG["sdf_model"]["pca"]["min_factors"],
        max_factors=CONFIG["sdf_model"]["pca"]["max_factors"],
        standardize=True,
    )
    pca.fit(train_ret_filled)
    factors = pca.get_factors(train_ret_filled.index)

    # Sparse rotation
    rotator = SparseRotation(max_iter=CONFIG["sdf_model"]["rotation"]["max_iter"])
    rotator.fit(pca.loadings_)
    sparse_loadings = SparseRotation.create_sparse_mask(rotator.rotated_loadings_, top_n=3)

    # VAR forecast
    forecaster = VARForecast(
        lag_order=CONFIG["sdf_model"]["var"]["lag_order"],
        use_kalman=CONFIG["sdf_model"]["var"]["use_kalman"],
    )
    forecasted_factors = forecaster.predict_factors(factors, train_mac_filled, horizon=1)

    # Reconstruct
    reconstructor = ReturnReconstructor(
        residual_penalty=residual_penalty,
        top_loadings_per_factor=3,
    )
    reconstructor.fit(train_ret_filled, factors, sparse_loadings)
    forecasted_returns, _ = reconstructor.reconstruct(forecasted_factors)

    # Scale to decimal
    def scale(v):
        v = float(v)
        if not np.isfinite(v):
            return 0.0
        if abs(v) > 1.0:
            v = v / 100.0
        return float(np.clip(v, -0.5, 0.5))

    forecasted_dict = {
        asset: scale(ret)
        for asset, ret in zip(assets, forecasted_returns)
    }

    # Score
    scorer = CrossSectionalScorer(
        factor_exposure_weight=factor_exposure_weight,
        residual_vol_penalty=residual_penalty,
    )
    scorer.fit(assets, factors.columns.tolist(), sparse_loadings, reconstructor.residual_std_)
    scores_df = scorer.compute_scores(forecasted_returns, forecasted_factors)
    selected = scorer.select_top_n(scores_df, top_n)
    scores_dict = {row["asset"]: float(row["composite_score"]) for _, row in scores_df.iterrows()}

    # Sort for top pick
    sorted_etfs = sorted(forecasted_dict.items(), key=lambda x: x[1], reverse=True)
    top_etf, top_ret = sorted_etfs[0]

    return {
        "universe": universe,
        "benchmark": benchmark,
        "signal_date": get_next_trading_date(),
        "generated_at": datetime.utcnow().isoformat(),
        "last_data_date": str(returns.index[-1].date()),
        "top_etf": top_etf,
        "top_return": round(top_ret, 6),
        "top_etfs": selected["asset"].tolist(),
        "forecasted_returns": {k: round(v, 6) for k, v in forecasted_dict.items()},
        "scores": {k: round(v, 6) for k, v in scores_dict.items()},
        # Backtest metrics from the best training run
        "sharpe_ratio": round(float(best_config.get("sharpe_ratio", 0)), 4),
        "annual_return": round(float(best_config.get("annual_return", 0)), 6),
        "max_drawdown": round(float(best_config.get("max_drawdown", 0)), 6),
        "volatility": round(float(best_config.get("volatility", 0)), 6),
        "win_rate": round(float(best_config.get("win_rate", 0)), 4),
        # Config provenance
        "best_fold": int(best_config.get("fold", -1)),
        "best_lr": float(best_config.get("learning_rate", 0)),
        "best_model": str(best_config.get("model_type", "")),
    }


def upload_signals(payload: dict, token: str) -> None:
    """Upload latest_signals.json to HF dataset."""
    api = HfApi(token=token)
    try:
        api.create_repo(RESULTS_REPO, repo_type="dataset", exist_ok=True, token=token)
    except Exception:
        pass

    data = json.dumps(payload, indent=2).encode("utf-8")
    api.upload_file(
        path_or_fileobj=BytesIO(data),
        path_in_repo=SIGNALS_FILE,
        repo_id=RESULTS_REPO,
        repo_type="dataset",
        commit_message=f"Update signals ({payload.get('generated_at', '')})",
    )
    print(f"\n[predict] Uploaded {SIGNALS_FILE} to {RESULTS_REPO}")


def main():
    token = os.getenv("HF_TOKEN")
    if not token:
        print("[ERROR] HF_TOKEN not set")
        sys.exit(1)

    print("=" * 60)
    print(f"SDF ENGINE — Daily Signal Generation")
    print(f"Run at: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 60)

    # ── Equity ──────────────────────────────────────────────────────────────
    eq_config = load_best_config("equity_results.parquet", token)
    if eq_config is None:
        print("[ERROR] No equity training results found — run training first")
        sys.exit(1)

    eq_signal = run_signal(
        universe="equity",
        assets=CONFIG["universes"]["equity"]["assets"],
        benchmark=CONFIG["universes"]["equity"]["benchmark"],
        best_config=eq_config,
        token=token,
    )

    # ── FI / Commodities ─────────────────────────────────────────────────────
    fi_config = load_best_config("fi_results.parquet", token)
    if fi_config is None:
        print("[ERROR] No FI training results found — run training first")
        sys.exit(1)

    fi_signal = run_signal(
        universe="fi_commodity",
        assets=CONFIG["universes"]["fi_commodities"]["assets"],
        benchmark=CONFIG["universes"]["fi_commodities"]["benchmark"],
        best_config=fi_config,
        token=token,
    )

    # ── Upload ───────────────────────────────────────────────────────────────
    payload = {
        "generated_at": datetime.utcnow().isoformat(),
        "equity": eq_signal,
        "fi_commodity": fi_signal,
    }
    upload_signals(payload, token)

    print("\n=== Signal generation complete ===")
    print(f"  Equity top pick : {eq_signal['top_etf']} "
          f"({eq_signal['top_return']*100:+.3f}%)  "
          f"Sharpe={eq_signal['sharpe_ratio']:.2f}  "
          f"AnnRet={eq_signal['annual_return']*100:.1f}%")
    print(f"  FI top pick     : {fi_signal['top_etf']} "
          f"({fi_signal['top_return']*100:+.3f}%)  "
          f"Sharpe={fi_signal['sharpe_ratio']:.2f}  "
          f"AnnRet={fi_signal['annual_return']*100:.1f}%")


if __name__ == "__main__":
    main()
