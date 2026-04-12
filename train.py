#!/usr/bin/env python3
"""
train.py - Production training with backtest metrics

Each matrix job writes ONE isolated JSON file to HF:
    results/fold{N}_{model}_{lr}.json

This completely eliminates the race condition from concurrent parquet appends.
predict.py reads all result files, picks the best config, and writes signals.
"""

import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
import json
from datetime import datetime
from io import BytesIO

from huggingface_hub import HfApi
from configs import CONFIG
from backtest_engine import BacktestEngine

warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels.tsa.base.tsa_model")

print("=" * 60)
print("train.py - Production")
print("=" * 60)
sys.stdout.flush()

RESULTS_REPO = "P2SAMAPA/p2-etf-sdf-engine-results"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold",       type=int,   required=True)
    parser.add_argument("--lr",         type=float, required=True)
    parser.add_argument("--model",      type=str,   required=True, choices=["rf", "xgb", "elasticnet"])
    parser.add_argument("--start-date", type=str,   default=None)
    parser.add_argument("--end-date",   type=str,   default=None)
    return parser.parse_args()


def override_config(lr: float):
    CONFIG['sdf_model']['signal']['residual_vol_penalty']    = lr
    CONFIG['sdf_model']['signal']['factor_exposure_weight']  = lr * 3
    print(f"Config: residual_vol_penalty={lr}, factor_exposure_weight={lr * 3}")


def safe_metrics(strategy_returns, benchmark_returns, universe_name=""):
    """Compute backtest metrics with overflow protection."""
    if len(strategy_returns) == 0:
        return {k: 0.0 for k in ['sharpe_ratio', 'annual_return', 'volatility',
                                   'max_drawdown', 'win_rate', 'total_return']}

    clean = strategy_returns.replace([np.inf, -np.inf], np.nan).fillna(0).clip(-0.99, 1.0)

    total_return = float(np.clip((1 + clean).prod() - 1, -0.9999, 10.0))
    if not np.isfinite(total_return):
        total_return = 0.0

    n_days = len(clean)
    years  = n_days / 252.0

    try:
        annual_return = (1 + total_return) ** (1 / years) - 1 if years >= 0.1 else total_return * (252.0 / n_days)
    except Exception:
        annual_return = total_return * (252.0 / n_days)

    annual_return = float(np.clip(annual_return if np.isfinite(annual_return) else 0.0, -1.0, 5.0))

    volatility = float(clean.std() * np.sqrt(252))
    if not np.isfinite(volatility):
        volatility = 0.0

    if volatility > 1e-10:
        sharpe = float(np.clip(annual_return / volatility, -10, 10))
    else:
        sharpe = 10.0 if annual_return > 0 else (-10.0 if annual_return < 0 else 0.0)

    cum         = (1 + clean).cumprod()
    max_drawdown = float((cum - cum.cummax()).div(cum.cummax()).min())
    win_rate     = float((clean > 0).mean())

    print(f"  [{universe_name}] AnnRet={annual_return*100:.2f}%  Sharpe={sharpe:.4f}  "
          f"MaxDD={max_drawdown*100:.2f}%  WinRate={win_rate*100:.1f}%")

    return {
        'sharpe_ratio': sharpe,
        'annual_return': annual_return,
        'volatility': volatility,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'total_return': total_return,
    }


def run_backtest(assets, benchmark, start_date, end_date, window_size, top_n,
                 max_windows=None, universe_name=""):
    """Run rolling window backtest, return (metrics, final_signals_row)."""
    print(f"\n[{universe_name}] Backtest: {len(assets)} assets  {start_date} → {end_date}")
    sys.stdout.flush()

    engine     = BacktestEngine(assets, benchmark, hf_token=os.getenv("HF_TOKEN"))
    results_df = engine.run_rolling_window(start_date, end_date, window_size, top_n,
                                           max_windows=max_windows)

    if len(results_df) == 0:
        print(f"[{universe_name}] ERROR: No results produced")
        return None, None

    print(f"[{universe_name}] {len(results_df)} rolling periods completed")

    _, _, benchmark_returns = engine.prepare_data(start_date, end_date)
    strat_returns = results_df.set_index('date')['strategy_return']
    metrics       = safe_metrics(strat_returns, benchmark_returns, universe_name)
    final_signals = results_df.iloc[-1]

    return metrics, final_signals


def upload_result_json(record: dict, fold: int, model: str, lr: float, token: str) -> bool:
    """
    Write this job's result to its own isolated file:
        results/fold{N}_{model}_{lr}.json

    Each job gets a unique filename → zero overlap → no race condition.
    predict.py scans all files in results/ and picks the best.
    """
    lr_str   = str(lr).replace(".", "_")
    filename = f"results/fold{fold}_{model}_{lr_str}.json"

    try:
        api  = HfApi(token=token)
        data = json.dumps(record, indent=2).encode("utf-8")
        api.upload_file(
            path_or_fileobj=BytesIO(data),
            path_in_repo=filename,
            repo_id=RESULTS_REPO,
            repo_type="dataset",
            commit_message=f"Result fold={fold} model={model} lr={lr}",
        )
        print(f"  Uploaded {filename}  Sharpe={record['equity']['sharpe_ratio']:.4f}  "
              f"AnnRet={record['equity']['annual_return']*100:.2f}%")
        return True
    except Exception as e:
        print(f"  ERROR uploading {filename}: {e}")
        return False


def main():
    args = parse_args()
    print(f"Args: fold={args.fold}, lr={args.lr}, model={args.model}")

    is_ci       = os.getenv('CI_MODE', '').lower() == 'true'
    max_windows = os.getenv('MAX_WINDOWS')
    max_windows = int(max_windows) if max_windows else None

    override_config(args.lr)
    np.random.seed(args.fold)

    start_date  = args.start_date or CONFIG['backtest']['start_date']
    end_date    = args.end_date   or CONFIG['backtest']['end_date']

    if is_ci and not args.start_date:
        start_date = '2024-01-01'
        print(f"CI MODE: Using {start_date}")

    window_size = CONFIG['backtest']['window_strategies']['rolling']['window_size']
    top_n       = CONFIG['backtest']['rebalance']['top_n']

    # ── Equity ────────────────────────────────────────────────────────────────
    eq_assets  = CONFIG['universes']['equity']['assets']
    eq_bench   = CONFIG['universes']['equity']['benchmark']
    eq_metrics, eq_final = run_backtest(
        eq_assets, eq_bench, start_date, end_date, window_size, top_n, max_windows, "EQUITY"
    )
    if eq_metrics is None:
        sys.exit(1)

    # ── FI / Commodities ──────────────────────────────────────────────────────
    fi_assets  = CONFIG['universes']['fi_commodities']['assets']
    fi_bench   = CONFIG['universes']['fi_commodities']['benchmark']
    fi_metrics, fi_final = run_backtest(
        fi_assets, fi_bench, start_date, end_date, window_size, top_n, max_windows, "FI"
    )
    if fi_metrics is None:
        sys.exit(1)

    # ── Build record ──────────────────────────────────────────────────────────
    def universe_block(metrics, final, benchmark):
        forecasted = dict(final.get('forecasted_returns', {})) if final is not None else {}
        scores     = dict(final.get('scores',             {})) if final is not None else {}
        return {
            "benchmark":        benchmark,
            "sharpe_ratio":     float(metrics['sharpe_ratio']),
            "annual_return":    float(metrics['annual_return']),
            "volatility":       float(metrics['volatility']),
            "max_drawdown":     float(metrics['max_drawdown']),
            "win_rate":         float(metrics['win_rate']),
            "total_return":     float(metrics['total_return']),
            "top_etfs":         [str(x) for x in final['selected_assets']] if final is not None else [],
            "forecasted_returns": {str(k): float(v) for k, v in forecasted.items()},
            "scores":             {str(k): float(v) for k, v in scores.items()},
            "last_data_date":   final['date'].strftime('%Y-%m-%d') if final is not None else None,
        }

    record = {
        "fold":          int(args.fold),
        "learning_rate": float(args.lr),
        "model_type":    str(args.model),
        "timestamp":     datetime.now().isoformat(),
        "start_date":    start_date,
        "end_date":      end_date,
        "equity":        universe_block(eq_metrics, eq_final, eq_bench),
        "fi_commodity":  universe_block(fi_metrics, fi_final, fi_bench),
    }

    # ── Upload ────────────────────────────────────────────────────────────────
    token = os.getenv("HF_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN not set")
        sys.exit(1)

    api = HfApi(token=token)
    try:
        api.create_repo(RESULTS_REPO, repo_type="dataset", exist_ok=True, token=token)
    except Exception:
        pass

    print("\nUploading result...")
    ok = upload_result_json(record, args.fold, args.model, args.lr, token)
    if not ok:
        sys.exit(1)

    print("\n=== Training complete ===")
    print(f"  Equity  Sharpe={eq_metrics['sharpe_ratio']:.4f}  "
          f"AnnRet={eq_metrics['annual_return']*100:.2f}%")
    print(f"  FI      Sharpe={fi_metrics['sharpe_ratio']:.4f}  "
          f"AnnRet={fi_metrics['annual_return']*100:.2f}%")


if __name__ == "__main__":
    main()
