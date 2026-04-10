#!/usr/bin/env python3
"""
train.py - Parallel matrix training with backtest metrics
Includes detailed debug logging to trace execution.
CI Mode: Set CI_MODE=true and MAX_WINDOWS=N for quick validation
"""

import os
import sys
import argparse
import time
import warnings
import numpy as np
import pandas as pd
from datetime import datetime

from datasets import Dataset
from huggingface_hub.errors import HfHubHTTPError

from configs import CONFIG
from backtest_engine import BacktestEngine

# Suppress harmless statsmodels warnings
warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels.tsa.base.tsa_model")

print("=" * 60)
print("train.py started")
print("=" * 60)
sys.stdout.flush()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True, help="Fold index (seed)")
    parser.add_argument("--lr", type=float, required=True, help="Learning rate")
    parser.add_argument("--model", type=str, required=True, choices=["rf", "xgb", "elasticnet"])
    parser.add_argument("--start-date", type=str, default=None, help="Override start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default=None, help="Override end date (YYYY-MM-DD)")
    return parser.parse_args()

def override_config(lr: float):
    CONFIG['sdf_model']['signal']['residual_vol_penalty'] = lr
    CONFIG['sdf_model']['signal']['factor_exposure_weight'] = lr * 3
    print(f"Config overridden: residual_vol_penalty={lr}, factor_exposure_weight={lr*3}")

def safe_metrics(strategy_returns, benchmark_returns):
    """Compute metrics with safety checks to avoid inf/nan."""
    print("Computing performance metrics...")
    if len(strategy_returns) == 0:
        print("WARNING: Empty strategy returns")
        return {k: np.nan for k in ['sharpe_ratio', 'annual_return', 'volatility',
                                     'max_drawdown', 'win_rate', 'total_return']}
    common_idx = strategy_returns.index.intersection(benchmark_returns.index)
    r = strategy_returns.loc[common_idx]
    b = benchmark_returns.loc[common_idx]
    if len(r) == 0:
        print("WARNING: No overlapping dates between strategy and benchmark")
        return {k: np.nan for k in ['sharpe_ratio', 'annual_return', 'volatility',
                                     'max_drawdown', 'win_rate', 'total_return']}
    
    total_return = (1 + r).prod() - 1
    
    # FIX: Handle overflow in annual return calculation
    years = len(r) / 252
    if years > 0 and np.isfinite(total_return) and total_return > -1:
        # Clamp total_return to avoid overflow
        total_return_clamped = np.clip(total_return, -0.9999, 100)  # Max 100x return
        annual_return = (1 + total_return_clamped) ** (1 / years) - 1
        if not np.isfinite(annual_return):
            annual_return = 0.0
    else:
        annual_return = 0.0
    
    volatility = r.std() * np.sqrt(252)
    sharpe = annual_return / (volatility + 1e-8) if volatility > 0 else 0.0
    # Clamp Sharpe to reasonable range
    sharpe = float(np.clip(sharpe, -10, 10))
    
    cum = (1 + r).cumprod()
    rolling_max = cum.cummax()
    drawdown = (cum - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    win_rate = (r > 0).mean()
    
    metrics = {
        'sharpe_ratio': float(sharpe),
        'annual_return': float(annual_return),
        'volatility': float(volatility),
        'max_drawdown': float(max_drawdown),
        'win_rate': float(win_rate),
        'total_return': float(total_return),
    }
    print(f"Metrics computed: Sharpe={sharpe:.3f}, AnnRet={annual_return:.4f}, MaxDD={max_drawdown:.4f}")
    return metrics

def run_backtest_for_universe(assets, benchmark, start_date, end_date, window_size, top_n, max_windows=None):
    """Run rolling window backtest and return metrics + final prediction."""
    print(f"\n--- Backtest for {len(assets)} assets, benchmark={benchmark} ---")
    print(f"Date range: {start_date} to {end_date}, window={window_size}, top_n={top_n}")
    if max_windows:
        print(f"CI MODE: Limiting to first {max_windows} windows")
    sys.stdout.flush()
    
    engine = BacktestEngine(assets, benchmark, hf_token=os.getenv("HF_TOKEN"))
    print("BacktestEngine created")
    
    print("Running rolling window...")
    results_df = engine.run_rolling_window(start_date, end_date, window_size, top_n, max_windows=max_windows)
    print(f"Results DataFrame shape: {results_df.shape}")
    if len(results_df) == 0:
        print("ERROR: No results from backtest")
        return None, None
    
    print("Loading benchmark returns for metric calculation...")
    _, _, benchmark_returns = engine.prepare_data(start_date, end_date)
    strat_returns = results_df.set_index('date')['strategy_return']
    print(f"Strategy returns length: {len(strat_returns)}")
    
    metrics = safe_metrics(strat_returns, benchmark_returns)
    final_signals = results_df.iloc[-1] if len(results_df) > 0 else None
    if final_signals is not None:
        print(f"Final signals date: {final_signals['date']}, selected: {final_signals['selected_assets']}")
    else:
        print("WARNING: No final signals")
    
    return metrics, final_signals

def push_with_retry(dataset, dataset_name, token, max_retries=5):
    """Push dataset with retry on 409/412."""
    for attempt in range(max_retries):
        try:
            dataset.push_to_hub(dataset_name, token=token, split="train")
            print(f"Successfully pushed to {dataset_name}")
            return True
        except HfHubHTTPError as e:
            if e.response.status_code in (409, 412):
                wait = 2 ** attempt
                print(f"Conflict {e.response.status_code}, retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"HTTP error {e.response.status_code}: {e}")
                raise
        except Exception as e:
            print(f"Unexpected error: {e}")
            raise
    print(f"Failed to push to {dataset_name} after {max_retries} attempts")
    return False

def main():
    print("\n=== Entering main() ===")
    args = parse_args()
    print(f"Arguments: fold={args.fold}, lr={args.lr}, model={args.model}")
    sys.stdout.flush()
    
    # CI MODE DETECTION
    is_ci = os.getenv('CI_MODE', '').lower() == 'true' or os.getenv('GITHUB_ACTIONS', '').lower() == 'true'
    max_windows = os.getenv('MAX_WINDOWS')
    if max_windows:
        max_windows = int(max_windows)
        print(f"CI MODE: MAX_WINDOWS set to {max_windows}")
    
    override_config(args.lr)
    np.random.seed(args.fold)
    
    # Get config values with CI overrides
    start_date = args.start_date or CONFIG['backtest']['start_date']
    end_date = args.end_date or CONFIG['backtest']['end_date']
    
    # CI OPTIMIZATION: If no explicit dates but CI mode, use last 2 years only
    if is_ci and not args.start_date and not args.end_date:
        start_date = '2024-01-01'  # Last ~1.5 years instead of 2008
        print(f"CI MODE: Auto-adjusted start_date to {start_date} for speed")
    
    window_size = CONFIG['backtest']['window_strategies']['rolling']['window_size']
    top_n = CONFIG['backtest']['rebalance']['top_n']
    print(f"Backtest config: {start_date} -> {end_date}, window={window_size}, top_n={top_n}")
    sys.stdout.flush()
    
    # ----- Equity Universe -----
    eq_assets = CONFIG['universes']['equity']['assets']
    eq_bench = CONFIG['universes']['equity']['benchmark']
    print(f"\nProcessing EQUITY universe with {len(eq_assets)} assets")
    eq_metrics, eq_final = run_backtest_for_universe(
        eq_assets, eq_bench, start_date, end_date, window_size, top_n, max_windows
    )
    if eq_metrics is None:
        print("FATAL: Equity backtest failed, exiting.")
        sys.exit(1)
    
    # ----- FI/Commodity Universe -----
    fi_assets = CONFIG['universes']['fi_commodities']['assets']
    fi_bench = CONFIG['universes']['fi_commodities']['benchmark']
    print(f"\nProcessing FI/COMMODITY universe with {len(fi_assets)} assets")
    fi_metrics, fi_final = run_backtest_for_universe(
        fi_assets, fi_bench, start_date, end_date, window_size, top_n, max_windows
    )
    if fi_metrics is None:
        print("FATAL: FI backtest failed, exiting.")
        sys.exit(1)
    
    # FIX: Convert all dict keys to strings for PyArrow compatibility
    def convert_scores(scores_dict):
        """Convert scores dict to have string keys for PyArrow."""
        if not scores_dict:
            return {}
        return {str(k): float(v) for k, v in scores_dict.items()}
    
    def convert_forecasted_returns(returns_dict):
        """Convert forecasted returns dict to have string keys."""
        if not returns_dict:
            return {}
        return {str(k): float(v) for k, v in returns_dict.items()}
    
    # Build records with explicit type conversion
    equity_record = {
        "fold": int(args.fold),
        "learning_rate": float(args.lr),
        "model_type": str(args.model),
        "timestamp": datetime.now().isoformat(),
        "date": eq_final['date'].strftime('%Y-%m-%d') if eq_final is not None else None,
        "sharpe_ratio": float(eq_metrics['sharpe_ratio']),
        "annual_return": float(eq_metrics['annual_return']),
        "volatility": float(eq_metrics['volatility']),
        "max_drawdown": float(eq_metrics['max_drawdown']),
        "win_rate": float(eq_metrics['win_rate']),
        "total_return": float(eq_metrics['total_return']),
        "top_etfs": [str(x) for x in eq_final['selected_assets']] if eq_final is not None else [],
        "forecasted_returns": convert_forecasted_returns(eq_final['forecasted_returns']) if eq_final is not None else {},
        "scores": convert_scores(eq_final['scores']) if eq_final is not None else {},
    }
    
    fi_record = {
        "fold": int(args.fold),
        "learning_rate": float(args.lr),
        "model_type": str(args.model),
        "timestamp": datetime.now().isoformat(),
        "date": fi_final['date'].strftime('%Y-%m-%d') if fi_final is not None else None,
        "sharpe_ratio": float(fi_metrics['sharpe_ratio']),
        "annual_return": float(fi_metrics['annual_return']),
        "volatility": float(fi_metrics['volatility']),
        "max_drawdown": float(fi_metrics['max_drawdown']),
        "win_rate": float(fi_metrics['win_rate']),
        "total_return": float(fi_metrics['total_return']),
        "top_etfs": [str(x) for x in fi_final['selected_assets']] if fi_final is not None else [],
        "forecasted_returns": convert_forecasted_returns(fi_final['forecasted_returns']) if fi_final is not None else {},
        "scores": convert_scores(fi_final['scores']) if fi_final is not None else {},
    }
    
    token = os.getenv("HF_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN environment variable not set")
        sys.exit(1)
    
    print("\n--- Pushing Equity Results ---")
    eq_ds = Dataset.from_pandas(pd.DataFrame([equity_record]))
    push_with_retry(eq_ds, "P2SAMAPA/p2-etf-sdf-engine-results-equity", token)
    
    print("\n--- Pushing FI Results ---")
    fi_ds = Dataset.from_pandas(pd.DataFrame([fi_record]))
    push_with_retry(fi_ds, "P2SAMAPA/p2-etf-sdf-engine-results-fi", token)
    
    print("\n=== Training completed successfully ===")

if __name__ == "__main__":
    main()
