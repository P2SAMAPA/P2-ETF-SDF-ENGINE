#!/usr/bin/env python3
"""
train.py - Parallel matrix training with backtest metrics
Runs rolling window backtest over the full date range from config,
calculates performance metrics, and saves results to HF dataset.
"""

import os
import argparse
import time
import pandas as pd
import numpy as np
from datetime import datetime

from datasets import load_dataset, Dataset
from huggingface_hub.errors import HfHubHTTPError

from configs import CONFIG
from backtest_engine import BacktestEngine

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True,
                        help="HF dataset name for input data")
    parser.add_argument("--output", type=str, required=True,
                        help="HF dataset name for results")
    parser.add_argument("--fold", type=int, required=True,
                        help="Fold index (used as random seed)")
    parser.add_argument("--lr", type=float, required=True,
                        help="Learning rate -> residual_penalty and factor_exposure_weight")
    parser.add_argument("--model", type=str, required=True,
                        choices=["rf", "xgb", "elasticnet"],
                        help="Model type (stored for reference)")
    return parser.parse_args()

def override_config(lr: float):
    """Map learning rate to model parameters."""
    CONFIG['sdf_model']['signal']['residual_vol_penalty'] = lr
    CONFIG['sdf_model']['signal']['factor_exposure_weight'] = lr * 3  # adjust as needed

def run_backtest_for_universe(assets, benchmark, start_date, end_date, lr, window_size, top_n):
    """
    Run rolling window backtest for a given universe.
    Returns (metrics_dict, final_signals_df)
    """
    engine = BacktestEngine(assets, benchmark, hf_token=os.getenv("HF_TOKEN"))
    
    # Run rolling window backtest
    results_df = engine.run_rolling_window(
        start_date=start_date,
        end_date=end_date,
        window_size=window_size,
        top_n=top_n
    )
    
    # Get benchmark returns for the same period
    _, _, benchmark_returns = engine.prepare_data(start_date, end_date)
    
    # Align dates
    strategy_returns = results_df.set_index('date')['strategy_return']
    common_idx = strategy_returns.index.intersection(benchmark_returns.index)
    strategy_returns = strategy_returns.loc[common_idx]
    benchmark_aligned = benchmark_returns.loc[common_idx]
    
    # Calculate metrics
    metrics = BacktestEngine.calculate_performance(strategy_returns, benchmark_aligned)
    
    # Get final prediction (last row of results_df)
    final_signals = results_df.iloc[-1] if len(results_df) > 0 else None
    
    return metrics, final_signals

def push_with_retry(dataset, dataset_name, token, max_retries=5):
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
                raise
    raise Exception("Failed after retries")

def main():
    args = parse_args()
    override_config(args.lr)
    np.random.seed(args.fold)   # for reproducibility
    
    # Get date range from config
    start_date = CONFIG['backtest']['start_date']
    end_date   = CONFIG['backtest']['end_date']
    window_size = CONFIG['backtest']['window_strategies']['rolling']['window_size']
    top_n = CONFIG['backtest']['rebalance']['top_n']
    
    print(f"Running backtest from {start_date} to {end_date} with window={window_size}, top_n={top_n}")
    
    # ----- Equity Universe -----
    eq_assets = CONFIG['universes']['equity']['assets']
    eq_bench = CONFIG['universes']['equity']['benchmark']
    eq_metrics, eq_final = run_backtest_for_universe(
        eq_assets, eq_bench, start_date, end_date, args.lr, window_size, top_n
    )
    
    # ----- FI/Commodity Universe -----
    fi_assets = CONFIG['universes']['fi_commodities']['assets']
    fi_bench = CONFIG['universes']['fi_commodities']['benchmark']
    fi_metrics, fi_final = run_backtest_for_universe(
        fi_assets, fi_bench, start_date, end_date, args.lr, window_size, top_n
    )
    
    # Prepare result record (metrics + final predictions)
    result_record = {
        "fold": args.fold,
        "learning_rate": args.lr,
        "model_type": args.model,
        "timestamp": datetime.now().isoformat(),
        "date": eq_final['date'].strftime('%Y-%m-%d') if eq_final is not None else None,
        # Equity metrics
        "equity_sharpe": eq_metrics.get("sharpe_ratio", np.nan),
        "equity_annual_return": eq_metrics.get("annual_return", np.nan),
        "equity_volatility": eq_metrics.get("volatility", np.nan),
        "equity_max_drawdown": eq_metrics.get("max_drawdown", np.nan),
        "equity_win_rate": eq_metrics.get("win_rate", np.nan),
        "equity_total_return": eq_metrics.get("total_return", np.nan),
        # FI metrics
        "fi_sharpe": fi_metrics.get("sharpe_ratio", np.nan),
        "fi_annual_return": fi_metrics.get("annual_return", np.nan),
        "fi_volatility": fi_metrics.get("volatility", np.nan),
        "fi_max_drawdown": fi_metrics.get("max_drawdown", np.nan),
        "fi_win_rate": fi_metrics.get("win_rate", np.nan),
        "fi_total_return": fi_metrics.get("total_return", np.nan),
        # Final predictions (top ETFs)
        "equity_top_etfs": eq_final['selected_assets'] if eq_final is not None else [],
        "fi_top_etfs": fi_final['selected_assets'] if fi_final is not None else [],
    }
    
    # Load existing dataset or create new
    token = os.getenv("HF_TOKEN")
    try:
        existing = load_dataset(args.output, split="train", token=token)
        combined_df = pd.concat([existing.to_pandas(), pd.DataFrame([result_record])], ignore_index=True)
        final_ds = Dataset.from_pandas(combined_df)
    except:
        final_ds = Dataset.from_pandas(pd.DataFrame([result_record]))
    
    push_with_retry(final_ds, args.output, token)
    print("Training and upload complete.")

if __name__ == "__main__":
    main()
