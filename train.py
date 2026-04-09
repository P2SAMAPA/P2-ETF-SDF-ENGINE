#!/usr/bin/env python3
"""
train.py - Parallel matrix training for P2-ETF-SDF-ENGINE
Includes retry logic for 409 and 412 conflicts when pushing to Hugging Face.
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
from data_loader import DataLoader
from backtest_engine import BacktestEngine

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True,
                        help="HF dataset name for input data")
    parser.add_argument("--output", type=str, required=True,
                        help="HF dataset name for results")
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--model", type=str, required=True,
                        choices=["rf", "xgb", "elasticnet"])
    return parser.parse_args()

def override_config(lr: float):
    CONFIG['sdf_model']['signal']['residual_vol_penalty'] = lr
    CONFIG['sdf_model']['signal']['factor_exposure_weight'] = lr * 3

def run_backtest_for_universe(assets, benchmark, start_date, end_date, lr, window_size=252, top_n=3):
    engine = BacktestEngine(assets, benchmark, hf_token=os.getenv("HF_TOKEN"))
    results_df = engine.run_rolling_window(start_date, end_date, window_size, top_n)
    _, _, benchmark_returns = engine.prepare_data(start_date, end_date)
    strategy_returns = results_df.set_index('date')['strategy_return']
    common_idx = strategy_returns.index.intersection(benchmark_returns.index)
    strategy_returns = strategy_returns.loc[common_idx]
    benchmark_aligned = benchmark_returns.loc[common_idx]
    metrics = BacktestEngine.calculate_performance(strategy_returns, benchmark_aligned)
    return metrics

def push_with_retry(result_record, dataset_name, token, max_retries=5, initial_delay=1):
    """Push result to Hub with exponential backoff on 409/412 conflicts."""
    for attempt in range(max_retries):
        try:
            # Always reload the latest existing dataset to avoid 412
            try:
                existing = load_dataset(dataset_name, split="train", token=token)
                combined_df = pd.concat([existing.to_pandas(), pd.DataFrame([result_record])], ignore_index=True)
                final_ds = Dataset.from_pandas(combined_df)
            except Exception:
                # First run: no existing dataset
                final_ds = Dataset.from_pandas(pd.DataFrame([result_record]))

            final_ds.push_to_hub(dataset_name, token=token, split="train")
            print(f"Successfully pushed to {dataset_name}")
            return True

        except HfHubHTTPError as e:
            # Retry on both 409 (concurrent commit) and 412 (branch updated)
            if e.response.status_code in (409, 412):
                wait_time = initial_delay * (2 ** attempt)
                print(f"Conflict {e.response.status_code} on attempt {attempt+1}/{max_retries}. Waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise
        except Exception as e:
            # Catch any other exception and retry
            wait_time = initial_delay * (2 ** attempt)
            print(f"Unexpected error: {e}. Retry {attempt+1}/{max_retries} in {wait_time}s...")
            time.sleep(wait_time)

    raise Exception(f"Failed to push after {max_retries} attempts due to persistent conflicts.")

def main():
    args = parse_args()
    override_config(args.lr)
    np.random.seed(args.fold)

    start_date = CONFIG['backtest']['start_date']
    end_date   = CONFIG['backtest']['end_date']
    window_size = CONFIG['backtest']['window_strategies']['rolling']['window_size']
    top_n = CONFIG['backtest']['rebalance']['top_n']

    # Equity universe
    equity_assets = CONFIG['universes']['equity']['assets']
    equity_benchmark = CONFIG['universes']['equity']['benchmark']
    equity_metrics = run_backtest_for_universe(
        equity_assets, equity_benchmark, start_date, end_date,
        args.lr, window_size, top_n
    )

    # FI/Commodity universe
    fi_assets = CONFIG['universes']['fi_commodities']['assets']
    fi_benchmark = CONFIG['universes']['fi_commodities']['benchmark']
    fi_metrics = run_backtest_for_universe(
        fi_assets, fi_benchmark, start_date, end_date,
        args.lr, window_size, top_n
    )

    # Prepare result record
    result_record = {
        "fold": args.fold,
        "learning_rate": args.lr,
        "model_type": args.model,
        "timestamp": datetime.now().isoformat(),
        "equity_sharpe": equity_metrics["sharpe_ratio"],
        "equity_annual_return": equity_metrics["annual_return"],
        "equity_volatility": equity_metrics["volatility"],
        "equity_max_drawdown": equity_metrics["max_drawdown"],
        "equity_win_rate": equity_metrics["win_rate"],
        "equity_total_return": equity_metrics["total_return"],
        "fi_sharpe": fi_metrics["sharpe_ratio"],
        "fi_annual_return": fi_metrics["annual_return"],
        "fi_volatility": fi_metrics["volatility"],
        "fi_max_drawdown": fi_metrics["max_drawdown"],
        "fi_win_rate": fi_metrics["win_rate"],
        "fi_total_return": fi_metrics["total_return"],
    }

    push_with_retry(result_record, args.output, os.getenv("HF_TOKEN"))
    print("Training and upload complete.")

if __name__ == "__main__":
    main()
