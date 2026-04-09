#!/usr/bin/env python3
"""
train.py - Parallel matrix training for P2-ETF-SDF-ENGINE

Loads data from:   P2SAMAPA/fi-etf-macro-signal-master-data
Saves results to:  P2SAMAPA/p2-etf-sdf-engine-results

Accepts:
  --fold   : fold index (used for train/test split or seed)
  --lr     : learning rate (maps to residual_penalty and factor_exposure_weight)
  --model  : model type (stored in results, not used in current pipeline)
"""

import os
import argparse
import pandas as pd
import numpy as np
from datetime import datetime

from datasets import load_dataset, Dataset
from huggingface_hub import HfApi

# Import your existing modules
from configs import CONFIG
from data_loader import DataLoader
from backtest_engine import BacktestEngine

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True,
                        help="HF dataset name for input data")
    parser.add_argument("--output", type=str, required=True,
                        help="HF dataset name for results")
    parser.add_argument("--fold", type=int, required=True,
                        help="Fold index (0..K-1)")
    parser.add_argument("--lr", type=float, required=True,
                        help="Learning rate (maps to residual_penalty)")
    parser.add_argument("--model", type=str, required=True,
                        choices=["rf", "xgb", "elasticnet"],
                        help="Model type (stored for reference)")
    return parser.parse_args()

def override_config(lr: float):
    """Override config parameters with matrix hyperparameters."""
    # Map learning rate to residual penalty and factor exposure weight
    CONFIG['sdf_model']['signal']['residual_vol_penalty'] = lr
    CONFIG['sdf_model']['signal']['factor_exposure_weight'] = lr * 3  # example mapping
    # You can adjust the mapping as needed

def run_backtest_for_universe(assets, benchmark, start_date, end_date, lr, window_size=252, top_n=3):
    """
    Run rolling window backtest for a given universe.
    Returns performance metrics dictionary.
    """
    engine = BacktestEngine(assets, benchmark, hf_token=os.getenv("HF_TOKEN"))
    
    # Override residual penalty in the engine's config? 
    # The BacktestEngine uses CONFIG directly, so we already updated CONFIG.
    
    results_df = engine.run_rolling_window(
        start_date=start_date,
        end_date=end_date,
        window_size=window_size,
        top_n=top_n
    )
    
    # Calculate performance metrics from strategy returns
    # First, get benchmark returns for the same period
    _, _, benchmark_returns = engine.prepare_data(start_date, end_date)
    
    # Align dates
    strategy_returns = results_df.set_index('date')['strategy_return']
    common_idx = strategy_returns.index.intersection(benchmark_returns.index)
    strategy_returns = strategy_returns.loc[common_idx]
    benchmark_aligned = benchmark_returns.loc[common_idx]
    
    metrics = BacktestEngine.calculate_performance(strategy_returns, benchmark_aligned)
    
    return metrics

def main():
    args = parse_args()
    
    # Override config with hyperparameters
    override_config(args.lr)
    
    # Set random seed based on fold for reproducibility
    np.random.seed(args.fold)
    
    # Define date range (from config)
    start_date = CONFIG['backtest']['start_date']
    end_date   = CONFIG['backtest']['end_date']
    window_size = CONFIG['backtest']['window_strategies']['rolling']['window_size']
    top_n = CONFIG['backtest']['rebalance']['top_n']
    
    # Run backtest for Equity universe
    equity_assets = CONFIG['universes']['equity']['assets']
    equity_benchmark = CONFIG['universes']['equity']['benchmark']
    
    print(f"Running equity backtest for fold {args.fold}, lr={args.lr}, model={args.model}")
    equity_metrics = run_backtest_for_universe(
        equity_assets, equity_benchmark, start_date, end_date,
        lr=args.lr, window_size=window_size, top_n=top_n
    )
    
    # Run backtest for FI/Commodity universe
    fi_assets = CONFIG['universes']['fi_commodities']['assets']
    fi_benchmark = CONFIG['universes']['fi_commodities']['benchmark']
    
    print(f"Running FI/Commodity backtest for fold {args.fold}")
    fi_metrics = run_backtest_for_universe(
        fi_assets, fi_benchmark, start_date, end_date,
        lr=args.lr, window_size=window_size, top_n=top_n
    )
    
    # Prepare result record
    result_record = {
        "fold": args.fold,
        "learning_rate": args.lr,
        "model_type": args.model,
        "timestamp": datetime.now().isoformat(),
        
        # Equity metrics
        "equity_sharpe": equity_metrics["sharpe_ratio"],
        "equity_annual_return": equity_metrics["annual_return"],
        "equity_volatility": equity_metrics["volatility"],
        "equity_max_drawdown": equity_metrics["max_drawdown"],
        "equity_win_rate": equity_metrics["win_rate"],
        "equity_total_return": equity_metrics["total_return"],
        
        # FI/Commodity metrics
        "fi_sharpe": fi_metrics["sharpe_ratio"],
        "fi_annual_return": fi_metrics["annual_return"],
        "fi_volatility": fi_metrics["volatility"],
        "fi_max_drawdown": fi_metrics["max_drawdown"],
        "fi_win_rate": fi_metrics["win_rate"],
        "fi_total_return": fi_metrics["total_return"],
    }
    
    # Save to output Hugging Face dataset
    print(f"Saving results to {args.output}")
    results_df = pd.DataFrame([result_record])
    results_ds = Dataset.from_pandas(results_df)
    
    # Append to existing dataset if present
    try:
        existing = load_dataset(args.output, split="train", token=os.getenv("HF_TOKEN"))
        combined_df = pd.concat([existing.to_pandas(), results_df], ignore_index=True)
        results_ds = Dataset.from_pandas(combined_df)
    except Exception:
        pass  # First run
    
    results_ds.push_to_hub(
        args.output,
        token=os.getenv("HF_TOKEN"),
        split="train"
    )
    
    print("Training and upload complete.")

if __name__ == "__main__":
    main()
