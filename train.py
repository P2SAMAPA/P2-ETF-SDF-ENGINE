#!/usr/bin/env python3
"""
train.py - MINIMAL TEST VERSION - Only 5 windows to debug
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
from datasets import Dataset

from configs import CONFIG
from backtest_engine import BacktestEngine

warnings.filterwarnings("ignore")

print("=" * 60)
print("MINIMAL TEST VERSION - MAX 5 WINDOWS")
print("=" * 60)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    return parser.parse_args()

args = parse_args()
print(f"Args: fold={args.fold}, lr={args.lr}, model={args.model}")

# Override config
CONFIG['sdf_model']['signal']['residual_vol_penalty'] = args.lr
CONFIG['sdf_model']['signal']['factor_exposure_weight'] = args.lr * 3

# Use very limited data
start_date = '2025-01-01'  # Only ~3 months
end_date = '2025-04-01'
window_size = 60  # Small window
top_n = 3

print(f"\nTesting with: {start_date} to {end_date}, window={window_size}")

# Test Equity only
print("\n" + "="*60)
print("TESTING EQUITY")
assets = CONFIG['universes']['equity']['assets']
benchmark = CONFIG['universes']['equity']['benchmark']

print(f"Assets: {assets}")
print(f"Benchmark: {benchmark}")

try:
    print("\nCreating engine...")
    engine = BacktestEngine(assets, benchmark, hf_token=os.getenv("HF_TOKEN"))
    print("Engine created")
    
    print("\nRunning rolling window (max 5)...")
    results_df = engine.run_rolling_window(start_date, end_date, window_size, top_n, max_windows=5)
    print(f"Results: {len(results_df)} rows")
    
    if len(results_df) == 0:
        print("ERROR: No results")
        sys.exit(1)
    
    print("\nCalculating metrics...")
    strat_returns = results_df.set_index('date')['strategy_return']
    print(f"Strategy returns: {len(strat_returns)}")
    print(f"Sample: {strat_returns.head()}")
    
    # Simple metrics
    total_return = (1 + strat_returns).prod() - 1
    n_days = len(strat_returns)
    annual_return = total_return * (252.0 / n_days) if n_days > 0 else 0
    volatility = strat_returns.std() * np.sqrt(252)
    sharpe = annual_return / volatility if volatility > 0 else 0
    max_dd = 0  # Simplified
    
    metrics = {
        'sharpe_ratio': float(sharpe),
        'annual_return': float(annual_return),
        'volatility': float(volatility),
        'max_drawdown': float(max_dd),
        'win_rate': float((strat_returns > 0).mean()),
        'total_return': float(total_return),
    }
    
    print(f"\nMETRICS: {metrics}")
    
    # Save minimal record
    final = results_df.iloc[-1]
    record = {
        "fold": int(args.fold),
        "learning_rate": float(args.lr),
        "model_type": str(args.model),
        "timestamp": datetime.now().isoformat(),
        "date": final['date'].strftime('%Y-%m-%d'),
        "benchmark": benchmark,
        "sharpe_ratio": metrics['sharpe_ratio'],
        "annual_return": metrics['annual_return'],
        "volatility": metrics['volatility'],
        "max_drawdown": metrics['max_drawdown'],
        "win_rate": metrics['win_rate'],
        "total_return": metrics['total_return'],
        "top_etfs": json.dumps([str(x) for x in final['selected_assets']]),
        "forecasted_returns": json.dumps({str(k): float(v) for k, v in final['forecasted_returns'].items()}),
        "scores
