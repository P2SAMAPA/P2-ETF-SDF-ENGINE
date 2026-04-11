#!/usr/bin/env python3
"""
train.py - ULTRA DEBUG VERSION - Shows every step
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
print("ULTRA DEBUG VERSION")
print("=" * 60)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--model", type=str, required=True)
    return parser.parse_args()

args = parse_args()
print(f"Args: fold={args.fold}, lr={args.lr}, model={args.model}")

# Config
CONFIG['sdf_model']['signal']['residual_vol_penalty'] = args.lr
CONFIG['sdf_model']['signal']['factor_exposure_weight'] = args.lr * 3

# Use minimal data
start_date = '2025-03-01'
end_date = '2025-04-01'
window_size = 20
top_n = 3
max_windows = 3

print(f"\nTesting: {start_date} to {end_date}, window={window_size}, max_windows={max_windows}")

# Test Equity
print("\n" + "="*60)
print("EQUITY TEST")
assets = CONFIG['universes']['equity']['assets']
benchmark = CONFIG['universes']['equity']['benchmark']

print(f"Assets: {len(assets)}")
print(f"Benchmark: {benchmark}")

engine = BacktestEngine(assets, benchmark, hf_token=os.getenv("HF_TOKEN"))
results_df = engine.run_rolling_window(start_date, end_date, window_size, top_n, max_windows=max_windows)

print(f"\nResults DataFrame: {len(results_df)} rows")
print(f"Columns: {list(results_df.columns)}")

if len(results_df) == 0:
    print("ERROR: No results")
    sys.exit(1)

# Check strategy returns
print(f"\nStrategy returns:")
print(results_df['strategy_return'].describe())
print(f"Sample values: {results_df['strategy_return'].head().tolist()}")

# Calculate metrics manually
strat_returns = results_df.set_index('date')['strategy_return']
print(f"\nManual calculation:")
print(f"  Count: {len(strat_returns)}")
print(f"  Mean: {strat_returns.mean()}")
print(f"  Sum: {strat_returns.sum()}")
print(f"  Prod of (1+r): {(1 + strat_returns).prod()}")

total_return = (1 + strat_returns).prod() - 1
n_days = len(strat_returns)
annual_return = total_return * (252.0 / n_days) if n_days > 0 else 0
volatility = strat_returns.std() * np.sqrt(252)
sharpe = annual_return / volatility if volatility > 0 else 0

print(f"\nCalculated metrics:")
print(f"  Total return: {total_return}")
print(f"  Annual return: {annual_return}")
print(f"  Volatility: {volatility}")
print(f"  Sharpe: {sharpe}")

# Build record
final = results_df.iloc[-1]
record = {
    "fold": int(args.fold),
    "learning_rate": float(args.lr),
    "model_type": str(args.model),
    "timestamp": datetime.now().isoformat(),
    "date": final['date'].strftime('%Y-%m-%d'),
    "benchmark": benchmark,
    "sharpe_ratio": float(sharpe),
    "annual_return": float(annual_return),
    "volatility": float(volatility),
    "max_drawdown": 0.0,
    "win_rate": float((strat_returns > 0).mean()),
    "total_return": float(total_return),
    "top_etfs": json.dumps(final['selected_assets']),
    "forecasted_returns": json.dumps(final['forecasted_returns']),
    "scores": json.dumps(final['scores']),
}

print(f"\nRecord to save:")
for k, v in record.items():
    if k not in ['forecasted_returns', 'scores', 'top_etfs']:
        print(f"  {k}: {v} (type: {type(v)})")

# Save
token = os.getenv("HF_TOKEN")
api = HfApi()
repo_id = "P2SAMAPA/p2-etf-sdf-engine-results"

try:
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True, token=token)
except:
    pass

# Delete old file first to ensure clean state
try:
    api.delete_file("debug_equity.parquet", repo_id=repo_id, repo_type="dataset", token=token)
    print("Deleted old debug file")
except:
    pass

# Save new
df = pd.DataFrame([record])
print(f"\nDataFrame to save:")
print(df[['sharpe_ratio', 'annual_return', 'max_drawdown']])

ds = Dataset.from_pandas(df)
buffer = BytesIO()
ds.to_parquet(buffer)
buffer.seek(0)

api.upload_file(
    path_or_fileobj=buffer,
    path_in_repo="debug_equity.parquet",
    repo_id=repo_id,
    repo_type="dataset",
    token=token
)

print("\nSAVED to debug_equity.parquet")

# Verify by downloading back
print("\nVerifying download...")
from huggingface_hub import hf_hub_download
downloaded = hf_hub_download(repo_id, "debug_equity.parquet", repo_type="dataset", token=token, force_download=True)
verify_df = pd.read_parquet(downloaded)
print(f"Verified DataFrame:")
print(verify_df[['sharpe_ratio', 'annual_return', 'max_drawdown']])
print(f"Values: Sharpe={verify_df['sharpe_ratio'].iloc[0]}, AnnRet={verify_df['annual_return'].iloc[0]}")

print("\n" + "="*60)
print("DONE")
