#!/usr/bin/env python3
"""
train.py - EMERGENCY DEBUG VERSION
"""

import os
import sys
import argparse
import time
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

warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels.tsa.base.tsa_model")

print("=" * 60)
print("train.py EMERGENCY DEBUG VERSION")
print("=" * 60)
sys.stdout.flush()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--model", type=str, required=True, choices=["rf", "xgb", "elasticnet"])
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    return parser.parse_args()


def override_config(lr: float):
    CONFIG['sdf_model']['signal']['residual_vol_penalty'] = lr
    CONFIG['sdf_model']['signal']['factor_exposure_weight'] = lr * 3
    print(f"Config: residual_vol_penalty={lr}, factor_exposure_weight={lr*3}")


def safe_metrics(strategy_returns, benchmark_returns, universe_name=""):
    """Compute metrics with EMERGENCY logging."""
    print(f"\n{'='*40}")
    print(f"[{universe_name}] METRICS CALCULATION START")
    print(f"[{universe_name}] Input returns count: {len(strategy_returns)}")
    
    if len(strategy_returns) == 0:
        print(f"[{universe_name}] ERROR: Empty returns!")
        return {k: 0.0 for k in ['sharpe_ratio', 'annual_return', 'volatility',
                                 'max_drawdown', 'win_rate', 'total_return']}
    
    # Log raw returns stats
    print(f"[{universe_name}] Raw returns stats:")
    print(f"  Min: {strategy_returns.min()}")
    print(f"  Max: {strategy_returns.max()}")
    print(f"  Mean: {strategy_returns.mean()}")
    print(f"  Std: {strategy_returns.std()}")
    print(f"  Any NaN: {strategy_returns.isna().any()}")
    print(f"  Any Inf: {np.isinf(strategy_returns).any()}")
    
    # STEP 1: Clean returns
    clean_returns = strategy_returns.replace([np.inf, -np.inf], np.nan).fillna(0)
    clean_returns = clean_returns.clip(-0.99, 1.0)
    
    print(f"[{universe_name}] After cleaning:")
    print(f"  Min: {clean_returns.min()}")
    print(f"  Max: {clean_returns.max()}")
    
    # STEP 2: Calculate total return
    gross_returns = 1 + clean_returns
    print(f"[{universe_name}] Gross returns (1+r):")
    print(f"  Min: {gross_returns.min()}")
    print(f"  Max: {gross_returns.max()}")
    
    total_gross = gross_returns.prod()
    print(f"[{universe_name}] Total gross return: {total_gross}")
    
    total_return = total_gross - 1
    print(f"[{universe_name}] Total return (before clip): {total_return}")
    
    # Handle non-finite
    if not np.isfinite(total_return):
        print(f"[{universe_name}] WARNING: Non-finite total_return!")
        total_return = 0.0
    
    total_return = float(np.clip(total_return, -0.9999, 10.0))
    print(f"[{universe_name}] Total return (after clip): {total_return}")
    
    # STEP 3: Annualize
    n_days = len(clean_returns)
    years = n_days / 252.0
    print(f"[{universe_name}] Days: {n_days}, Years: {years}")
    
    if years > 0 and n_days > 0:
        # Simple annualization (safer)
        annual_return = total_return * (252.0 / n_days)
        print(f"[{universe_name}] Annual return (simple scaling): {annual_return}")
    else:
        annual_return = 0.0
        print(f"[{universe_name}] WARNING: No time period!")
    
    # Validate
    if not np.isfinite(annual_return):
        print(f"[{universe_name}] WARNING: Non-finite annual_return!")
        annual_return = 0.0
    
    annual_return = float(np.clip(annual_return, -2.0, 5.0))
    print(f"[{universe_name}] Annual return (final): {annual_return}")
    
    # STEP 4: Volatility
    volatility = clean_returns.std() * np.sqrt(252)
    volatility = float(volatility) if np.isfinite(volatility) else 0.0
    print(f"[{universe_name}] Volatility: {volatility}")
    
    # STEP 5: Sharpe
    if volatility > 1e-10:
        sharpe = annual_return / volatility
        print(f"[{universe_name}] Sharpe (raw): {sharpe}")
    else:
        sharpe = 0.0
        print(f"[{universe_name}] Sharpe: zero volatility, set to 0")
    
    if not np.isfinite(sharpe):
        print(f"[{universe_name}] WARNING: Non-finite sharpe!")
        sharpe = 0.0
    
    sharpe = float(np.clip(sharpe, -10, 10))
    print(f"[{universe_name}] Sharpe (final): {sharpe}")
    
    # STEP 6: Max drawdown
    cum = (1 + clean_returns).cumprod()
    rolling_max = cum.cummax()
    drawdown = (cum - rolling_max) / rolling_max
    max_drawdown = float(drawdown.min()) if len(drawdown) > 0 else 0.0
    print(f"[{universe_name}] Max drawdown: {max_drawdown}")
    
    # STEP 7: Win rate
    win_rate = float((clean_returns > 0).mean())
    print(f"[{universe_name}] Win rate: {win_rate}")
    
    # FINAL RESULTS
    result = {
        'sharpe_ratio': sharpe,
        'annual_return': annual_return,
        'volatility': volatility,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'total_return': total_return,
    }
    
    print(f"[{universe_name}] FINAL METRICS:")
    for k, v in result.items():
        print(f"  {k}: {v}")
    print(f"{'='*40}")
    
    return result


def run_backtest(assets, benchmark, start_date, end_date, window_size, top_n, max_windows=None, universe_name=""):
    """Run rolling window backtest with debug."""
    print(f"\n{'='*60}")
    print(f"[{universe_name}] BACKTEST START")
    print(f"[{universe_name}] Assets: {len(assets)}, Benchmark: {benchmark}")
    sys.stdout.flush()
    
    engine = BacktestEngine(assets, benchmark, hf_token=os.getenv("HF_TOKEN"))
    results_df = engine.run_rolling_window(start_date, end_date, window_size, top_n, max_windows=max_windows)
    
    print(f"[{universe_name}] Results shape: {results_df.shape}")
    
    if len(results_df) == 0:
        print(f"[{universe_name}] ERROR: No results!")
        return None, None
    
    # Get strategy returns
    strat_returns = results_df.set_index('date')['strategy_return']
    print(f"[{universe_name}] Strategy returns extracted: {len(strat_returns)}")
    
    # Get benchmark
    _, _, benchmark_returns = engine.prepare_data(start_date, end_date)
    
    # Calculate metrics
    metrics = safe_metrics(strat_returns, benchmark_returns, universe_name)
    
    final_signals = results_df.iloc[-1] if len(results_df) > 0 else None
    
    print(f"[{universe_name}] BACKTEST COMPLETE")
    return metrics, final_signals


def save_universe_to_parquet(repo_id, filename, record, token):
    """Save with verification."""
    print(f"\n{'='*40}")
    print(f"SAVING {filename}")
    print(f"Record contents:")
    for k, v in record.items():
        if k not in ['forecasted_returns', 'scores', 'top_etfs']:
            print(f"  {k}: {v} (type: {type(v)})")
    
    try:
        # Try to load existing
        try:
            from datasets import load_dataset
            existing_ds = load_dataset(repo_id, data_files=filename, split="train", token=token)
            existing_df = existing_ds.to_pandas()
            print(f"  Existing rows: {len(existing_df)}")
        except Exception as e:
            print(f"  No existing file (or error): {e}")
            existing_df = pd.DataFrame()
        
        # Create new row
        new_df = pd.DataFrame([record])
        print(f"  New row metrics:")
        print(f"    sharpe_ratio: {new_df['sharpe_ratio'].iloc[0]}")
        print(f"    annual_return: {new_df['annual_return'].iloc[0]}")
        print(f"    max_drawdown: {new_df['max_drawdown'].iloc[0]}")
        
        # Combine
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        print(f"  Combined rows: {len(combined_df)}")
        
        # Save
        ds = Dataset.from_pandas(combined_df)
        buffer = BytesIO()
        ds.to_parquet(buffer)
        buffer.seek(0)
        
        api = HfApi()
        api.upload_file(
            path_or_fileobj=buffer,
            path_in_repo=filename,
            repo_id=repo_id,
            repo_type="dataset",
            token=token
        )
        
        print(f"  UPLOADED SUCCESSFULLY")
        
        # Verify by re-reading
        print(f"  Verifying upload...")
        verify_path = f"/tmp/verify_{filename}"
        try:
            from huggingface_hub import hf_hub_download
            downloaded = hf_hub_download(repo_id, filename, repo_type="dataset", token=token, local_dir="/tmp", force_download=True)
            verify_df = pd.read_parquet(downloaded)
            last_row = verify_df.iloc[-1]
            print(f"  Verification - Last row metrics:")
            print(f"    sharpe_ratio: {last_row['sharpe_ratio']}")
            print(f"    annual_return: {last_row['annual_return']}")
            print(f"    max_drawdown: {last_row['max_drawdown']}")
        except Exception as e:
            print(f"  Verification failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    args = parse_args()
    print(f"\nArgs: fold={args.fold}, lr={args.lr}, model={args.model}")
    
    is_ci = os.getenv('CI_MODE', '').lower() == 'true'
    max_windows = os.getenv('MAX_WINDOWS')
    max_windows = int(max_windows) if max_windows else None
    
    override_config(args.lr)
    np.random.seed(args.fold)
    
    # Date setup
    start_date = args.start_date or CONFIG['backtest']['start_date']
    end_date = args.end_date or CONFIG['backtest']['end_date']
    
    if is_ci and not args.start_date:
        start_date = '2024-01-01'
        print(f"CI MODE: Using {start_date}")
    
    window_size = CONFIG['backtest']['window_strategies']['rolling']['window_size']
    top_n = CONFIG['backtest']['rebalance']['top_n']
    
    # Process Equity
    print(f"\n{'='*60}")
    print("EQUITY PROCESSING")
    eq_assets = CONFIG['universes']['equity']['assets']
    eq_bench = CONFIG['universes']['equity']['benchmark']
    eq_metrics, eq_final = run_backtest(eq_assets, eq_bench, start_date, end_date, window_size, top_n, max_windows, "EQUITY")
    
    # Process FI
    print(f"\n{'='*60}")
    print("FI PROCESSING")
    fi_assets = CONFIG['universes']['fi_commodities']['assets']
    fi_bench = CONFIG['universes']['fi_commodities']['benchmark']
    fi_metrics, fi_final = run_backtest(fi_assets, fi_bench, start_date, end_date, window_size, top_n, max_windows, "FI")
    
    if eq_metrics is None or fi_metrics is None:
        print("FATAL: One universe failed")
        sys.exit(1)
    
    # Build records
    timestamp = datetime.now().isoformat()
    
    def build_record(metrics, final, universe, benchmark):
        print(f"\nBuilding record for {universe}:")
        print(f"  Input metrics: {metrics}")
        
        forecasted = final.get('forecasted_returns', {}) if final is not None else {}
        scores = final.get('scores', {}) if final is not None else {}
        
        record = {
            "fold": int(args.fold),
            "learning_rate": float(args.lr),
            "model_type": str(args.model),
            "timestamp": timestamp,
            "date": final['date'].strftime('%Y-%m-%d') if final is not None else None,
            "benchmark": str(benchmark),
            "sharpe_ratio": float(metrics['sharpe_ratio']),
            "annual_return": float(metrics['annual_return']),
            "volatility": float(metrics['volatility']),
            "max_drawdown": float(metrics['max_drawdown']),
            "win_rate": float(metrics['win_rate']),
            "total_return": float(metrics['total_return']),
            "top_etfs": json.dumps([str(x) for x in final['selected_assets']]) if final is not None else "[]",
            "forecasted_returns": json.dumps({str(k): float(v) for k, v in forecasted.items()}),
            "scores": json.dumps({str(k): float(v) for k, v in scores.items()}),
        }
        
        print(f"  Record metrics:")
        print(f"    sharpe_ratio: {record['sharpe_ratio']}")
        print(f"    annual_return: {record['annual_return']}")
        print(f"    max_drawdown: {record['max_drawdown']}")
        
        return record
    
    eq_record = build_record(eq_metrics, eq_final, "equity", eq_bench)
    fi_record = build_record(fi_metrics, fi_final, "fi", fi_bench)
    
    # Upload
    token = os.getenv("HF_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN not set")
        sys.exit(1)
    
    repo_id = "P2SAMAPA/p2-etf-sdf-engine-results"
    
    api = HfApi()
    try:
        api.create_repo(repo_id, repo_type="dataset", exist_ok=True, token=token)
    except:
        pass
    
    print(f"\n{'='*60}")
    print("SAVING EQUITY...")
    save_universe_to_parquet(repo_id, "equity_results.parquet", eq_record, token)
    
    print(f"\n{'='*60}")
    print("SAVING FI...")
    save_universe_to_parquet(repo_id, "fi_results.parquet", fi_record, token)
    
    print(f"\n{'='*60}")
    print("ALL DONE")

if __name__ == "__main__":
    main()
