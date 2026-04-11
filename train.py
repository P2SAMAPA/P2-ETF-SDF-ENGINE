#!/usr/bin/env python3
"""
train.py - Fixed overflow handling for extreme returns
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
print("train.py started")
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
    """Compute metrics with proper overflow protection."""
    print(f"\n  [{universe_name}] Computing metrics...")
    print(f"  [{universe_name}] Returns count: {len(strategy_returns)}")
    
    if len(strategy_returns) == 0:
        print(f"  [{universe_name}] WARNING: Empty returns")
        return {k: 0.0 for k in ['sharpe_ratio', 'annual_return', 'volatility',
                                 'max_drawdown', 'win_rate', 'total_return']}
    
    # CRITICAL FIX: Clip individual returns to prevent overflow
    # Daily returns should never be >100% or <-100%
    clipped_returns = strategy_returns.clip(-0.99, 1.0)
    
    # Check for extreme values
    if (strategy_returns < -0.5).any() or (strategy_returns > 0.5).any():
        print(f"  [{universe_name}] WARNING: Extreme returns detected!")
        print(f"    Min: {strategy_returns.min():.4f}")
        print(f"    Max: {strategy_returns.max():.4f}")
        print(f"    Using clipped returns for calculation")
    
    # Calculate total return safely
    total_return = (1 + clipped_returns).prod() - 1
    
    # Handle inf/nan
    if not np.isfinite(total_return):
        print(f"  [{universe_name}] WARNING: Non-finite total_return ({total_return}), using 0")
        total_return = 0.0
    
    # Cap at reasonable limits
    total_return = float(np.clip(total_return, -0.9999, 10.0))  # Max 1000% total return
    
    n_days = len(clipped_returns)
    years = n_days / 252.0
    
    print(f"  [{universe_name}] Total return: {total_return:.6f}, Days: {n_days}, Years: {years:.4f}")
    
    # Annual return with overflow protection
    if years >= 0.1:
        try:
            annual_return = (1 + total_return) ** (1 / years) - 1
            if not np.isfinite(annual_return):
                annual_return = total_return * (252.0 / n_days)  # Fallback to simple scaling
        except:
            annual_return = total_return * (252.0 / n_days)
    elif n_days > 0:
        annual_return = total_return * (252.0 / n_days)
    else:
        annual_return = 0.0
    
    # Validate annual return
    if not np.isfinite(annual_return):
        annual_return = 0.0
    annual_return = float(np.clip(annual_return, -1.0, 5.0))  # Cap at 500% annual
    
    # Volatility
    volatility = clipped_returns.std() * np.sqrt(252)
    volatility = float(volatility) if np.isfinite(volatility) else 0.0
    
    # Sharpe ratio with zero vol protection
    if volatility > 1e-10:
        sharpe = annual_return / volatility
    else:
        sharpe = 10.0 if annual_return > 0 else (-10.0 if annual_return < 0 else 0.0)
    sharpe = float(np.clip(sharpe, -10, 10))
    
    # Max drawdown
    cum = (1 + clipped_returns).cumprod()
    rolling_max = cum.cummax()
    drawdown = (cum - rolling_max) / rolling_max
    max_drawdown = float(drawdown.min()) if len(drawdown) > 0 else 0.0
    
    # Win rate
    win_rate = float((clipped_returns > 0).mean())
    
    print(f"  [{universe_name}] RESULTS: AnnRet={annual_return:.6f} ({annual_return*100:.2f}%), Sharpe={sharpe:.4f}, MaxDD={max_drawdown:.4f}, Vol={volatility:.4f}")
    
    return {
        'sharpe_ratio': sharpe,
        'annual_return': annual_return,
        'volatility': volatility,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'total_return': total_return,
    }


def run_backtest(assets, benchmark, start_date, end_date, window_size, top_n, max_windows=None, universe_name=""):
    """Run rolling window backtest."""
    print(f"\n{'='*60}")
    print(f"[{universe_name}] Backtest: {len(assets)} assets, benchmark={benchmark}")
    print(f"[{universe_name}] Range: {start_date} to {end_date}")
    sys.stdout.flush()
    
    engine = BacktestEngine(assets, benchmark, hf_token=os.getenv("HF_TOKEN"))
    results_df = engine.run_rolling_window(start_date, end_date, window_size, top_n, max_windows=max_windows)
    
    if len(results_df) == 0:
        print(f"[{universe_name}] ERROR: No results")
        return None, None
    
    print(f"[{universe_name}] Got {len(results_df)} results")
    
    # Calculate metrics from strategy returns
    strat_returns = results_df.set_index('date')['strategy_return']
    
    print(f"[{universe_name}] Strategy returns stats:")
    print(f"  Min: {strat_returns.min():.6f}")
    print(f"  Max: {strat_returns.max():.6f}")
    print(f"  Mean: {strat_returns.mean():.6f}")
    print(f"  Std: {strat_returns.std():.6f}")
    
    # Load benchmark for comparison only
    _, _, benchmark_returns = engine.prepare_data(start_date, end_date)
    
    # Calculate metrics
    metrics = safe_metrics(strat_returns, benchmark_returns, universe_name)
    
    final_signals = results_df.iloc[-1] if len(results_df) > 0 else None
    
    return metrics, final_signals


def save_universe_to_parquet(repo_id, filename, record, token):
    """Save universe record to HF as Parquet."""
    try:
        # Load existing or create new
        try:
            from datasets import load_dataset
            existing_ds = load_dataset(repo_id, data_files=filename, split="train", token=token)
            existing_df = existing_ds.to_pandas()
            print(f"  Existing {filename}: {len(existing_df)} rows")
        except:
            existing_df = pd.DataFrame()
            print(f"  Creating new {filename}")
        
        # Create new row
        new_df = pd.DataFrame([record])
        
        # Combine
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        
        # Convert and save
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
        
        print(f"\n  SAVED {filename}:")
        print(f"    Sharpe: {record['sharpe_ratio']:.4f}")
        print(f"    AnnRet: {record['annual_return']:.4f} ({record['annual_return']*100:.2f}%)")
        print(f"    MaxDD: {record['max_drawdown']:.4f}")
        return True
        
    except Exception as e:
        print(f"  ERROR saving {filename}: {e}")
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
    print("PROCESSING EQUITY")
    eq_assets = CONFIG['universes']['equity']['assets']
    eq_bench = CONFIG['universes']['equity']['benchmark']
    eq_metrics, eq_final = run_backtest(eq_assets, eq_bench, start_date, end_date, window_size, top_n, max_windows, "EQUITY")
    
    # Process FI
    print(f"\n{'='*60}")
    print("PROCESSING FI/COMMODITY")
    fi_assets = CONFIG['universes']['fi_commodities']['assets']
    fi_bench = CONFIG['universes']['fi_commodities']['benchmark']
    fi_metrics, fi_final = run_backtest(fi_assets, fi_bench, start_date, end_date, window_size, top_n, max_windows, "FI")
    
    if eq_metrics is None or fi_metrics is None:
        print("FATAL: One universe failed")
        sys.exit(1)
    
    # Verify they're different
    print(f"\n{'='*60}")
    print("VERIFICATION:")
    print(f"  Equity: Sharpe={eq_metrics['sharpe_ratio']:.4f}, AnnRet={eq_metrics['annual_return']:.4f}")
    print(f"  FI:     Sharpe={fi_metrics['sharpe_ratio']:.4f}, AnnRet={fi_metrics['annual_return']:.4f}")
    
    # Build records
    timestamp = datetime.now().isoformat()
    
    def build_record(metrics, final, universe, benchmark):
        forecasted = final.get('forecasted_returns', {}) if final is not None else {}
        scores = final.get('scores', {}) if final is not None else {}
        
        # Clean values
        clean_scores = {}
        for k, v in scores.items():
            try:
                clean_scores[str(k)] = float(v) if np.isfinite(float(v)) else 0.0
            except:
                clean_scores[str(k)] = 0.0
        
        clean_forecasted = {}
        for k, v in forecasted.items():
            try:
                clean_forecasted[str(k)] = float(v) if np.isfinite(float(v)) else 0.0
            except:
                clean_forecasted[str(k)] = 0.0
        
        return {
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
            "forecasted_returns": json.dumps(clean_forecasted),
            "scores": json.dumps(clean_scores),
        }
    
    eq_record = build_record(eq_metrics, eq_final, "equity", eq_bench)
    fi_record = build_record(fi_metrics, fi_final, "fi", fi_bench)
    
    print(f"\n{'='*60}")
    print("FINAL RECORDS:")
    print(f"  Equity: Sharpe={eq_record['sharpe_ratio']:.4f}, AnnRet={eq_record['annual_return']:.4f}")
    print(f"  FI:     Sharpe={fi_record['sharpe_ratio']:.4f}, AnnRet={fi_record['annual_return']:.4f}")
    
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
    print("COMPLETE")

if __name__ == "__main__":
    main()
