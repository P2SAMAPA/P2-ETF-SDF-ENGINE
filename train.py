#!/usr/bin/env python3
"""
train.py - Parallel matrix training with backtest metrics
Outputs Parquet files to P2SAMAPA/p2-etf-sdf-engine-results
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


def safe_metrics(strategy_returns, benchmark_returns):
    """Compute metrics with overflow protection."""
    print("Computing performance metrics...")
    
    if len(strategy_returns) == 0:
        print("WARNING: Empty strategy returns")
        return {k: 0.0 for k in ['sharpe_ratio', 'annual_return', 'volatility',
                                 'max_drawdown', 'win_rate', 'total_return']}
    
    common_idx = strategy_returns.index.intersection(benchmark_returns.index)
    r = strategy_returns.loc[common_idx]
    
    if len(r) == 0:
        print("WARNING: No overlapping dates")
        return {k: 0.0 for k in ['sharpe_ratio', 'annual_return', 'volatility',
                                 'max_drawdown', 'win_rate', 'total_return']}
    
    total_return = (1 + r).prod() - 1
    total_return = float(np.clip(total_return, -0.9999, 100.0))
    
    n_days = len(r)
    years = n_days / 252.0
    
    print(f"  Period: {n_days} days ({years:.3f} years), Total return: {total_return:.4f}")
    
    # Annualization
    if years >= 1.0:
        annual_return = (1 + total_return) ** (1 / years) - 1
    elif n_days > 0:
        annual_return = total_return * (252.0 / n_days)
    else:
        annual_return = 0.0
    
    if not np.isfinite(annual_return):
        annual_return = 0.0
    
    annual_return = float(np.clip(annual_return, -1.0, 10.0))
    
    volatility = r.std() * np.sqrt(252)
    volatility = float(volatility) if np.isfinite(volatility) else 0.0
    
    sharpe = annual_return / (volatility + 1e-8) if volatility > 0 else 0.0
    sharpe = float(np.clip(sharpe, -10, 10))
    
    cum = (1 + r).cumprod()
    rolling_max = cum.cummax()
    drawdown = (cum - rolling_max) / rolling_max
    max_drawdown = float(drawdown.min()) if len(drawdown) > 0 else 0.0
    win_rate = float((r > 0).mean())
    
    print(f"  Results: AnnRet={annual_return:.4f}, Sharpe={sharpe:.3f}, MaxDD={max_drawdown:.4f}")
    
    return {
        'sharpe_ratio': sharpe,
        'annual_return': annual_return,
        'volatility': volatility,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'total_return': total_return,
    }


def run_backtest(assets, benchmark, start_date, end_date, window_size, top_n, max_windows=None):
    """Run rolling window backtest."""
    print(f"\nBacktest: {len(assets)} assets, benchmark={benchmark}")
    print(f"Range: {start_date} to {end_date}, window={window_size}, top_n={top_n}")
    if max_windows:
        print(f"CI MODE: {max_windows} windows")
    sys.stdout.flush()
    
    engine = BacktestEngine(assets, benchmark, hf_token=os.getenv("HF_TOKEN"))
    results_df = engine.run_rolling_window(start_date, end_date, window_size, top_n, max_windows=max_windows)
    
    if len(results_df) == 0:
        print("ERROR: No results")
        return None, None
    
    _, _, benchmark_returns = engine.prepare_data(start_date, end_date)
    strat_returns = results_df.set_index('date')['strategy_return']
    
    print(f"Strategy returns: {len(strat_returns)} periods")
    
    metrics = safe_metrics(strat_returns, benchmark_returns)
    final_signals = results_df.iloc[-1] if len(results_df) > 0 else None
    
    if final_signals is not None:
        scores = final_signals.get('scores', {})
        print(f"Final scores count: {len(scores)}")
    
    return metrics, final_signals


def save_to_hf_parquet(repo_id, record, token):
    """
    Save record to HF as Parquet dataset.
    Appends to existing dataset or creates new one.
    """
    try:
        # Try to load existing dataset
        from datasets import load_dataset
        try:
            existing_ds = load_dataset(repo_id, split="train", token=token)
            existing_df = existing_ds.to_pandas()
            print(f"Loaded existing dataset: {len(existing_df)} rows")
        except:
            existing_df = pd.DataFrame()
            print("Creating new dataset")
        
        # Create new row
        new_df = pd.DataFrame([record])
        
        # Combine
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        
        # Convert to Dataset and push
        ds = Dataset.from_pandas(combined_df)
        ds.push_to_hub(repo_id, token=token, split="train")
        print(f"Saved to {repo_id}: {len(combined_df)} total rows")
        return True
        
    except Exception as e:
        print(f"Error saving to HF: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    args = parse_args()
    print(f"Args: fold={args.fold}, lr={args.lr}, model={args.model}")
    
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
    
    # Equity
    eq_assets = CONFIG['universes']['equity']['assets']
    eq_bench = CONFIG['universes']['equity']['benchmark']
    eq_metrics, eq_final = run_backtest(eq_assets, eq_bench, start_date, end_date, window_size, top_n, max_windows)
    if eq_metrics is None:
        sys.exit(1)
    
    # FI/Commodity
    fi_assets = CONFIG['universes']['fi_commodities']['assets']
    fi_bench = CONFIG['universes']['fi_commodities']['benchmark']
    fi_metrics, fi_final = run_backtest(fi_assets, fi_bench, start_date, end_date, window_size, top_n, max_windows)
    if fi_metrics is None:
        sys.exit(1)
    
    # Build records
    timestamp = datetime.now().isoformat()
    
    def build_record(metrics, final, universe, benchmark):
        # Convert dicts to JSON strings for Parquet compatibility
        forecasted = final.get('forecasted_returns', {}) if final is not None else {}
        scores = final.get('scores', {}) if final is not None else {}
        
        # Ensure simple float values
        clean_scores = {str(k): float(v) if isinstance(v, (int, float, np.number)) and np.isfinite(v) else 0.0 
                       for k, v in scores.items()}
        clean_forecasted = {str(k): float(v) if isinstance(v, (int, float, np.number)) and np.isfinite(v) else 0.0 
                          for k, v in forecasted.items()}
        
        return {
            "fold": int(args.fold),
            "learning_rate": float(args.lr),
            "model_type": str(args.model),
            "timestamp": timestamp,
            "date": final['date'].strftime('%Y-%m-%d') if final is not None else None,
            "universe": str(universe),
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
    
    equity_record = build_record(eq_metrics, eq_final, "equity", eq_bench)
    fi_record = build_record(fi_metrics, fi_final, "fi_commodity", fi_bench)
    
    # Upload
    token = os.getenv("HF_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN not set")
        sys.exit(1)
    
    repo_id = "P2SAMAPA/p2-etf-sdf-engine-results"
    
    print("\n--- Saving Equity Results ---")
    save_to_hf_parquet(repo_id, equity_record, token)
    
    print("\n--- Saving FI Results ---")
    save_to_hf_parquet(repo_id, fi_record, token)
    
    print("\n=== Training complete ===")


if __name__ == "__main__":
    main()
