#!/usr/bin/env python3
"""
train.py - Production training with backtest metrics
Outputs two Parquet files: equity_results.parquet and fi_results.parquet
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
print("train.py - Production")
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
    """Compute metrics with overflow protection."""
    if len(strategy_returns) == 0:
        return {k: 0.0 for k in ['sharpe_ratio', 'annual_return', 'volatility',
                                 'max_drawdown', 'win_rate', 'total_return']}
    
    # Clean returns
    clean_returns = strategy_returns.replace([np.inf, -np.inf], np.nan).fillna(0)
    clean_returns = clean_returns.clip(-0.99, 1.0)
    
    # Calculate
    total_return = (1 + clean_returns).prod() - 1
    if not np.isfinite(total_return):
        total_return = 0.0
    
    total_return = float(np.clip(total_return, -0.9999, 10.0))
    
    n_days = len(clean_returns)
    years = n_days / 252.0
    
    if years >= 0.1:
        try:
            annual_return = (1 + total_return) ** (1 / years) - 1
            if not np.isfinite(annual_return):
                annual_return = total_return * (252.0 / n_days)
        except:
            annual_return = total_return * (252.0 / n_days)
    elif n_days > 0:
        annual_return = total_return * (252.0 / n_days)
    else:
        annual_return = 0.0
    
    if not np.isfinite(annual_return):
        annual_return = 0.0
    annual_return = float(np.clip(annual_return, -1.0, 5.0))
    
    volatility = clean_returns.std() * np.sqrt(252)
    volatility = float(volatility) if np.isfinite(volatility) else 0.0
    
    if volatility > 1e-10:
        sharpe = annual_return / volatility
    else:
        sharpe = 10.0 if annual_return > 0 else (-10.0 if annual_return < 0 else 0.0)
    sharpe = float(np.clip(sharpe, -10, 10))
    
    cum = (1 + clean_returns).cumprod()
    rolling_max = cum.cummax()
    drawdown = (cum - rolling_max) / rolling_max
    max_drawdown = float(drawdown.min()) if len(drawdown) > 0 else 0.0
    win_rate = float((clean_returns > 0).mean())
    
    print(f"  [{universe_name}] AnnRet={annual_return:.4f} ({annual_return*100:.2f}%), Sharpe={sharpe:.4f}")
    
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
    print(f"\n[{universe_name}] Backtest: {len(assets)} assets, {start_date} to {end_date}")
    sys.stdout.flush()
    
    engine = BacktestEngine(assets, benchmark, hf_token=os.getenv("HF_TOKEN"))
    results_df = engine.run_rolling_window(start_date, end_date, window_size, top_n, max_windows=max_windows)
    
    if len(results_df) == 0:
        print(f"[{universe_name}] ERROR: No results")
        return None, None
    
    print(f"[{universe_name}] Got {len(results_df)} results")
    
    _, _, benchmark_returns = engine.prepare_data(start_date, end_date)
    strat_returns = results_df.set_index('date')['strategy_return']
    metrics = safe_metrics(strat_returns, benchmark_returns, universe_name)
    final_signals = results_df.iloc[-1] if len(results_df) > 0 else None
    
    return metrics, final_signals


def save_universe_to_parquet(repo_id, filename, record, token):
    """Save universe record to HF as Parquet."""
    try:
        from datasets import load_dataset
        
        try:
            existing_ds = load_dataset(repo_id, data_files=filename, split="train", token=token)
            existing_df = existing_ds.to_pandas()
        except:
            existing_df = pd.DataFrame()
        
        new_df = pd.DataFrame([record])
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        
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
        print(f"Saved {filename}: Sharpe={record['sharpe_ratio']:.4f}, AnnRet={record['annual_return']:.4f}")
        return True
        
    except Exception as e:
        print(f"Error saving {filename}: {e}")
        return False


def main():
    args = parse_args()
    print(f"Args: fold={args.fold}, lr={args.lr}, model={args.model}")
    
    is_ci = os.getenv('CI_MODE', '').lower() == 'true'
    max_windows = os.getenv('MAX_WINDOWS')
    max_windows = int(max_windows) if max_windows else None
    
    override_config(args.lr)
    np.random.seed(args.fold)
    
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
    eq_metrics, eq_final = run_backtest(eq_assets, eq_bench, start_date, end_date, window_size, top_n, max_windows, "EQUITY")
    if eq_metrics is None:
        sys.exit(1)
    
    # FI
    fi_assets = CONFIG['universes']['fi_commodities']['assets']
    fi_bench = CONFIG['universes']['fi_commodities']['benchmark']
    fi_metrics, fi_final = run_backtest(fi_assets, fi_bench, start_date, end_date, window_size, top_n, max_windows, "FI")
    if fi_metrics is None:
        sys.exit(1)
    
    # Build records
    timestamp = datetime.now().isoformat()
    
    def build_record(metrics, final, universe, benchmark):
        forecasted = final.get('forecasted_returns', {}) if final is not None else {}
        scores = final.get('scores', {}) if final is not None else {}
        
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
            "forecasted_returns": json.dumps({str(k): float(v) for k, v in forecasted.items()}),
            "scores": json.dumps({str(k): float(v) for k, v in scores.items()}),
        }
    
    eq_record = build_record(eq_metrics, eq_final, "equity", eq_bench)
    fi_record = build_record(fi_metrics, fi_final, "fi_commodity", fi_bench)
    
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
    
    print("\nSaving...")
    save_universe_to_parquet(repo_id, "equity_results.parquet", eq_record, token)
    save_universe_to_parquet(repo_id, "fi_results.parquet", fi_record, token)
    
    print("\n=== Training complete ===")


if __name__ == "__main__":
    main()
