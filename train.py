#!/usr/bin/env python3
"""
train.py - Parallel matrix training with backtest metrics
Outputs JSON files to P2SAMAPA/p2-etf-sdf-engine-results
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
from huggingface_hub.errors import HfHubHTTPError

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
    if len(strategy_returns) == 0:
        return {k: 0.0 for k in ['sharpe_ratio', 'annual_return', 'volatility',
                                 'max_drawdown', 'win_rate', 'total_return']}
    
    common_idx = strategy_returns.index.intersection(benchmark_returns.index)
    r = strategy_returns.loc[common_idx]
    
    if len(r) == 0:
        return {k: 0.0 for k in ['sharpe_ratio', 'annual_return', 'volatility',
                                 'max_drawdown', 'win_rate', 'total_return']}
    
    total_return = (1 + r).prod() - 1
    years = len(r) / 252
    
    # Clamp to prevent overflow
    total_return = np.clip(total_return, -0.9999, 100.0)
    annual_return = (1 + total_return) ** (1 / max(years, 0.01)) - 1
    annual_return = 0.0 if not np.isfinite(annual_return) else float(annual_return)
    
    volatility = r.std() * np.sqrt(252)
    sharpe = annual_return / (volatility + 1e-8) if volatility > 0 else 0.0
    sharpe = float(np.clip(sharpe, -10, 10))
    
    cum = (1 + r).cumprod()
    rolling_max = cum.cummax()
    drawdown = (cum - rolling_max) / rolling_max
    max_drawdown = float(drawdown.min())
    win_rate = float((r > 0).mean())
    
    return {
        'sharpe_ratio': sharpe,
        'annual_return': annual_return,
        'volatility': float(volatility),
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'total_return': float(total_return),
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
    metrics = safe_metrics(strat_returns, benchmark_returns)
    final_signals = results_df.iloc[-1] if len(results_df) > 0 else None
    
    print(f"Completed: Sharpe={metrics['sharpe_ratio']:.3f}, Return={metrics['annual_return']:.4f}")
    return metrics, final_signals


def convert_scores(scores_dict):
    """Flatten scores to simple float dict."""
    if not scores_dict:
        return {}
    
    result = {}
    for k, v in scores_dict.items():
        key = str(k)
        if isinstance(v, dict):
            # Take 'score' or first numeric value
            val = v.get('score', next((x for x in v.values() if isinstance(x, (int, float))), 0))
            result[key] = float(val) if np.isfinite(val) else 0.0
        elif isinstance(v, (int, float)):
            result[key] = float(v) if np.isfinite(v) else 0.0
        else:
            result[key] = 0.0
    return result


def save_json_to_hf(api, repo_id, filename, data, token):
    """Upload JSON to HF dataset."""
    json_str = json.dumps(data, indent=2, default=str)
    json_bytes = json_str.encode('utf-8')
    
    for attempt in range(5):
        try:
            api.upload_file(
                path_or_fileobj=BytesIO(json_bytes),
                path_in_repo=filename,
                repo_id=repo_id,
                repo_type="dataset",
                token=token
            )
            print(f"Saved: {filename}")
            return True
        except Exception as e:
            wait = 2 ** attempt
            print(f"Retry {attempt+1} in {wait}s: {e}")
            time.sleep(wait)
    
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
        return {
            "fold": int(args.fold),
            "learning_rate": float(args.lr),
            "model_type": str(args.model),
            "timestamp": timestamp,
            "date": final['date'].strftime('%Y-%m-%d') if final is not None else None,
            "universe": universe,
            "benchmark": benchmark,
            "sharpe_ratio": metrics['sharpe_ratio'],
            "annual_return": metrics['annual_return'],
            "volatility": metrics['volatility'],
            "max_drawdown": metrics['max_drawdown'],
            "win_rate": metrics['win_rate'],
            "total_return": metrics['total_return'],
            "top_etfs": [str(x) for x in final['selected_assets']] if final is not None else [],
            "forecasted_returns": {str(k): float(v) for k, v in final['forecasted_returns'].items()} if final is not None else {},
            "scores": convert_scores(final['scores']) if final is not None else {},
        }
    
    equity_record = build_record(eq_metrics, eq_final, "equity", eq_bench)
    fi_record = build_record(fi_metrics, fi_final, "fi_commodity", fi_bench)
    
    # Upload
    token = os.getenv("HF_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN not set")
        sys.exit(1)
    
    api = HfApi()
    repo_id = "P2SAMAPA/p2-etf-sdf-engine-results"
    
    try:
        api.create_repo(repo_id, repo_type="dataset", exist_ok=True, token=token)
    except:
        pass
    
    equity_file = f"equity_fold{args.fold}_lr{args.lr}_{args.model}.json"
    fi_file = f"fi_fold{args.fold}_lr{args.lr}_{args.model}.json"
    
    save_json_to_hf(api, repo_id, equity_file, equity_record, token)
    save_json_to_hf(api, repo_id, fi_file, fi_record, token)
    
    print("\n=== Training complete ===")

if __name__ == "__main__":
    main()
