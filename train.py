#!/usr/bin/env python3
"""
train.py - Parallel matrix training with backtest metrics
Saves separate datasets for equity and FI/commodities.
"""

import os
import argparse
import time
import pandas as pd
import numpy as np
from datetime import datetime

from datasets import Dataset
from huggingface_hub.errors import HfHubHTTPError

from configs import CONFIG
from backtest_engine import BacktestEngine

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--model", type=str, required=True,
                        choices=["rf", "xgb", "elasticnet"])
    return parser.parse_args()

def override_config(lr: float):
    CONFIG['sdf_model']['signal']['residual_vol_penalty'] = lr
    CONFIG['sdf_model']['signal']['factor_exposure_weight'] = lr * 3

def safe_metrics(strategy_returns, benchmark_returns):
    """Compute metrics with safety checks to avoid inf/nan."""
    if len(strategy_returns) == 0:
        return {k: np.nan for k in ['sharpe_ratio', 'annual_return', 'volatility',
                                     'max_drawdown', 'win_rate', 'total_return']}
    # Align
    common = strategy_returns.index.intersection(benchmark_returns.index)
    r = strategy_returns.loc[common]
    b = benchmark_returns.loc[common]
    if len(r) == 0:
        return {k: np.nan for k in ['sharpe_ratio', 'annual_return', 'volatility',
                                     'max_drawdown', 'win_rate', 'total_return']}
    # Total return
    total_return = (1 + r).prod() - 1
    # Annualized return (252 trading days)
    years = len(r) / 252
    annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
    # Volatility
    volatility = r.std() * np.sqrt(252)
    # Sharpe (add epsilon to avoid division by zero)
    sharpe = annual_return / (volatility + 1e-8)
    # Max drawdown
    cum = (1 + r).cumprod()
    rolling_max = cum.cummax()
    drawdown = (cum - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    # Win rate
    win_rate = (r > 0).mean()
    return {
        'sharpe_ratio': sharpe,
        'annual_return': annual_return,
        'volatility': volatility,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'total_return': total_return,
    }

def run_backtest_for_universe(assets, benchmark, start_date, end_date, window_size, top_n):
    engine = BacktestEngine(assets, benchmark, hf_token=os.getenv("HF_TOKEN"))
    results_df = engine.run_rolling_window(start_date, end_date, window_size, top_n)
    _, _, benchmark_returns = engine.prepare_data(start_date, end_date)
    strat_returns = results_df.set_index('date')['strategy_return']
    metrics = safe_metrics(strat_returns, benchmark_returns)
    final_signals = results_df.iloc[-1] if len(results_df) > 0 else None
    return metrics, final_signals

def push_with_retry(dataset, dataset_name, token):
    for attempt in range(5):
        try:
            dataset.push_to_hub(dataset_name, token=token, split="train")
            print(f"Pushed to {dataset_name}")
            return
        except HfHubHTTPError as e:
            if e.response.status_code in (409, 412):
                time.sleep(2 ** attempt)
            else:
                raise
    raise Exception("Failed to push after 5 attempts")

def main():
    args = parse_args()
    override_config(args.lr)
    np.random.seed(args.fold)

    start_date = CONFIG['backtest']['start_date']
    end_date   = CONFIG['backtest']['end_date']
    window_size = CONFIG['backtest']['window_strategies']['rolling']['window_size']
    top_n = CONFIG['backtest']['rebalance']['top_n']

    # Equity
    eq_assets = CONFIG['universes']['equity']['assets']
    eq_bench = CONFIG['universes']['equity']['benchmark']
    eq_metrics, eq_final = run_backtest_for_universe(eq_assets, eq_bench, start_date, end_date, window_size, top_n)

    # FI/Commodity
    fi_assets = CONFIG['universes']['fi_commodities']['assets']
    fi_bench = CONFIG['universes']['fi_commodities']['benchmark']
    fi_metrics, fi_final = run_backtest_for_universe(fi_assets, fi_bench, start_date, end_date, window_size, top_n)

    # Prepare records
    equity_record = {
        "fold": args.fold, "learning_rate": args.lr, "model_type": args.model,
        "timestamp": datetime.now().isoformat(),
        "date": eq_final['date'].strftime('%Y-%m-%d') if eq_final is not None else None,
        "sharpe_ratio": eq_metrics['sharpe_ratio'],
        "annual_return": eq_metrics['annual_return'],
        "volatility": eq_metrics['volatility'],
        "max_drawdown": eq_metrics['max_drawdown'],
        "win_rate": eq_metrics['win_rate'],
        "total_return": eq_metrics['total_return'],
        "top_etfs": eq_final['selected_assets'] if eq_final is not None else [],
        "forecasted_returns": eq_final['forecasted_returns'] if eq_final is not None else {},
        "scores": eq_final['scores'] if eq_final is not None else {},
    }

    fi_record = {
        "fold": args.fold, "learning_rate": args.lr, "model_type": args.model,
        "timestamp": datetime.now().isoformat(),
        "date": fi_final['date'].strftime('%Y-%m-%d') if fi_final is not None else None,
        "sharpe_ratio": fi_metrics['sharpe_ratio'],
        "annual_return": fi_metrics['annual_return'],
        "volatility": fi_metrics['volatility'],
        "max_drawdown": fi_metrics['max_drawdown'],
        "win_rate": fi_metrics['win_rate'],
        "total_return": fi_metrics['total_return'],
        "top_etfs": fi_final['selected_assets'] if fi_final is not None else [],
        "forecasted_returns": fi_final['forecasted_returns'] if fi_final is not None else {},
        "scores": fi_final['scores'] if fi_final is not None else {},
    }

    token = os.getenv("HF_TOKEN")
    # Push equity results
    eq_ds = Dataset.from_pandas(pd.DataFrame([equity_record]))
    push_with_retry(eq_ds, "P2SAMAPA/p2-etf-sdf-engine-results-equity", token)
    # Push FI results
    fi_ds = Dataset.from_pandas(pd.DataFrame([fi_record]))
    push_with_retry(fi_ds, "P2SAMAPA/p2-etf-sdf-engine-results-fi", token)

    print("Done.")

if __name__ == "__main__":
    main()
