#!/usr/bin/env python3
"""
train.py - Parallel matrix training for P2-ETF-SDF-ENGINE
Stores ETF predictions (selected assets, scores) into Hugging Face dataset.
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
from equity_engine import EquityEngine
from fi_commodity_engine import FICommodityEngine

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--model", type=str, required=True,
                        choices=["rf", "xgb", "elasticnet"])
    return parser.parse_args()

def override_config(lr: float):
    CONFIG['sdf_model']['signal']['residual_vol_penalty'] = lr
    CONFIG['sdf_model']['signal']['factor_exposure_weight'] = lr * 3

def get_latest_predictions(engine_class, assets, benchmark, lr, window_size=252, top_n=3):
    """Run engine on latest window and return predictions."""
    engine = engine_class(hf_token=os.getenv("HF_TOKEN"))
    returns, macro, _ = engine.prepare_data(
        start_date=CONFIG['backtest']['start_date'],
        end_date=datetime.now().strftime('%Y-%m-%d')
    )
    # Use most recent window
    train_returns = returns.iloc[-window_size:]
    train_macro = macro.loc[train_returns.index]
    result = engine.generate_signals_pipeline(train_returns, train_macro, top_n)
    return result

def push_with_retry(dataset, dataset_name, token, max_retries=5):
    for attempt in range(max_retries):
        try:
            dataset.push_to_hub(dataset_name, token=token, split="train")
            return True
        except HfHubHTTPError as e:
            if e.response.status_code in (409, 412):
                time.sleep(2 ** attempt)
            else:
                raise
    raise Exception("Failed after retries")

def main():
    args = parse_args()
    override_config(args.lr)
    np.random.seed(args.fold)

    # Get equity predictions
    equity_result = get_latest_predictions(
        EquityEngine,
        CONFIG['universes']['equity']['assets'],
        CONFIG['universes']['equity']['benchmark'],
        args.lr,
        top_n=3
    )

    # Get FI/Commodity predictions
    fi_result = get_latest_predictions(
        FICommodityEngine,
        CONFIG['universes']['fi_commodities']['assets'],
        CONFIG['universes']['fi_commodities']['benchmark'],
        args.lr,
        top_n=3
    )

    # Build prediction records
    equity_selected = equity_result['signals'][['asset', 'expected_return', 'composite_score']].to_dict(orient='records')
    fi_selected = fi_result['signals'][['asset', 'expected_return', 'composite_score']].to_dict(orient='records')

    result_record = {
        "fold": args.fold,
        "learning_rate": args.lr,
        "model_type": args.model,
        "timestamp": datetime.now().isoformat(),
        "date": equity_result['date'].strftime('%Y-%m-%d'),
        "equity_top_etfs": [r['asset'] for r in equity_selected],
        "equity_forecasted_returns": {r['asset']: r['expected_return'] for r in equity_selected},
        "equity_scores": {r['asset']: r['composite_score'] for r in equity_selected},
        "equity_n_factors": equity_result['n_factors'],
        "fi_top_etfs": [r['asset'] for r in fi_selected],
        "fi_forecasted_returns": {r['asset']: r['expected_return'] for r in fi_selected},
        "fi_scores": {r['asset']: r['composite_score'] for r in fi_selected},
        "fi_n_factors": fi_result['n_factors'],
    }

    # Load existing dataset or create new
    try:
        existing = load_dataset(args.output, split="train", token=os.getenv("HF_TOKEN"))
        combined_df = pd.concat([existing.to_pandas(), pd.DataFrame([result_record])], ignore_index=True)
        final_ds = Dataset.from_pandas(combined_df)
    except:
        final_ds = Dataset.from_pandas(pd.DataFrame([result_record]))

    push_with_retry(final_ds, args.output, os.getenv("HF_TOKEN"))
    print("Saved ETF predictions to HF dataset.")

if __name__ == "__main__":
    main()
