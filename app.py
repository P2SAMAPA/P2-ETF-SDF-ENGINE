import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import shutil
from datetime import datetime
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="SDF Engine Debug", layout="wide")

# Clear cache completely
cache_dir = os.path.expanduser("~/.cache/huggingface")
if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir, ignore_errors=True)

st.title("DEBUG: Compare Equity vs FI Data")

def get_hf_token():
    return st.secrets.get("HF_TOKEN") or os.getenv("HF_TOKEN")

def load_raw_parquet(filename):
    """Load parquet and return raw dataframe."""
    token = get_hf_token()
    try:
        file_path = hf_hub_download(
            repo_id="P2SAMAPA/p2-etf-sdf-engine-results",
            filename=filename,
            repo_type="dataset",
            token=token,
            force_download=True
        )
        return pd.read_parquet(file_path)
    except Exception as e:
        st.error(f"Error loading {filename}: {e}")
        return None

# Load both files
st.subheader("Raw Data Comparison")
col1, col2 = st.columns(2)

with col1:
    st.markdown("**Equity File**")
    eq_df = load_raw_parquet("equity_results.parquet")
    if eq_df is not None:
        st.write(f"Shape: {eq_df.shape}")
        st.write("Columns:", list(eq_df.columns))
        st.write("Raw values:")
        st.dataframe(eq_df[['sharpe_ratio', 'annual_return', 'max_drawdown', 'win_rate', 'total_return']])
        
        # Show JSON fields
        if len(eq_df) > 0:
            st.write("Top ETFs:", json.loads(eq_df.iloc[0]['top_etfs']))
            st.write("Date:", eq_df.iloc[0]['date'])
            st.write("Benchmark:", eq_df.iloc[0]['benchmark'])

with col2:
    st.markdown("**FI File**")
    fi_df = load_raw_parquet("fi_results.parquet")
    if fi_df is not None:
        st.write(f"Shape: {fi_df.shape}")
        st.write("Columns:", list(fi_df.columns))
        st.write("Raw values:")
        st.dataframe(fi_df[['sharpe_ratio', 'annual_return', 'max_drawdown', 'win_rate', 'total_return']])
        
        # Show JSON fields
        if len(fi_df) > 0:
            st.write("Top ETFs:", json.loads(fi_df.iloc[0]['top_etfs']))
            st.write("Date:", fi_df.iloc[0]['date'])
            st.write("Benchmark:", fi_df.iloc[0]['benchmark'])

# Check if they're identical
if eq_df is not None and fi_df is not None:
    st.subheader("Data Comparison")
    
    if eq_df.shape == fi_df.shape:
        # Compare metric columns
        metric_cols = ['sharpe_ratio', 'annual_return', 'max_drawdown']
        are_identical = True
        
        for col in metric_cols:
            if col in eq_df.columns and col in fi_df.columns:
                eq_val = eq_df[col].iloc[0]
                fi_val = fi_df[col].iloc[0]
                if abs(eq_val - fi_val) > 1e-10:
                    are_identical = False
                    st.write(f"❌ {col} DIFFERENT: Equity={eq_val}, FI={fi_val}")
                else:
                    st.write(f"✅ {col} SAME: {eq_val}")
        
        if are_identical:
            st.error("🚨 FILES CONTAIN IDENTICAL DATA!")
            st.info("This means the training saved the same metrics to both files, or the same data was copied to both.")
    else:
        st.write(f"Different shapes: Equity {eq_df.shape} vs FI {fi_df.shape}")

# Now render the actual UI
st.markdown("---")
st.title("SDF Engine – Actual Display")

# Check cache issue
st.write("Cache cleared:", not os.path.exists(cache_dir))

def get_best_record(df):
    if df is None or len(df) == 0:
        return None
    idx = df['sharpe_ratio'].idxmax()
    row = df.loc[idx]
    return {
        'sharpe': float(row['sharpe_ratio']),
        'ann_ret': float(row['annual_return']),
        'max_dd': float(row['max_drawdown']),
        'top_etfs': json.loads(row['top_etfs']),
        'forecasted': json.loads(row['forecasted_returns']),
        'scores': json.loads(row['scores'])
    }

eq_best = get_best_record(eq_df)
fi_best = get_best_record(fi_df)

col3, col4 = st.columns(2)

with col3:
    st.header("Equity Display")
    if eq_best:
        st.write(f"Sharpe: {eq_best['sharpe']:.6f}")
        st.write(f"AnnRet: {eq_best['ann_ret']:.6f}")
        st.write(f"MaxDD: {eq_best['max_dd']:.6f}")

with col4:
    st.header("FI Display")
    if fi_best:
        st.write(f"Sharpe: {fi_best['sharpe']:.6f}")
        st.write(f"AnnRet: {fi_best['ann_ret']:.6f}")
        st.write(f"MaxDD: {fi_best['max_dd']:.6f}")

# Check if display values match
if eq_best and fi_best:
    if abs(eq_best['sharpe'] - fi_best['sharpe']) < 0.01:
        st.error("Display shows same Sharpe!")
    if abs(eq_best['ann_ret'] - fi_best['ann_ret']) < 0.01:
        st.error("Display shows same AnnRet!")
