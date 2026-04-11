import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import shutil
from datetime import datetime
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="SDF Engine Debug", layout="wide")

st.title("🔍 DEBUG: Check Parquet File Contents")

def get_hf_token():
    return st.secrets.get("HF_TOKEN") or os.getenv("HF_TOKEN")

def load_raw_parquet(filename):
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

col1, col2 = st.columns(2)

with col1:
    st.header("Equity File")
    eq_df = load_raw_parquet("equity_results.parquet")
    if eq_df is not None:
        st.write(f"Shape: {eq_df.shape}")
        st.write("All columns:", list(eq_df.columns))
        
        if len(eq_df) > 0:
            st.subheader("Last row (most recent):")
            last_row = eq_df.iloc[-1]
            
            st.write("Raw metric values:")
            for col in ['sharpe_ratio', 'annual_return', 'max_drawdown', 'volatility', 'win_rate', 'total_return']:
                val = last_row[col]
                st.write(f"  {col}: {val} (type: {type(val)})")
            
            st.subheader("JSON fields:")
            st.write("top_etfs:", json.loads(last_row['top_etfs']))
            st.write("forecasted_returns:", json.loads(last_row['forecasted_returns']))
            
            # Check if all metrics are zero
            metrics = [last_row['sharpe_ratio'], last_row['annual_return'], last_row['max_drawdown']]
            if all(abs(m) < 0.0001 for m in metrics):
                st.error("🚨 ALL METRICS ARE ZERO!")
                st.info("This means the training calculated zeros or failed to calculate properly.")
        else:
            st.error("File is empty!")

with col2:
    st.header("FI File")
    fi_df = load_raw_parquet("fi_results.parquet")
    if fi_df is not None:
        st.write(f"Shape: {fi_df.shape}")
        st.write("All columns:", list(fi_df.columns))
        
        if len(fi_df) > 0:
            st.subheader("Last row (most recent):")
            last_row = fi_df.iloc[-1]
            
            st.write("Raw metric values:")
            for col in ['sharpe_ratio', 'annual_return', 'max_drawdown', 'volatility', 'win_rate', 'total_return']:
                val = last_row[col]
                st.write(f"  {col}: {val} (type: {type(val)})")
            
            st.subheader("JSON fields:")
            st.write("top_etfs:", json.loads(last_row['top_etfs']))
            st.write("forecasted_returns:", json.loads(last_row['forecasted_returns']))
            
            # Check if all metrics are zero
            metrics = [last_row['sharpe_ratio'], last_row['annual_return'], last_row['max_drawdown']]
            if all(abs(m) < 0.0001 for m in metrics):
                st.error("🚨 ALL METRICS ARE ZERO!")
        else:
            st.error("File is empty!")

# Comparison
if eq_df is not None and fi_df is not None and len(eq_df) > 0 and len(fi_df) > 0:
    st.header("Comparison")
    
    eq_last = eq_df.iloc[-1]
    fi_last = fi_df.iloc[-1]
    
    st.write("Equity vs FI (last row):")
    comparison_data = {
        'Metric': ['sharpe_ratio', 'annual_return', 'max_drawdown', 'volatility', 'win_rate', 'total_return'],
        'Equity': [eq_last[col] for col in ['sharpe_ratio', 'annual_return', 'max_drawdown', 'volatility', 'win_rate', 'total_return']],
        'FI': [fi_last[col] for col in ['sharpe_ratio', 'annual_return', 'max_drawdown', 'volatility', 'win_rate', 'total_return']]
    }
    st.table(pd.DataFrame(comparison_data))

st.markdown("---")
st.header("🔧 Next Steps")

if eq_df is not None and len(eq_df) > 0:
    eq_metrics = [eq_df.iloc[-1]['sharpe_ratio'], eq_df.iloc[-1]['annual_return'], eq_df.iloc[-1]['max_drawdown']]
    if all(abs(m) < 0.0001 for m in eq_metrics):
        st.error("""
        **All metrics are zero!** This means:
        
        1. Check GitHub Actions training logs - did it complete successfully?
        2. Look for lines starting with `[EQUITY] RESULTS:` and `[FI] RESULTS:`
        3. If those show non-zero values, the save failed
        4. If those show zeros too, the calculation failed
        
        Common causes:
        - `safe_metrics()` received empty returns
        - Division by zero in calculation
        - All strategy returns were NaN
        """)
    else:
        st.success("Metrics look good! The issue is in the display app.")
