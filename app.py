import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime
from pandas.tseries.offsets import CustomBusinessDay
from pandas.tseries.holiday import USFederalHolidayCalendar
from datasets import load_dataset

st.set_page_config(page_title="SDF Engine", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    [data-testid="stSidebar"] { display: none; }
    .hero-card {
        background: linear-gradient(135deg, #1E3A5F 0%, #2C5282 100%);
        border-radius: 20px; padding: 2rem; color: white; margin-bottom: 1rem;
    }
    .hero-ticker { font-size: 3rem; font-weight: 800; }
    .hero-return { font-size: 2rem; font-weight: 600; }
    .metric-card {
        background: white; border-radius: 12px; padding: 1rem; text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    .metric-value { font-size: 1.8rem; font-weight: 700; color: #1E3A5F; }
    .footer { text-align: center; margin-top: 2rem; font-size: 0.8rem; color: #718096; }
</style>
""", unsafe_allow_html=True)

def get_hf_token():
    return st.secrets.get("HF_TOKEN") or os.getenv("HF_TOKEN")

@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_best_result(dataset_name):
    token = get_hf_token()
    if not token:
        st.error("HF_TOKEN not found in secrets or environment")
        return None
    try:
        ds = load_dataset(dataset_name, split="train", token=token)
        df = ds.to_pandas()
        if len(df) == 0:
            return None
        # Choose best Sharpe ratio
        if 'sharpe_ratio' in df.columns and not df['sharpe_ratio'].isna().all():
            best = df.loc[df['sharpe_ratio'].idxmax()]
        else:
            best = df.iloc[0]
        return best.to_dict()
    except Exception as e:
        st.error(f"Error loading {dataset_name}: {str(e)}")
        return None

def get_next_trading_date():
    us_cal = USFederalHolidayCalendar()
    nyse = CustomBusinessDay(calendar=us_cal)
    return (datetime.now().date() + nyse).strftime('%Y-%m-%d')

def safe_get(data, key, default=None):
    """Safely get value from dict, handling None values."""
    if data is None:
        return default
    value = data.get(key, default)
    # Handle NaN, None, or empty values
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return default
    return value

def render_universe(title, dataset_name, benchmark):
    st.header(title)
    st.markdown(f"**Benchmark:** {benchmark}")
    data = load_best_result(dataset_name)
    if not data:
        st.warning("No results found. Run training first.")
        return

    next_date = get_next_trading_date()
    
    # FIX: Use safe_get to handle None/NaN values properly
    top_etfs = safe_get(data, 'top_etfs', [])
    forecasted = safe_get(data, 'forecasted_returns', {})
    scores = safe_get(data, 'scores', {})
    
    # Ensure top_etfs is a list
    if not isinstance(top_etfs, list):
        top_etfs = []
    
    # Ensure forecasted is a dict
    if not isinstance(forecasted, dict):
        forecasted = {}
    
    # Ensure scores is a dict
    if not isinstance(scores, dict):
        scores = {}

    if top_etfs and len(top_etfs) > 0:
        top = top_etfs[0]
        top_return = forecasted.get(top, 0)
        if not isinstance(top_return, (int, float)) or np.isnan(top_return):
            top_return = 0
            
        secondary = top_etfs[1] if len(top_etfs) > 1 else None
        sec_return = forecasted.get(secondary, 0) if secondary else 0
        if not isinstance(sec_return, (int, float)) or np.isnan(sec_return):
            sec_return = 0

        col1, col2 = st.columns([2,1])
        with col1:
            st.markdown(f"""
            <div class="hero-card">
                <div class="hero-ticker">{top}</div>
                <div class="hero-return">+{top_return*100:.3f}%</div>
                <div class="hero-label">Expected Return – {next_date}</div>
            </div>
            """, unsafe_allow_html=True)
            if secondary:
                st.markdown(f"📈 **{secondary}** +{sec_return*100:.3f}%", unsafe_allow_html=True)
        with col2:
            # FIX: Safe metric access with defaults
            ann_return = safe_get(data, 'annual_return', 0)
            sharpe = safe_get(data, 'sharpe_ratio', 0)
            max_dd = safe_get(data, 'max_drawdown', 0)
            
            # Handle inf/nan values
            if not isinstance(ann_return, (int, float)) or not np.isfinite(ann_return):
                ann_return = 0
            if not isinstance(sharpe, (int, float)) or not np.isfinite(sharpe):
                sharpe = 0
            if not isinstance(max_dd, (int, float)) or not np.isfinite(max_dd):
                max_dd = 0
            
            st.markdown(f"""
            <div class="metric-card"><div class="metric-value">{ann_return*100:.1f}%</div><div>Annual Return</div></div>
            <div class="metric-card" style="margin-top:12px"><div class="metric-value">{sharpe:.2f}</div><div>Sharpe Ratio</div></div>
            <div class="metric-card" style="margin-top:12px"><div class="metric-value">{-max_dd*100:.1f}%</div><div>Max Drawdown</div></div>
            """, unsafe_allow_html=True)
    else:
        st.info("No ETF predictions available.")

    st.markdown("---")
    st.subheader("All ETF Rankings")
    
    # FIX: Safely build rankings DataFrame
    if forecasted and len(forecasted) > 0:
        try:
            rank_data = []
            for k, v in forecasted.items():
                # Handle various value types
                if isinstance(v, (int, float)) and np.isfinite(v):
                    ret_val = v * 100
                else:
                    ret_val = 0.0
                    
                # Get score safely
                score_val = scores.get(k, 0)
                if not isinstance(score_val, (int, float)) or not np.isfinite(score_val):
                    score_val = 0.0
                    
                rank_data.append({
                    "ETF": str(k), 
                    "Expected Return (%)": ret_val, 
                    "Score": score_val
                })
            
            if rank_data:
                df_rank = pd.DataFrame(rank_data)
                st.dataframe(df_rank.sort_values("Expected Return (%)", ascending=False), 
                           use_container_width=True, hide_index=True)
            else:
                st.info("No ranking data available.")
        except Exception as e:
            st.error(f"Error displaying rankings: {str(e)}")
    else:
        st.info("No forecasted returns available.")

def main():
    st.title("SDF Engine – ETF Signal Generator")
    tab1, tab2 = st.tabs(["📈 Equity ETFs", "🏦 FI & Commodities"])
    with tab1:
        render_universe("US Equity ETFs", "P2SAMAPA/p2-etf-sdf-engine-results-equity", "SPY")
    with tab2:
        render_universe("Fixed Income & Commodities", "P2SAMAPA/p2-etf-sdf-engine-results-fi", "AGG")
    st.markdown(f'<div class="footer">Model trained on data up to {datetime.now().strftime("%d %b %Y")}</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
