import streamlit as st
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from pandas.tseries.offsets import CustomBusinessDay
from pandas.tseries.holiday import USFederalHolidayCalendar
from huggingface_hub import hf_hub_download, list_repo_files

st.set_page_config(page_title="SDF Engine", layout="wide")

st.markdown("""
<style>
    [data-testid="stSidebar"] {display: none;}
    .hero-card {background: linear-gradient(135deg, #1E3A5F 0%, #2C5282 100%); border-radius: 20px; padding: 2rem; color: white; margin-bottom: 1rem;}
    .hero-ticker {font-size: 3rem; font-weight: 800;}
    .hero-return {font-size: 2rem; font-weight: 600;}
    .metric-card {background: white; border-radius: 12px; padding: 1rem; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.05);}
    .metric-value {font-size: 1.8rem; font-weight: 700; color: #1E3A5F;}
</style>
""", unsafe_allow_html=True)


def get_hf_token():
    return st.secrets.get("HF_TOKEN") or os.getenv("HF_TOKEN")


@st.cache_data(ttl=300)
def load_best_result(universe_prefix):
    """Load best result from JSON files."""
    token = get_hf_token()
    if not token:
        return None
    
    try:
        files = list_repo_files("P2SAMAPA/p2-etf-sdf-engine-results", repo_type="dataset", token=token)
        json_files = [f for f in files if f.startswith(universe_prefix) and f.endswith('.json')]
        
        if not json_files:
            return None
        
        best_record = None
        best_sharpe = -np.inf
        
        for filename in json_files:
            try:
                path = hf_hub_download("P2SAMAPA/p2-etf-sdf-engine-results", filename, repo_type="dataset", token=token)
                with open(path, 'r') as f:
                    record = json.load(f)
                
                sharpe = record.get('sharpe_ratio', -np.inf)
                if isinstance(sharpe, (int, float)) and np.isfinite(sharpe) and sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_record = record
            except:
                continue
        
        return best_record
        
    except Exception as e:
        st.error(f"Load error: {e}")
        return None


def get_next_trading_date():
    us_cal = USFederalHolidayCalendar()
    nyse = CustomBusinessDay(calendar=us_cal)
    return (datetime.now().date() + nyse).strftime('%Y-%m-%d')


def render_universe(title, universe_prefix, benchmark):
    st.header(title)
    st.markdown(f"**Benchmark:** {benchmark}")
    
    data = load_best_result(universe_prefix)
    if not data:
        st.warning("No results found.")
        return
    
    # Extract data with defaults
    top_etfs = data.get('top_etfs', []) or []
    forecasted = data.get('forecasted_returns', {}) or {}
    scores = data.get('scores', {}) or {}
    
    # Get metrics with proper defaults
    sharpe = data.get('sharpe_ratio', 0)
    ann_return = data.get('annual_return', 0)
    max_dd = data.get('max_drawdown', 0)
    
    # Validate metrics
    if not isinstance(sharpe, (int, float)) or not np.isfinite(sharpe):
        sharpe = 0.0
    if not isinstance(ann_return, (int, float)) or not np.isfinite(ann_return):
        ann_return = 0.0
    if not isinstance(max_dd, (int, float)) or not np.isfinite(max_dd):
        max_dd = 0.0
    
    # FIX 1: Sort by forecasted return to find true top ETF (not just first in list)
    if forecasted and len(forecasted) > 0:
        # Create sorted list by return
        sorted_etfs = sorted(
            [(etf, forecasted.get(etf, 0)) for etf in forecasted.keys()],
            key=lambda x: x[1],
            reverse=True
        )
        
        # Hero card shows highest return ETF
        if sorted_etfs:
            top, top_ret = sorted_etfs[0]
            top_ret_pct = top_ret * 100
            
            # Secondary is second highest or first in top_etfs list
            sec = sorted_etfs[1][0] if len(sorted_etfs) > 1 else None
            sec_ret_pct = forecasted.get(sec, 0) * 100 if sec else 0
        else:
            top, top_ret_pct = "N/A", 0
            sec, sec_ret_pct = None, 0
    else:
        top, top_ret_pct = "N/A", 0
        sec, sec_ret_pct = None, 0
    
    # Display hero card
    if top != "N/A":
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"""
            <div class="hero-card">
                <div class="hero-ticker">{top}</div>
                <div class="hero-return">+{top_ret_pct:.3f}%</div>
                <div>Expected Return – {get_next_trading_date()}</div>
            </div>
            """, unsafe_allow_html=True)
            if sec:
                st.markdown(f"📈 **{sec}** +{sec_ret_pct:.3f}%")
        
        with col2:
            # FIX 2: Display actual metrics (not hardcoded)
            st.markdown(f"""
            <div class="metric-card"><div class="metric-value">{ann_return*100:.1f}%</div><div>Annual Return</div></div>
            <div class="metric-card" style="margin-top:12px"><div class="metric-value">{sharpe:.2f}</div><div>Sharpe Ratio</div></div>
            <div class="metric-card" style="margin-top:12px"><div class="metric-value">{-max_dd*100:.1f}%</div><div>Max Drawdown</div></div>
            """, unsafe_allow_html=True)
    else:
        st.info("No predictions available.")
        return
    
    # Rankings table
    st.markdown("---")
    st.subheader("All ETF Rankings")
    
    if forecasted:
        # FIX 3: Ensure scores are properly matched to ETFs
        table_data = []
        for etf in forecasted.keys():
            ret = forecasted.get(etf, 0)
            score = scores.get(etf, 0)  # Direct lookup by ETF ticker
            
            # Validate values
            if not isinstance(ret, (int, float)) or not np.isfinite(ret):
                ret = 0
            if not isinstance(score, (int, float)) or not np.isfinite(score):
                score = 0
                
            table_data.append({
                "ETF": str(etf),
                "Expected Return (%)": ret * 100,
                "Score": score
            })
        
        if table_data:
            df = pd.DataFrame(table_data)
            df = df.sort_values("Expected Return (%)", ascending=False)
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No ranking data.")
    else:
        st.info("No forecasted returns.")


def main():
    st.title("SDF Engine – ETF Signal Generator")
    tab1, tab2 = st.tabs(["📈 Equity ETFs", "🏦 FI & Commodities"])
    with tab1:
        render_universe("US Equity ETFs", "equity", "SPY")
    with tab2:
        render_universe("Fixed Income & Commodities", "fi", "AGG")
    st.markdown(f'<div style="text-align:center; margin-top:2rem; color:#718096;">Model trained on data up to {datetime.now().strftime("%d %b %Y")}</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
