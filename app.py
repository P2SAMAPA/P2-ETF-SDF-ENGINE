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
    
    top_etfs = data.get('top_etfs', []) or []
    forecasted = data.get('forecasted_returns', {}) or {}
    scores = data.get('scores', {}) or {}
    
    if not top_etfs:
        st.info("No predictions available.")
        return
    
    top = top_etfs[0]
    top_ret = forecasted.get(top, 0) * 100
    sec = top_etfs[1] if len(top_etfs) > 1 else None
    sec_ret = forecasted.get(sec, 0) * 100 if sec else 0
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f"""
        <div class="hero-card">
            <div class="hero-ticker">{top}</div>
            <div class="hero-return">+{top_ret:.3f}%</div>
            <div>Expected Return – {get_next_trading_date()}</div>
        </div>
        """, unsafe_allow_html=True)
        if sec:
            st.markdown(f"📈 **{sec}** +{sec_ret:.3f}%")
    
    with col2:
        ann_ret = data.get('annual_return', 0) * 100
        sharpe = data.get('sharpe_ratio', 0)
        max_dd = data.get('max_drawdown', 0) * 100
        
        st.markdown(f"""
        <div class="metric-card"><div class="metric-value">{ann_ret:.1f}%</div><div>Annual Return</div></div>
        <div class="metric-card" style="margin-top:12px"><div class="metric-value">{sharpe:.2f}</div><div>Sharpe Ratio</div></div>
        <div class="metric-card" style="margin-top:12px"><div class="metric-value">{-max_dd:.1f}%</div><div>Max Drawdown</div></div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.subheader("All ETF Rankings")
    
    if forecasted:
        df = pd.DataFrame([
            {"ETF": k, "Expected Return (%)": v * 100, "Score": scores.get(k, 0)}
            for k, v in forecasted.items()
        ])
        st.dataframe(df.sort_values("Expected Return (%)", ascending=False), use_container_width=True, hide_index=True)


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
