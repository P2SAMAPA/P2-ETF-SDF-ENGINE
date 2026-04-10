import streamlit as st
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from pandas.tseries.offsets import CustomBusinessDay
from pandas.tseries.holiday import USFederalHolidayCalendar
from datasets import load_dataset

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
def load_best_result_from_parquet(parquet_file):
    """Load best result from specific Parquet file."""
    token = get_hf_token()
    if not token:
        return None
    
    try:
        ds = load_dataset(
            "P2SAMAPA/p2-etf-sdf-engine-results", 
            data_files=parquet_file,
            split="train", 
            token=token
        )
        df = ds.to_pandas()
        
        if len(df) == 0:
            return None
        
        # Find best Sharpe
        best_idx = df['sharpe_ratio'].idxmax()
        best = df.loc[best_idx].to_dict()
        
        # Parse JSON strings back to objects
        best['top_etfs'] = json.loads(best.get('top_etfs', '[]'))
        best['forecasted_returns'] = json.loads(best.get('forecasted_returns', '{}'))
        best['scores'] = json.loads(best.get('scores', '{}'))
        
        return best
        
    except Exception as e:
        st.error(f"Load error for {parquet_file}: {e}")
        return None


def get_next_trading_date():
    us_cal = USFederalHolidayCalendar()
    nyse = CustomBusinessDay(calendar=us_cal)
    return (datetime.now().date() + nyse).strftime('%Y-%m-%d')


def render_universe(title, parquet_file, benchmark):
    st.header(title)
    st.markdown(f"**Benchmark:** {benchmark}")
    
    data = load_best_result_from_parquet(parquet_file)
    if not data:
        st.warning("No results found.")
        return
    
    # Extract data
    top_etfs = data.get('top_etfs', [])
    forecasted = data.get('forecasted_returns', {})
    scores = data.get('scores', {})
    
    metrics = {
        'annual_return': data.get('annual_return', 0),
        'sharpe_ratio': data.get('sharpe_ratio', 0),
        'max_drawdown': data.get('max_drawdown', 0)
    }
    
    # Validate
    for k in metrics:
        if not isinstance(metrics[k], (int, float)) or not np.isfinite(metrics[k]):
            metrics[k] = 0.0
    
    # Sort by forecasted return for hero card
    if forecasted:
        sorted_etfs = sorted(forecasted.items(), key=lambda x: x[1], reverse=True)
        top, top_ret = sorted_etfs[0]
        top_ret_pct = top_ret * 100
        
        sec = sorted_etfs[1][0] if len(sorted_etfs) > 1 else None
        sec_ret_pct = forecasted.get(sec, 0) * 100 if sec else 0
    else:
        top, top_ret_pct = "N/A", 0
        sec, sec_ret_pct = None, 0
    
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
            st.markdown(f"""
            <div class="metric-card"><div class="metric-value">{metrics['annual_return']*100:.1f}%</div><div>Annual Return</div></div>
            <div class="metric-card" style="margin-top:12px"><div class="metric-value">{metrics['sharpe_ratio']:.2f}</div><div>Sharpe Ratio</div></div>
            <div class="metric-card" style="margin-top:12px"><div class="metric-value">{-metrics['max_drawdown']*100:.1f}%</div><div>Max Drawdown</div></div>
            """, unsafe_allow_html=True)
    else:
        st.info("No predictions available.")
        return
    
    # Rankings table
    st.markdown("---")
    st.subheader("All ETF Rankings")
    
    if forecasted:
        df = pd.DataFrame([
            {"ETF": k, "Expected Return (%)": v * 100, "Score": scores.get(k, 0)}
            for k, v in forecasted.items()
        ])
        st.dataframe(df.sort_values("Expected Return (%)", ascending=False), 
                   use_container_width=True, hide_index=True)


def main():
    st.title("SDF Engine – ETF Signal Generator")
    tab1, tab2 = st.tabs(["📈 Equity ETFs", "🏦 FI & Commodities"])
    with tab1:
        render_universe("US Equity ETFs", "equity_results.parquet", "SPY")
    with tab2:
        render_universe("Fixed Income & Commodities", "fi_results.parquet", "AGG")
    st.markdown(f'<div style="text-align:center; margin-top:2rem; color:#718096;">Last updated: {datetime.now().strftime("%d %b %Y")}</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
