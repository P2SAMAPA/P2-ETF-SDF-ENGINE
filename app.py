# =============================================================================
# Streamlit App - SDF Engine Dashboard (Professional)
# =============================================================================

import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs import CONFIG
from data_loader import DataLoader
from equity_engine import EquityEngine
from fi_commodity_engine import FICommodityEngine
from datasets import load_dataset

st.set_page_config(
    page_title=CONFIG['streamlit']['title'],
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for hero cards
st.markdown("""
<style>
    .hero-card {
        background: linear-gradient(135deg, #1E3A5F 0%, #2C5282 100%);
        border-radius: 20px;
        padding: 2rem;
        color: white;
        margin-bottom: 1rem;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
    }
    .hero-ticker {
        font-size: 3rem;
        font-weight: 800;
        letter-spacing: -0.02em;
    }
    .hero-return {
        font-size: 2rem;
        font-weight: 600;
        margin-top: 0.5rem;
    }
    .hero-label {
        font-size: 0.9rem;
        opacity: 0.8;
        margin-top: 0.25rem;
    }
    .secondary-card {
        background-color: #F7FAFC;
        border-radius: 16px;
        padding: 1rem;
        margin-top: 1rem;
        border-left: 4px solid #2C5282;
    }
    .metric-card {
        background-color: white;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1E3A5F;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #4A5568;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .positive {
        color: #2E7D32;
    }
    .negative {
        color: #C62828;
    }
</style>
""", unsafe_allow_html=True)

def get_hf_token():
    token = os.getenv('HF_TOKEN')
    if not token:
        try:
            token = st.secrets.get('HF_TOKEN', None)
        except:
            pass
    return token

@st.cache_data(ttl=3600)
def load_best_metrics():
    """Load best metrics from training results dataset."""
    token = get_hf_token()
    if not token:
        return None
    try:
        ds = load_dataset(CONFIG['huggingface']['dataset_results'], split="train", token=token)
        df = ds.to_pandas()
        if len(df) == 0:
            return None
        # Find best sharpe ratio across all runs
        best_row = df.loc[df['equity_sharpe'].idxmax()] if 'equity_sharpe' in df.columns else None
        if best_row is not None:
            return {
                'equity_sharpe': best_row.get('equity_sharpe', 0),
                'equity_annual_return': best_row.get('equity_annual_return', 0),
                'equity_max_drawdown': best_row.get('equity_max_drawdown', 0),
                'fi_sharpe': best_row.get('fi_sharpe', 0),
                'fi_annual_return': best_row.get('fi_annual_return', 0),
                'fi_max_drawdown': best_row.get('fi_max_drawdown', 0),
            }
    except:
        return None
    return None

@st.cache_data(ttl=3600)
def generate_equity_signals(window_size=252, top_n=3):
    hf_token = get_hf_token()
    if not hf_token:
        raise ValueError("HF_TOKEN not set")
    engine = EquityEngine(hf_token=hf_token)
    end_date = datetime.now().strftime('%Y-%m-%d')
    returns, macro, _ = engine.prepare_data(start_date='2020-01-01', end_date=end_date)
    if len(returns) < window_size:
        raise ValueError(f"Insufficient data: {len(returns)} rows")
    train_returns = returns.iloc[-window_size:]
    train_macro = macro.loc[train_returns.index]
    result = engine.generate_signals_pipeline(train_returns, train_macro, top_n)
    return result, engine

@st.cache_data(ttl=3600)
def generate_fi_commodity_signals(window_size=252, top_n=3):
    hf_token = get_hf_token()
    if not hf_token:
        raise ValueError("HF_TOKEN not set")
    engine = FICommodityEngine(hf_token=hf_token)
    end_date = datetime.now().strftime('%Y-%m-%d')
    returns, macro, _ = engine.prepare_data(start_date='2020-01-01', end_date=end_date)
    if len(returns) < window_size:
        raise ValueError(f"Insufficient data: {len(returns)} rows")
    train_returns = returns.iloc[-window_size:]
    train_macro = macro.loc[train_returns.index]
    result = engine.generate_signals_pipeline(train_returns, train_macro, top_n)
    return result, engine

def render_hero_card(selected_df, metrics, universe_name, benchmark_name):
    """Render hero card with top positive ETF and secondary."""
    if selected_df is None or len(selected_df) == 0:
        st.warning("No positive expected returns available.")
        return
    
    # Filter positive expected returns
    pos_df = selected_df[selected_df['expected_return'] > 0].copy()
    if len(pos_df) == 0:
        st.warning("No ETFs with positive expected return at this time.")
        return
    
    pos_df = pos_df.sort_values('expected_return', ascending=False)
    top = pos_df.iloc[0]
    secondary = pos_df.iloc[1] if len(pos_df) > 1 else None
    
    col_hero, col_metrics = st.columns([2, 1])
    
    with col_hero:
        st.markdown(f"""
        <div class="hero-card">
            <div class="hero-ticker">{top['asset']}</div>
            <div class="hero-return">+{top['expected_return']*100:.3f}%</div>
            <div class="hero-label">Expected Return – Next Trading Day</div>
        </div>
        """, unsafe_allow_html=True)
        
        if secondary is not None:
            st.markdown(f"""
            <div class="secondary-card">
                <strong>📈 {secondary['asset']}</strong><br>
                Expected Return: +{secondary['expected_return']*100:.3f}%
            </div>
            """, unsafe_allow_html=True)
    
    with col_metrics:
        if metrics:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{metrics.get('annual_return', 0)*100:.1f}%</div>
                <div class="metric-label">Annualized Return</div>
            </div>
            <div class="metric-card" style="margin-top: 12px;">
                <div class="metric-value">{metrics.get('sharpe', 0):.2f}</div>
                <div class="metric-label">Sharpe Ratio</div>
            </div>
            <div class="metric-card" style="margin-top: 12px;">
                <div class="metric-value">{-metrics.get('max_drawdown', 0)*100:.1f}%</div>
                <div class="metric-label">Max Drawdown</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Performance metrics will appear after training completes.")

def render_equity_tab():
    st.header("US Equity ETFs")
    st.markdown(f"**Benchmark:** {CONFIG['universes']['equity']['benchmark']}")
    
    try:
        with st.spinner("Analyzing market signals..."):
            result, engine = generate_equity_signals(window_size=252, top_n=3)
        
        # Get metrics from training results
        metrics_data = load_best_metrics()
        equity_metrics = {
            'annual_return': metrics_data['equity_annual_return'] if metrics_data else None,
            'sharpe': metrics_data['equity_sharpe'] if metrics_data else None,
            'max_drawdown': metrics_data['equity_max_drawdown'] if metrics_data else None,
        } if metrics_data else None
        
        render_hero_card(result['signals'], equity_metrics, "Equity", "SPY")
        
        st.markdown("---")
        st.subheader("📊 All ETF Rankings")
        scores_df = result['signals'].copy()
        scores_df['Expected Return (%)'] = scores_df['expected_return'] * 100
        scores_df['Score'] = scores_df['composite_score'].round(4)
        st.dataframe(scores_df[['asset', 'Expected Return (%)', 'Score']], 
                     use_container_width=True, hide_index=True)
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.code(traceback.format_exc())

def render_fi_commodity_tab():
    st.header("Fixed Income & Commodities ETFs")
    st.markdown(f"**Benchmark:** {CONFIG['universes']['fi_commodities']['benchmark']}")
    
    try:
        with st.spinner("Analyzing market signals..."):
            result, engine = generate_fi_commodity_signals(window_size=252, top_n=3)
        
        metrics_data = load_best_metrics()
        fi_metrics = {
            'annual_return': metrics_data['fi_annual_return'] if metrics_data else None,
            'sharpe': metrics_data['fi_sharpe'] if metrics_data else None,
            'max_drawdown': metrics_data['fi_max_drawdown'] if metrics_data else None,
        } if metrics_data else None
        
        render_hero_card(result['signals'], fi_metrics, "FI/Commodity", "AGG")
        
        st.markdown("---")
        st.subheader("📊 All ETF Rankings")
        scores_df = result['signals'].copy()
        scores_df['Expected Return (%)'] = scores_df['expected_return'] * 100
        scores_df['Score'] = scores_df['composite_score'].round(4)
        st.dataframe(scores_df[['asset', 'Expected Return (%)', 'Score']], 
                     use_container_width=True, hide_index=True)
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.code(traceback.format_exc())

def main():
    st.title("SDF Engine – ETF Signal Generator")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        window_size = st.slider("Training Window (days)", 126, 504, 252, 21)
        top_n = st.slider("Top ETFs to Display", 1, 5, 3)
        st.divider()
        st.markdown("**Model:** Sparse Dynamic Factor Model")
        st.markdown("**Data Source:** Hugging Face Datasets")
        st.caption(f"Last refresh: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    
    # Tabs
    tab1, tab2 = st.tabs(["📈 Equity ETFs", "🏦 FI & Commodities"])
    with tab1:
        render_equity_tab()
    with tab2:
        render_fi_commodity_tab()

if __name__ == "__main__":
    main()
