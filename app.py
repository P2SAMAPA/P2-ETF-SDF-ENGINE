# =============================================================================
# Streamlit App - SDF Engine Dashboard
# =============================================================================
"""
Streamlit application for displaying SDF Engine signals.
Two tabs: Equity ETFs and FI/Commodities ETFs
"""

import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import traceback

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs import CONFIG
from data_loader import DataLoader
from equity_engine import EquityEngine
from fi_commodity_engine import FICommodityEngine
from backtest_engine import BacktestEngine

# Page configuration
st.set_page_config(
    page_title=CONFIG['streamlit']['title'],
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for large fonts and white background
st.markdown("""
<style>
    /* Main background - white/light shade */
    .stApp {
        background-color: #FFFFFF;
    }

    /* Large font size for hero tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 20px;
        font-weight: 600;
        padding: 16px 32px;
    }

    /* Headers */
    h1 {
        font-size: 2.5rem !important;
        color: #1E3A5F !important;
    }
    h2 {
        font-size: 1.8rem !important;
        color: #1E3A5F !important;
    }
    h3 {
        font-size: 1.4rem !important;
        color: #2C5282 !important;
    }

    /* Cards */
    .metric-card {
        background-color: #F7FAFC;
        border-radius: 10px;
        padding: 20px;
        border-left: 4px solid #1E3A5F;
    }

    /* Selected ETF highlight */
    .selected-etf {
        background-color: #EBF8FF;
        border: 2px solid #3182CE;
        border-radius: 8px;
        padding: 12px;
        margin: 8px 0;
    }
</style>
""", unsafe_allow_html=True)


def get_hf_token():
    """Get HF token from environment or Streamlit secrets."""
    token = os.getenv('HF_TOKEN')
    if not token:
        try:
            token = st.secrets.get('HF_TOKEN', None)
        except:
            pass
    return token


@st.cache_data(ttl=3600)
def load_data_universe():
    """Load data for both universes."""
    hf_token = get_hf_token()
    if not hf_token:
        st.error("HF_TOKEN not found. Please set it in secrets or environment.")
        return None
    
    loader = DataLoader(hf_token=hf_token)
    raw_data = loader.load_raw_data()
    return raw_data


@st.cache_data(ttl=3600)
def generate_equity_signals(window_size=252, top_n=3):
    """Generate equity signals."""
    hf_token = get_hf_token()
    if not hf_token:
        raise ValueError("HF_TOKEN not set")
    
    engine = EquityEngine(hf_token=hf_token)
    
    # Get latest data
    end_date = datetime.now().strftime('%Y-%m-%d')
    returns, macro, _ = engine.prepare_data(
        start_date='2020-01-01',  # Use longer history for better training
        end_date=end_date
    )
    
    # Ensure we have enough data
    if len(returns) < window_size:
        raise ValueError(f"Insufficient data: only {len(returns)} rows, need {window_size}")
    
    # Use most recent window
    train_returns = returns.iloc[-window_size:]
    train_macro = macro.loc[train_returns.index]
    
    # Generate signals
    result = engine.generate_signals_pipeline(train_returns, train_macro, top_n)
    
    return result, engine


@st.cache_data(ttl=3600)
def generate_fi_commodity_signals(window_size=252, top_n=3):
    """Generate FI/Commodity signals."""
    hf_token = get_hf_token()
    if not hf_token:
        raise ValueError("HF_TOKEN not set")
    
    engine = FICommodityEngine(hf_token=hf_token)
    
    # Get latest data
    end_date = datetime.now().strftime('%Y-%m-%d')
    returns, macro, _ = engine.prepare_data(
        start_date='2020-01-01',
        end_date=end_date
    )
    
    # Ensure we have enough data
    if len(returns) < window_size:
        raise ValueError(f"Insufficient data: only {len(returns)} rows, need {window_size}")
    
    # Use most recent window
    train_returns = returns.iloc[-window_size:]
    train_macro = macro.loc[train_returns.index]
    
    # Generate signals
    result = engine.generate_signals_pipeline(train_returns, train_macro, top_n)
    
    return result, engine


def plot_factor_interpretations(factor_df):
    """Plot factor interpretations as a heatmap."""
    if factor_df is None or len(factor_df) == 0:
        return None
    
    try:
        # Try to extract actual loadings if available
        if 'loadings' in factor_df.columns:
            # Use actual loadings data
            loadings_data = np.array(factor_df['loadings'].tolist())
            fig = px.imshow(
                loadings_data,
                title='Factor Loadings Heatmap',
                labels=dict(x="Factors", y="Assets", color="Loading"),
                color_continuous_scale='RdBu'
            )
        else:
            # Create sample heatmap data for visualization
            n_factors = len(factor_df)
            n_assets = min(8, len(CONFIG['universes']['equity']['assets']))
            
            np.random.seed(42)
            data = np.random.randn(n_factors, n_assets) * 0.5
            
            assets = CONFIG['universes']['equity']['assets'][:n_assets]
            factors = [f'Factor {i+1}' for i in range(n_factors)]
            
            fig = px.imshow(
                data,
                x=assets,
                y=factors,
                color_continuous_scale='RdBu',
                title='Factor Loadings Heatmap (Illustrative)',
                labels=dict(x="Assets", y="Factors", color="Loading")
            )
        
        fig.update_layout(template='plotly_white', height=400)
        return fig
    except Exception as e:
        st.warning(f"Could not create heatmap: {e}")
        return None


def render_equity_tab():
    """Render Equity ETFs tab."""
    st.header("US Equity ETFs")
    st.markdown(f"**Benchmark:** {CONFIG['universes']['equity']['benchmark']}")
    st.markdown(f"**Universe:** {', '.join(CONFIG['universes']['equity']['assets'])}")
    
    col1, col2, col3, col4 = st.columns(4)
    
    try:
        # Get signals
        with st.spinner('Generating equity signals... This may take a minute...'):
            result, engine = generate_equity_signals(window_size=252, top_n=3)
        
        # Display metrics
        with col1:
            date_str = result['date'].strftime('%Y-%m-%d') if hasattr(result['date'], 'strftime') else str(result['date'])
            st.metric("Date", date_str)
        
        with col2:
            st.metric("Factors", result['n_factors'])
        
        with col3:
            explained = np.mean(result['explained_variance']) * 100
            st.metric("Explained Var.", f"{explained:.1f}%")
        
        with col4:
            st.metric("Selected ETFs", len(result['selected_assets']))
        
        st.divider()
        
        # Selected ETFs
        st.subheader("📊 Selected ETFs (Top 3)")
        
        selected = result['signals']
        if len(selected) > 0:
            for _, row in selected.iterrows():
                with st.container():
                    st.markdown(f"""
                    <div class="selected-etf">
                        <h3>📈 {row['asset']}</h3>
                        <p><strong>Expected Return:</strong> {row['expected_return']*100:.4f}%<br>
                        <strong>Composite Score:</strong> {row['composite_score']:.4f}</p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.warning("No ETFs selected")
        
        st.divider()
        
        # All ETF Scores
        st.subheader("📋 All ETF Rankings")
        
        scores_df = result['signals'].copy()
        scores_df['Expected Return (%)'] = scores_df['expected_return'] * 100
        scores_df['Score'] = scores_df['composite_score'].round(4)
        
        display_cols = ['asset', 'Expected Return (%)', 'Score']
        st.dataframe(
            scores_df[display_cols],
            use_container_width=True,
            hide_index=True
        )
        
        st.divider()
        
        # Factor Interpretations
        st.subheader("🔍 Factor Interpretations")
        
        factor_df = result['factor_interpretations']
        if len(factor_df) > 0:
            st.dataframe(
                factor_df[['factor', 'top_assets', 'interpretation']],
                use_container_width=True,
                hide_index=True
            )
            
            fig = plot_factor_interpretations(factor_df)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No factor interpretations available")
            
    except Exception as e:
        st.error(f"Error generating signals: {str(e)}")
        st.code(traceback.format_exc())
        st.info("Make sure HF_TOKEN is set in your environment or Streamlit secrets.")


def render_fi_commodity_tab():
    """Render FI/Commodities tab."""
    st.header("Fixed Income & Commodities ETFs")
    st.markdown(f"**Benchmark:** {CONFIG['universes']['fi_commodities']['benchmark']}")
    st.markdown(f"**Universe:** {', '.join(CONFIG['universes']['fi_commodities']['assets'])}")
    
    col1, col2, col3, col4 = st.columns(4)
    
    try:
        # Get signals
        with st.spinner('Generating FI/Commodity signals... This may take a minute...'):
            result, engine = generate_fi_commodity_signals(window_size=252, top_n=3)
        
        # Display metrics
        with col1:
            date_str = result['date'].strftime('%Y-%m-%d') if hasattr(result['date'], 'strftime') else str(result['date'])
            st.metric("Date", date_str)
        
        with col2:
            st.metric("Factors", result['n_factors'])
        
        with col3:
            explained = np.mean(result['explained_variance']) * 100
            st.metric("Explained Var.", f"{explained:.1f}%")
        
        with col4:
            st.metric("Selected ETFs", len(result['selected_assets']))
        
        st.divider()
        
        # Selected ETFs
        st.subheader("📊 Selected ETFs (Top 3)")
        
        selected = result['signals']
        if len(selected) > 0:
            for _, row in selected.iterrows():
                with st.container():
                    st.markdown(f"""
                    <div class="selected-etf">
                        <h3>📈 {row['asset']}</h3>
                        <p><strong>Expected Return:</strong> {row['expected_return']*100:.4f}%<br>
                        <strong>Composite Score:</strong> {row['composite_score']:.4f}</p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.warning("No ETFs selected")
        
        st.divider()
        
        # All ETF Scores
        st.subheader("📋 All ETF Rankings")
        
        scores_df = result['signals'].copy()
        scores_df['Expected Return (%)'] = scores_df['expected_return'] * 100
        scores_df['Score'] = scores_df['composite_score'].round(4)
        
        display_cols = ['asset', 'Expected Return (%)', 'Score']
        st.dataframe(
            scores_df[display_cols],
            use_container_width=True,
            hide_index=True
        )
        
        st.divider()
        
        # Factor Interpretations
        st.subheader("🔍 Factor Interpretations")
        
        factor_df = result['factor_interpretations']
        if len(factor_df) > 0:
            st.dataframe(
                factor_df[['factor', 'top_assets', 'interpretation']],
                use_container_width=True,
                hide_index=True
            )
            
            fig = plot_factor_interpretations(factor_df)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No factor interpretations available")
            
    except Exception as e:
        st.error(f"Error generating signals: {str(e)}")
        st.code(traceback.format_exc())
        st.info("Make sure HF_TOKEN is set in your environment or Streamlit secrets.")


def main():
    """Main application."""
    st.title(CONFIG['streamlit']['title'])
    
    # Check for HF_TOKEN at startup
    hf_token = get_hf_token()
    if not hf_token:
        st.error("""
        ⚠️ **HF_TOKEN not found!**
        
        Please set your Hugging Face token to use this app.
        
        **For Streamlit Cloud:**
        1. Go to your app settings
        2. Add a secret named `HF_TOKEN` with your Hugging Face token
        3. Restart the app
        
        **For local development:**
        ```bash
        export HF_TOKEN="your_token_here"
        streamlit run app.py
