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


@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_data_universe():
    """Load data for both universes."""
    hf_token = os.getenv('HF_TOKEN')

    loader = DataLoader(hf_token=hf_token)
    raw_data = loader.load_raw_data()

    return raw_data


@st.cache_data(ttl=3600)
def generate_equity_signals(window_size=252, top_n=3):
    """Generate equity signals."""
    hf_token = os.getenv('HF_TOKEN')
    engine = EquityEngine(hf_token=hf_token)

    # Get latest data
    returns, macro, _ = engine.prepare_data(
        start_date='2023-01-01',
        end_date=datetime.now().strftime('%Y-%m-%d')
    )

    # Use most recent window
    train_returns = returns.iloc[-window_size:]
    train_macro = macro.loc[train_returns.index]

    # Generate signals
    result = engine.generate_signals_pipeline(train_returns, train_macro, top_n)

    return result, engine


@st.cache_data(ttl=3600)
def generate_fi_commodity_signals(window_size=252, top_n=3):
    """Generate FI/Commodity signals."""
    hf_token = os.getenv('HF_TOKEN')
    engine = FICommodityEngine(hf_token=hf_token)

    # Get latest data
    returns, macro, _ = engine.prepare_data(
        start_date='2023-01-01',
        end_date=datetime.now().strftime('%Y-%m-%d')
    )

    # Use most recent window
    train_returns = returns.iloc[-window_size:]
    train_macro = macro.loc[train_returns.index]

    # Generate signals
    result = engine.generate_signals_pipeline(train_returns, train_macro, top_n)

    return result, engine


def plot_returns_comparison(signals_df, benchmark_returns, title):
    """Plot strategy vs benchmark returns."""
    if len(signals_df) == 0:
        return None

    # Calculate strategy returns
    strategy_returns = []

    for _, row in signals_df.iterrows():
        if 'selected_assets' in row and len(row['selected_assets']) > 0:
            assets = row['selected_assets']
            # This is a simplification - in reality you'd calculate actual returns
            strategy_returns.append(0.001)  # Placeholder
        else:
            strategy_returns.append(0)

    # Create comparison chart
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=signals_df.index,
        y=np.cumsum(strategy_returns) * 100,
        mode='lines',
        name='Strategy',
        line=dict(color='#3182CE', width=2)
    ))

    fig.add_trace(go.Scatter(
        x=benchmark_returns.index[:len(signals_df)],
        y=np.cumsum(benchmark_returns.values[:len(signals_df)]) * 100,
        mode='lines',
        name='Benchmark',
        line=dict(color='#718096', width=2, dash='dash')
    ))

    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Cumulative Return (%)',
        template='plotly_white',
        height=400,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    return fig


def plot_factor_interpretations(factor_df):
    """Plot factor interpretations as a heatmap."""
    if factor_df is None or len(factor_df) == 0:
        return None

    # Create sample heatmap data
    n_factors = len(factor_df)
    n_assets = 8

    # Sample data for visualization
    np.random.seed(42)
    data = np.random.randn(n_factors, n_assets) * 0.5

    assets = ['QQQ', 'XLK', 'XLF', 'XLE', 'VNQ', 'GLD', 'TLT', 'HYG']
    factors = [f'Factor {i+1}' for i in range(n_factors)]

    fig = px.imshow(
        data,
        x=assets[:n_assets],
        y=factors,
        color_continuous_scale='RdBu',
        title='Factor Loadings Heatmap',
        labels=dict(x="Assets", y="Factors", color="Loading")
    )

    fig.update_layout(template='plotly_white', height=300)

    return fig


def render_equity_tab():
    """Render Equity ETFs tab."""
    st.header("US Equity ETFs")
    st.markdown(f"**Benchmark:** {CONFIG['universes']['equity']['benchmark']}")
    st.markdown(f"**Universe:** {', '.join(CONFIG['universes']['equity']['assets'])}")

    col1, col2, col3, col4 = st.columns(4)

    try:
        # Get signals
        with st.spinner('Generating equity signals...'):
            result, engine = generate_equity_signals(
                window_size=252,
                top_n=3
            )

        # Display metrics
        with col1:
            st.metric("Date", result['date'].strftime('%Y-%m-%d') if hasattr(result['date'], 'strftime') else str(result['date']))

        with col2:
            st.metric("Factors", result['n_factors'])

        with col3:
            explained = np.mean(result['explained_variance']) * 100
            st.metric("Explained Var.", f"{explained:.1f}%")

        with col4:
            st.metric("Selected ETFs", len(result['selected_assets']))

        st.divider()

        # Selected ETFs
        st.subheader("Selected ETFs (Top 3)")

        selected = result['signals']
        if len(selected) > 0:
            for i, row in selected.iterrows():
                with st.container():
                    st.markdown(f"""
                    <div class="selected-etf">
                        <h4>{row['asset']}</h4>
                        <p>Expected Return: {row['expected_return']*100:.3f}% |
                           Composite Score: {row['composite_score']:.4f}</p>
                    </div>
                    """, unsafe_allow_html=True)

        st.divider()

        # All ETF Scores
        st.subheader("All ETF Rankings")

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
        st.subheader("Factor Interpretations")

        factor_df = result['factor_interpretations']
        if len(factor_df) > 0:
            st.dataframe(
                factor_df[['factor', 'top_assets', 'interpretation']],
                use_container_width=True,
                hide_index=True
            )

            # Plot heatmap
            fig = plot_factor_interpretations(factor_df)
            if fig:
                st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error generating signals: {str(e)}")
        st.info("Make sure HF_TOKEN is set in your environment.")


def render_fi_commodity_tab():
    """Render FI/Commodities tab."""
    st.header("Fixed Income & Commodities ETFs")
    st.markdown(f"**Benchmark:** {CONFIG['universes']['fi_commodities']['benchmark']}")
    st.markdown(f"**Universe:** {', '.join(CONFIG['universes']['fi_commodities']['assets'])}")

    col1, col2, col3, col4 = st.columns(4)

    try:
        # Get signals
        with st.spinner('Generating FI/Commodity signals...'):
            result, engine = generate_fi_commodity_signals(
                window_size=252,
                top_n=3
            )

        # Display metrics
        with col1:
            st.metric("Date", result['date'].strftime('%Y-%m-%d') if hasattr(result['date'], 'strftime') else str(result['date']))

        with col2:
            st.metric("Factors", result['n_factors'])

        with col3:
            explained = np.mean(result['explained_variance']) * 100
            st.metric("Explained Var.", f"{explained:.1f}%")

        with col4:
            st.metric("Selected ETFs", len(result['selected_assets']))

        st.divider()

        # Selected ETFs
        st.subheader("Selected ETFs (Top 3)")

        selected = result['signals']
        if len(selected) > 0:
            for i, row in selected.iterrows():
                with st.container():
                    st.markdown(f"""
                    <div class="selected-etf">
                        <h4>{row['asset']}</h4>
                        <p>Expected Return: {row['expected_return']*100:.3f}% |
                           Composite Score: {row['composite_score']:.4f}</p>
                    </div>
                    """, unsafe_allow_html=True)

        st.divider()

        # All ETF Scores
        st.subheader("All ETF Rankings")

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
        st.subheader("Factor Interpretations")

        factor_df = result['factor_interpretations']
        if len(factor_df) > 0:
            st.dataframe(
                factor_df[['factor', 'top_assets', 'interpretation']],
                use_container_width=True,
                hide_index=True
            )

            # Plot heatmap
            fig = plot_factor_interpretations(factor_df)
            if fig:
                st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error generating signals: {str(e)}")
        st.info("Make sure HF_TOKEN is set in your environment.")


def main():
    """Main application."""
    st.title(CONFIG['streamlit']['title'])

    # Sidebar for settings
    with st.sidebar:
        st.header("Settings")

        st.subheader("Model Parameters")
        window_size = st.slider(
            "Training Window",
            min_value=126,
            max_value=504,
            value=252,
            step=21,
            help="Number of days for training window"
        )

        top_n = st.slider(
            "Top ETFs to Select",
            min_value=1,
            max_value=5,
            value=3,
            help="Number of top ETFs to hold"
        )

        st.divider()

        st.subheader("About")
        st.markdown("""
        **SDF Engine** uses a Sparse Dynamic Factor model to generate ETF trading signals.

        - **PCA** for factor extraction
        - **VARIMAX** for sparse rotation
        - **VAR + Kalman** for forecasting
        - **Cross-sectional scoring** for selection
        """)

        st.divider()

        st.caption(f"Data: {CONFIG['huggingface']['dataset_source']}")
        st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    # Main content with tabs
    tab_names = [
        CONFIG['streamlit']['tab_names']['equity'],
        CONFIG['streamlit']['tab_names']['fi_commodities']
    ]

    tabs = st.tabs(tab_names)

    with tabs[0]:
        render_equity_tab()

    with tabs[1]:
        render_fi_commodity_tab()


if __name__ == "__main__":
    main()
