import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import shutil
from datetime import datetime, timedelta
from huggingface_hub import hf_hub_download
import pandas_market_calendars as mcal  # <-- NEW IMPORT

st.set_page_config(page_title="SDF Engine", layout="wide")

cache_dir = os.path.expanduser("~/.cache/huggingface")
if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir, ignore_errors=True)

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

nyse = mcal.get_calendar("NYSE")  # <-- NEW CALENDAR

def get_nyse_next_trading_day(base_date_str):
    """
    Returns the next valid NYSE trading day.
    If the input date is already a trading day (weekday, not holiday),
    it returns the same date. Otherwise, it finds the next trading day.
    """
    try:
        current_dt = pd.to_datetime(base_date_str).normalize()
    except Exception:
        current_dt = pd.Timestamp.now(tz='US/Eastern').normalize()

    # Official 2026 NYSE Holiday Schedule (extendable)
    nyse_holidays = [
        '2026-01-01', # New Year's Day
        '2026-01-19', # Martin Luther King, Jr. Day
        '2026-02-16', # Washington's Birthday
        '2026-04-03', # Good Friday
        '2026-05-25', # Memorial Day
        '2026-06-19', # Juneteenth National Independence Day
        '2026-07-03', # Independence Day (Observed)
        '2026-09-07', # Labor Day
        '2026-11-26', # Thanksgiving Day
        '2026-12-25'  # Christmas Day
    ]

    # Check if current date is a valid trading day
    is_weekday = current_dt.weekday() < 5  # Monday=0, Friday=4
    is_holiday = current_dt.strftime('%Y-%m-%d') in nyse_holidays
    if is_weekday and not is_holiday:
        # Already a trading day – use it directly
        return current_dt.strftime('%Y-%m-%d')

    # Otherwise, increment to the next business day
    next_biz_day = current_dt + pd.offsets.BusinessDay(1)
    while next_biz_day.strftime('%Y-%m-%d') in nyse_holidays:
        next_biz_day = next_biz_day + pd.offsets.BusinessDay(1)

    return next_biz_day.strftime('%Y-%m-%d')

def get_hf_token():
    return st.secrets.get("HF_TOKEN") or os.getenv("HF_TOKEN")

@st.cache_data(ttl=300)
def load_signals() -> dict:
    token = get_hf_token()
    if not token:
        st.error("HF_TOKEN not configured in Streamlit secrets.")
        return {}
    try:
        path = hf_hub_download(
            repo_id="P2SAMAPA/p2-etf-sdf-engine-results",
            filename="latest_signals.json",
            repo_type="dataset",
            token=token,
            force_download=True,
        )
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Could not load signals: {e}")
        return {}

def render_universe(signal: dict, title: str):
    st.header(title)
    benchmark = signal.get("benchmark", "")
    st.markdown(f"**Benchmark:** {benchmark}")

    if not signal or "top_etf" not in signal:
        st.info("Signal not available yet — run the training + predict workflow first.")
        return

    forecasted = signal.get("forecasted_returns", {})
    scores     = signal.get("scores", {})

    sharpe  = float(signal.get("sharpe_ratio",  0))
    ann_ret = float(signal.get("annual_return", 0))
    max_dd  = float(signal.get("max_drawdown",  0))

    if not np.isfinite(sharpe):  sharpe  = 0.0
    if not np.isfinite(ann_ret): ann_ret = 0.0
    if not np.isfinite(max_dd):  max_dd  = 0.0

    if forecasted:
        sorted_etfs      = sorted(forecasted.items(), key=lambda x: x[1], reverse=True)
        top, top_ret     = sorted_etfs[0]
        top_pct          = float(top_ret) * 100
        sec              = sorted_etfs[1][0] if len(sorted_etfs) > 1 else None
        sec_pct          = float(forecasted.get(sec, 0)) * 100 if sec else 0
    else:
        top, top_pct     = "N/A", 0
        sec, sec_pct     = None, 0

    raw_date  = signal.get("signal_date", "—")
    # --- APPLY NYSE CORRECTION ---
    if raw_date != "—":
        try:
            corrected_date = get_nyse_next_trading_day(raw_date)
        except Exception:
            corrected_date = raw_date
    else:
        corrected_date = raw_date

    generated_at = signal.get("generated_at", "")
    try:
        generated_at = datetime.fromisoformat(generated_at).strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        pass

    if top != "N/A":
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown(f"""
            <div class="hero-card">
                <div class="hero-ticker">{top}</div>
                <div class="hero-return">+{top_pct:.3f}%</div>
                <div>Expected Return – {corrected_date}</div>
            </div>
            """, unsafe_allow_html=True)

            if sec:
                st.markdown(f"📈 **{sec}** +{sec_pct:.3f}%")

            st.caption(f"Generated {generated_at} · "
                       f"Best config: fold={signal.get('best_fold')}  "
                       f"lr={signal.get('best_lr')}  "
                       f"model={signal.get('best_model')}")

        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{ann_ret*100:.1f}%</div>
                <div>Annual Return</div>
            </div>
            <div class="metric-card" style="margin-top:12px">
                <div class="metric-value">{sharpe:.2f}</div>
                <div>Sharpe Ratio</div>
            </div>
            <div class="metric-card" style="margin-top:12px">
                <div class="metric-value">{-max_dd*100:.1f}%</div>
                <div>Max Drawdown</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No predictions available.")
        return

    st.markdown("---")
    st.subheader("All ETF Rankings")
    if forecasted:
        df = pd.DataFrame([
            {
                "ETF":                  k,
                "Expected Return (%)":  round(float(v) * 100, 4),
                "Score":                round(float(scores.get(k, 0)), 4),
            }
            for k, v in forecasted.items()
        ])
        st.dataframe(
            df.sort_values("Expected Return (%)", ascending=False),
            use_container_width=True,
            hide_index=True,
        )

def main():
    st.title("SDF Engine – ETF Signal Generator")

    col1, col2 = st.columns([6, 1])
    with col2:
        if st.button("🔄 Refresh", help="Clear cache and reload signals"):
            st.cache_data.clear()
            st.rerun()

    with st.spinner("Loading signals..."):
        signals = load_signals()

    tab1, tab2 = st.tabs(["📈 Equity ETFs", "🏦 FI & Commodities"])

    with tab1:
        render_universe(signals.get("equity", {}), "US Equity ETFs")

    with tab2:
        render_universe(signals.get("fi_commodity", {}), "Fixed Income & Commodities")

    st.markdown(
        f'<div style="text-align:center; margin-top:2rem; color:#718096;">'
        f'Dashboard loaded: {datetime.now().strftime("%d %b %Y %H:%M")}</div>',
        unsafe_allow_html=True,
    )

if __name__ == "__main__":
    main()
