# ETF Macro Signal Engine

Quantitative ETF selection using Sparse Dynamic Factor Model with macro signals.

## Strategy Logic

```
Return Panel (T×N) → PCA Factor Extraction → Sparse Rotation (VARIMAX)
       ↓
VAR Forecast on Latent Factors → Reconstruct ETF Returns
       ↓
Cross-sectional Scoring (R' × Factor Exposure × Vol Penalty)
       ↓
ETF Picks per Module
```

## Components

| Step | Component | Description |
|------|-----------|-------------|
| 1 | `pca_extractor.py` | Rolling PCA with Bai-Ng IC selection (k=3-5) |
| 2 | `sparse_rotation.py` | VARIMAX/SPCA for sparse loadings |
| 3 | `var_forecast.py` | VAR + Kalman smoother for factor forecast |
| 4 | `return_reconstruction.py` | Reconstruct R'(t+1) from factors |
| 5 | `cross_sectional_score.py` | Rank by R' × exposure × vol penalty |

## Modules

| Module | Engine | Benchmark |
|--------|--------|----------|
| Equity | `equity_engine.py` | SPY |
| FI/Commodity | `fi_commodity_engine.py` | AGG |

## Data

- **Source**: `P2SAMAPA/fi-etf-macro-signal-master-data`
- **Range**: 2008 - 2026 YTD (daily updates)

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Output

- **Streamlit**: Two tabs (Equity | FI/Commodity)
- **HuggingFace**: Signals, scores, and backtest results

## Tech Stack

`sklearn` `statsmodels` `pykalman` `streamlit` `pandas`

*CPU-only, runs <2 min on free tier.*
