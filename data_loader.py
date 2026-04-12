# =============================================================================
# Data Loader Module - HuggingFace Dataset Integration
# =============================================================================
"""
Loads ETF price data from HuggingFace dataset.
Dataset: P2SAMAPA/fi-etf-macro-signal-master-data
Date range: 2008-01-01 to 2026 YTD (updated daily)

FIX: Dataset stores closing prices (e.g. 150.23). All ETF and benchmark
columns are converted to log returns (ln(P_t / P_{t-1})) before being
returned. Macro indicator columns are returned as-is since they are
already in level/rate form (VIX, yield spreads, etc.).
"""

import os
import pandas as pd
import numpy as np
from typing import Optional, Dict, List
import warnings

try:
    from datasets import load_dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    warnings.warn("datasets library not available. Install with: pip install datasets")

from configs import CONFIG


class DataLoader:
    """Load and manage ETF data from HuggingFace."""

    def __init__(self, hf_token: Optional[str] = None):
        self.dataset_name = CONFIG['huggingface']['dataset_source']
        self.hf_token = hf_token or os.getenv(CONFIG['huggingface']['token_env'])
        self._cache = {}

    def load_raw_data(self) -> pd.DataFrame:
        """
        Load raw data from HuggingFace dataset.
        Returns DataFrame with all columns and date index.
        Values are stored as closing prices — use the helper methods
        below which apply the correct transformation per column type.
        """
        if 'raw_data' in self._cache:
            return self._cache['raw_data'].copy()

        if not HF_AVAILABLE:
            raise ImportError("datasets library required. Run: pip install datasets")

        print(f"Loading dataset from HuggingFace: {self.dataset_name}")

        try:
            dataset = load_dataset(
                self.dataset_name,
                split='train',
                token=self.hf_token
            )

            df = dataset.to_pandas()

            # Handle index column
            if '__index_level_0__' in df.columns:
                df['date'] = pd.to_datetime(df['__index_level_0__'])
                df = df.drop(columns=['__index_level_0__'])
            elif 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            else:
                raise ValueError("No date column found in dataset")

            df = df.set_index('date').sort_index()

            print(f"Loaded {len(df)} rows from {df.index.min()} to {df.index.max()}")
            self._cache['raw_data'] = df

            return df.copy()

        except Exception as e:
            print(f"Error loading dataset: {e}")
            raise

    @staticmethod
    def _prices_to_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
        """
        Convert a DataFrame of closing prices to log returns.
        log_return(t) = ln(P_t / P_{t-1})
        First row becomes NaN and is dropped.
        Any inf/-inf (from zero or negative prices) is set to NaN then ffilled.
        """
        log_ret = np.log(prices / prices.shift(1))
        log_ret = log_ret.replace([np.inf, -np.inf], np.nan)
        log_ret = log_ret.dropna(how='all')          # drop the all-NaN first row
        log_ret = log_ret.fillna(method='ffill')     # forward-fill any remaining NaNs
        log_ret = log_ret.fillna(0.0)                # fill any leading NaNs with 0
        return log_ret

    @staticmethod
    def _series_prices_to_log_returns(prices: pd.Series) -> pd.Series:
        """Same as above but for a single Series (benchmark)."""
        log_ret = np.log(prices / prices.shift(1))
        log_ret = log_ret.replace([np.inf, -np.inf], np.nan)
        log_ret = log_ret.dropna()
        log_ret = log_ret.fillna(0.0)
        return log_ret

    def get_universe_data(
        self,
        assets: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get log returns for a specific asset universe.
        Prices are converted to log returns before returning.

        Returns:
            DataFrame of log returns (decimal form, e.g. 0.0082 = 0.82%)
        """
        df = self.load_raw_data()

        available_assets = [col for col in assets if col in df.columns]
        missing = set(assets) - set(available_assets)
        if missing:
            print(f"Warning: Assets not in dataset: {missing}")

        # Convert prices → log returns BEFORE date filtering so we don't
        # lose the first return at the start of the requested window
        prices = df[available_assets].copy()
        returns = self._prices_to_log_returns(prices)

        # Now apply date filter
        if start_date:
            returns = returns[returns.index >= pd.to_datetime(start_date)]
        if end_date:
            returns = returns[returns.index <= pd.to_datetime(end_date)]

        print(f"Universe data: {len(available_assets)} assets, {len(returns)} dates  "
              f"(log returns, mean={returns.mean().mean():.5f})")

        return returns

    def get_macro_data(
        self,
        indicators: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get macro indicator data.
        Macro columns (VIX, spreads, yields) are levels/rates — returned as-is,
        NOT converted to returns, since the VAR uses them as control variables.
        """
        if indicators is None:
            indicators = CONFIG['macro_indicators']

        df = self.load_raw_data()

        available = [col for col in indicators if col in df.columns]
        missing = set(indicators) - set(available)
        if missing:
            print(f"Warning: Macro indicators not in dataset: {missing}")

        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]

        return df[available].copy()

    def get_benchmark_data(
        self,
        benchmark: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.Series:
        """
        Get benchmark log returns.
        Prices are converted to log returns before returning.

        Returns:
            Series of log returns (decimal form)
        """
        df = self.load_raw_data()

        if benchmark not in df.columns:
            raise ValueError(f"Benchmark {benchmark} not in dataset")

        # Convert prices → log returns before date filtering
        log_ret = self._series_prices_to_log_returns(df[benchmark])

        if start_date:
            log_ret = log_ret[log_ret.index >= pd.to_datetime(start_date)]
        if end_date:
            log_ret = log_ret[log_ret.index <= pd.to_datetime(end_date)]

        return log_ret

    @staticmethod
    def get_us_trading_days(start_date: str, end_date: str) -> pd.DatetimeIndex:
        """Get US trading day calendar (excludes weekends and major holidays)."""
        all_days = pd.date_range(start=start_date, end=end_date, freq='D')
        trading_days = all_days[all_days.dayofweek < 5]

        holidays = {
            pd.Timestamp('2008-01-01'), pd.Timestamp('2009-01-01'),
            pd.Timestamp('2010-01-01'), pd.Timestamp('2011-01-01'),
            pd.Timestamp('2012-01-02'), pd.Timestamp('2013-01-01'),
            pd.Timestamp('2014-01-01'), pd.Timestamp('2015-01-01'),
            pd.Timestamp('2016-01-01'), pd.Timestamp('2017-01-02'),
            pd.Timestamp('2018-01-01'), pd.Timestamp('2019-01-01'),
            pd.Timestamp('2020-01-01'), pd.Timestamp('2021-01-01'),
            pd.Timestamp('2022-01-03'), pd.Timestamp('2023-01-02'),
            pd.Timestamp('2024-01-01'), pd.Timestamp('2025-01-01'),
            pd.Timestamp('2026-01-01'),
        }

        return trading_days.difference(holidays)


def load_hf_data(
    assets: List[str],
    benchmark: Optional[str] = None,
    macro: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    hf_token: Optional[str] = None
) -> Dict[str, pd.DataFrame]:
    """Convenience function to load all required data as log returns."""
    loader = DataLoader(hf_token=hf_token)

    result = {
        'returns': loader.get_universe_data(assets, start_date, end_date),
        'macro':   loader.get_macro_data(macro, start_date, end_date),
    }

    if benchmark:
        result['benchmark'] = loader.get_benchmark_data(benchmark, start_date, end_date)

    return result


if __name__ == "__main__":
    loader = DataLoader()
    df = loader.load_raw_data()
    print(f"\nAvailable columns: {list(df.columns)}")
    print(f"\nSample raw (prices):\n{df.head()}")

    # Quick sanity check on returns
    from configs import CONFIG
    eq_assets = CONFIG['universes']['equity']['assets']
    returns = loader.get_universe_data(eq_assets, '2024-01-01', '2024-12-31')
    print(f"\nSample log returns:\n{returns.head()}")
    print(f"\nReturn stats:\n{returns.describe()}")
