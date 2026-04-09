# =============================================================================
# Data Loader Module - HuggingFace Dataset Integration
# =============================================================================
"""
Loads ETF price data from HuggingFace dataset.
Dataset: P2SAMAPA/fi-etf-macro-signal-master-data
Date range: 2008-01-01 to 2026-04-08 (YTD, updated daily)
"""

import os
import pandas as pd
from datetime import datetime
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
        """
        Initialize data loader.

        Args:
            hf_token: HuggingFace API token (optional, uses HF_TOKEN env var if not provided)
        """
        self.dataset_name = CONFIG['huggingface']['dataset_source']
        self.hf_token = hf_token or os.getenv(CONFIG['huggingface']['token_env'])
        self._cache = {}

    def load_raw_data(self) -> pd.DataFrame:
        """
        Load raw data from HuggingFace dataset.

        Returns:
            DataFrame with all ETF columns and date index
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

    def get_universe_data(
        self,
        assets: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get data for specific asset universe.

        Args:
            assets: List of asset ticker symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with specified assets filtered to date range
        """
        df = self.load_raw_data()

        # Filter available columns
        available_assets = [col for col in assets if col in df.columns]
        missing = set(assets) - set(available_assets)
        if missing:
            print(f"Warning: Assets not in dataset: {missing}")

        # Filter date range
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]

        # Return only requested columns
        result = df[available_assets].copy()

        print(f"Universe data: {len(available_assets)} assets, {len(result)} dates")

        return result

    def get_macro_data(
        self,
        indicators: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get macro indicator data.

        Args:
            indicators: List of macro indicator names
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with macro indicators
        """
        if indicators is None:
            indicators = CONFIG['macro_indicators']

        df = self.load_raw_data()

        # Filter available indicators
        available = [col for col in indicators if col in df.columns]
        missing = set(indicators) - set(available)
        if missing:
            print(f"Warning: Macro indicators not in dataset: {missing}")

        # Filter date range
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
        Get benchmark returns.

        Args:
            benchmark: Benchmark ticker symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            Series of benchmark returns
        """
        df = self.load_raw_data()

        if benchmark not in df.columns:
            raise ValueError(f"Benchmark {benchmark} not in dataset")

        # Filter date range
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]

        return df[benchmark].copy()

    @staticmethod
    def get_us_trading_days(start_date: str, end_date: str) -> pd.DatetimeIndex:
        """
        Get US trading day calendar (excludes weekends and major holidays).

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DatetimeIndex of US trading days
        """
        # Generate all calendar days
        all_days = pd.date_range(start=start_date, end=end_date, freq='D')

        # Remove weekends (5=Saturday, 6=Sunday)
        trading_days = all_days[all_days.dayofweek < 5]

        # Major US market holidays (simplified list)
        holidays = {
            # New Year's Day
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

        trading_days = trading_days.difference(holidays)

        return trading_days


def load_hf_data(
    assets: List[str],
    benchmark: Optional[str] = None,
    macro: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    hf_token: Optional[str] = None
) -> Dict[str, pd.DataFrame]:
    """
    Convenience function to load all required data.

    Args:
        assets: List of asset tickers
        benchmark: Benchmark ticker
        macro: List of macro indicators
        start_date: Start date
        end_date: End date
        hf_token: HuggingFace token

    Returns:
        Dictionary with keys: 'returns', 'benchmark', 'macro'
    """
    loader = DataLoader(hf_token=hf_token)

    result = {
        'returns': loader.get_universe_data(assets, start_date, end_date),
        'macro': loader.get_macro_data(macro, start_date, end_date),
    }

    if benchmark:
        result['benchmark'] = loader.get_benchmark_data(benchmark, start_date, end_date)

    return result


# Example usage
if __name__ == "__main__":
    # Test loading
    loader = DataLoader()
    df = loader.load_raw_data()
    print(f"\nAvailable columns: {list(df.columns)}")
    print(f"\nSample data:\n{df.head()}")
