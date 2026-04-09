# =============================================================================
# Preprocessor Module - Data Cleaning & Standardization
# =============================================================================
"""
Handles data preprocessing for the SDF model:
- Fill missing values (forward-fill VCIT)
- Calculate daily returns
- Standardize data
- Handle outliers
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
import warnings


class Preprocessor:
    """Preprocess ETF data for SDF model."""

    def __init__(self, fill_method: str = 'ffill'):
        """
        Initialize preprocessor.

        Args:
            fill_method: Method for filling missing values ('ffill', 'bfill', 'interpolate')
        """
        self.fill_method = fill_method

    def fill_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fill missing values in the dataset.

        Note: VCIT has null values throughout - will be forward-filled.

        Args:
            df: DataFrame with price data

        Returns:
            DataFrame with filled missing values
        """
        result = df.copy()

        # Count initial missing
        initial_missing = result.isnull().sum().sum()

        if self.fill_method == 'ffill':
            result = result.ffill()
        elif self.fill_method == 'bfill':
            result = result.bfill()
        elif self.fill_method == 'interpolate':
            result = result.interpolate(method='time')
        else:
            raise ValueError(f"Unknown fill method: {self.fill_method}")

        # Fill any remaining NaN (e.g., at start of series)
        result = result.fillna(method='ffill').fillna(method='bfill')

        final_missing = result.isnull().sum().sum()
        if final_missing > 0:
            warnings.warn(f"{final_missing} missing values could not be filled")

        if initial_missing > 0:
            print(f"Filled {initial_missing} missing values using {self.fill_method}")

        return result

    @staticmethod
    def calculate_returns(prices: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate daily returns from prices.

        Args:
            prices: DataFrame of prices

        Returns:
            DataFrame of daily returns
        """
        returns = prices.pct_change()

        # Handle infinite values (from zero division)
        returns = returns.replace([np.inf, -np.inf], np.nan)

        return returns

    @staticmethod
    def standardize_returns(
        returns: pd.DataFrame,
        window: Optional[int] = None
    ) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """
        Standardize returns (z-score normalization).

        Args:
            returns: DataFrame of returns
            window: Rolling window for mean/std estimation (None = full period)

        Returns:
            Tuple of (standardized returns, rolling mean, rolling std)
        """
        if window is None:
            # Use full period statistics
            mean = returns.mean()
            std = returns.std()
        else:
            # Use rolling statistics
            mean = returns.rolling(window=window).mean()
            std = returns.rolling(window=window).std()

            # Fill initial NaN with full period stats
            full_mean = returns.mean()
            full_std = returns.std()
            mean = mean.fillna(full_mean)
            std = std.fillna(full_std)

        # Standardize
        standardized = (returns - mean) / std

        # Replace inf/nan
        standardized = standardized.replace([np.inf, -np.inf], np.nan)

        return standardized, mean, std

    @staticmethod
    def winsorize_returns(returns: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
        """
        Winsorize returns to remove extreme values.

        Args:
            returns: DataFrame of returns
            threshold: Number of standard deviations to cap at

        Returns:
            DataFrame with winsorized returns
        """
        result = returns.copy()

        # Calculate z-scores
        z_scores = np.abs((returns - returns.mean()) / returns.std())

        # Cap at threshold
        result = result.where(z_scores <= threshold, returns.mean() + threshold * returns.std() * np.sign(returns))
        result = result.where(z_scores <= threshold, returns.mean() + threshold * returns.std() * np.sign(returns))

        return result

    def preprocess_pipeline(
        self,
        prices: pd.DataFrame,
        calculate_returns_flag: bool = True,
        standardize: bool = True,
        winsorize_threshold: Optional[float] = 3.0
    ) -> Tuple[pd.DataFrame, dict]:
        """
        Full preprocessing pipeline.

        Args:
            prices: DataFrame of prices
            calculate_returns_flag: Whether to calculate returns
            standardize: Whether to standardize
            winsorize_threshold: Winsorization threshold (None to skip)

        Returns:
            Tuple of (processed data, metadata dict)
        """
        result = prices.copy()
        metadata = {}

        # Step 1: Fill missing values
        result = self.fill_missing_values(result)

        # Step 2: Calculate returns if needed
        if calculate_returns_flag:
            result = self.calculate_returns(result)
            metadata['data_type'] = 'returns'

            # Step 3: Winsorize
            if winsorize_threshold is not None:
                result = self.winsorize_returns(result, winsorize_threshold)
                metadata['winsorized'] = winsorize_threshold

            # Step 4: Standardize
            if standardize:
                result, mean, std = self.standardize_returns(result)
                metadata['standardized'] = True
                metadata['mean'] = mean
                metadata['std'] = std
        else:
            metadata['data_type'] = 'prices'

        # Drop first row (NaN from returns calculation)
        result = result.dropna(how='all')

        print(f"Preprocessed data shape: {result.shape}")

        return result, metadata


def get_train_test_dates(
    full_start: str,
    full_end: str,
    test_pct: float = 0.2
) -> Tuple[str, str, str, str]:
    """
    Get train/test split dates for walk-forward validation.

    Args:
        full_start: Full period start date
        full_end: Full period end date
        test_pct: Percentage of data for test period

    Returns:
        Tuple of (train_start, train_end, test_start, test_end)
    """
    all_dates = pd.date_range(start=full_start, end=full_end, freq='B')  # Business days
    n = len(all_dates)

    split_idx = int(n * (1 - test_pct))

    train_end = all_dates[split_idx - 1]
    test_start = all_dates[split_idx]

    return full_start, str(train_end.date()), str(test_start.date()), full_end


# Example usage
if __name__ == "__main__":
    from data_loader import DataLoader

    # Load sample data
    loader = DataLoader()
    prices = loader.get_universe_data(
        assets=['QQQ', 'XLK', 'GLD', 'TLT'],
        start_date='2024-01-01',
        end_date='2024-12-31'
    )

    print(f"Raw prices shape: {prices.shape}")
    print(f"Raw prices missing:\n{prices.isnull().sum()}")

    # Preprocess
    preprocessor = Preprocessor()
    processed, metadata = preprocessor.preprocess_pipeline(prices)

    print(f"\nProcessed shape: {processed.shape}")
    print(f"Processed missing:\n{processed.isnull().sum()}")
    print(f"\nMetadata: {metadata}")
