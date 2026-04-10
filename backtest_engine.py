# =============================================================================
# Backtest Engine Module - Walk-Forward Testing with Rolling Window
# =============================================================================

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
import warnings
import os

from data_loader import DataLoader
from preprocessor import Preprocessor
from pca_extractor import PCAExtractor
from sparse_rotation import SparseRotation
from var_forecast import VARForecast
from return_reconstruction import ReturnReconstructor
from cross_sectional_score import CrossSectionalScorer
from configs import CONFIG


@dataclass
class BacktestConfig:
    """Configuration for backtest."""
    start_date: str
    end_date: str
    window_type: str  # 'rolling', 'shrinking', 'expanding'
    window_size: int = 252
    initial_size: int = 504
    decay_factor: float = 0.99
    min_size: int = 126
    rebalance_frequency: str = 'daily'
    top_n: int = 3
    residual_penalty: float = 0.1
    factor_exposure_weight: float = 0.3


class BacktestEngine:
    """Walk-forward backtest engine with multiple window strategies."""

    def __init__(
        self,
        assets: list,
        benchmark: str,
        macro_indicators: Optional[list] = None,
        hf_token: Optional[str] = None
    ):
        """
        Initialize backtest engine.

        Args:
            assets: List of asset tickers
            benchmark: Benchmark ticker (not in assets)
            macro_indicators: List of macro indicators to use
            hf_token: HuggingFace token
        """
        self.assets = assets
        self.benchmark = benchmark
        self.macro_indicators = macro_indicators or CONFIG['macro_indicators']

        # Load data
        self.loader = DataLoader(hf_token=hf_token)
        self.preprocessor = Preprocessor()  # For filling missing values
        self.raw_data = self.loader.load_raw_data()

        # Processed data will be stored
        self.returns_ = None
        self.macro_ = None
        self.benchmark_returns_ = None

    def prepare_data(
        self,
        start_date: str,
        end_date: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        """
        Prepare data for backtesting.

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            Tuple of (returns, macro, benchmark_returns)
        """
        # Get returns
        self.returns_ = self.loader.get_universe_data(
            self.assets, start_date, end_date
        )

        # Get macro data
        self.macro_ = self.loader.get_macro_data(
            self.macro_indicators, start_date, end_date
        )

        # Get benchmark returns
        self.benchmark_returns_ = self.loader.get_benchmark_data(
            self.benchmark, start_date, end_date
        )

        return self.returns_, self.macro_, self.benchmark_returns_

    def _run_single_period(
        self,
        train_returns: pd.DataFrame,
        train_macro: pd.DataFrame,
        test_returns: pd.Series,
        config: BacktestConfig
    ) -> Dict:
        """
        Run single backtest period.

        Args:
            train_returns: Training period returns
            train_macro: Training macro data
            test_returns: Test period (single day) returns
            config: Backtest configuration

        Returns:
            Dictionary with results
        """
        try:
            # ----- FILL MISSING VALUES -----
            train_returns_filled = self.preprocessor.fill_missing_values(train_returns)
            train_macro_filled = self.preprocessor.fill_missing_values(train_macro)

            # Step 1: Extract factors with PCA
            pca = PCAExtractor(
                min_factors=CONFIG['sdf_model']['pca']['min_factors'],
                max_factors=CONFIG['sdf_model']['pca']['max_factors'],
                standardize=True
            )
            pca.fit(train_returns_filled)

            factors = pca.get_factors(train_returns_filled.index)

            # Step 2: Sparse rotation
            rotator = SparseRotation(max_iter=CONFIG['sdf_model']['rotation']['max_iter'])
            rotator.fit(pca.loadings_)
            sparse_loadings = SparseRotation.create_sparse_mask(
                rotator.rotated_loadings_,
                top_n=3
            )

            # Step 3: VAR forecast
            forecaster = VARForecast(
                lag_order=CONFIG['sdf_model']['var']['lag_order'],
                use_kalman=CONFIG['sdf_model']['var']['use_kalman']
            )
            forecasted_factors = forecaster.predict_factors(factors, train_macro_filled, horizon=1)

            # Step 4: Reconstruct returns
            reconstructor = ReturnReconstructor(
                residual_penalty=config.residual_penalty,
                top_loadings_per_factor=3
            )
            reconstructor.fit(train_returns_filled, factors, sparse_loadings)
            forecasted_returns, _ = reconstructor.reconstruct(forecasted_factors)

            # Step 5: Score and rank
            scorer = CrossSectionalScorer(
                factor_exposure_weight=config.factor_exposure_weight,
                residual_vol_penalty=config.residual_penalty
            )
            scorer.fit(self.assets, factors.columns.tolist(), sparse_loadings, reconstructor.residual_std_)
            scores = scorer.compute_scores(forecasted_returns, forecasted_factors)
            selected = scorer.select_top_n(scores, config.top_n)

            # Calculate strategy return (equal weight of top N)
            selected_assets = selected['asset'].tolist()
            if len(selected_assets) > 0:
                strategy_return = test_returns[selected_assets].mean()
            else:
                strategy_return = 0

            return {
                'date': test_returns.name,
                'selected_assets': selected_assets,
                'strategy_return': strategy_return,
                'forecasted_returns': dict(zip(self.assets, forecasted_returns)),
                'scores': scores.to_dict()
            }

        except Exception as e:
            warnings.warn(f"Error in period {test_returns.name}: {e}")
            return {
                'date': test_returns.name,
                'selected_assets': [],
                'strategy_return': 0,
                'error': str(e)
            }

    def run_rolling_window(
        self,
        start_date: str,
        end_date: str,
        window_size: int = 252,
        top_n: int = 3,
        max_windows: Optional[int] = None,  # NEW: Limit windows for CI
        rebalance_freq: int = 1  # NEW: Process every Nth day (1=daily, 5=weekly, 21=monthly)
    ) -> pd.DataFrame:
        """
        Run rolling window backtest.

        Args:
            start_date: Backtest start date
            end_date: Backtest end date
            window_size: Rolling window size
            top_n: Number of top ETFs to hold
            max_windows: Maximum number of windows to process (for CI/testing)
            rebalance_freq: Rebalance every N days (default 1=daily, 5=weekly)

        Returns:
            DataFrame with backtest results
        """
        # Prepare data
        self.prepare_data(start_date, end_date)

        config = BacktestConfig(
            start_date=start_date,
            end_date=end_date,
            window_type='rolling',
            window_size=window_size,
            top_n=top_n,
            rebalance_frequency='daily'
        )

        results = []
        all_dates = self.returns_.index[window_size:]
        
        # NEW: Apply rebalance frequency (skip days)
        if rebalance_freq > 1:
            all_dates = all_dates[::rebalance_freq]
            print(f"Rebalancing every {rebalance_freq} days: {len(all_dates)} periods")
        
        # NEW: Limit windows if specified
        if max_windows and len(all_dates) > max_windows:
            print(f"CI MODE: Limiting from {len(all_dates)} to {max_windows} windows")
            all_dates = all_dates[:max_windows]

        total = len(all_dates)
        is_ci = os.getenv('CI_MODE', '').lower() == 'true' or os.getenv('GITHUB_ACTIONS', '').lower() == 'true'
        
        for i, date in enumerate(all_dates):
            # Progress logging (less frequent in CI to reduce overhead)
            if i % (10 if is_ci else 50) == 0 or i == total - 1:
                print(f"Rolling: Processing {date.date()} ({i+1}/{total})")
                
                # CI SAFETY: Check for timeout approaching (8 min mark)
                if is_ci and i > 0 and i % 50 == 0:
                    import time
                    # Note: GitHub Actions has 10 min default timeout, warn at 8 min
                    print(f"CI progress check: {i}/{total} windows processed")

            # Find the index in returns for this date
            date_idx = self.returns_.index.get_loc(date)
            train_returns = self.returns_.iloc[date_idx - window_size:date_idx]
            train_macro = self.macro_.loc[train_returns.index]
            test_returns = self.returns_.iloc[date_idx]

            result = self._run_single_period(
                train_returns, train_macro, test_returns, config
            )
            results.append(result)

        return pd.DataFrame(results)

    @staticmethod
    def calculate_performance(
        returns: pd.Series,
        benchmark_returns: pd.Series
    ) -> Dict:
        """
        Calculate performance metrics.

        Args:
            returns: Strategy returns
            benchmark_returns: Benchmark returns

        Returns:
            Dictionary of performance metrics
        """
        # Align indices
        common_idx = returns.index.intersection(benchmark_returns.index)
        returns = returns.loc[common_idx]
        benchmark = benchmark_returns.loc[common_idx]

        if len(returns) == 0:
            return {
                'total_return': np.nan,
                'annual_return': np.nan,
                'volatility': np.nan,
                'sharpe_ratio': np.nan,
                'max_drawdown': np.nan,
                'win_rate': np.nan,
                'total_trades': 0
            }

        # Cumulative returns
        cum_returns = (1 + returns).cumprod()
        cum_benchmark = (1 + benchmark).cumprod()

        # Metrics
        total_return = cum_returns.iloc[-1] - 1
        years = len(returns) / 252
        annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

        volatility = returns.std() * np.sqrt(252)
        sharpe = annual_return / (volatility + 1e-8)  # add small epsilon to avoid division by zero

        # Max drawdown
        rolling_max = cum_returns.cummax()
        drawdown = (cum_returns - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        # Win rate
        win_rate = (returns > 0).mean()

        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': len(returns),
            'cum_returns': cum_returns,
            'cum_benchmark': cum_benchmark
        }


# Example usage
if __name__ == "__main__":
    # Sample backtest for equity universe
    assets = CONFIG['universes']['equity']['assets']
    benchmark = CONFIG['universes']['equity']['benchmark']

    engine = BacktestEngine(assets, benchmark)

    # Run single strategy
    results = engine.run_rolling_window(
        start_date='2020-01-01',
        end_date='2024-12-31',
        window_size=252,
        top_n=3
    )

    print(f"Backtest completed: {len(results)} periods")
    print(results.head())
