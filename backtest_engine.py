# =============================================================================
# Backtest Engine Module - Walk-Forward Testing with 3 Window Strategies
# =============================================================================
"""
Performs walk-forward backtesting using three window strategies:
A) Rolling Window - Fixed size, slides forward
B) Shrinking Window - Starts large, exponentially decays
C) Expanding Window - Grows from start, never shrinks
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Callable
from dataclasses import dataclass
import warnings

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
        self.raw_data = self.loader.load_raw_data()

        # Preprocessor
        self.preprocessor = Preprocessor()

        # Store processed data
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
            train_macro: Training period macro data
            test_returns: Test period (single day) returns
            config: Backtest configuration

        Returns:
            Dictionary with results
        """
        try:
            # Step 1: Extract factors with PCA
            pca = PCAExtractor(
                min_factors=CONFIG['sdf_model']['pca']['min_factors'],
                max_factors=CONFIG['sdf_model']['pca']['max_factors'],
                standardize=True
            )
            pca.fit(train_returns)

            factors = pca.get_factors(train_returns.index)

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
            forecasted_factors = forecaster.predict_factors(factors, train_macro, horizon=1)

            # Step 4: Reconstruct returns
            reconstructor = ReturnReconstructor(
                residual_penalty=config.residual_penalty,
                top_loadings_per_factor=3
            )
            reconstructor.fit(train_returns, factors, sparse_loadings)
            forecasted_returns, residual_std = reconstructor.reconstruct(forecasted_factors)

            # Step 5: Score and rank
            scorer = CrossSectionalScorer(
                factor_exposure_weight=config.factor_exposure_weight,
                residual_vol_penalty=config.residual_penalty
            )
            scorer.fit(self.assets, factors.columns.tolist(), sparse_loadings, residual_std)
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
        rebalance_freq: str = 'daily',
        top_n: int = 3
    ) -> pd.DataFrame:
        """
        Run rolling window backtest.

        Args:
            start_date: Backtest start date
            end_date: Backtest end date
            window_size: Rolling window size
            rebalance_freq: Rebalancing frequency
            top_n: Number of top ETFs to hold

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
            rebalance_frequency=rebalance_freq
        )

        results = []
        dates = self.returns_.index[window_size:]

        for i, date in enumerate(dates):
            if i % 50 == 0:
                print(f"Rolling: Processing {date.date()} ({i+1}/{len(dates)})")

            train_returns = self.returns_.iloc[i:i + window_size]
            train_macro = self.macro_.loc[train_returns.index]
            test_returns = self.returns_.iloc[i + window_size]

            result = self._run_single_period(
                train_returns, train_macro, test_returns, config
            )
            results.append(result)

        return pd.DataFrame(results)

    def run_shrinking_window(
        self,
        start_date: str,
        end_date: str,
        initial_size: int = 504,
        decay_factor: float = 0.99,
        min_size: int = 126,
        rebalance_freq: str = 'daily',
        top_n: int = 3
    ) -> pd.DataFrame:
        """
        Run shrinking window backtest.

        Window starts at initial_size and shrinks exponentially.

        Args:
            start_date: Backtest start date
            end_date: Backtest end date
            initial_size: Initial window size
            decay_factor: Exponential decay factor
            min_size: Minimum window size
            rebalance_freq: Rebalancing frequency
            top_n: Number of top ETFs to hold

        Returns:
            DataFrame with backtest results
        """
        self.prepare_data(start_date, end_date)

        results = []
        dates = self.returns_.index[initial_size:]
        current_window = initial_size

        for i, date in enumerate(dates):
            if i % 50 == 0:
                print(f"Shrinking: Processing {date.date()} window={current_window:.0f}")

            # Update window size with decay
            current_window = max(min_size, current_window * decay_factor)

            config = BacktestConfig(
                start_date=start_date,
                end_date=end_date,
                window_type='shrinking',
                window_size=int(current_window),
                top_n=top_n
            )

            train_returns = self.returns_.iloc[i:i + int(current_window)]
            train_macro = self.macro_.loc[train_returns.index]
            test_returns = self.returns_.iloc[i + int(current_window)]

            result = self._run_single_period(
                train_returns, train_macro, test_returns, config
            )
            result['window_size'] = current_window
            results.append(result)

        return pd.DataFrame(results)

    def run_expanding_window(
        self,
        start_date: str,
        end_date: str,
        min_size: int = 126,
        initial_gap: int = 252,
        rebalance_freq: str = 'daily',
        top_n: int = 3
    ) -> pd.DataFrame:
        """
        Run expanding window backtest.

        Window starts at min_size and grows to include all history.

        Args:
            start_date: Backtest start date
            end_date: Backtest end date
            min_size: Minimum starting window
            initial_gap: Initial skip for burn-in
            rebalance_freq: Rebalancing frequency
            top_n: Number of top ETFs to hold

        Returns:
            DataFrame with backtest results
        """
        self.prepare_data(start_date, end_date)

        results = []
        dates = self.returns_.index[initial_gap:]

        for i, date in enumerate(dates):
            if i % 50 == 0:
                print(f"Expanding: Processing {date.date()} (obs={i + initial_gap})")

            # Window grows: from 0 to current index
            window_end = i + initial_gap

            config = BacktestConfig(
                start_date=start_date,
                end_date=end_date,
                window_type='expanding',
                window_size=window_end,
                top_n=top_n
            )

            train_returns = self.returns_.iloc[:window_end]
            train_macro = self.macro_.loc[train_returns.index]
            test_returns = self.returns_.iloc[window_end]

            result = self._run_single_period(
                train_returns, train_macro, test_returns, config
            )
            result['window_size'] = window_end
            results.append(result)

        return pd.DataFrame(results)

    def run_all_strategies(
        self,
        start_date: str,
        end_date: str,
        **kwargs
    ) -> Dict[str, pd.DataFrame]:
        """
        Run all three window strategies.

        Args:
            start_date: Backtest start date
            end_date: Backtest end date
            **kwargs: Additional arguments for individual strategies

        Returns:
            Dictionary with results for each strategy
        """
        results = {}

        # Rolling
        print("\n=== Running Rolling Window ===")
        results['rolling'] = self.run_rolling_window(
            start_date, end_date,
            window_size=kwargs.get('window_size', 252)
        )

        # Shrinking
        print("\n=== Running Shrinking Window ===")
        results['shrinking'] = self.run_shrinking_window(
            start_date, end_date,
            initial_size=kwargs.get('initial_size', 504),
            decay_factor=kwargs.get('decay_factor', 0.99)
        )

        # Expanding
        print("\n=== Running Expanding Window ===")
        results['expanding'] = self.run_expanding_window(
            start_date, end_date,
            initial_gap=kwargs.get('initial_gap', 252)
        )

        return results

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

        # Cumulative returns
        cum_returns = (1 + returns).cumprod()
        cum_benchmark = (1 + benchmark).cumprod()

        # Metrics
        total_return = cum_returns.iloc[-1] - 1
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1

        volatility = returns.std() * np.sqrt(252)
        sharpe = annual_return / volatility if volatility > 0 else 0

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
