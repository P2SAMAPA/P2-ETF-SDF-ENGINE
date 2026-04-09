# =============================================================================
# Equity Engine Module - US Equity ETF Signal Generation
# =============================================================================
"""
SDF Engine for US Equity ETFs.
Uses SPY as benchmark for performance comparison.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from datetime import datetime
import warnings

from data_loader import DataLoader
from preprocessor import Preprocessor
from pca_extractor import PCAExtractor
from sparse_rotation import SparseRotation
from var_forecast import VARForecast
from return_reconstruction import ReturnReconstructor
from cross_sectional_score import CrossSectionalScorer
from configs import CONFIG


class EquityEngine:
    """Signal generation engine for US Equity ETFs."""

    def __init__(self, hf_token: Optional[str] = None):
        """
        Initialize Equity Engine.

        Args:
            hf_token: HuggingFace token
        """
        self.assets = CONFIG['universes']['equity']['assets']
        self.benchmark = CONFIG['universes']['equity']['benchmark']
        self.name = CONFIG['universes']['equity']['name']

        self.loader = DataLoader(hf_token=hf_token)
        self.preprocessor = Preprocessor()

        # Fitted components
        self.pca_ = None
        self.rotator_ = None
        self.forecaster_ = None
        self.reconstructor_ = None
        self.scorer_ = None
        self.sparse_loadings_ = None

    def prepare_data(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        """
        Load and prepare data.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            Tuple of (returns, macro, benchmark)
        """
        if start_date is None:
            start_date = CONFIG['backtest']['start_date']
        if end_date is None:
            end_date = CONFIG['backtest']['end_date']

        returns = self.loader.get_universe_data(self.assets, start_date, end_date)
        macro = self.loader.get_macro_data(CONFIG['macro_indicators'], start_date, end_date)
        benchmark = self.loader.get_benchmark_data(self.benchmark, start_date, end_date)

        return returns, macro, benchmark

    def fit(self, train_returns: pd.DataFrame, train_macro: pd.DataFrame) -> 'EquityEngine':
        """
        Fit the SDF model on training data.

        Args:
            train_returns: Training period returns
            train_macro: Training period macro data

        Returns:
            Self
        """
        # Step 1: PCA Factor Extraction
        self.pca_ = PCAExtractor(
            min_factors=CONFIG['sdf_model']['pca']['min_factors'],
            max_factors=CONFIG['sdf_model']['pca']['max_factors'],
            standardize=True
        )
        self.pca_.fit(train_returns)
        factors = self.pca_.get_factors(train_returns.index)

        print(f"Equity: Extracted {self.pca_.optimal_k_} factors")
        print(f"  Loadings shape: {self.pca_.loadings_.shape}")  # Should be (11, 5)

        # Step 2: Sparse Rotation
        self.rotator_ = SparseRotation(
            max_iter=CONFIG['sdf_model']['rotation']['max_iter']
        )
        self.rotator_.fit(self.pca_.loadings_)
        
        # Get rotated loadings and ensure correct shape
        rotated_loadings = self.rotator_.get_rotated_loadings()
        print(f"  Rotated loadings shape: {rotated_loadings.shape}")  # Should be (11, 5)
        
        # Create sparse mask
        self.sparse_loadings_ = SparseRotation.create_sparse_mask(
            rotated_loadings,
            top_n=3
        )
        print(f"  Sparse loadings shape: {self.sparse_loadings_.shape}")  # Should be (11, 5)

        # Step 3: VAR Forecast
        self.forecaster_ = VARForecast(
            lag_order=CONFIG['sdf_model']['var']['lag_order'],
            use_kalman=CONFIG['sdf_model']['var']['use_kalman']
        )
        self.forecaster_.fit(factors, train_macro)

        # Step 4: Fit Reconstructor
        self.reconstructor_ = ReturnReconstructor(
            residual_penalty=CONFIG['sdf_model']['signal']['residual_vol_penalty'],
            top_loadings_per_factor=3
        )
        self.reconstructor_.fit(train_returns, factors, self.sparse_loadings_)

        # Step 5: Fit Scorer
        self.scorer_ = CrossSectionalScorer(
            factor_exposure_weight=CONFIG['sdf_model']['signal']['factor_exposure_weight'],
            residual_vol_penalty=CONFIG['sdf_model']['signal']['residual_vol_penalty']
        )
        self.scorer_.fit(
            self.assets,
            factors.columns.tolist(),
            self.sparse_loadings_,
            self.reconstructor_.residual_std_
        )

        return self

    def predict(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate forecasts for next period.

        Returns:
            Tuple of (forecasted_returns, forecasted_factors)
        """
        if self.pca_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Get last factors from PCA
        factors = self.pca_.get_factors(self.pca_.factors_)

        # Forecast factors
        forecasted_factors = self.forecaster_.predict_factors(
            factors, self.forecaster_.data_, horizon=1
        )

        # Reconstruct returns
        forecasted_returns, residual_std = self.reconstructor_.reconstruct(
            forecasted_factors
        )

        return forecasted_returns, forecasted_factors

    def get_signals(self, top_n: int = 3) -> pd.DataFrame:
        """
        Get trading signals for current period.

        Args:
            top_n: Number of top ETFs to select

        Returns:
            DataFrame with signals and scores
        """
        forecasted_returns, forecasted_factors = self.predict()

        scores = self.scorer_.compute_scores(
            forecasted_returns, forecasted_factors
        )

        selected = self.scorer_.select_top_n(scores, top_n)

        return selected

    def get_factor_interpretations(self) -> pd.DataFrame:
        """
        Get factor interpretations.

        Returns:
            DataFrame with factor interpretations
        """
        if self.rotator_ is None:
            return pd.DataFrame()

        from sparse_rotation import interpret_factors

        return interpret_factors(
            self.rotator_.rotated_loadings_,
            self.assets,
            top_n=3
        )

    def generate_signals_pipeline(
        self,
        train_window: pd.DataFrame,
        train_macro: pd.DataFrame,
        top_n: int = 3
    ) -> Dict:
        """
        Full pipeline: fit and generate signals.

        Args:
            train_window: Training window returns
            train_macro: Training window macro data
            top_n: Number of top ETFs

        Returns:
            Dictionary with signals, scores, and model info
        """
        # Fit model
        self.fit(train_window, train_macro)

        # Generate signals
        forecasted_returns, forecasted_factors = self.predict()
        signals = self.get_signals(top_n)
        factor_interpretations = self.get_factor_interpretations()

        return {
            'date': train_window.index[-1],
            'selected_assets': signals['asset'].tolist(),
            'signals': signals,
            'forecasted_returns': dict(zip(self.assets, forecasted_returns)),
            'factor_interpretations': factor_interpretations,
            'n_factors': self.pca_.optimal_k_,
            'explained_variance': self.pca_.explained_variance_.tolist()
        }

    def run_historical_signals(
        self,
        start_date: str,
        end_date: str,
        window_size: int = 252,
        top_n: int = 3
    ) -> pd.DataFrame:
        """
        Run historical signal generation.

        Args:
            start_date: Start date
            end_date: End date
            window_size: Training window size
            top_n: Number of top ETFs

        Returns:
            DataFrame with historical signals
        """
        # Load data
        returns, macro, _ = self.prepare_data(start_date, end_date)

        results = []
        dates = returns.index[window_size:]

        for i, date in enumerate(dates):
            if i % 50 == 0:
                print(f"Equity: {date.date()} ({i+1}/{len(dates)})")

            train_returns = returns.iloc[i:i + window_size]
            train_macro = macro.loc[train_returns.index]

            try:
                signal_result = self.generate_signals_pipeline(
                    train_returns, train_macro, top_n
                )
                results.append(signal_result)
            except Exception as e:
                warnings.warn(f"Error at {date}: {e}")

        return pd.DataFrame(results)


# Allow warnings import
import warnings


# Example usage
if __name__ == "__main__":
    engine = EquityEngine()

    # Load sample data
    returns, macro, benchmark = engine.prepare_data(
        start_date='2024-01-01',
        end_date='2024-12-31'
    )

    print(f"Assets: {engine.assets}")
    print(f"Benchmark: {engine.benchmark}")
    print(f"Returns shape: {returns.shape}")

    # Run single period
    train_returns = returns.iloc[:200]
    train_macro = macro.loc[train_returns.index]

    result = engine.generate_signals_pipeline(train_returns, train_macro, top_n=3)

    print(f"\nSelected assets: {result['selected_assets']}")
    print(f"Number of factors: {result['n_factors']}")
