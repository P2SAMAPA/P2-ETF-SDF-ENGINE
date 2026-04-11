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
    window_type: str
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
        self.assets = assets
        self.benchmark = benchmark
        self.macro_indicators = macro_indicators or CONFIG['macro_indicators']
        self.loader = DataLoader(hf_token=hf_token)
        self.preprocessor = Preprocessor()
        self.raw_data = self.loader.load_raw_data()
        self.returns_ = None
        self.macro_ = None
        self.benchmark_returns_ = None

    def prepare_data(
        self,
        start_date: str,
        end_date: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        self.returns_ = self.loader.get_universe_data(self.assets, start_date, end_date)
        self.macro_ = self.loader.get_macro_data(self.macro_indicators, start_date, end_date)
        self.benchmark_returns_ = self.loader.get_benchmark_data(self.benchmark, start_date, end_date)
        return self.returns_, self.macro_, self.benchmark_returns_

    def _scale_returns(self, returns_value):
        """
        Scale returns to decimal form.
        Detects if returns are in percentage (>10) and converts to decimal.
        """
        if abs(returns_value) > 10:
            # Returns are in percentage points (e.g., 85.86 = 85.86%)
            return returns_value / 100.0
        elif abs(returns_value) > 1:
            # Returns might be in decimal * 100 (e.g., 1.47 = 1.47%)
            return returns_value / 100.0
        else:
            # Returns are already in decimal form (e.g., 0.0147 = 1.47%)
            return returns_value

    def _run_single_period(
        self,
        train_returns: pd.DataFrame,
        train_macro: pd.DataFrame,
        test_returns: pd.Series,
        config: BacktestConfig
    ) -> Dict:
        try:
            train_returns_filled = self.preprocessor.fill_missing_values(train_returns)
            train_macro_filled = self.preprocessor.fill_missing_values(train_macro)

            # Step 1: PCA
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
            sparse_loadings = SparseRotation.create_sparse_mask(rotator.rotated_loadings_, top_n=3)

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

            # FIX: Scale forecasted returns to decimal form
            scaled_forecasted = {}
            for asset, ret in zip(self.assets, forecasted_returns):
                scaled_ret = self._scale_returns(ret)
                # Safety clip: daily returns should never exceed ±50%
                scaled_forecasted[asset] = float(np.clip(scaled_ret, -0.5, 0.5))

            # Step 5: Score and rank
            scorer = CrossSectionalScorer(
                factor_exposure_weight=config.factor_exposure_weight,
                residual_vol_penalty=config.residual_penalty
            )
            scorer.fit(self.assets, factors.columns.tolist(), sparse_loadings, reconstructor.residual_std_)
            
            # Get scores DataFrame
            scores_df = scorer.compute_scores(forecasted_returns, forecasted_factors)
            selected = scorer.select_top_n(scores_df, config.top_n)

            # Convert scores to simple {asset: score} dict
            scores_dict = {}
            for _, row in scores_df.iterrows():
                asset_name = row['asset']
                scores_dict[asset_name] = float(row['composite_score'])
            
            # FIX: Scale strategy return from test period
            selected_assets = selected['asset'].tolist()
            if len(selected_assets) > 0:
                raw_strategy_return = test_returns[selected_assets].mean()
                strategy_return = self._scale_returns(raw_strategy_return)
                strategy_return = float(np.clip(strategy_return, -0.5, 0.5))
            else:
                strategy_return = 0

            return {
                'date': test_returns.name,
                'selected_assets': selected_assets,
                'strategy_return': strategy_return,
                'forecasted_returns': scaled_forecasted,
                'scores': scores_dict
            }

        except Exception as e:
            warnings.warn(f"Error in period {test_returns.name}: {e}")
            import traceback
            traceback.print_exc()
            return {
                'date': test_returns.name,
                'selected_assets': [],
                'strategy_return': 0,
                'forecasted_returns': {},
                'scores': {},
                'error': str(e)
            }

    def run_rolling_window(
        self,
        start_date: str,
        end_date: str,
        window_size: int = 252,
        top_n: int = 3,
        max_windows: Optional[int] = None,
        rebalance_freq: int = 1
    ) -> pd.DataFrame:
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
        
        if rebalance_freq > 1:
            all_dates = all_dates[::rebalance_freq]
            print(f"Rebalancing every {rebalance_freq} days: {len(all_dates)} periods")
        
        if max_windows and len(all_dates) > max_windows:
            print(f"CI MODE: Limiting from {len(all_dates)} to {max_windows} windows")
            all_dates = all_dates[:max_windows]

        total = len(all_dates)
        is_ci = os.getenv('CI_MODE', '').lower() == 'true' or os.getenv('GITHUB_ACTIONS', '').lower() == 'true'
        
        for i, date in enumerate(all_dates):
            if i % (10 if is_ci else 50) == 0 or i == total - 1:
                print(f"Rolling: Processing {date.date()} ({i+1}/{total})")

            date_idx = self.returns_.index.get_loc(date)
            train_returns = self.returns_.iloc[date_idx - window_size:date_idx]
            train_macro = self.macro_.loc[train_returns.index]
            test_returns = self.returns_.iloc[date_idx]

            result = self._run_single_period(train_returns, train_macro, test_returns, config)
            results.append(result)

        return pd.DataFrame(results)

    @staticmethod
    def calculate_performance(returns: pd.Series, benchmark_returns: pd.Series) -> Dict:
        common_idx = returns.index.intersection(benchmark_returns.index)
        returns = returns.loc[common_idx]
        benchmark = benchmark_returns.loc[common_idx]

        if len(returns) == 0:
            return {
                'total_return': 0.0,
                'annual_return': 0.0,
                'volatility': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'total_trades': 0
            }

        cum_returns = (1 + returns).cumprod()
        total_return = cum_returns.iloc[-1] - 1
        years = len(returns) / 252
        
        # Better annualization for short periods
        if years > 0.1:
            annual_return = (1 + total_return) ** (1 / years) - 1
        else:
            annual_return = total_return * (252 / len(returns))
        
        annual_return = 0.0 if not np.isfinite(annual_return) else float(annual_return)

        volatility = returns.std() * np.sqrt(252)
        sharpe = annual_return / (volatility + 1e-8) if volatility > 0 else 0.0
        sharpe = float(np.clip(sharpe, -10, 10))

        rolling_max = cum_returns.cummax()
        drawdown = (cum_returns - rolling_max) / rolling_max
        max_drawdown = float(drawdown.min())
        win_rate = float((returns > 0).mean())

        return {
            'total_return': float(total_return),
            'annual_return': annual_return,
            'volatility': float(volatility),
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': len(returns),
            'cum_returns': cum_returns,
            'cum_benchmark': (1 + benchmark).cumprod()
        }


if __name__ == "__main__":
    assets = CONFIG['universes']['equity']['assets']
    benchmark = CONFIG['universes']['equity']['benchmark']
    engine = BacktestEngine(assets, benchmark)
    results = engine.run_rolling_window('2020-01-01', '2024-12-31', window_size=252, top_n=3)
    print(f"Backtest completed: {len(results)} periods")
    print(results.head())
