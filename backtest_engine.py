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

    def prepare_data(self, start_date: str, end_date: str):
        # get_universe_data and get_benchmark_data now return log returns
        self.returns_           = self.loader.get_universe_data(self.assets, start_date, end_date)
        self.macro_             = self.loader.get_macro_data(self.macro_indicators, start_date, end_date)
        self.benchmark_returns_ = self.loader.get_benchmark_data(self.benchmark, start_date, end_date)
        return self.returns_, self.macro_, self.benchmark_returns_

    def _safe_return(self, v) -> float:
        """
        Safety clamp only — data_loader already returns decimal log returns.
        This just guards against any residual outliers from the reconstruction.
        """
        v = float(v)
        if not np.isfinite(v):
            return 0.0
        # Clip to sane daily range (±50% daily move is extreme but possible in crises)
        return float(np.clip(v, -0.5, 0.5))

    def _run_single_period(
        self,
        train_returns: pd.DataFrame,
        train_macro: pd.DataFrame,
        test_returns: pd.Series,
        config: BacktestConfig
    ) -> Dict:
        try:
            tr = self.preprocessor.fill_missing_values(train_returns)
            tm = self.preprocessor.fill_missing_values(train_macro)

            # PCA
            pca = PCAExtractor(
                min_factors=CONFIG['sdf_model']['pca']['min_factors'],
                max_factors=CONFIG['sdf_model']['pca']['max_factors'],
                standardize=True
            )
            pca.fit(tr)
            factors = pca.get_factors(tr.index)

            # Sparse rotation
            rotator = SparseRotation(max_iter=CONFIG['sdf_model']['rotation']['max_iter'])
            rotator.fit(pca.loadings_)
            sparse_loadings = SparseRotation.create_sparse_mask(rotator.rotated_loadings_, top_n=3)

            # VAR forecast
            forecaster = VARForecast(
                lag_order=CONFIG['sdf_model']['var']['lag_order'],
                use_kalman=CONFIG['sdf_model']['var']['use_kalman']
            )
            forecasted_factors = forecaster.predict_factors(factors, tm, horizon=1)

            # Reconstruct returns
            reconstructor = ReturnReconstructor(
                residual_penalty=config.residual_penalty,
                top_loadings_per_factor=3
            )
            reconstructor.fit(tr, factors, sparse_loadings)
            forecasted_returns, _ = reconstructor.reconstruct(forecasted_factors)

            # Apply safety clamp (data is already in decimal return form)
            forecasted_dict = {
                asset: self._safe_return(ret)
                for asset, ret in zip(self.assets, forecasted_returns)
            }

            # Score
            scorer = CrossSectionalScorer(
                factor_exposure_weight=config.factor_exposure_weight,
                residual_vol_penalty=config.residual_penalty
            )
            scorer.fit(self.assets, factors.columns.tolist(), sparse_loadings, reconstructor.residual_std_)
            scores_df = scorer.compute_scores(forecasted_returns, forecasted_factors)
            selected  = scorer.select_top_n(scores_df, config.top_n)

            scores_dict    = {row['asset']: float(row['composite_score']) for _, row in scores_df.iterrows()}
            selected_assets = selected['asset'].tolist()

            # test_returns are already decimal log returns from data_loader
            strategy_return = float(np.clip(
                test_returns[selected_assets].mean(), -0.5, 0.5
            )) if selected_assets else 0.0

            return {
                'date':               test_returns.name,
                'selected_assets':    selected_assets,
                'strategy_return':    strategy_return,
                'forecasted_returns': forecasted_dict,
                'scores':             scores_dict,
            }

        except Exception as e:
            warnings.warn(f"Error in period {test_returns.name}: {e}")
            import traceback; traceback.print_exc()
            return {
                'date': test_returns.name,
                'selected_assets': [],
                'strategy_return': 0.0,
                'forecasted_returns': {},
                'scores': {},
                'error': str(e),
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
        )

        results   = []
        all_dates = self.returns_.index[window_size:]

        if rebalance_freq > 1:
            all_dates = all_dates[::rebalance_freq]

        if max_windows and len(all_dates) > max_windows:
            print(f"CI MODE: Limiting to {max_windows} windows")
            all_dates = all_dates[:max_windows]

        total = len(all_dates)
        is_ci = (os.getenv('CI_MODE', '').lower() == 'true'
                 or os.getenv('GITHUB_ACTIONS', '').lower() == 'true')

        for i, date in enumerate(all_dates):
            if i % (10 if is_ci else 50) == 0 or i == total - 1:
                print(f"Rolling: {date.date()} ({i+1}/{total})")

            date_idx     = self.returns_.index.get_loc(date)
            train_returns = self.returns_.iloc[date_idx - window_size:date_idx]
            train_macro   = self.macro_.loc[train_returns.index]
            test_returns  = self.returns_.iloc[date_idx]

            results.append(self._run_single_period(train_returns, train_macro, test_returns, config))

        return pd.DataFrame(results)

    @staticmethod
    def calculate_performance(returns: pd.Series, benchmark_returns: pd.Series) -> Dict:
        common_idx = returns.index.intersection(benchmark_returns.index)
        returns    = returns.loc[common_idx]
        benchmark  = benchmark_returns.loc[common_idx]

        if len(returns) == 0:
            return {'total_return': 0.0, 'annual_return': 0.0, 'volatility': 0.0,
                    'sharpe_ratio': 0.0, 'max_drawdown': 0.0, 'win_rate': 0.0, 'total_trades': 0}

        cum_returns  = (1 + returns).cumprod()
        total_return = cum_returns.iloc[-1] - 1
        years        = len(returns) / 252

        annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0.1 else total_return * (252 / len(returns))
        annual_return = 0.0 if not np.isfinite(annual_return) else float(annual_return)

        volatility = returns.std() * np.sqrt(252)
        sharpe     = float(np.clip(annual_return / (volatility + 1e-8), -10, 10))

        rolling_max  = cum_returns.cummax()
        max_drawdown = float(((cum_returns - rolling_max) / rolling_max).min())

        return {
            'total_return':  float(total_return),
            'annual_return': annual_return,
            'volatility':    float(volatility),
            'sharpe_ratio':  sharpe,
            'max_drawdown':  max_drawdown,
            'win_rate':      float((returns > 0).mean()),
            'total_trades':  len(returns),
            'cum_returns':   cum_returns,
            'cum_benchmark': (1 + benchmark).cumprod(),
        }


if __name__ == "__main__":
    assets    = CONFIG['universes']['equity']['assets']
    benchmark = CONFIG['universes']['equity']['benchmark']
    engine    = BacktestEngine(assets, benchmark)
    results   = engine.run_rolling_window('2020-01-01', '2024-12-31', window_size=252, top_n=3)
    print(f"Backtest completed: {len(results)} periods")
    print(results.head())
