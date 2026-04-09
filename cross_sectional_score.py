# =============================================================================
# Cross-Sectional Scoring Module - ETF Ranking & Selection
# =============================================================================
"""
Computes cross-sectional scores for ETF selection.
Score = R_hat * Factor Exposure * Residual Vol Penalty
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
import warnings


class CrossSectionalScorer:
    """Score and rank ETFs based on forecasted returns and factor exposure."""

    def __init__(
        self,
        factor_exposure_weight: float = 0.3,
        residual_vol_penalty: float = 0.1
    ):
        """
        Initialize cross-sectional scorer.

        Args:
            factor_exposure_weight: Weight for factor alignment score
            residual_vol_penalty: Weight for residual volatility penalty
        """
        self.factor_exposure_weight = factor_exposure_weight
        self.residual_vol_penalty = residual_vol_penalty

        self.asset_names_ = None
        self.factor_names_ = None
        self.sparse_loadings_ = None
        self.scores_ = None

    def fit(
        self,
        asset_names: List[str],
        factor_names: List[str],
        sparse_loadings: np.ndarray,
        residual_std: np.ndarray
    ) -> 'CrossSectionalScorer':
        """
        Fit scorer with loadings and residual information.

        Args:
            asset_names: List of asset names
            factor_names: List of factor names
            sparse_loadings: Sparse loadings matrix (N x k)
            residual_std: Residual standard deviation (N,)

        Returns:
            Self
        """
        self.asset_names_ = asset_names
        self.factor_names_ = factor_names
        self.sparse_loadings_ = sparse_loadings
        self.residual_std_ = residual_std

        return self

    def compute_factor_exposure_score(
        self,
        forecasted_factors: np.ndarray,
        loadings: np.ndarray
    ) -> np.ndarray:
        """
        Compute factor exposure score for each asset.

        Measures how much each asset benefits from predicted factor movements.

        Args:
            forecasted_factors: Array of forecasted factors (k,)
            loadings: Loading matrix (N x k)

        Returns:
            Array of factor exposure scores (N,)
        """
        # For each asset: sum of loading * forecasted factor
        # Higher score = asset benefits from favorable factor movements
        exposures = loadings * forecasted_factors  # Element-wise
        factor_scores = np.sum(exposures, axis=1)

        # Normalize to [0, 1]
        if np.std(factor_scores) > 0:
            factor_scores = (factor_scores - np.min(factor_scores)) / (np.max(factor_scores) - np.min(factor_scores))

        return factor_scores

    def compute_residual_vol_penalty(
        self,
        residual_std: np.ndarray
    ) -> np.ndarray:
        """
        Compute residual volatility penalty.

        Penalizes assets with high idiosyncratic volatility.

        Args:
            residual_std: Residual standard deviation (N,)

        Returns:
            Array of penalties in [0, 1] range
        """
        # Lower residual vol = higher score
        penalty = 1 / (1 + self.residual_vol_penalty * residual_std)

        return penalty

    def compute_scores(
        self,
        forecasted_returns: np.ndarray,
        forecasted_factors: np.ndarray,
        sparse_loadings: Optional[np.ndarray] = None,
        residual_std: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        """
        Compute composite scores for all assets.

        Score = R_hat * factor_exposure * residual_vol_penalty

        Args:
            forecasted_returns: Array of forecasted returns (N,)
            forecasted_factors: Array of forecasted factors (k,)
            sparse_loadings: Optional sparse loadings (if not fitted)
            residual_std: Optional residual std (if not fitted)

        Returns:
            DataFrame with scores for each asset
        """
        # Use fitted or provided values
        loadings = sparse_loadings if sparse_loadings is not None else self.sparse_loadings_
        res_std = residual_std if residual_std is not None else self.residual_std_

        if loadings is None or res_std is None:
            raise ValueError("Provide loadings/residual or fit the model first")

        assets = self.asset_names_ if self.asset_names_ else [f'Asset_{i}' for i in range(len(forecasted_returns))]

        # Component scores
        factor_exposure = self.compute_factor_exposure_score(forecasted_factors, loadings)
        vol_penalty = self.compute_residual_vol_penalty(res_std)

        # Normalize forecasted returns to [0, 1]
        if np.std(forecasted_returns) > 0:
            norm_returns = (forecasted_returns - np.min(forecasted_returns)) / (np.max(forecasted_returns) - np.min(forecasted_returns))
        else:
            norm_returns = np.ones_like(forecasted_returns) * 0.5

        # Composite score
        composite = norm_returns * (1 + self.factor_exposure_weight * factor_exposure) * vol_penalty

        # Create results DataFrame
        results = pd.DataFrame({
            'asset': assets,
            'expected_return': forecasted_returns,
            'norm_return': norm_returns,
            'factor_exposure_score': factor_exposure,
            'residual_vol_penalty': vol_penalty,
            'composite_score': composite
        })

        # Sort by composite score
        results = results.sort_values('composite_score', ascending=False)

        self.scores_ = results

        return results

    def select_top_n(
        self,
        scores_df: pd.DataFrame,
        top_n: int = 3
    ) -> pd.DataFrame:
        """
        Select top N ETFs based on composite score.

        Args:
            scores_df: DataFrame with scores
            top_n: Number of top ETFs to select

        Returns:
            DataFrame of selected ETFs
        """
        return scores_df.head(top_n)

    def get_factor_loadings_summary(
        self,
        top_assets: List[str],
        loadings: np.ndarray,
        forecasted_factors: np.ndarray
    ) -> pd.DataFrame:
        """
        Get factor loading summary for selected assets.

        Args:
            top_assets: List of selected asset names
            loadings: Full loadings matrix
            forecasted_factors: Forecasted factor values

        Returns:
            DataFrame showing why each asset was selected
        """
        if self.factor_names_ is None:
            factor_names = [f'Factor_{i+1}' for i in range(loadings.shape[1])]
        else:
            factor_names = self.factor_names_

        asset_indices = [self.asset_names_.index(a) for a in top_assets if a in self.asset_names_]

        summary_data = []
        for idx in asset_indices:
            loading = loadings[idx]
            contribution = loading * forecasted_factors

            summary_data.append({
                'asset': self.asset_names_[idx],
                'factor_loadings': dict(zip(factor_names, loading.round(3))),
                'factor_contributions': dict(zip(factor_names, contribution.round(6))),
                'total_exposure': np.sum(contribution)
            })

        return pd.DataFrame(summary_data)


def rank_etfs_pipeline(
    assets: List[str],
    forecasted_returns: np.ndarray,
    forecasted_factors: np.ndarray,
    sparse_loadings: np.ndarray,
    residual_std: np.ndarray,
    factor_names: List[str],
    top_n: int = 3,
    factor_exposure_weight: float = 0.3,
    residual_vol_penalty: float = 0.1
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full pipeline for ETF ranking.

    Args:
        assets: List of asset names
        forecasted_returns: Forecasted returns
        forecasted_factors: Forecasted factors
        sparse_loadings: Sparse loadings matrix
        residual_std: Residual standard deviations
        factor_names: Factor names
        top_n: Number of top ETFs
        factor_exposure_weight: Factor exposure weight
        residual_vol_penalty: Residual penalty weight

    Returns:
        Tuple of (all scores DataFrame, selected ETFs DataFrame)
    """
    scorer = CrossSectionalScorer(
        factor_exposure_weight=factor_exposure_weight,
        residual_vol_penalty=residual_vol_penalty
    )

    scorer.fit(assets, factor_names, sparse_loadings, residual_std)

    scores = scorer.compute_scores(
        forecasted_returns,
        forecasted_factors
    )

    selected = scorer.select_top_n(scores, top_n)

    return scores, selected


# Example usage
if __name__ == "__main__":
    # Sample data
    assets = ['QQQ', 'XLK', 'XLF', 'GLD', 'TLT', 'HYG']
    factors = ['F1_Rates', 'F2_Credit', 'F3_Growth']
    n_assets = len(assets)
    n_factors = len(factors)

    # Sample loadings (3 factors, 6 assets)
    loadings = np.array([
        [-0.5, 0.3, 0.8],  # QQQ
        [-0.3, 0.2, 0.9],  # XLK
        [0.2, 0.7, 0.4],   # XLF
        [0.4, -0.2, 0.3],  # GLD
        [0.7, 0.1, -0.2],  # TLT
        [0.1, 0.8, 0.2]    # HYG
    ])

    # Sample forecasted factors
    forecasted_factors = np.array([0.01, -0.005, 0.02])

    # Sample forecasted returns
    forecasted_returns = loadings @ forecasted_factors + np.random.randn(n_assets) * 0.002

    # Sample residual std
    residual_std = np.array([0.01, 0.012, 0.015, 0.008, 0.01, 0.018])

    # Rank
    scores, selected = rank_etfs_pipeline(
        assets=assets,
        forecasted_returns=forecasted_returns,
        forecasted_factors=forecasted_factors,
        sparse_loadings=loadings,
        residual_std=residual_std,
        factor_names=factors,
        top_n=3
    )

    print("All ETF Scores:")
    print(scores)
    print("\nSelected ETFs:")
    print(selected)
