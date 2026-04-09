# =============================================================================
# Return Reconstruction Module - Sparse Return Forecast from Factors
# =============================================================================
"""
Reconstructs individual ETF returns from forecasted factors using sparse loadings.
Return(t+1) = Lambda @ F_hat(t+1) + residual
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
import warnings


class ReturnReconstructor:
    """Reconstruct ETF returns from latent factors."""

    def __init__(
        self,
        residual_penalty: float = 0.1,
        top_loadings_per_factor: int = 3
    ):
        """
        Initialize return reconstructor.

        Args:
            residual_penalty: Penalty weight for high residual volatility
            top_loadings_per_factor: Number of top loadings to keep per factor
        """
        self.residual_penalty = residual_penalty
        self.top_loadings_per_factor = top_loadings_per_factor

        self.sparse_loadings_ = None
        self.residual_variance_ = None

    def fit(
        self,
        returns: pd.DataFrame,
        factors: pd.DataFrame,
        loadings: np.ndarray
    ) -> 'ReturnReconstructor':
        """
        Fit reconstructor to historical data.

        Args:
            returns: Historical returns (T x N)
            factors: Historical factor values (T x k)
            loadings: Loading matrix (N x k)

        Returns:
            Self
        """
        # Create sparse loadings
        self.sparse_loadings_ = self._create_sparse_loadings(loadings)

        # Compute residual variance for each asset
        # residual = actual_return - predicted_return
        predicted = factors.values @ self.sparse_loadings_.T
        residuals = returns.values - predicted

        self.residual_variance_ = np.var(residuals, axis=0)
        self.residual_std_ = np.sqrt(self.residual_variance_)

        # Store mean returns for shrinkage
        self.mean_returns_ = returns.mean().values

        print(f"Fitted reconstructor:")
        print(f"  Sparse loadings shape: {self.sparse_loadings_.shape}")
        print(f"  Mean residual std: {np.mean(self.residual_std_):.6f}")

        return self

    def _create_sparse_loadings(self, loadings: np.ndarray) -> np.ndarray:
        """
        Create sparse loadings by keeping only top loadings per factor.

        Args:
            loadings: Full loadings matrix (N x k)

        Returns:
            Sparse loadings matrix
        """
        sparse = loadings.copy()
        n, k = sparse.shape

        for j in range(k):
            abs_loadings = np.abs(sparse[:, j])

            # Get threshold for top_n
            threshold_idx = np.argsort(abs_loadings)[::-1][self.top_loadings_per_factor]
            threshold = abs_loadings[threshold_idx]

            # Zero out smaller loadings
            sparse[abs_loadings < threshold, j] = 0

        return sparse

    def reconstruct(
        self,
        forecasted_factors: np.ndarray,
        returns_std: Optional[pd.DataFrame] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reconstruct returns from forecasted factors.

        Args:
            forecasted_factors: Array of forecasted factors (k,)
            returns_std: Optional standard deviations for unstandardization

        Returns:
            Tuple of (reconstructed returns, residual volatilities)
        """
        if self.sparse_loadings_ is None:
            raise ValueError("Model not fitted yet")

        # Reconstruct returns: R_hat = F_hat @ Lambda^T
        reconstructed = forecasted_factors @ self.sparse_loadings_.T

        # Apply residual volatility penalty
        # Penalize assets with high idiosyncratic volatility
        penalty = 1 / (1 + self.residual_penalty * self.residual_variance_)
        adjusted_returns = reconstructed * penalty

        return adjusted_returns, self.residual_std_

    def get_sparse_loadings_df(
        self,
        asset_names: list,
        factor_names: Optional[list] = None
    ) -> pd.DataFrame:
        """
        Get sparse loadings as DataFrame.

        Args:
            asset_names: List of asset names
            factor_names: Optional list of factor names

        Returns:
            DataFrame of sparse loadings
        """
        if self.sparse_loadings_ is None:
            raise ValueError("Model not fitted yet")

        if factor_names is None:
            factor_names = [f'Factor_{i+1}' for i in range(self.sparse_loadings_.shape[1])]

        return pd.DataFrame(
            self.sparse_loadings_,
            index=asset_names,
            columns=factor_names
        )

    def get_residual_variance_df(
        self,
        asset_names: list
    ) -> pd.DataFrame:
        """
        Get residual variance for each asset.

        Args:
            asset_names: List of asset names

        Returns:
            DataFrame of residual variances
        """
        if self.residual_variance_ is None:
            raise ValueError("Model not fitted yet")

        return pd.DataFrame({
            'asset': asset_names,
            'residual_variance': self.residual_variance_,
            'residual_std': self.residual_std_,
            'penalty': 1 / (1 + self.residual_penalty * self.residual_variance_)
        }).set_index('asset')


def reconstruct_returns_pipeline(
    returns: pd.DataFrame,
    factors: pd.DataFrame,
    loadings: np.ndarray,
    forecasted_factors: np.ndarray,
    residual_penalty: float = 0.1,
    top_loadings: int = 3
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Full pipeline for return reconstruction.

    Args:
        returns: Historical returns
        factors: Historical factors
        loadings: Loading matrix
        forecasted_factors: Forecasted factors
        residual_penalty: Residual penalty weight
        top_loadings: Number of top loadings per factor

    Returns:
        Tuple of (reconstructed returns, residual std, sparse loadings)
    """
    reconstructor = ReturnReconstructor(
        residual_penalty=residual_penalty,
        top_loadings_per_factor=top_loadings
    )

    reconstructor.fit(returns, factors, loadings)

    reconstructed, residual_std = reconstructor.reconstruct(forecasted_factors)

    return reconstructed, residual_std, reconstructor.sparse_loadings_


def unstandardize_returns(
    standardized_returns: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray
) -> np.ndarray:
    """
    Convert standardized returns back to actual returns.

    Args:
        standardized_returns: Standardized returns
        mean: Mean of original returns
        std: Std of original returns

    Returns:
        Unstandardized returns
    """
    return standardized_returns * std + mean


# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    n_assets = 10
    n_factors = 3
    n_obs = 100

    # Sample returns
    returns = pd.DataFrame(
        np.random.randn(n_obs, n_assets) * 0.01,
        columns=[f'ETF{i+1}' for i in range(n_assets)]
    )

    # Sample factors
    factors = pd.DataFrame(
        np.random.randn(n_obs, n_factors) * 0.01,
        columns=[f'F{i+1}' for i in range(n_factors)]
    )

    # Sample loadings
    loadings = np.random.randn(n_assets, n_factors) * 0.5

    # Forecasted factors (e.g., from VAR)
    forecasted_factors = np.random.randn(n_factors) * 0.01

    # Reconstruct
    reconstructed, residual_std, sparse = reconstruct_returns_pipeline(
        returns, factors, loadings, forecasted_factors,
        residual_penalty=0.1, top_loadings=3
    )

    print(f"Reconstructed returns: {reconstructed}")
    print(f"Residual std: {residual_std}")
    print(f"Sparse loadings:\n{sparse}")
