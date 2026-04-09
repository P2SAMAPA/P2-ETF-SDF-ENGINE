# =============================================================================
# PCA Extractor Module - Factor Extraction with Bai-Ng IC Criterion
# =============================================================================
"""
Rolling PCA with automatic factor selection using Bai-Ng Information Criterion.
Estimates the optimal number of factors k based on statistical criteria.
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from typing import Tuple, Optional, Dict
import warnings


class PCAExtractor:
    """Extract latent factors using PCA with IC-based selection."""

    def __init__(
        self,
        min_factors: int = 3,
        max_factors: int = 5,
        ic_criterion: str = 'bai_ng',
        standardize: bool = True
    ):
        """
        Initialize PCA extractor.

        Args:
            min_factors: Minimum number of factors to consider
            max_factors: Maximum number of factors to consider
            ic_criterion: Information criterion ('bai_ng', 'aic', 'bic')
            standardize: Whether to standardize data before PCA
        """
        self.min_factors = min_factors
        self.max_factors = max_factors
        self.ic_criterion = ic_criterion
        self.standardize = standardize

        # Fitted attributes
        self.optimal_k_ = None
        self.loadings_ = None
        self.factors_ = None
        self.explained_variance_ = None

    def _compute_ic(
        self,
        eigenvalues: np.ndarray,
        n_samples: int,
        n_features: int
    ) -> Tuple[np.ndarray, int]:
        """
        Compute Bai-Ng Information Criterion for factor selection.

        IC(k) = ln(V_k) + k * (T^{-1} * sum_{i=k+1}^{T} 1/(i-1)) * sigma^2_T

        where V_k = sum_{j=k+1}^{T} lambda_j^2

        Args:
            eigenvalues: Eigenvalues from covariance matrix (descending order)
            n_samples: Number of observations (T)
            n_features: Number of variables (N)

        Returns:
            Tuple of (IC values for each k, optimal k)
        """
        T, N = n_samples, n_features
        eigenvalues = eigenvalues[:self.max_factors + 1]  # Include extra for comparison

        # Residual variance
        eigs_sq = eigenvalues ** 2
        total_var = np.sum(eigs_sq)

        ic_values = []

        for k in range(self.min_factors, self.max_factors + 1):
            # Residual sum of squares
            # RSS(k) = sum of squared eigenvalues for factors k+1 to N
            residual = np.sum(eigs_sq[k:])

            # Bai-Ng penalty term
            # penalty = (N^2 * T) / (T - k - 1) * sigma^2 * sum_{i=k+1}^{N} 1/(i - k)
            sigma_sq = np.mean(eigs_sq[:k]) if k > 0 else total_var / N

            # Penalty factor
            numerator = N ** 2
            denominator = max(T - k - 1, 1)
            penalty_factor = numerator / denominator

            # Sum of inverse eigenvalues for penalty
            inv_sum = np.sum(1.0 / (np.arange(k + 1, N + 1) - k))

            penalty = penalty_factor * sigma_sq * inv_sum / N

            # BIC-style criterion
            ic = np.log(residual / N) + penalty

            ic_values.append(ic)

        # Find optimal k (minimum IC)
        ic_values = np.array(ic_values)
        optimal_idx = np.argmin(ic_values)
        optimal_k = self.min_factors + optimal_idx

        return ic_values, optimal_k

    def fit(
        self,
        returns: pd.DataFrame,
        current_date: Optional[pd.Timestamp] = None
    ) -> 'PCAExtractor':
        """
        Fit PCA model to returns data.

        Args:
            returns: DataFrame of returns (T x N)
            current_date: Current date for logging

        Returns:
            Self
        """
        T, N = returns.shape

        if T < self.max_factors + 10:
            warnings.warn(
                f"Insufficient data points ({T}) for PCA with {self.max_factors} max factors"
            )
            self.optimal_k_ = min(self.min_factors, max(1, T // 10))
        else:
            # Compute covariance matrix
            cov_matrix = returns.cov().values

            # Eigenvalue decomposition
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

            # Sort descending
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]

            # Compute IC to find optimal k
            ic_values, optimal_k = self._compute_ic(eigenvalues, T, N)

            self.optimal_k_ = optimal_k

            if current_date:
                print(f"{current_date.date()}: Optimal factors k={optimal_k} "
                      f"(IC={ic_values[optimal_k - self.min_factors]:.4f})")

        # Fit final PCA with optimal k
        self.pca_ = PCA(n_components=self.optimal_k_)
        data = returns.values

        if self.standardize:
            mean = data.mean(axis=0)
            std = data.std(axis=0)
            std[std == 0] = 1  # Avoid division by zero
            data = (data - mean) / std

        self.factors_ = self.pca_.fit_transform(data)
        self.loadings_ = self.pca_.components_.T  # (N x k)
        self.explained_variance_ = self.pca_.explained_variance_ratio_

        return self

    def transform(self, returns: pd.DataFrame) -> np.ndarray:
        """
        Transform returns to factor space.

        Args:
            returns: DataFrame of returns

        Returns:
            Array of factor values
        """
        data = returns.values

        if self.standardize:
            mean = data.mean(axis=0)
            std = data.std(axis=0)
            std[std == 0] = 1
            data = (data - mean) / std

        return self.pca_.transform(data)

    def get_loadings(self) -> pd.DataFrame:
        """
        Get factor loadings matrix.

        Returns:
            DataFrame (N assets x k factors)
        """
        if self.loadings_ is None:
            raise ValueError("Model not fitted yet")

        return pd.DataFrame(
            self.loadings_,
            index=self.factors_df_.index if hasattr(self, 'factors_df_') else None,
            columns=[f'Factor_{i+1}' for i in range(self.loadings_.shape[1])]
        )

    def get_factors(self, dates: pd.Index) -> pd.DataFrame:
        """
        Get factor time series.

        Args:
            dates: Date index

        Returns:
            DataFrame of factor values
        """
        if self.factors_ is None:
            raise ValueError("Model not fitted yet")

        return pd.DataFrame(
            self.factors_,
            index=dates,
            columns=[f'F{i+1}' for i in range(self.factors_.shape[1])]
        )


def rolling_pca(
    returns: pd.DataFrame,
    window_size: int,
    min_factors: int = 3,
    max_factors: int = 5
) -> Tuple[pd.DataFrame, pd.DataFrame, list]:
    """
    Perform rolling PCA on returns.

    Args:
        returns: DataFrame of returns (T x N)
        window_size: Size of rolling window
        min_factors: Minimum number of factors
        max_factors: Maximum number of factors

    Returns:
        Tuple of (factors_df, loadings_df, optimal_k_list)
    """
    dates = returns.index
    n = len(returns)

    all_factors = []
    all_loadings = []
    optimal_k_list = []

    for i in range(window_size, n):
        window_data = returns.iloc[i - window_size:i]

        extractor = PCAExtractor(
            min_factors=min_factors,
            max_factors=max_factors,
            standardize=True
        )

        extractor.fit(window_data, dates[i])

        factors = extractor.transform(returns.iloc[i:i+1])
        all_factors.append(factors[0])
        all_loadings.append(extractor.loadings_)
        optimal_k_list.append(extractor.optimal_k_)

    # Combine results
    factors_df = pd.DataFrame(
        all_factors,
        index=dates[window_size:],
        columns=[f'F{i+1}' for i in range(all_factors[0].shape[0])]
    )

    loadings_df = pd.DataFrame(
        all_loadings,
        index=dates[window_size:],
        columns=[f'Factor_{i+1}' for i in range(all_loadings[0].shape[1])]
    ).T

    return factors_df, loadings_df, optimal_k_list


# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', '2024-12-31', freq='B')
    n = len(dates)

    # Generate correlated returns
    returns = pd.DataFrame(
        np.random.randn(n, 5) @ np.random.randn(5, 5) * 0.02 + 0.0001,
        index=dates,
        columns=['ETF1', 'ETF2', 'ETF3', 'ETF4', 'ETF5']
    )

    print(f"Returns shape: {returns.shape}")

    # Fit PCA
    extractor = PCAExtractor(min_factors=2, max_factors=4)
    extractor.fit(returns.iloc[:100], dates[99])

    print(f"Optimal k: {extractor.optimal_k_}")
    print(f"Explained variance: {extractor.explained_variance_}")
    print(f"Loadings shape: {extractor.loadings_.shape}")
