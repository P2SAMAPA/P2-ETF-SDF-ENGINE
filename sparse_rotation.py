# =============================================================================
# Sparse Rotation Module - VARIMAX Rotation for Interpretable Factors
# =============================================================================
"""
Applies VARIMAX rotation to PCA loadings for interpretability.
Makes each factor load primarily on a subset of assets.
"""

import numpy as np
import pandas as pd
from scipy.linalg import svd
from typing import Tuple, Optional
import warnings


class SparseRotation:
    """Apply VARIMAX rotation for sparse factor loadings."""

    def __init__(
        self,
        max_iter: int = 500,
        tol: float = 1e-6,
        gamma: float = 1.0
    ):
        """
        Initialize sparse rotation.

        Args:
            max_iter: Maximum iterations for convergence
            tol: Convergence tolerance
            gamma: Rotation parameter (1 = varimax, 0 = quartimax)
        """
        self.max_iter = max_iter
        self.tol = tol
        self.gamma = gamma

        self.rotation_matrix_ = None
        self.rotated_loadings_ = None

    def _varimax(self, loadings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply VARIMAX rotation.

        VARIMAX maximizes the sum of variances of squared loadings,
        resulting in sparse, interpretable factors.

        Args:
            loadings: Initial loadings matrix (N x k)

        Returns:
            Tuple of (rotated loadings, rotation matrix)
        """
        n, k = loadings.shape
        rotation = np.eye(k)

        for iteration in range(self.max_iter):
            old_rotation = rotation.copy()

            # Current rotated loadings
            L = loadings @ rotation

            # Compute the varimax criterion gradient
            # d(L_kj^2) / d(rotation_mj)
            L_sq = L ** 2

            # Diagonal term
            diag = np.diag(L_sq.T @ L_sq)

            # Off-diagonal term
            off_diag = L_sq.T @ L

            # Update rotation matrix
            numerator = loadings.T @ (L_sq - self.gamma / n * diag)
            denominator = loadings.T @ off_diag

            # Solve using SVD for orthogonal rotation
            try:
                U, S, Vt = svd(numerator @ np.linalg.inv(denominator + 1e-10))
                rotation = Vt.T @ U.T
            except np.linalg.LinAlgError:
                warnings.warn("SVD convergence issue in VARIMAX")
                break

            # Check convergence
            delta = np.max(np.abs(rotation - old_rotation))
            if delta < self.tol:
                print(f"VARIMAX converged in {iteration + 1} iterations")
                break
        else:
            print(f"VARIMAX did not converge in {self.max_iter} iterations")

        rotated = loadings @ rotation

        return rotated, rotation

    def fit(self, loadings: np.ndarray) -> 'SparseRotation':
        """
        Fit sparse rotation to loadings.

        Args:
            loadings: Initial loadings matrix (N x k)

        Returns:
            Self
        """
        self.rotation_matrix_, self.rotated_loadings_ = self._varimax(loadings)
        return self

    def transform(self, loadings: np.ndarray) -> np.ndarray:
        """
        Apply rotation to loadings.

        Args:
            loadings: Loadings matrix

        Returns:
            Rotated loadings
        """
        return loadings @ self.rotation_matrix_

    def get_rotated_loadings(self) -> np.ndarray:
        """
        Get rotated loadings.

        Returns:
            Rotated loadings matrix
        """
        if self.rotated_loadings_ is None:
            raise ValueError("Model not fitted yet")
        return self.rotated_loadings_

    @staticmethod
    def create_sparse_mask(
        loadings: np.ndarray,
        top_n: int = 3,
        threshold: float = 0.1
    ) -> np.ndarray:
        """
        Create sparse mask by keeping only top loadings per factor.

        Args:
            loadings: Rotated loadings matrix (N x k)
            top_n: Number of top loadings to keep per factor
            threshold: Minimum absolute value to keep

        Returns:
            Sparse loadings matrix
        """
        sparse = loadings.copy()
        n, k = sparse.shape

        for j in range(k):
            # Get absolute loadings for this factor
            abs_loadings = np.abs(sparse[:, j])

            # Find indices of top_n loadings
            if top_n is not None and top_n < n:
                threshold_idx = np.argsort(abs_loadings)[::-1][top_n]
                threshold_value = abs_loadings[threshold_idx]
                mask = abs_loadings >= threshold_value
            else:
                mask = abs_loadings >= threshold

            # Zero out loadings below threshold
            sparse[~mask, j] = 0

        return sparse


def interpret_factors(
    loadings: np.ndarray,
    asset_names: list,
    top_n: int = 3
) -> pd.DataFrame:
    """
    Interpret factors by showing top loading assets.

    Args:
        loadings: Loadings matrix (N x k)
        asset_names: List of asset names
        top_n: Number of top assets per factor

    Returns:
        DataFrame with factor interpretations
    """
    n, k = loadings.shape

    interpretations = []

    for j in range(k):
        abs_loadings = np.abs(loadings[:, j])
        top_indices = np.argsort(abs_loadings)[::-1][:top_n]

        top_assets = [(asset_names[i], loadings[i, j]) for i in top_indices]
        interpretations.append({
            'factor': f'Factor_{j+1}',
            'top_assets': ', '.join([f"{a}({v:.3f})" for a, v in top_assets]),
            'interpretation': _infer_interpretation(top_assets)
        })

    return pd.DataFrame(interpretations)


def _infer_interpretation(top_assets: list) -> str:
    """
    Infer factor interpretation from top assets.

    Args:
        top_assets: List of (asset_name, loading) tuples

    Returns:
        String interpretation
    """
    asset_list = [a.lower() for a, _ in top_assets]

    # Simple rule-based interpretation
    if any('gld' in a or 'slv' in a or 'gdx' in a or 'xme' in a for a in asset_list):
        return "Commodities/Metals"
    elif any('tlt' in a or 'agg' in a or 'vcit' in a or 'lqd' in a or 'hyg' in a for a in asset_list):
        return "Fixed Income"
    elif any('xlk' in a or 'qqq' in a for a in asset_list):
        return "Technology/Growth"
    elif any('xlf' in a for a in asset_list):
        return "Financials"
    elif any('xle' in a for a in asset_list):
        return "Energy"
    elif any('xlu' in a for a in asset_list):
        return "Utilities"
    elif any('vnq' in a for a in asset_list):
        return "Real Estate"
    else:
        return "Equity"


# Example usage
if __name__ == "__main__":
    # Create sample loadings
    np.random.seed(42)
    loadings = np.random.randn(10, 4) * 0.5

    print("Original loadings:")
    print(loadings[:5, :3])

    # Apply VARIMAX rotation
    rotator = SparseRotation(max_iter=100)
    rotator.fit(loadings)

    print("\nRotated loadings:")
    print(rotator.rotated_loadings_[:5, :3])

    # Create sparse version
    sparse_loadings = SparseRotation.create_sparse_mask(
        rotator.rotated_loadings_,
        top_n=3
    )

    print("\nSparse loadings:")
    print(sparse_loadings[:5, :3])
