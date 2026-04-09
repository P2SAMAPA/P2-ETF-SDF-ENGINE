# =============================================================================
# VAR Forecast Module - Vector Autoregression + Kalman Filter
# =============================================================================
"""
Applies VAR model on latent factors with optional Kalman filter/smoother
for state estimation and forecasting.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict
import warnings

try:
    from statsmodels.tsa.api import VAR
    from statsmodels.tsa.statespace.kalman_filter import KalmanFilter
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    warnings.warn("statsmodels not available. Install with: pip install statsmodels")


class VARForecast:
    """VAR model with Kalman filter for factor forecasting."""

    def __init__(
        self,
        lag_order: int = 2,
        use_kalman: bool = True,
        use_smoother: bool = True,
        include_macro: bool = True
    ):
        """
        Initialize VAR forecast model.

        Args:
            lag_order: Number of lags for VAR model
            use_kalman: Whether to use Kalman filter
            use_smoother: Whether to use Kalman smoother
            include_macro: Whether to include macro variables
        """
        self.lag_order = lag_order
        self.use_kalman = use_kalman
        self.use_smoother = use_smoother
        self.include_macro = include_macro

        self.model_ = None
        self.results_ = None
        self.kalman_filter_ = None
        self.data_ = None

    def fit(
        self,
        factors: pd.DataFrame,
        macro: Optional[pd.DataFrame] = None
    ) -> 'VARForecast':
        """
        Fit VAR model to factors.

        F(t) = A1*F(t-1) + A2*F(t-2) + ... + B*macro(t) + epsilon

        Args:
            factors: DataFrame of factor time series (T x k)
            macro: Optional macro variable time series

        Returns:
            Self
        """
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels required for VAR. Install: pip install statsmodels")

        # Combine factors with macro if provided
        if self.include_macro and macro is not None:
            # Rename macro columns to avoid overlap with factor columns
            macro_renamed = macro.copy()
            macro_renamed.columns = [f'macro_{col}' for col in macro.columns]
            
            # Align indices
            combined = factors.join(macro_renamed, how='inner')
            self.data_ = combined.dropna()
        else:
            self.data_ = factors.dropna()

        # Ensure we have enough data
        if len(self.data_) < self.lag_order + 5:
            warnings.warn(f"Insufficient data ({len(self.data_)} rows) for VAR with {self.lag_order} lags")
            # Use simple autoregressive model instead
            self.lag_order_ = max(1, len(self.data_) // 10)
        else:
            self.lag_order_ = self.lag_order

        try:
            # Fit VAR model
            self.model_ = VAR(self.data_)
            self.results_ = self.model_.fit(maxlags=self.lag_order_, ic='aic', trend='c')
            print(f"VAR fitted with {self.results_.k_ar} lags")
            print(f"AIC: {self.results_.aic:.4f}, BIC: {self.results_.bic:.4f}")
        except (np.linalg.LinAlgError, ValueError) as e:
            warnings.warn(f"VAR fitting failed: {e}. Using simple autoregressive model.")
            self.results_ = None
            # Fit simple AR model for each factor independently
            self._fit_ar_models(factors)

        return self

    def _fit_ar_models(self, factors: pd.DataFrame):
        """
        Fit simple AR models when VAR fails.
        
        Args:
            factors: Factor time series
        """
        from statsmodels.tsa.ar_model import AutoReg
        
        self.ar_models_ = []
        self.ar_coeffs_ = []
        
        for col in factors.columns:
            try:
                model = AutoReg(factors[col].dropna(), lags=min(2, len(factors) // 5))
                result = model.fit()
                self.ar_models_.append(result)
                self.ar_coeffs_.append(result.params)
            except:
                # If AR fails, use mean forecast
                self.ar_models_.append(None)
                self.ar_coeffs_.append(np.array([factors[col].mean()]))
        
        print(f"Fitted AR models for {len(self.ar_models_)} factors")

    def forecast(
        self,
        steps: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate factor forecasts.

        Args:
            steps: Number of steps ahead to forecast

        Returns:
            Tuple of (forecasted factors, forecast covariance)
        """
        if self.results_ is not None:
            # Get last observations
            y = self.data_.values[-self.results_.k_ar:]
            
            # Forecast
            forecast = self.results_.forecast(y, steps=steps)
            
            # Get number of factor columns (exclude macro)
            n_factors = len([col for col in self.data_.columns if not col.startswith('macro_')])
            return forecast[:, :n_factors], None
        else:
            # Use AR models for forecast
            forecast = np.zeros((steps, len(self.ar_models_)))
            for i, model in enumerate(self.ar_models_):
                if model is not None:
                    # Get last values
                    last_vals = self.data_.iloc[-model.nobs:, i].values
                    forecast[0, i] = model.forecast(steps=1)[0]
                else:
                    forecast[0, i] = self.ar_coeffs_[i][0]
            return forecast, None

    def kalman_filter_state(
        self,
        factors: pd.DataFrame,
        transition_matrix: Optional[np.ndarray] = None,
        obs_matrix: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply Kalman filter to factor states.

        Args:
            factors: Observed factor values
            transition_matrix: State transition matrix (A)
            obs_matrix: Observation matrix (H)

        Returns:
            Tuple of (filtered states, state covariance)
        """
        k = factors.shape[1]

        # Default state transition: identity
        if transition_matrix is None:
            A = np.eye(k)
        else:
            A = transition_matrix

        # Default observation: identity (we observe the state directly)
        if obs_matrix is None:
            H = np.eye(k)
        else:
            H = obs_matrix

        # Estimate process and observation noise
        diff = factors.diff().dropna()
        if len(diff) > 0:
            Q = np.cov(diff.T)
            # Ensure positive definite
            Q = Q + np.eye(k) * 1e-6
        else:
            Q = np.eye(k) * 0.01
            
        R = np.eye(k) * 0.01  # Observation noise (small)

        # Initialize state
        x0 = factors.iloc[0].values
        P0 = np.eye(k) * 1.0  # Initial state covariance

        # Kalman filter
        n = len(factors)
        states = np.zeros((n, k))
        covariances = np.zeros((n, k, k))

        x = x0.copy()
        P = P0.copy()

        for t in range(n):
            if t > 0:
                # Prediction
                x_pred = A @ x
                P_pred = A @ P @ A.T + Q

                # Update
                y = factors.iloc[t].values
                innovation = y - H @ x_pred
                S = H @ P_pred @ H.T + R
                
                # Add small regularization to ensure invertibility
                S = S + np.eye(k) * 1e-8
                
                try:
                    K = P_pred @ H.T @ np.linalg.inv(S)
                except np.linalg.LinAlgError:
                    K = np.zeros((k, k))

                x = x_pred + K @ innovation
                P = (np.eye(k) - K @ H) @ P_pred
            else:
                x = x0
                P = P0

            states[t] = x
            covariances[t] = P

        return states, covariances

    def kalman_smooth(
        self,
        factors: pd.DataFrame,
        states: np.ndarray,
        covariances: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply RTS smoother to Kalman filter output.

        Args:
            factors: Observed factor values
            states: Filtered states
            covariances: State covariances

        Returns:
            Tuple of (smoothed states, smoothed covariances)
        """
        k = factors.shape[1]
        n = len(factors)

        # Get state space model matrices
        A = np.eye(k)  # Transition matrix
        diff = factors.diff().dropna()
        if len(diff) > 0:
            Q = np.cov(diff.T)
            Q = Q + np.eye(k) * 1e-6
        else:
            Q = np.eye(k) * 0.01

        # Smoother gains
        smoothed_states = np.zeros_like(states)
        smoothed_covs = np.zeros_like(covariances)

        smoothed_states[-1] = states[-1]
        smoothed_covs[-1] = covariances[-1]

        # Backward pass
        for t in range(n - 2, -1, -1):
            # Prediction
            P_pred = A @ covariances[t] @ A.T + Q

            # Smoother gain
            try:
                G = covariances[t] @ A.T @ np.linalg.inv(P_pred)
            except np.linalg.LinAlgError:
                G = np.zeros((k, k))

            # Smooth
            smoothed_states[t] = states[t] + G @ (smoothed_states[t + 1] - A @ states[t])
            smoothed_covs[t] = covariances[t] + G @ (smoothed_covs[t + 1] - P_pred) @ G.T

        return smoothed_states, smoothed_covs

    def predict_factors(
        self,
        factors: pd.DataFrame,
        macro: Optional[pd.DataFrame] = None,
        horizon: int = 1
    ) -> np.ndarray:
        """
        Predict next period factors using VAR + Kalman.

        Args:
            factors: Historical factor values
            macro: Historical macro values
            horizon: Forecast horizon

        Returns:
            Array of forecasted factors
        """
        # Fit VAR
        self.fit(factors, macro)

        # Get VAR forecast
        var_forecast, var_cov = self.forecast(steps=horizon)

        # If using Kalman smoother, combine with observed data
        if self.use_kalman:
            try:
                states, covariances = self.kalman_filter_state(factors)

                if self.use_smoother:
                    states, covariances = self.kalman_smooth(factors, states, covariances)

                # Combine VAR forecast with smoothed state
                last_smoothed = states[-1]
                predicted_factors = 0.7 * var_forecast[0] + 0.3 * last_smoothed
            except Exception as e:
                warnings.warn(f"Kalman filter failed: {e}. Using VAR forecast only.")
                predicted_factors = var_forecast[0] if len(var_forecast) > 0 else var_forecast
        else:
            predicted_factors = var_forecast[0] if len(var_forecast) > 0 else var_forecast

        return predicted_factors


def estimate_var_coefficients(
    factors: pd.DataFrame,
    macro: Optional[pd.DataFrame] = None
) -> Dict[str, np.ndarray]:
    """
    Estimate VAR coefficient matrices.

    Args:
        factors: Factor time series
        macro: Optional macro variables

    Returns:
        Dictionary with coefficient matrices A, B
    """
    if not STATSMODELS_AVAILABLE:
        raise ImportError("statsmodels required")

    # Rename macro columns to avoid overlap
    if macro is not None:
        macro = macro.copy()
        macro.columns = [f'macro_{col}' for col in macro.columns]
    
    model = VAR(factors.join(macro, how='inner') if macro is not None else factors)
    results = model.fit(maxlags=2)

    return {
        'A': results.params.values,
        'residuals': results.resid.values,
        'sigma': results.resid.cov().values
    }


# Example usage
if __name__ == "__main__":
    # Create sample factor data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', '2024-12-31', freq='B')
    n = len(dates)

    factors = pd.DataFrame(
        np.random.randn(n, 3) * 0.01,
        index=dates,
        columns=['F1', 'F2', 'F3']
    )

    macro = pd.DataFrame(
        np.random.randn(n, 2) * 0.005,
        index=dates,
        columns=['VIX', 'DXY']
    )

    print(f"Factors shape: {factors.shape}")
    print(f"Macro shape: {macro.shape}")

    # Fit and forecast
    forecaster = VARForecast(lag_order=2, use_kalman=True, use_smoother=True)
    predicted = forecaster.predict_factors(factors, macro, horizon=1)

    print(f"\nPredicted factors: {predicted}")
    print(f"Predicted factors shape: {predicted.shape}")
