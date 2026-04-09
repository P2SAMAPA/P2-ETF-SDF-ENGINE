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
        self.lag_order = lag_order
        self.use_kalman = use_kalman
        self.use_smoother = use_smoother
        self.include_macro = include_macro
        self.model_ = None
        self.results_ = None
        self.kalman_filter_ = None
        self.data_ = None
        self.ar_models_ = None

    def fit(self, factors: pd.DataFrame, macro: Optional[pd.DataFrame] = None) -> 'VARForecast':
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels required")

        if self.include_macro and macro is not None:
            macro_renamed = macro.copy()
            macro_renamed.columns = [f'macro_{col}' for col in macro.columns]
            combined = factors.join(macro_renamed, how='inner')
            self.data_ = combined.dropna()
        else:
            self.data_ = factors.dropna()

        if len(self.data_) < self.lag_order + 5:
            warnings.warn(f"Insufficient data ({len(self.data_)} rows) for VAR with {self.lag_order} lags")
            self.lag_order_ = max(1, len(self.data_) // 10)
        else:
            self.lag_order_ = self.lag_order

        try:
            self.model_ = VAR(self.data_)
            self.results_ = self.model_.fit(maxlags=self.lag_order_, ic='aic', trend='c')
            print(f"VAR fitted with {self.results_.k_ar} lags")
            print(f"AIC: {self.results_.aic:.4f}, BIC: {self.results_.bic:.4f}")
        except (np.linalg.LinAlgError, ValueError) as e:
            warnings.warn(f"VAR fitting failed: {e}. Using simple autoregressive model.")
            self.results_ = None
            self._fit_ar_models(factors)

        return self

    def _fit_ar_models(self, factors: pd.DataFrame):
        from statsmodels.tsa.ar_model import AutoReg
        self.ar_models_ = []
        for col in factors.columns:
            try:
                model = AutoReg(factors[col].dropna(), lags=min(2, len(factors) // 5))
                result = model.fit()
                self.ar_models_.append(result)
            except:
                self.ar_models_.append(None)
        print(f"Fitted AR models for {len(self.ar_models_)} factors")

    def forecast(self, steps: int = 1) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if self.results_ is not None:
            y = self.data_.values[-self.results_.k_ar:]
            forecast = self.results_.forecast(y, steps=steps)
            n_factors = len([col for col in self.data_.columns if not col.startswith('macro_')])
            return forecast[:, :n_factors], None
        else:
            # Use AR models
            forecast = np.zeros((steps, len(self.ar_models_)))
            for i, model in enumerate(self.ar_models_):
                if model is not None:
                    pred = model.forecast(steps=1)
                    # Use .iloc[0] for positional access (fixes KeyError: 0)
                    forecast[0, i] = pred.iloc[0] if hasattr(pred, 'iloc') else pred[0]
                else:
                    forecast[0, i] = 0.0
            return forecast, None

    def kalman_filter_state(self, factors: pd.DataFrame, transition_matrix: Optional[np.ndarray] = None, obs_matrix: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        k = factors.shape[1]
        if transition_matrix is None:
            A = np.eye(k)
        else:
            A = transition_matrix
        if obs_matrix is None:
            H = np.eye(k)
        else:
            H = obs_matrix
        diff = factors.diff().dropna()
        if len(diff) > 0:
            Q = np.cov(diff.T) + np.eye(k) * 1e-6
        else:
            Q = np.eye(k) * 0.01
        R = np.eye(k) * 0.01
        x0 = factors.iloc[0].values
        P0 = np.eye(k) * 1.0
        n = len(factors)
        states = np.zeros((n, k))
        covariances = np.zeros((n, k, k))
        x = x0.copy()
        P = P0.copy()
        for t in range(n):
            if t > 0:
                x_pred = A @ x
                P_pred = A @ P @ A.T + Q
                y = factors.iloc[t].values
                innovation = y - H @ x_pred
                S = H @ P_pred @ H.T + R + np.eye(k) * 1e-8
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

    def kalman_smooth(self, factors: pd.DataFrame, states: np.ndarray, covariances: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        k = factors.shape[1]
        n = len(factors)
        A = np.eye(k)
        diff = factors.diff().dropna()
        if len(diff) > 0:
            Q = np.cov(diff.T) + np.eye(k) * 1e-6
        else:
            Q = np.eye(k) * 0.01
        smoothed_states = np.zeros_like(states)
        smoothed_covs = np.zeros_like(covariances)
        smoothed_states[-1] = states[-1]
        smoothed_covs[-1] = covariances[-1]
        for t in range(n - 2, -1, -1):
            P_pred = A @ covariances[t] @ A.T + Q
            try:
                G = covariances[t] @ A.T @ np.linalg.inv(P_pred)
            except np.linalg.LinAlgError:
                G = np.zeros((k, k))
            smoothed_states[t] = states[t] + G @ (smoothed_states[t + 1] - A @ states[t])
            smoothed_covs[t] = covariances[t] + G @ (smoothed_covs[t + 1] - P_pred) @ G.T
        return smoothed_states, smoothed_covs

    def predict_factors(self, factors: pd.DataFrame, macro: Optional[pd.DataFrame] = None, horizon: int = 1) -> np.ndarray:
        self.fit(factors, macro)
        var_forecast, _ = self.forecast(steps=horizon)
        if self.use_kalman:
            try:
                states, covariances = self.kalman_filter_state(factors)
                if self.use_smoother:
                    states, _ = self.kalman_smooth(factors, states, covariances)
                last_smoothed = states[-1]
                predicted_factors = 0.7 * var_forecast[0] + 0.3 * last_smoothed
            except Exception as e:
                warnings.warn(f"Kalman filter failed: {e}. Using VAR forecast only.")
                predicted_factors = var_forecast[0] if len(var_forecast) > 0 else var_forecast
        else:
            predicted_factors = var_forecast[0] if len(var_forecast) > 0 else var_forecast
        return predicted_factors


def estimate_var_coefficients(factors: pd.DataFrame, macro: Optional[pd.DataFrame] = None) -> Dict[str, np.ndarray]:
    if not STATSMODELS_AVAILABLE:
        raise ImportError("statsmodels required")
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
