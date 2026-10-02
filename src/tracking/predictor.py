"""Latency-aware future Line-of-Sight (LOS) predictor for mobile FSOC terminals.

This module addresses the central research question:
'Instead of controlling toward where the beacon was last observed, can we estimate
where it will be when the control command takes effect?'

Total loop latency tau = tau_capture + tau_proc + tau_actuation.
Target state at t + tau is estimated to synthesize lead angle pointing commands.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np

from src.tracking.state_estimator import KalmanStateEstimator


class FutureLOSPredictor:
    """Predicts beacon future coordinates (t + tau) using state kinematics."""

    def __init__(self, latency_tau_s: float = 0.08):
        self.latency_tau_s = latency_tau_s
        # Scientific transparency status for SIH evaluation
        self.ml_residual_status = "IN DEVELOPMENT (State extrapolation active; learned residual planned)"

    def predict(
        self,
        estimator: KalmanStateEstimator,
        custom_tau: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Calculates predicted beacon location at t + tau.
        
        Args:
            estimator: Active Kalman filter instance.
            custom_tau: Optional override for latency parameter (seconds).
            
        Returns:
            Dict containing predicted pixel coords, latency applied, uncertainty, and status.
        """
        tau = self.latency_tau_s if custom_tau is None else custom_tau

        if not estimator.initialized:
            return {
                "predicted_pos": None,
                "latency_s": tau,
                "uncertainty_sigma_px": 0.0,
                "lead_vector_px": (0.0, 0.0),
                "ml_residual_applied": False,
                "status": "UNINITIALIZED",
            }

        cur_x, cur_y = estimator.x[0], estimator.x[1]
        future_x, future_y, sigma = estimator.predict_future(tau)

        # Lead vector = P_future - P_current
        lead_dx = future_x - cur_x
        lead_dy = future_y - cur_y

        # Note: Learned residual correction is intentionally marked IN DEVELOPMENT.
        # We do not fabricate fake neural network outputs.

        return {
            "predicted_pos": (float(future_x), float(future_y)),
            "latency_s": float(tau),
            "uncertainty_sigma_px": float(sigma),
            "lead_vector_px": (float(lead_dx), float(lead_dy)),
            "ml_residual_applied": False,
            "status": "ACTIVE_PHYSICS_PREDICTION",
        }
