"""Linear Kalman Filter for beacon 2D position and velocity state estimation."""

import numpy as np
from typing import Optional, Tuple, Dict, Any


class KalmanStateEstimator:
    """Estimates beacon state [x, y, vx, vy]^T in sensor pixel space with covariance propagation."""

    def __init__(
        self,
        dt: float = 0.02,
        process_noise_pos: float = 0.02,
        process_noise_vel: float = 0.8,
        measurement_noise: float = 1.5,
    ):
        self.dt = dt
        self.process_noise_pos = process_noise_pos
        self.process_noise_vel = process_noise_vel
        self.measurement_noise = measurement_noise

        # State vector: [x, y, vx, vy]^T
        self.x = np.zeros(4, dtype=np.float64)
        self.initialized = False
        self.missed_frames = 0

        # State transition matrix F
        self.F = np.array([
            [1.0, 0.0, dt,  0.0],
            [0.0, 1.0, 0.0, dt ],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float64)

        # Measurement matrix H (observing x, y positions)
        self.H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0]
        ], dtype=np.float64)

        # Process noise covariance Q (discrete white noise model)
        q_pos = self.process_noise_pos
        q_vel = self.process_noise_vel
        self.Q = np.diag([q_pos, q_pos, q_vel, q_vel]).astype(np.float64)

        # Measurement noise covariance R
        r_var = self.measurement_noise ** 2
        self.R = np.diag([r_var, r_var]).astype(np.float64)

        # State error covariance matrix P
        self.P = np.diag([10.0, 10.0, 50.0, 50.0]).astype(np.float64)

    def reset(self) -> None:
        """Resets the state estimator to uninitialized state."""
        self.x = np.zeros(4, dtype=np.float64)
        self.P = np.diag([10.0, 10.0, 50.0, 50.0]).astype(np.float64)
        self.initialized = False
        self.missed_frames = 0

    def step(
        self,
        measurement: Optional[Tuple[float, float]],
        ego_motion_px: Tuple[float, float] = (0.0, 0.0),
    ) -> Dict[str, Any]:
        """Runs one Kalman predict-update cycle.
        
        Args:
            measurement: (x, y) observed centroid, or None if beacon was not detected.
            ego_motion_px: (dx, dy) pixel shift caused by camera pan/tilt gimbal movement.
            
        Returns:
            Dict containing estimated position, velocity, and 1-sigma uncertainty.
        """
        # 1. State prediction
        if self.initialized:
            # Predict state forward by dt
            self.x = self.F @ self.x
            # Compensate camera platform ego-motion
            self.x[0] += ego_motion_px[0]
            self.x[1] += ego_motion_px[1]
            # Propagate error covariance
            self.P = self.F @ self.P @ self.F.T + self.Q

        # 2. Measurement update
        if measurement is not None:
            z = np.array([measurement[0], measurement[1]], dtype=np.float64)

            if not self.initialized:
                # Initialize state with first observation
                self.x = np.array([z[0], z[1], 0.0, 0.0], dtype=np.float64)
                self.P = np.diag([self.measurement_noise ** 2, self.measurement_noise ** 2, 25.0, 25.0])
                self.initialized = True
                self.missed_frames = 0
            else:
                # Innovation / measurement residual
                y = z - self.H @ self.x
                # Innovation covariance
                S = self.H @ self.P @ self.H.T + self.R
                # Kalman gain
                K = self.P @ self.H.T @ np.linalg.inv(S)
                # State update
                self.x = self.x + K @ y
                # Covariance update (Joseph form for numerical stability)
                I_KH = np.eye(4) - K @ self.H
                self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T
                self.missed_frames = 0
        else:
            # Observation missing (occlusion or out-of-frame)
            self.missed_frames += 1

        sigma_x = float(np.sqrt(max(self.P[0, 0], 0.0)))
        sigma_y = float(np.sqrt(max(self.P[1, 1], 0.0)))

        return {
            "estimated_pos": (float(self.x[0]), float(self.x[1])),
            "estimated_vel": (float(self.x[2]), float(self.x[3])),
            "uncertainty_sigma": (sigma_x, sigma_y),
            "missed_frames": self.missed_frames,
            "initialized": self.initialized,
        }

    def predict_future(self, tau_s: float) -> Tuple[float, float, float]:
        """Extrapolates state forward in time by tau seconds (compensating loop latency).
        
        Returns:
            (future_x, future_y, future_sigma_px)
        """
        if not self.initialized:
            return 0.0, 0.0, 0.0

        future_x = self.x[0] + self.x[2] * tau_s
        future_y = self.x[1] + self.x[3] * tau_s

        # Propagate position uncertainty forward with velocity uncertainty
        var_x = self.P[0, 0] + (tau_s ** 2) * self.P[2, 2] + 2.0 * tau_s * self.P[0, 2]
        var_y = self.P[1, 1] + (tau_s ** 2) * self.P[3, 3] + 2.0 * tau_s * self.P[1, 3]
        total_sigma = float(np.sqrt(max((var_x + var_y) * 0.5, 0.0)))

        return float(future_x), float(future_y), total_sigma
