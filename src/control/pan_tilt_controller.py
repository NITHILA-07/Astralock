"""PID controller for closed-loop pan/tilt gimbal coarse alignment."""

import numpy as np
from typing import Tuple, Dict, Any, Optional


class PanTiltPIDController:
    """Computes pan and tilt angular rate commands to align the virtual camera optical axis."""

    def __init__(
        self,
        kp: float = 0.60,
        ki: float = 0.015,
        kd: float = 0.06,
        max_rate_deg_s: float = 8.0,
        integral_limit_deg: float = 1.5,
    ):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_rate_deg_s = max_rate_deg_s
        self.integral_limit_deg = integral_limit_deg

        # Integrator and differentiator states
        self.integral_pan = 0.0
        self.integral_tilt = 0.0
        self.prev_err_pan = 0.0
        self.prev_err_tilt = 0.0
        self.initialized = False

    def reset(self) -> None:
        """Resets controller integrators and memory."""
        self.integral_pan = 0.0
        self.integral_tilt = 0.0
        self.prev_err_pan = 0.0
        self.prev_err_tilt = 0.0
        self.initialized = False

    def compute(
        self,
        target_pixel: Optional[Tuple[float, float]],
        camera_center_px: Tuple[float, float],
        pixel_pitch_deg: float,
        dt: float,
    ) -> Dict[str, Any]:
        """Calculates pan and tilt angle corrections for the current time step.
        
        Args:
            target_pixel: Target pixel coordinate (predicted or detected). If None, no command issued.
            camera_center_px: (cx, cy) boresight center pixel.
            pixel_pitch_deg: Angular scale per pixel.
            dt: Time step in seconds.
            
        Returns:
            Dict containing delta_pan_deg, delta_tilt_deg, and control diagnostics.
        """
        if target_pixel is None:
            # When target is lost, maintain current orientation (zero rate command)
            return {
                "delta_pan_deg": 0.0,
                "delta_tilt_deg": 0.0,
                "error_px": 0.0,
                "error_deg": 0.0,
                "active": False,
            }

        tx, ty = target_pixel
        cx, cy = camera_center_px

        # Pixel error from boresight
        err_x_px = tx - cx
        err_y_px = ty - cy
        error_dist_px = float(np.hypot(err_x_px, err_y_px))

        # Convert to angular errors
        err_pan_deg = err_x_px * pixel_pitch_deg
        err_tilt_deg = -err_y_px * pixel_pitch_deg
        error_dist_deg = float(np.hypot(err_pan_deg, err_tilt_deg))

        if not self.initialized:
            self.prev_err_pan = err_pan_deg
            self.prev_err_tilt = err_tilt_deg
            self.initialized = True

        # Integrator with anti-windup clamping
        self.integral_pan = float(np.clip(
            self.integral_pan + err_pan_deg * dt,
            -self.integral_limit_deg,
            self.integral_limit_deg
        ))
        self.integral_tilt = float(np.clip(
            self.integral_tilt + err_tilt_deg * dt,
            -self.integral_limit_deg,
            self.integral_limit_deg
        ))

        # Derivative
        deriv_pan = (err_pan_deg - self.prev_err_pan) / max(dt, 1e-5)
        deriv_tilt = (err_tilt_deg - self.prev_err_tilt) / max(dt, 1e-5)

        self.prev_err_pan = err_pan_deg
        self.prev_err_tilt = err_tilt_deg

        # PID control law (commanded angular rate in deg/s)
        rate_cmd_pan = self.kp * err_pan_deg + self.ki * self.integral_pan + self.kd * deriv_pan
        rate_cmd_tilt = self.kp * err_tilt_deg + self.ki * self.integral_tilt + self.kd * deriv_tilt

        # Motor rate limit
        rate_pan = float(np.clip(rate_cmd_pan, -self.max_rate_deg_s, self.max_rate_deg_s))
        rate_tilt = float(np.clip(rate_cmd_tilt, -self.max_rate_deg_s, self.max_rate_deg_s))

        # Commanded angle delta for this time step
        delta_pan = rate_pan * dt
        delta_tilt = rate_tilt * dt

        return {
            "delta_pan_deg": delta_pan,
            "delta_tilt_deg": delta_tilt,
            "error_px": error_dist_px,
            "error_deg": error_dist_deg,
            "active": True,
            "rate_pan_deg_s": rate_pan,
            "rate_tilt_deg_s": rate_tilt,
        }
