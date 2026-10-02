"""Disturbance models for FSOC coarse-alignment simulation.

Models platform jitter, sensor noise, occlusion, and simplified optical beam wander.
NOTE: Atmospheric disturbance is modeled as a simplified turbulence-like beam wander
and intensity fluctuation; it is not a full split-step Fourier optical wave propagation model.
"""

import math
import numpy as np
from typing import List, Tuple, Optional


class DisturbanceGenerator:
    """Generates realistic physical and optical disturbances on the FSOC terminal."""

    def __init__(
        self,
        sensor_noise: bool = True,
        sensor_noise_sigma: float = 5.0,
        platform_jitter: bool = True,
        jitter_freq_hz: float = 3.0,
        jitter_amp_deg: float = 0.06,
        turbulence_disturbance: bool = True,
        turbulence_amp_px: float = 1.2,
        occlusion_intervals: Optional[List[List[float]]] = None,
        seed: int = 42,
    ):
        self.sensor_noise = sensor_noise
        self.sensor_noise_sigma = sensor_noise_sigma
        self.platform_jitter = platform_jitter
        self.jitter_freq_hz = jitter_freq_hz
        self.jitter_amp_deg = jitter_amp_deg
        self.turbulence_disturbance = turbulence_disturbance
        self.turbulence_amp_px = turbulence_amp_px
        self.occlusion_intervals = occlusion_intervals or []
        self.rng = np.random.RandomState(seed)

    def get_platform_jitter(self, t: float) -> Tuple[float, float]:
        """Calculates angular platform jitter (deg) along pan and tilt axes."""
        if not self.platform_jitter:
            return 0.0, 0.0

        # Deterministic harmonic vibration + small band-limited stochastic vibration
        omega = 2.0 * math.pi * self.jitter_freq_hz
        jitter_x = self.jitter_amp_deg * math.sin(omega * t) + 0.3 * self.jitter_amp_deg * math.sin(2.3 * omega * t)
        jitter_y = self.jitter_amp_deg * math.cos(omega * t * 0.9) + 0.3 * self.jitter_amp_deg * math.sin(1.7 * omega * t)

        return jitter_x, jitter_y

    def get_optical_wander_px(self, t: float) -> Tuple[float, float]:
        """Simplified turbulence-like beam centroid wander displacement in pixels."""
        if not self.turbulence_disturbance:
            return 0.0, 0.0

        # Slow optical path fluctuations
        wander_x = self.turbulence_amp_px * (
            0.6 * math.sin(1.3 * t) + 0.4 * math.cos(3.7 * t)
        )
        wander_y = self.turbulence_amp_px * (
            0.6 * math.cos(1.1 * t) + 0.4 * math.sin(4.1 * t)
        )
        return wander_x, wander_y

    def is_occluded(self, t: float) -> bool:
        """Checks if optical line of sight is temporarily blocked at time t."""
        for start_t, end_t in self.occlusion_intervals:
            if start_t <= t <= end_t:
                return True
        return False

    def apply_sensor_noise(self, frame: np.ndarray) -> np.ndarray:
        """Adds Gaussian readout and thermal noise to the camera sensor frame."""
        if not self.sensor_noise or self.sensor_noise_sigma <= 0.0:
            return frame

        noise = self.rng.normal(0.0, self.sensor_noise_sigma, frame.shape)
        noisy = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return noisy
