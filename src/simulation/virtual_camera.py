"""Virtual camera module that generates realistic synthetic optical frames for beacon tracking."""

import numpy as np
import cv2
from typing import Tuple, Optional, Dict, Any


class VirtualCamera:
    """Simulates a coarse-alignment optical sensor mounted on a 2-DOF pan-tilt gimbal.
    
    Generates realistic 2D frames containing the optical beacon point spread function (PSF).
    """

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fov_x_deg: float = 10.0,
        exposure_intensity: float = 255.0,
        psf_sigma: float = 2.5,
        ambient_noise_std: float = 6.0,
        initial_pan_deg: float = 0.0,
        initial_tilt_deg: float = 0.0,
    ):
        self.width = width
        self.height = height
        self.fov_x_deg = fov_x_deg
        self.fov_y_deg = fov_x_deg * (height / width)
        self.pixel_pitch_deg = fov_x_deg / float(width)  # degrees per pixel

        self.cx = width / 2.0
        self.cy = height / 2.0

        self.exposure_intensity = exposure_intensity
        self.psf_sigma = psf_sigma
        self.ambient_noise_std = ambient_noise_std

        # Gimbal orientation (degrees in world frame)
        self.pan_deg = initial_pan_deg
        self.tilt_deg = initial_tilt_deg

    def world_to_pixel(
        self,
        theta_x_deg: float,
        theta_y_deg: float,
        jitter_deg: Tuple[float, float] = (0.0, 0.0),
        wander_px: Tuple[float, float] = (0.0, 0.0),
    ) -> Tuple[float, float]:
        """Maps world angular position into camera sensor pixel coordinates."""
        rel_pan = theta_x_deg - (self.pan_deg + jitter_deg[0])
        rel_tilt = theta_y_deg - (self.tilt_deg + jitter_deg[1])

        # Angular to pixel projection (optical boresight at cx, cy)
        px_x = self.cx + (rel_pan / self.pixel_pitch_deg) + wander_px[0]
        # Tilt up corresponds to moving higher in world, which is lower row index (upwards) in image
        px_y = self.cy - (rel_tilt / self.pixel_pitch_deg) + wander_px[1]

        return px_x, px_y

    def pixel_to_angular_offset(self, px_x: float, px_y: float) -> Tuple[float, float]:
        """Calculates angular pointing error (deg) relative to camera boresight."""
        err_x_deg = (px_x - self.cx) * self.pixel_pitch_deg
        err_y_deg = -(px_y - self.cy) * self.pixel_pitch_deg
        return err_x_deg, err_y_deg

    def render_frame(
        self,
        gt_pixel_pos: Optional[Tuple[float, float]],
        is_occluded: bool = False,
        beacon_peak_intensity: float = 240.0,
    ) -> np.ndarray:
        """Renders an actual image sensor frame (uint8 grayscale) containing optical beacon PSF."""
        # Baseline background sensor dark current level
        frame = np.full((self.height, self.width), fill_value=12, dtype=np.uint8)

        if is_occluded or gt_pixel_pos is None:
            return frame

        u, v = gt_pixel_pos

        # Check if beacon center is anywhere near the sensor FOV
        margin = int(self.psf_sigma * 4)
        if -margin <= u < self.width + margin and -margin <= v < self.height + margin:
            # Render Gaussian spot
            x_min = max(0, int(np.floor(u - margin)))
            x_max = min(self.width, int(np.ceil(u + margin + 1)))
            y_min = max(0, int(np.floor(v - margin)))
            y_max = min(self.height, int(np.ceil(v + margin + 1)))

            if x_max > x_min and y_max > y_min:
                ys, xs = np.mgrid[y_min:y_max, x_min:x_max]
                dist_sq = (xs - u) ** 2 + (ys - v) ** 2
                spot = beacon_peak_intensity * np.exp(-dist_sq / (2.0 * self.psf_sigma ** 2))
                
                # Composite with ambient level
                composite = np.clip(frame[y_min:y_max, x_min:x_max].astype(np.float32) + spot, 0, 255)
                frame[y_min:y_max, x_min:x_max] = composite.astype(np.uint8)

        return frame

    def apply_pan_tilt_command(
        self,
        delta_pan_deg: float,
        delta_tilt_deg: float,
        max_rate_deg_s: float,
        dt: float,
    ) -> Tuple[float, float]:
        """Applies gimbal velocity-limited pan/tilt corrections to update camera orientation."""
        max_delta = max_rate_deg_s * dt

        # Enforce gimbal slew rate limits
        actual_delta_pan = np.clip(delta_pan_deg, -max_delta, max_delta)
        actual_delta_tilt = np.clip(delta_tilt_deg, -max_delta, max_delta)

        self.pan_deg += float(actual_delta_pan)
        self.tilt_deg += float(actual_delta_tilt)

        return float(actual_delta_pan), float(actual_delta_tilt)
