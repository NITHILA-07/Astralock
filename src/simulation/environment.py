"""Environment coordinator for the AstraLock FSOC coarse-alignment testbed."""

from typing import Dict, Any, Tuple, Optional
import numpy as np

from src.simulation.target import OpticalBeaconTarget
from src.simulation.virtual_camera import VirtualCamera
from src.simulation.disturbances import DisturbanceGenerator


class FSOCTestbedEnvironment:
    """Manages the full closed-loop simulation state: target, disturbances, and virtual camera."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        sim_cfg = config.get("simulation", {})
        cam_cfg = config.get("camera", {})
        tgt_cfg = config.get("target", {})
        dst_cfg = config.get("disturbances", {})
        ctrl_cfg = config.get("control", {})

        self.dt = sim_cfg.get("dt", 0.02)
        self.duration = sim_cfg.get("duration", 15.0)
        self.seed = sim_cfg.get("seed", 42)
        self.max_angular_rate = ctrl_cfg.get("max_angular_rate_deg_s", 8.0)

        self.time = 0.0
        self.step_idx = 0

        # Initialize submodules
        self.target = OpticalBeaconTarget(
            trajectory_type=tgt_cfg.get("trajectory_type", "curved"),
            speed_deg_s=tgt_cfg.get("speed_deg_s", 1.5),
            center_offset_deg=tuple(tgt_cfg.get("center_offset_deg", [0.0, 0.0])),
            beacon_intensity=tgt_cfg.get("beacon_intensity", 240.0),
            seed=self.seed,
        )

        self.camera = VirtualCamera(
            width=cam_cfg.get("width", 640),
            height=cam_cfg.get("height", 480),
            fov_x_deg=cam_cfg.get("fov_x_deg", 10.0),
            exposure_intensity=cam_cfg.get("exposure_intensity", 255.0),
            psf_sigma=cam_cfg.get("psf_sigma", 2.5),
            ambient_noise_std=cam_cfg.get("ambient_noise_std", 6.0),
        )

        self.disturbances = DisturbanceGenerator(
            sensor_noise=dst_cfg.get("sensor_noise", True),
            sensor_noise_sigma=dst_cfg.get("sensor_noise_sigma", 5.0),
            platform_jitter=dst_cfg.get("platform_jitter", True),
            jitter_freq_hz=dst_cfg.get("jitter_freq_hz", 3.0),
            jitter_amp_deg=dst_cfg.get("jitter_amp_deg", 0.06),
            turbulence_disturbance=dst_cfg.get("turbulence_disturbance", True),
            turbulence_amp_px=dst_cfg.get("turbulence_amp_px", 1.2),
            occlusion_intervals=dst_cfg.get("occlusion_intervals", []),
            seed=self.seed,
        )

    def reset(self) -> Dict[str, Any]:
        """Resets the environment to initial conditions at t=0."""
        self.time = 0.0
        self.step_idx = 0
        self.camera.pan_deg = 0.0
        self.camera.tilt_deg = 0.0
        return self._get_observation()

    def step(self, delta_pan_deg: float, delta_tilt_deg: float) -> Tuple[Dict[str, Any], bool]:
        """Applies control commands, updates physics, and advances simulation by dt."""
        # 1. Update camera orientation based on gimbal motor commands (CLOSED LOOP)
        self.camera.apply_pan_tilt_command(
            delta_pan_deg=delta_pan_deg,
            delta_tilt_deg=delta_tilt_deg,
            max_rate_deg_s=self.max_angular_rate,
            dt=self.dt,
        )

        # 2. Advance time
        self.time += self.dt
        self.step_idx += 1
        done = self.time >= self.duration

        # 3. Generate new observation frame
        obs = self._get_observation()
        return obs, done

    def _get_observation(self) -> Dict[str, Any]:
        """Renders the current frame and packages telemetry."""
        # Ground truth world position of target
        gt_theta_x, gt_theta_y = self.target.get_position(self.time)

        # Disturbance values
        jitter_deg = self.disturbances.get_platform_jitter(self.time)
        wander_px = self.disturbances.get_optical_wander_px(self.time)
        is_occluded = self.disturbances.is_occluded(self.time)

        # Project world position onto sensor coordinates
        px_x, px_y = self.camera.world_to_pixel(
            theta_x_deg=gt_theta_x,
            theta_y_deg=gt_theta_y,
            jitter_deg=jitter_deg,
            wander_px=wander_px,
        )

        # Check field-of-view visibility
        is_in_fov = (0 <= px_x < self.camera.width) and (0 <= px_y < self.camera.height)
        gt_pixel = (px_x, px_y) if is_in_fov and not is_occluded else None

        # Render genuine pixel frame
        clean_frame = self.camera.render_frame(
            gt_pixel_pos=gt_pixel,
            is_occluded=is_occluded,
            beacon_peak_intensity=self.target.beacon_intensity,
        )
        noisy_frame = self.disturbances.apply_sensor_noise(clean_frame)

        return {
            "time": self.time,
            "step": self.step_idx,
            "frame": noisy_frame,
            "camera_pan_deg": self.camera.pan_deg,
            "camera_tilt_deg": self.camera.tilt_deg,
            "camera_center_px": (self.camera.cx, self.camera.cy),
            "pixel_pitch_deg": self.camera.pixel_pitch_deg,
            # Ground truth is provided strictly for metrics calculation / evaluation
            "gt_target_pixel": (px_x, px_y),
            "gt_target_visible_pixel": gt_pixel,
            "gt_world_target_deg": (gt_theta_x, gt_theta_y),
            "is_occluded": is_occluded,
            "is_in_fov": is_in_fov,
            "jitter_deg": jitter_deg,
        }
