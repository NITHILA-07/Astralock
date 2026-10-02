"""Target module for simulating optical beacon trajectories in mobile FSOC scenarios."""

import math
from typing import Tuple, Dict, Any


class OpticalBeaconTarget:
    """Simulates a remote optical beacon on a moving platform (UAV, satellite, vehicle).
    
    Coordinates are in world angular space (degrees) relative to initial boresight.
    """

    def __init__(
        self,
        trajectory_type: str = "curved",
        speed_deg_s: float = 1.5,
        center_offset_deg: Tuple[float, float] = (0.0, 0.0),
        beacon_intensity: float = 240.0,
        seed: int = 42,
    ):
        self.trajectory_type = trajectory_type
        self.speed = speed_deg_s
        self.offset_x, self.offset_y = center_offset_deg
        self.beacon_intensity = beacon_intensity
        self.seed = seed

    def get_position(self, t: float) -> Tuple[float, float]:
        """Calculates ground truth angular position (theta_x, theta_y) in degrees at time t."""
        if self.trajectory_type == "constant_velocity":
            # Linear trajectory across FOV
            vx = 0.70 * self.speed
            vy = 0.35 * self.speed
            theta_x = self.offset_x + vx * t
            theta_y = self.offset_y + vy * t
            return theta_x, theta_y

        elif self.trajectory_type == "curved":
            # Sinusoidal trajectory representing smooth curved flight path
            # Frequency chosen so peak velocity matches self.speed: v_max = A * 2 * pi * f
            freq_x = 0.18
            freq_y = 0.36
            omega_x = 2.0 * math.pi * freq_x
            omega_y = 2.0 * math.pi * freq_y
            amp_x = (0.75 * self.speed) / omega_x
            amp_y = (0.45 * self.speed) / omega_y
            theta_x = self.offset_x + amp_x * math.sin(omega_x * t)
            theta_y = self.offset_y + amp_y * math.sin(omega_y * t)
            return theta_x, theta_y

        elif self.trajectory_type == "sudden_change":
            # Target cruises with constant velocity, then executes an abrupt evasion maneuver at t=6.0s
            switch_time = 6.0
            vx1 = 0.65 * self.speed
            vy1 = 0.30 * self.speed
            vx2 = -0.75 * self.speed
            vy2 = 0.55 * self.speed

            if t < switch_time:
                theta_x = self.offset_x + vx1 * t
                theta_y = self.offset_y + vy1 * t
            else:
                dt_post = t - switch_time
                theta_x = self.offset_x + vx1 * switch_time + vx2 * dt_post
                theta_y = self.offset_y + vy1 * switch_time + vy2 * dt_post
            return theta_x, theta_y

        elif self.trajectory_type == "spiral":
            radius = 0.08 * self.speed * t
            angular_freq = 0.25
            omega = 2.0 * math.pi * angular_freq
            theta_x = self.offset_x + radius * math.cos(omega * t)
            theta_y = self.offset_y + radius * math.sin(omega * t)
            return theta_x, theta_y

        else:
            # Fallback stationary beacon
            return self.offset_x, self.offset_y

    def get_velocity(self, t: float, dt: float = 0.001) -> Tuple[float, float]:
        """Calculates ground truth velocity (d_theta_x/dt, d_theta_y/dt) in deg/s via finite difference."""
        p0 = self.get_position(t - dt * 0.5)
        p1 = self.get_position(t + dt * 0.5)
        vx = (p1[0] - p0[0]) / dt
        vy = (p1[1] - p0[1]) / dt
        return vx, vy
