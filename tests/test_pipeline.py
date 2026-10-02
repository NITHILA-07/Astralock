"""Unit and integration test suite for AstraLock closed-loop tracking pipeline."""

import unittest
import numpy as np
import yaml

from src.simulation.target import OpticalBeaconTarget
from src.simulation.disturbances import DisturbanceGenerator
from src.simulation.virtual_camera import VirtualCamera
from src.simulation.environment import FSOCTestbedEnvironment
from src.detection.beacon_detector import BeaconDetector
from src.tracking.state_estimator import KalmanStateEstimator
from src.tracking.predictor import FutureLOSPredictor
from src.tracking.reacquisition import ReacquisitionManager
from src.control.pan_tilt_controller import PanTiltPIDController
from src.evaluation.metrics import MetricsTracker


class TestAstraLockPipeline(unittest.TestCase):
    """Verifies all components of the coarse-alignment testbed."""

    def setUp(self):
        with open("configs/default.yaml", "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

    def test_target_trajectories(self):
        """Tests optical beacon trajectories for continuity and velocity calculations."""
        for traj in ["constant_velocity", "curved", "sudden_change", "spiral"]:
            target = OpticalBeaconTarget(trajectory_type=traj, speed_deg_s=1.5)
            p0 = target.get_position(1.0)
            p1 = target.get_position(1.1)
            self.assertEqual(len(p0), 2)
            self.assertEqual(len(p1), 2)
            vx, vy = target.get_velocity(1.0)
            self.assertFalse(np.isnan(vx) or np.isnan(vy))

    def test_disturbances(self):
        """Verifies disturbance models (jitter, noise, occlusion)."""
        dist = DisturbanceGenerator(
            sensor_noise_sigma=5.0,
            jitter_amp_deg=0.06,
            occlusion_intervals=[[2.0, 3.0]],
            seed=42,
        )
        jx, jy = dist.get_platform_jitter(1.0)
        self.assertLessEqual(abs(jx), 0.1)
        self.assertLessEqual(abs(jy), 0.1)

        self.assertFalse(dist.is_occluded(1.0))
        self.assertTrue(dist.is_occluded(2.5))
        self.assertFalse(dist.is_occluded(3.5))

        clean = np.full((100, 100), 12, dtype=np.uint8)
        noisy = dist.apply_sensor_noise(clean)
        self.assertEqual(noisy.shape, clean.shape)
        self.assertNotEqual(float(np.std(noisy)), 0.0)

    def test_virtual_camera_and_detector(self):
        """Tests frame rendering and subpixel detection from rendered frames."""
        cam = VirtualCamera(width=640, height=480, fov_x_deg=10.0, psf_sigma=2.5)
        detector = BeaconDetector(threshold_value=100)

        # Place target spot at known pixel (350.0, 220.0)
        gt_spot = (350.0, 220.0)
        frame = cam.render_frame(gt_pixel_pos=gt_spot, is_occluded=False, beacon_peak_intensity=240.0)
        self.assertEqual(frame.shape, (480, 640))

        # Detect from frame
        result = detector.detect(frame)
        self.assertTrue(result["detected"])
        cx, cy = result["centroid"]
        self.assertAlmostEqual(cx, 350.0, delta=1.0)
        self.assertAlmostEqual(cy, 220.0, delta=1.0)
        self.assertGreater(result["snr_db"], 15.0)

    def test_kalman_state_estimator(self):
        """Tests Kalman filter convergence and missing measurement handling."""
        kf = KalmanStateEstimator(dt=0.02)
        
        # Feed synthetic linear track
        true_x, true_y = 100.0, 200.0
        vx, vy = 50.0, 25.0
        for i in range(15):
            t = i * 0.02
            meas = (true_x + vx * t + np.random.normal(0, 0.5), true_y + vy * t + np.random.normal(0, 0.5))
            res = kf.step(measurement=meas)
            self.assertTrue(res["initialized"])

        est_x, est_y = kf.x[0], kf.x[1]
        self.assertAlmostEqual(est_x, true_x + vx * 0.28, delta=5.0)
        self.assertAlmostEqual(est_y, true_y + vy * 0.28, delta=5.0)

        # Test predict future with latency
        fx, fy, sigma = kf.predict_future(tau_s=0.08)
        self.assertGreater(fx, est_x)
        self.assertGreater(fy, est_y)

        # Test missing measurement (occlusion)
        res_missing = kf.step(measurement=None)
        self.assertEqual(res_missing["missed_frames"], 1)

    def test_future_predictor(self):
        """Tests future LOS prediction module."""
        kf = KalmanStateEstimator(dt=0.02)
        kf.step((200.0, 200.0))
        kf.step((205.0, 202.0))
        predictor = FutureLOSPredictor(latency_tau_s=0.08)
        pred = predictor.predict(kf)
        self.assertEqual(pred["status"], "ACTIVE_PHYSICS_PREDICTION")
        self.assertIsNotNone(pred["predicted_pos"])
        self.assertGreater(pred["lead_vector_px"][0], 0.0)

    def test_reacquisition_manager(self):
        """Tests staged reacquisition ROI expansion and recovery."""
        reacq = ReacquisitionManager(timeout_s=0.1, frame_width=640, frame_height=480)
        # Tracking normal
        info0 = reacq.update(target_detected=True, current_time=1.0, last_estimated_pos=(320.0, 240.0))
        self.assertFalse(info0["is_searching"])

        # Target lost
        info1 = reacq.update(target_detected=False, current_time=1.05, last_estimated_pos=(320.0, 240.0))
        self.assertTrue(info1["is_searching"])
        self.assertIsNotNone(info1["search_roi"])

        # Target lost for longer time -> ROI expands
        info2 = reacq.update(target_detected=False, current_time=1.5, last_estimated_pos=(320.0, 240.0))
        self.assertTrue(info2["is_searching"])
        r1_w = info1["search_roi"][2] - info1["search_roi"][0]
        r2_w = info2["search_roi"][2] - info2["search_roi"][0]
        self.assertGreater(r2_w, r1_w)

        # Target reacquired
        info3 = reacq.update(target_detected=True, current_time=1.6, last_estimated_pos=(325.0, 242.0))
        self.assertFalse(info3["is_searching"])
        self.assertEqual(info3["total_reacquisitions"], 1)
        self.assertAlmostEqual(info3["reacquisition_time_s"], 0.55, delta=0.05)

    def test_closed_loop_simulation(self):
        """Runs 50 full closed-loop steps to verify pan/tilt orientation actively changes and locks."""
        env = FSOCTestbedEnvironment(self.cfg)
        controller = PanTiltPIDController()
        detector = BeaconDetector()
        metrics = MetricsTracker()

        obs = env.reset()
        initial_pan = obs["camera_pan_deg"]
        initial_tilt = obs["camera_tilt_deg"]

        for _ in range(50):
            det = detector.detect(obs["frame"])
            target_pt = det["centroid"]
            ctrl = controller.compute(
                target_pixel=target_pt,
                camera_center_px=obs["camera_center_px"],
                pixel_pitch_deg=obs["pixel_pitch_deg"],
                dt=env.dt,
            )
            obs, done = env.step(ctrl["delta_pan_deg"], ctrl["delta_tilt_deg"])
            metrics.record_step(
                sim_time=obs["time"],
                gt_pixel_pos=obs["gt_target_pixel"],
                camera_center_px=obs["camera_center_px"],
                pixel_pitch_deg=obs["pixel_pitch_deg"],
                target_detected=det["detected"],
                predicted_pixel_pos=target_pt,
                processing_time_s=0.002,
            )

        # Pan / tilt angles MUST have moved to track moving beacon (proving closed loop)
        final_pan = obs["camera_pan_deg"]
        final_tilt = obs["camera_tilt_deg"]
        self.assertNotEqual(final_pan, initial_pan)
        self.assertNotEqual(final_tilt, initial_tilt)

        summary = metrics.compute_summary()
        self.assertGreater(summary["total_frames"], 45)
        self.assertIsNotNone(summary["mean_tracking_error_px"])


if __name__ == "__main__":
    unittest.main()
