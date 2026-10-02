"""AstraLock: Closed-Loop Virtual Camera Coarse-Alignment Testbed.

Interactive Engineering Demonstration for SIH26169.
Visualizes detection, Kalman state estimation, latency-aware future LOS prediction,
pan/tilt closed-loop gimbal orientation updates, and real-time telemetry.
"""

import os
import sys
import time
import argparse
import json
import yaml
import numpy as np
import cv2

from src.simulation.environment import FSOCTestbedEnvironment
from src.detection.beacon_detector import BeaconDetector
from src.tracking.state_estimator import KalmanStateEstimator
from src.tracking.predictor import FutureLOSPredictor
from src.tracking.reacquisition import ReacquisitionManager
from src.control.pan_tilt_controller import PanTiltPIDController
from src.evaluation.metrics import MetricsTracker
from src.evaluation.benchmark import run_benchmark_suite
from src.ui.dashboard import AstraLockDashboard


# Global mouse action state for OpenCV callback
pending_action = None


def on_mouse_click(event, x, y, flags, param):
    """Handles mouse click events on dashboard buttons."""
    global pending_action
    if event == cv2.EVENT_LBUTTONDOWN:
        dashboard = param
        for btn in dashboard.buttons:
            bx0, by0, bx1, by1 = btn["bbox"]
            if bx0 <= x <= bx1 and by0 <= y <= by1:
                pending_action = btn["id"]
                break


def main():
    global pending_action
    parser = argparse.ArgumentParser(description="AstraLock: Coarse Alignment Predictive Testbed for Mobile FSOC")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to configuration YAML")
    parser.add_argument("--method", type=str, choices=["baseline", "predictive"], default="predictive", help="Tracking method")
    parser.add_argument("--trajectory", type=str, default=None, help="Override trajectory: constant_velocity, curved, sudden_change, spiral")
    parser.add_argument("--speed", type=float, default=None, help="Override target angular speed (deg/s)")
    parser.add_argument("--latency", type=float, default=None, help="Override loop latency tau (seconds)")
    parser.add_argument("--headless", action="store_true", help="Run without graphical display window")
    parser.add_argument("--export-dir", type=str, default="outputs/runs", help="Directory to save run results")
    args = parser.parse_args()

    # Load configuration
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.method:
        cfg["tracking"]["method"] = args.method
    if args.trajectory:
        cfg["target"]["trajectory_type"] = args.trajectory
    if args.speed is not None:
        cfg["target"]["speed_deg_s"] = args.speed
    if args.latency is not None:
        cfg["tracking"]["latency_tau_s"] = args.latency

    os.makedirs(args.export_dir, exist_ok=True)
    screenshot_dir = os.path.join("docs", "screenshots")
    os.makedirs(screenshot_dir, exist_ok=True)

    # Initialize testbed modules
    env = FSOCTestbedEnvironment(cfg)
    detector = BeaconDetector(
        threshold_value=cfg["detection"]["threshold_value"],
        min_area_px=cfg["detection"]["min_area_px"],
        max_area_px=cfg["detection"]["max_area_px"],
    )
    estimator = KalmanStateEstimator(
        dt=cfg["simulation"]["dt"],
        process_noise_pos=cfg["tracking"]["process_noise_pos"],
        process_noise_vel=cfg["tracking"]["process_noise_vel"],
        measurement_noise=cfg["tracking"]["measurement_noise"],
    )
    predictor = FutureLOSPredictor(latency_tau_s=cfg["tracking"]["latency_tau_s"])
    reacquisition = ReacquisitionManager(
        timeout_s=cfg["reacquisition"]["timeout_s"],
        initial_roi_half_size=cfg["reacquisition"]["initial_roi_half_size"],
        expansion_rate_px_s=cfg["reacquisition"]["expansion_rate_px_s"],
        max_roi_half_size=cfg["reacquisition"]["max_roi_half_size"],
        frame_width=cfg["camera"]["width"],
        frame_height=cfg["camera"]["height"],
    )
    controller = PanTiltPIDController(
        kp=cfg["control"]["kp"],
        ki=cfg["control"]["ki"],
        kd=cfg["control"]["kd"],
        max_rate_deg_s=cfg["control"]["max_angular_rate_deg_s"],
    )
    metrics = MetricsTracker(
        lock_threshold_px=cfg["lock"]["error_threshold_px"],
        min_lock_consecutive_frames=cfg["lock"]["min_consecutive_frames"],
        scenario_id=f"{cfg['target']['trajectory_type']}_{cfg['tracking']['method']}",
        method=cfg["tracking"]["method"],
    )

    # Initialize professional engineering dashboard
    dashboard = AstraLockDashboard(width=1440, height=810)

    # Load existing benchmark results if available
    benchmark_path = os.path.join("outputs", "benchmarks", "benchmark_summary.json")
    benchmark_data = None
    if os.path.exists(benchmark_path):
        try:
            with open(benchmark_path, "r", encoding="utf-8") as f:
                benchmark_data = {"scenarios": json.load(f)}
        except Exception:
            benchmark_data = None

    print("=" * 70)
    print("ASTRALOCK: Adaptive Predictive Coarse-Alignment Testbed for Mobile FSOC")
    print(f"Tracking Method: {cfg['tracking']['method'].upper()}")
    print(f"Trajectory: {cfg['target']['trajectory_type']} (Speed: {cfg['target']['speed_deg_s']} deg/s)")
    print(f"Simulated Loop Latency (tau): {cfg['tracking']['latency_tau_s']*1000.0:.0f} ms")
    print(f"Lock Threshold: <= {cfg['lock']['error_threshold_px']} px")
    print("=" * 70)

    obs = env.reset()
    prev_pan = obs["camera_pan_deg"]
    prev_tilt = obs["camera_tilt_deg"]
    done = False
    paused = False
    eval_mode = False
    fps_estimate = 50.0

    trajectories = ["curved", "constant_velocity", "sudden_change", "spiral"]
    traj_idx = trajectories.index(cfg["target"]["trajectory_type"]) if cfg["target"]["trajectory_type"] in trajectories else 0

    # Rolling error history and event log
    error_history: list[float] = []
    max_history = 150
    event_log: list[str] = [
        f"[{obs['time']:.2f}s] System initialized: {cfg['target']['trajectory_type']} trajectory ({cfg['target']['speed_deg_s']} deg/s)"
    ]

    window_name = "AstraLock: FSOC Coarse Alignment Testbed (SIH26169)"
    if not args.headless:
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(window_name, on_mouse_click, dashboard)

    main_dashboard_saved = False
    predictive_saved = False
    baseline_saved = False
    reacq_saved = False

    while not done:
        loop_start = time.perf_counter()

        # Handle pending mouse clicks from UI
        if pending_action:
            action = pending_action
            pending_action = None

            if action == "pause":
                paused = not paused
                event_log.append(f"[{obs['time']:.2f}s] Simulation {'paused' if paused else 'resumed'}")
            elif action == "method":
                cfg["tracking"]["method"] = "baseline" if cfg["tracking"]["method"] == "predictive" else "predictive"
                metrics.method = cfg["tracking"]["method"]
                event_log.append(f"[{obs['time']:.2f}s] Tracking method switched to: {cfg['tracking']['method'].upper()}")
            elif action == "trajectory":
                traj_idx = (traj_idx + 1) % len(trajectories)
                cfg["target"]["trajectory_type"] = trajectories[traj_idx]
                env = FSOCTestbedEnvironment(cfg)
                estimator.reset()
                controller.reset()
                reacquisition.reset()
                metrics.reset()
                error_history.clear()
                dashboard.trail.clear()
                obs = env.reset()
                prev_pan = obs["camera_pan_deg"]
                prev_tilt = obs["camera_tilt_deg"]
                event_log.append(f"[{obs['time']:.2f}s] Switched trajectory to: {cfg['target']['trajectory_type']}")
            elif action == "eval":
                eval_mode = not eval_mode
                event_log.append(f"[{obs['time']:.2f}s] Evaluation mode: {'ENABLED' if eval_mode else 'DISABLED'}")
            elif action == "reset":
                obs = env.reset()
                estimator.reset()
                controller.reset()
                reacquisition.reset()
                metrics.reset()
                error_history.clear()
                dashboard.trail.clear()
                prev_pan = obs["camera_pan_deg"]
                prev_tilt = obs["camera_tilt_deg"]
                event_log.append(f"[{obs['time']:.2f}s] Simulation reset to initial state")
            elif action == "benchmark":
                event_log.append(f"[{obs['time']:.2f}s] Executing empirical benchmark suite...")
                print("\n[Running Benchmark Suite]...")
                benchmark_res = run_benchmark_suite(output_dir="outputs/benchmarks")
                benchmark_data = benchmark_res
                event_log.append(f"[{obs['time']:.2f}s] Benchmark completed successfully")

        if not paused:
            frame = obs["frame"]
            sim_time = obs["time"]
            cx, cy = obs["camera_center_px"]
            pitch = obs["pixel_pitch_deg"]

            # Camera ego-motion compensation
            d_pan = obs["camera_pan_deg"] - prev_pan
            d_tilt = obs["camera_tilt_deg"] - prev_tilt
            ego_motion_px = (-d_pan / pitch, d_tilt / pitch)
            prev_pan = obs["camera_pan_deg"]
            prev_tilt = obs["camera_tilt_deg"]

            # 1. Staged reacquisition state update
            est_pos = (estimator.x[0], estimator.x[1]) if estimator.initialized else None
            reacq_info = reacquisition.update(
                target_detected=False,
                current_time=sim_time,
                last_estimated_pos=est_pos,
                uncertainty_sigma_px=float(np.sqrt(max(estimator.P[0, 0], 1.0))) if estimator.initialized else 20.0,
            )

            # 2. Beacon detection from sensor frame
            det_result = detector.detect(frame, roi=reacq_info["search_roi"])

            # Inform reacquisition manager of detection result
            reacq_updated = reacquisition.update(
                target_detected=det_result["detected"],
                current_time=sim_time,
                last_estimated_pos=det_result["centroid"],
            )

            # Log reacquisition / loss events
            if reacq_updated.get("reacquisition_time_s", 0) > 0 and (not reacq_saved or len(event_log) < 25):
                event_log.append(f"[{sim_time:.2f}s] Beacon reacquired after {reacq_updated['reacquisition_time_s']:.2f}s search")
            if obs.get("is_occluded") and "occlusion" not in event_log[-1]:
                event_log.append(f"[{sim_time:.2f}s] Line-of-sight occlusion: Local search initiated")

            # 3. State estimation (Kalman filter)
            meas = det_result["centroid"] if det_result["detected"] else None
            est_result = estimator.step(measurement=meas, ego_motion_px=ego_motion_px)

            # 4. Latency-aware future LOS prediction
            pred_info = {"predicted_pos": None, "status": "BASELINE"}
            if cfg["tracking"]["method"] == "predictive":
                pred_info = predictor.predict(estimator)
                target_pt = pred_info["predicted_pos"]
            else:
                # Baseline reactive tracking
                target_pt = est_result["estimated_pos"] if estimator.initialized else det_result["centroid"]

            # 5. Closed-loop Pan/Tilt PID control
            ctrl_result = controller.compute(
                target_pixel=target_pt,
                camera_center_px=(cx, cy),
                pixel_pitch_deg=pitch,
                dt=env.dt,
            )

            proc_time = time.perf_counter() - loop_start

            # 6. Record step metrics
            is_locked = metrics.record_step(
                sim_time=sim_time,
                gt_pixel_pos=obs["gt_target_pixel"],
                camera_center_px=(cx, cy),
                pixel_pitch_deg=pitch,
                target_detected=det_result["detected"],
                predicted_pixel_pos=pred_info.get("predicted_pos"),
                processing_time_s=proc_time,
            )

            if is_locked and metrics.consecutive_locked == metrics.min_consecutive:
                event_log.append(f"[{sim_time:.2f}s] Coarse lock established (error <= {metrics.lock_threshold_px} px)")

            # Record tracking error into rolling history
            err_px = ctrl_result.get("error_px", 0.0)
            error_history.append(err_px)
            if len(error_history) > max_history:
                error_history.pop(0)

            # 7. Step simulation environment (Closed loop gimbal orientation update)
            obs, done = env.step(
                delta_pan_deg=ctrl_result["delta_pan_deg"],
                delta_tilt_deg=ctrl_result["delta_tilt_deg"],
            )

            fps_estimate = 0.9 * fps_estimate + 0.1 * (1.0 / max(proc_time, 1e-4))

        # Render complete engineering dashboard
        summary = metrics.compute_summary()
        canvas = dashboard.render(
            obs=obs,
            det_result=det_result,
            est_result=est_result,
            pred_info=pred_info,
            ctrl_result=ctrl_result,
            reacq_info=reacq_info,
            metrics_summary=summary,
            error_history=error_history,
            event_log=event_log,
            benchmark_data=benchmark_data,
            cfg=cfg,
            is_locked=is_locked,
            is_paused=paused,
            eval_mode=eval_mode,
            fps=fps_estimate,
        )

        # Auto-save presentation screenshots for SIH submission
        if not main_dashboard_saved and obs["time"] >= 2.5 and is_locked:
            main_shot = os.path.join(screenshot_dir, "main_dashboard.png")
            cv2.imwrite(main_shot, canvas)
            print(f"[Dashboard screenshot saved]: {main_shot}")
            main_dashboard_saved = True

        if not predictive_saved and cfg["tracking"]["method"] == "predictive" and obs["time"] >= 3.0 and is_locked:
            pred_shot = os.path.join(screenshot_dir, "predictive_tracking.png")
            cv2.imwrite(pred_shot, canvas)
            print(f"[Predictive screenshot saved]: {pred_shot}")
            predictive_saved = True

        if not baseline_saved and cfg["tracking"]["method"] == "baseline" and obs["time"] >= 3.0:
            base_shot = os.path.join(screenshot_dir, "baseline_comparison.png")
            cv2.imwrite(base_shot, canvas)
            print(f"[Baseline screenshot saved]: {base_shot}")
            baseline_saved = True

        # Capture reacquisition during occlusion window
        if obs["time"] >= 5.3 and obs["time"] <= 5.5 and reacq_info.get("is_searching") and not reacq_saved:
            reacq_shot = os.path.join(screenshot_dir, "tracking_reacquisition_demo.png")
            cv2.imwrite(reacq_shot, canvas)
            print(f"[Reacquisition screenshot saved]: {reacq_shot}")
            reacq_saved = True

        if not args.headless:
            cv2.imshow(window_name, canvas)
            key = cv2.waitKey(max(1, int(env.dt * 1000))) & 0xFF

            if key in [ord("q"), ord("Q"), 27]:  # Q or ESC
                break
            elif key == ord(" "):  # Pause / Resume
                paused = not paused
                event_log.append(f"[{obs['time']:.2f}s] Simulation {'paused' if paused else 'resumed'}")
            elif key in [ord("m"), ord("M")]:  # Toggle method
                cfg["tracking"]["method"] = "baseline" if cfg["tracking"]["method"] == "predictive" else "predictive"
                metrics.method = cfg["tracking"]["method"]
                event_log.append(f"[{obs['time']:.2f}s] Method toggled to: {cfg['tracking']['method'].upper()}")
            elif key in [ord("t"), ord("T")]:  # Cycle trajectory
                traj_idx = (traj_idx + 1) % len(trajectories)
                cfg["target"]["trajectory_type"] = trajectories[traj_idx]
                env = FSOCTestbedEnvironment(cfg)
                estimator.reset()
                controller.reset()
                reacquisition.reset()
                metrics.reset()
                error_history.clear()
                dashboard.trail.clear()
                obs = env.reset()
                prev_pan = obs["camera_pan_deg"]
                prev_tilt = obs["camera_tilt_deg"]
                event_log.append(f"[{obs['time']:.2f}s] Trajectory changed to: {cfg['target']['trajectory_type']}")
            elif key in [ord("e"), ord("E")]:  # Toggle evaluation mode
                eval_mode = not eval_mode
                event_log.append(f"[{obs['time']:.2f}s] Eval mode: {'ON' if eval_mode else 'OFF'}")
            elif key in [ord("b"), ord("B")]:  # Run benchmark
                event_log.append(f"[{obs['time']:.2f}s] Running benchmark suite...")
                benchmark_data = run_benchmark_suite(output_dir="outputs/benchmarks")
                event_log.append(f"[{obs['time']:.2f}s] Benchmark completed")
            elif key in [ord("r"), ord("R")]:  # Reset
                obs = env.reset()
                estimator.reset()
                controller.reset()
                reacquisition.reset()
                metrics.reset()
                error_history.clear()
                dashboard.trail.clear()
                prev_pan = obs["camera_pan_deg"]
                prev_tilt = obs["camera_tilt_deg"]
                event_log.append(f"[{obs['time']:.2f}s] Reset run")
            elif key in [ord("s"), ord("S")]:  # Manual screenshot
                shot_path = os.path.join(screenshot_dir, f"screenshot_{int(time.time())}.png")
                cv2.imwrite(shot_path, canvas)
                event_log.append(f"[{obs['time']:.2f}s] Manual screenshot captured")
                print(f"Saved screenshot: {shot_path}")
            elif key in [ord("+"), ord("=")]:  # Increase speed
                cfg["target"]["speed_deg_s"] += 0.2
                env.target.speed = cfg["target"]["speed_deg_s"]
                event_log.append(f"[{obs['time']:.2f}s] Target speed: {cfg['target']['speed_deg_s']:.1f} deg/s")
            elif key in [ord("-"), ord("_")]:  # Decrease speed
                cfg["target"]["speed_deg_s"] = max(0.2, cfg["target"]["speed_deg_s"] - 0.2)
                env.target.speed = cfg["target"]["speed_deg_s"]
                event_log.append(f"[{obs['time']:.2f}s] Target speed: {cfg['target']['speed_deg_s']:.1f} deg/s")

    if not args.headless:
        cv2.destroyAllWindows()

    # Final summary export
    summary = metrics.compute_summary()
    json_path = os.path.join(args.export_dir, f"run_{summary['scenario_id']}.json")
    csv_path = os.path.join(args.export_dir, f"run_{summary['scenario_id']}_telemetry.csv")
    metrics.export_json(json_path)
    metrics.export_csv(csv_path)

    print("\n" + "=" * 70)
    print("SIMULATION RUN COMPLETED")
    print("=" * 70)
    print(f"Scenario: {summary['scenario_id']}")
    print(f"Method: {summary['method'].upper()}")
    print(f"Total Simulation Time: {summary['total_simulation_time_s']} s ({summary['total_frames']} frames)")
    print(f"Acquisition Time: {summary['acquisition_time_s']} s")
    print(f"Mean Tracking Error: {summary['mean_tracking_error_px']} px")
    print(f"Max Tracking Error: {summary['max_tracking_error_px']} px")
    print(f"Lock Retention: {summary['lock_retention_percent']} %")
    print(f"Lock Loss Count: {summary['lock_loss_count']}")
    print(f"Mean Processing Time: {summary['mean_processing_time_ms']} ms ({summary['measured_fps']} FPS)")
    print(f"Telemetry saved to: {csv_path}")
    print(f"Summary JSON saved to: {json_path}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
