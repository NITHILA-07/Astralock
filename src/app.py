"""AstraLock: Closed-Loop Virtual Camera Coarse-Alignment Testbed.

Interactive Engineering Demonstration for SIH26169.
Visualizes detection, Kalman state estimation, latency-aware future LOS prediction,
pan/tilt closed-loop gimbal orientation updates, and real-time telemetry.
"""

import os
import sys
import time
import argparse
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


def draw_hud(
    display_img: np.ndarray,
    obs: dict,
    det_result: dict,
    est_result: dict,
    pred_info: dict,
    ctrl_result: dict,
    reacq_info: dict,
    metrics: MetricsTracker,
    is_locked: bool,
    fps: float,
    method: str,
    latency_tau_s: float,
) -> np.ndarray:
    """Renders visual overlays, crosshairs, prediction vectors, and engineering telemetry panel."""
    h, w = obs["frame"].shape[:2]
    cx, cy = obs["camera_center_px"]
    panel_w = 340
    total_w = w + panel_w

    # Create composite canvas: Camera viewport (left) + Engineering panel (right)
    canvas = np.zeros((h, total_w, 3), dtype=np.uint8)

    # 1. Convert sensor frame to BGR and place on left
    raw_bgr = cv2.cvtColor(obs["frame"], cv2.COLOR_GRAY2BGR)
    canvas[:, :w] = raw_bgr

    # Draw boresight crosshair (optical axis target)
    cv2.drawMarker(canvas, (int(cx), int(cy)), (180, 180, 180), cv2.MARKER_CROSS, 28, 1)
    cv2.circle(canvas, (int(cx), int(cy)), int(metrics.lock_threshold_px), (80, 80, 80), 1)
    cv2.putText(canvas, "OPTICAL AXIS", (int(cx) + 12, int(cy) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)

    # Draw search ROI if reacquisition is active
    if reacq_info.get("is_searching") and reacq_info.get("search_roi"):
        rx0, ry0, rx1, ry1 = reacq_info["search_roi"]
        cv2.rectangle(canvas, (rx0, ry0), (rx1, ry1), (0, 165, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"ROI SEARCH: {reacq_info['state']}", (rx0, max(15, ry0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 165, 255), 1)

    # Draw detected centroid (GREEN)
    if det_result["detected"] and det_result["centroid"]:
        dx, dy = int(det_result["centroid"][0]), int(det_result["centroid"][1])
        cv2.circle(canvas, (dx, dy), 7, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(canvas, f"DETECTED ({det_result['snr_db']:.1f} dB)", (dx + 10, dy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 0), 1)

    # Draw future predicted position (RED) in predictive mode
    if method == "predictive" and pred_info.get("predicted_pos"):
        px, py = int(pred_info["predicted_pos"][0]), int(pred_info["predicted_pos"][1])
        cv2.circle(canvas, (px, py), 6, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, f"PREDICTED (t+{int(latency_tau_s*1000)}ms)", (px + 10, py - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 255), 1)

        # Draw lead vector from detected to predicted
        if det_result["detected"] and det_result["centroid"]:
            cv2.arrowedLine(canvas, (dx, dy), (px, py), (0, 100, 255), 1, tipLength=0.25)

    # Draw line from target to optical axis (tracking error visual)
    target_pt = pred_info.get("predicted_pos") if method == "predictive" else det_result["centroid"]
    if target_pt is not None:
        tx, ty = int(target_pt[0]), int(target_pt[1])
        color_err = (0, 255, 0) if is_locked else (0, 165, 255)
        cv2.line(canvas, (int(cx), int(cy)), (tx, ty), color_err, 1, cv2.LINE_AA)

    # 2. Render Engineering & Telemetry Panel (Right)
    panel_x = w + 15
    cv2.line(canvas, (w, 0), (w, h), (50, 50, 50), 1)

    y = 28
    cv2.putText(canvas, "ASTRALOCK FSOC TESTBED", (panel_x, y), cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 255, 255), 1)
    y += 18
    cv2.putText(canvas, "Team: Trojan Titans | SIH26169", (panel_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1)
    y += 24
    cv2.line(canvas, (panel_x, y - 8), (total_w - 15, y - 8), (60, 60, 60), 1)

    # Mode Indicator
    mode_color = (255, 180, 50) if method == "predictive" else (200, 200, 200)
    cv2.putText(canvas, f"MODE: {method.upper()}", (panel_x, y), cv2.FONT_HERSHEY_DUPLEX, 0.48, mode_color, 1)
    y += 22

    # Lock Status Badge
    if is_locked:
        badge_text = "STATUS: [ LOCKED ]"
        badge_color = (0, 255, 0)
    elif reacq_info.get("is_searching"):
        badge_text = f"STATUS: [ {reacq_info['state']} ]"
        badge_color = (0, 165, 255)
    else:
        badge_text = "STATUS: [ UNLOCKED / ACQUIRING ]"
        badge_color = (0, 0, 255)
    cv2.putText(canvas, badge_text, (panel_x, y), cv2.FONT_HERSHEY_DUPLEX, 0.44, badge_color, 1)
    y += 26

    # Telemetry metrics
    tracking_err_px = ctrl_result.get("error_px", 0.0)
    tracking_err_deg = ctrl_result.get("error_deg", 0.0)
    pan_deg = obs["camera_pan_deg"]
    tilt_deg = obs["camera_tilt_deg"]
    summary = metrics.compute_summary()

    telemetry_items = [
        ("Sim Time", f"{obs['time']:.2f} s / {summary['total_simulation_time_s']:.1f} s"),
        ("Frame Rate", f"{fps:.1f} FPS (loop: {metrics.processing_times_s[-1]*1000.0:.1f} ms)"),
        ("Tracking Error", f"{tracking_err_px:.1f} px ({tracking_err_deg:.3f} deg)"),
        ("Mean Error (Run)", f"{summary['mean_tracking_error_px'] or 0.0:.1f} px"),
        ("Lock Threshold", f"<= {metrics.lock_threshold_px:.1f} px"),
        ("Lock Retention", f"{summary['lock_retention_percent']:.1f} %"),
        ("Lock Loss Events", f"{summary['lock_loss_count']}"),
        ("Initial Acq Time", f"{summary['acquisition_time_s'] or 'N/A'} s"),
        ("Pan / Tilt Angles", f"{pan_deg:+.2f} / {tilt_deg:+.2f} deg"),
        ("Loop Latency (tau)", f"{latency_tau_s * 1000.0:.0f} ms"),
        ("Signal SNR", f"{det_result.get('snr_db', 0.0):.1f} dB" if det_result['detected'] else "LOST"),
        ("Platform Jitter", f"+/- {np.hypot(obs['jitter_deg'][0], obs['jitter_deg'][1]):.3f} deg"),
        ("ML Residual Status", "IN DEV (State est. active)"),
    ]

    for label, val in telemetry_items:
        cv2.putText(canvas, f"{label}:", (panel_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (140, 140, 140), 1)
        cv2.putText(canvas, str(val), (panel_x + 130, y), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (220, 220, 220), 1)
        y += 18

    # Keyboard Controls Guide
    y += 12
    cv2.line(canvas, (panel_x, y - 6), (total_w - 15, y - 6), (60, 60, 60), 1)
    cv2.putText(canvas, "HOTKEYS:", (panel_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)
    y += 18
    hotkeys = [
        "[M] Toggle Baseline / Predictive",
        "[T] Switch Trajectory Pattern",
        "[R] Reset Run",
        "[S] Save Screenshot",
        "[SPACE] Pause / Resume",
        "[Q / ESC] Quit & Export",
    ]
    for hk in hotkeys:
        cv2.putText(canvas, hk, (panel_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (120, 120, 120), 1)
        y += 15

    return canvas


def main():
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
    fps_estimate = 50.0
    trajectories = ["curved", "constant_velocity", "sudden_change", "spiral"]
    traj_idx = trajectories.index(cfg["target"]["trajectory_type"]) if cfg["target"]["trajectory_type"] in trajectories else 0

    window_name = "AstraLock: FSOC Coarse Alignment Testbed (SIH26169)"
    if not args.headless:
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    first_frame_saved = False

    while not done:
        loop_start = time.perf_counter()

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
            reacquisition.update(
                target_detected=det_result["detected"],
                current_time=sim_time,
                last_estimated_pos=det_result["centroid"],
            )

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

            # 7. Step simulation environment (Closed loop gimbal orientation update)
            obs, done = env.step(
                delta_pan_deg=ctrl_result["delta_pan_deg"],
                delta_tilt_deg=ctrl_result["delta_tilt_deg"],
            )

            fps_estimate = 0.9 * fps_estimate + 0.1 * (1.0 / max(proc_time, 1e-4))

        canvas = draw_hud(
            display_img=obs["frame"],
            obs=obs,
            det_result=det_result,
            est_result=est_result,
            pred_info=pred_info,
            ctrl_result=ctrl_result,
            reacq_info=reacq_info,
            metrics=metrics,
            is_locked=is_locked,
            fps=fps_estimate,
            method=cfg["tracking"]["method"],
            latency_tau_s=cfg["tracking"]["latency_tau_s"],
        )

        # Auto-save representative screenshots for documentation
        if not first_frame_saved and obs["time"] >= 2.5:
            shot_path = os.path.join(screenshot_dir, f"tracking_{cfg['tracking']['method']}_demo.png")
            cv2.imwrite(shot_path, canvas)
            print(f"[Screenshot saved]: {shot_path}")
            first_frame_saved = True

        # Capture reacquisition during occlusion window
        if obs["time"] >= 5.3 and obs["time"] <= 5.5 and reacq_info.get("is_searching"):
            reacq_shot = os.path.join(screenshot_dir, "tracking_reacquisition_demo.png")
            if not os.path.exists(reacq_shot):
                cv2.imwrite(reacq_shot, canvas)
                print(f"[Reacquisition screenshot saved]: {reacq_shot}")

        if not args.headless:
            cv2.imshow(window_name, canvas)
            key = cv2.waitKey(max(1, int(env.dt * 1000))) & 0xFF

            if key in [ord("q"), ord("Q"), 27]:  # Q or ESC
                break
            elif key == ord(" "):  # Pause / Resume
                paused = not paused
                print(f"Simulation {'Paused' if paused else 'Resumed'}")
            elif key in [ord("m"), ord("M")]:  # Toggle method
                cfg["tracking"]["method"] = "baseline" if cfg["tracking"]["method"] == "predictive" else "predictive"
                metrics.method = cfg["tracking"]["method"]
                print(f"Switched tracking method to: {cfg['tracking']['method'].upper()}")
            elif key in [ord("t"), ord("T")]:  # Cycle trajectory
                traj_idx = (traj_idx + 1) % len(trajectories)
                cfg["target"]["trajectory_type"] = trajectories[traj_idx]
                env = FSOCTestbedEnvironment(cfg)
                estimator.reset()
                controller.reset()
                reacquisition.reset()
                metrics.reset()
                obs = env.reset()
                prev_pan = obs["camera_pan_deg"]
                prev_tilt = obs["camera_tilt_deg"]
                print(f"Switched trajectory to: {cfg['target']['trajectory_type']}")
            elif key in [ord("r"), ord("R")]:  # Reset
                obs = env.reset()
                estimator.reset()
                controller.reset()
                reacquisition.reset()
                metrics.reset()
                prev_pan = obs["camera_pan_deg"]
                prev_tilt = obs["camera_tilt_deg"]
                print("Simulation reset.")
            elif key in [ord("s"), ord("S")]:  # Manual screenshot
                shot_path = os.path.join(screenshot_dir, f"screenshot_{int(time.time())}.png")
                cv2.imwrite(shot_path, canvas)
                print(f"Saved screenshot: {shot_path}")

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
