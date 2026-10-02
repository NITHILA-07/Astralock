"""Automated benchmark comparing Baseline (reactive) vs Predictive (AstraLock) coarse tracking."""

import os
import copy
import time
import json
import yaml
import numpy as np
from typing import Dict, Any, List, Tuple

from src.simulation.environment import FSOCTestbedEnvironment
from src.detection.beacon_detector import BeaconDetector
from src.tracking.state_estimator import KalmanStateEstimator
from src.tracking.predictor import FutureLOSPredictor
from src.tracking.reacquisition import ReacquisitionManager
from src.control.pan_tilt_controller import PanTiltPIDController
from src.evaluation.metrics import MetricsTracker


def run_single_simulation(
    base_config: Dict[str, Any],
    method: str = "predictive",
    scenario_name: str = "test",
) -> Tuple[Dict[str, Any], MetricsTracker]:
    """Executes a full closed-loop simulation run under either baseline or predictive tracking.
    
    Both methods execute with identical random seeds and physical disturbances.
    """
    cfg = copy.deepcopy(base_config)
    cfg["tracking"]["method"] = method

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
        scenario_id=f"{scenario_name}_{method}",
        method=method,
    )

    obs = env.reset()
    done = False
    prev_pan = obs["camera_pan_deg"]
    prev_tilt = obs["camera_tilt_deg"]

    while not done:
        t_start = time.perf_counter()

        frame = obs["frame"]
        sim_time = obs["time"]
        cx, cy = obs["camera_center_px"]
        pitch = obs["pixel_pitch_deg"]

        # Calculate camera ego-motion in pixel space since previous step
        d_pan = obs["camera_pan_deg"] - prev_pan
        d_tilt = obs["camera_tilt_deg"] - prev_tilt
        ego_motion_px = (-d_pan / pitch, d_tilt / pitch)
        prev_pan = obs["camera_pan_deg"]
        prev_tilt = obs["camera_tilt_deg"]

        # 1. State-guided Reacquisition / ROI selection
        est_pos = (estimator.x[0], estimator.x[1]) if estimator.initialized else None
        reacq_info = reacquisition.update(
            target_detected=False,  # Updated after detection
            current_time=sim_time,
            last_estimated_pos=est_pos,
            uncertainty_sigma_px=float(np.sqrt(max(estimator.P[0, 0], 1.0))) if estimator.initialized else 20.0,
        )

        # 2. Beacon Detection from rendered sensor frame
        search_roi = reacq_info["search_roi"]
        det_result = detector.detect(frame, roi=search_roi)

        # Update reacquisition manager with actual detection result
        reacquisition.update(
            target_detected=det_result["detected"],
            current_time=sim_time,
            last_estimated_pos=det_result["centroid"],
        )

        # 3. State Estimation (Kalman Filter)
        meas = det_result["centroid"] if det_result["detected"] else None
        est_result = estimator.step(measurement=meas, ego_motion_px=ego_motion_px)

        # 4. Target Pointing Command Synthesis
        predicted_pos = None
        if method == "predictive":
            # AstraLock: Predict beacon LOS at t + tau to compensate loop latency
            pred_info = predictor.predict(estimator)
            target_pt = pred_info["predicted_pos"]
            predicted_pos = target_pt
        else:
            # Baseline: React solely to current estimated/detected position (tau = 0)
            target_pt = est_result["estimated_pos"] if estimator.initialized else det_result["centroid"]

        # 5. Pan/Tilt PID Control
        ctrl_result = controller.compute(
            target_pixel=target_pt,
            camera_center_px=(cx, cy),
            pixel_pitch_deg=pitch,
            dt=env.dt,
        )

        t_elapsed = time.perf_counter() - t_start

        # 6. Record verified performance metrics
        metrics.record_step(
            sim_time=sim_time,
            gt_pixel_pos=obs["gt_target_pixel"],
            camera_center_px=(cx, cy),
            pixel_pitch_deg=pitch,
            target_detected=det_result["detected"],
            predicted_pixel_pos=predicted_pos,
            processing_time_s=t_elapsed,
        )

        # 7. Step simulation environment (Closed loop gimbal update)
        obs, done = env.step(
            delta_pan_deg=ctrl_result["delta_pan_deg"],
            delta_tilt_deg=ctrl_result["delta_tilt_deg"],
        )

    summary = metrics.compute_summary()
    return summary, metrics


def run_benchmark_suite(config_path: str = "configs/default.yaml", output_dir: str = "outputs/benchmarks") -> Dict[str, Any]:
    """Runs rigorous side-by-side benchmark comparing Baseline vs Predictive tracking."""
    os.makedirs(output_dir, exist_ok=True)

    with open(config_path, "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)

    # Scenarios tested with identical seeds and disturbances
    test_scenarios = [
        {"name": "curved_flight", "traj": "curved", "speed": 1.5, "latency": 0.08, "jitter": 0.06},
        {"name": "high_speed_orbit", "traj": "curved", "speed": 2.5, "latency": 0.08, "jitter": 0.06},
        {"name": "sudden_maneuver", "traj": "sudden_change", "speed": 1.6, "latency": 0.08, "jitter": 0.05},
        {"name": "high_latency_channel", "traj": "curved", "speed": 1.8, "latency": 0.14, "jitter": 0.06},
    ]

    results = []

    print("\n" + "=" * 80)
    print("ASTRALOCK BENCHMARK: BASELINE (REACTIVE) VS PREDICTIVE (ASTRALOCK)")
    print("=" * 80)

    for sc in test_scenarios:
        cfg = copy.deepcopy(base_cfg)
        cfg["target"]["trajectory_type"] = sc["traj"]
        cfg["target"]["speed_deg_s"] = sc["speed"]
        cfg["tracking"]["latency_tau_s"] = sc["latency"]
        cfg["disturbances"]["jitter_amp_deg"] = sc["jitter"]

        # Run Baseline (Reactive, tau=0)
        base_res, base_metrics = run_single_simulation(cfg, method="baseline", scenario_name=sc["name"])
        base_metrics.export_csv(os.path.join(output_dir, f"{sc['name']}_baseline_telemetry.csv"))

        # Run Predictive (AstraLock, latency-aware lead angle)
        pred_res, pred_metrics = run_single_simulation(cfg, method="predictive", scenario_name=sc["name"])
        pred_metrics.export_csv(os.path.join(output_dir, f"{sc['name']}_predictive_telemetry.csv"))

        comparison = {
            "scenario": sc["name"],
            "trajectory": sc["traj"],
            "target_speed_deg_s": sc["speed"],
            "simulated_latency_s": sc["latency"],
            "baseline": base_res,
            "predictive": pred_res,
            "error_reduction_pct": round(
                ((base_res["mean_tracking_error_px"] - pred_res["mean_tracking_error_px"]) / base_res["mean_tracking_error_px"]) * 100.0, 2
            ) if base_res["mean_tracking_error_px"] else 0.0,
            "lock_retention_gain_pct": round(
                pred_res["lock_retention_percent"] - base_res["lock_retention_percent"], 2
            ),
        }
        results.append(comparison)

        print(f"\nScenario: {sc['name']} (Speed={sc['speed']} deg/s, Latency={sc['latency']*1000:.0f} ms)")
        print(f"  {'Metric':<28} | {'Baseline (Reactive)':<20} | {'Predictive (AstraLock)':<22} | {'Improvement'}")
        print("  " + "-" * 78)
        print(f"  {'Mean Tracking Error (px)':<28} | {base_res['mean_tracking_error_px']:<20} | {pred_res['mean_tracking_error_px']:<22} | {comparison['error_reduction_pct']}% error reduction")
        print(f"  {'Max Tracking Error (px)':<28} | {base_res['max_tracking_error_px']:<20} | {pred_res['max_tracking_error_px']:<22} |")
        print(f"  {'Lock Retention (%)':<28} | {base_res['lock_retention_percent']:<20} | {pred_res['lock_retention_percent']:<22} | +{comparison['lock_retention_gain_pct']}% retention")
        print(f"  {'Lock Loss Events':<28} | {base_res['lock_loss_count']:<20} | {pred_res['lock_loss_count']:<22} |")
        print(f"  {'Processing Rate (FPS)':<28} | {base_res['measured_fps']:<20} | {pred_res['measured_fps']:<22} |")

    out_file = os.path.join(output_dir, "benchmark_summary.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print(f"Benchmark completed successfully! Results exported to: {out_file}")
    print("=" * 80 + "\n")

    return {"scenarios": results}


if __name__ == "__main__":
    import numpy as np
    run_benchmark_suite()
