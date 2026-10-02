"""Comprehensive performance metrics tracker for FSOC coarse alignment."""

import json
import csv
import time
import numpy as np
from typing import Dict, Any, List, Optional, Tuple


class MetricsTracker:
    """Tracks and calculates verified quantitative performance metrics from actual simulation runs."""

    def __init__(
        self,
        lock_threshold_px: float = 15.0,
        min_lock_consecutive_frames: int = 3,
        scenario_id: str = "default_run",
        method: str = "predictive",
    ):
        self.lock_threshold_px = lock_threshold_px
        self.min_consecutive = min_lock_consecutive_frames
        self.scenario_id = scenario_id
        self.method = method

        self.reset()

    def reset(self) -> None:
        """Clears all logged history."""
        self.timestamps: List[float] = []
        self.tracking_errors_px: List[float] = []
        self.tracking_errors_deg: List[float] = []
        self.prediction_errors_px: List[float] = []
        self.processing_times_s: List[float] = []
        self.detected_flags: List[bool] = []
        self.locked_flags: List[bool] = []

        self.initial_acquisition_time_s: Optional[float] = None
        self.consecutive_locked = 0
        self.is_currently_locked = False
        self.lock_loss_count = 0
        self.reacquisition_times: List[float] = []
        self._unlock_timestamp: Optional[float] = None

    def record_step(
        self,
        sim_time: float,
        gt_pixel_pos: Optional[Tuple[float, float]],
        camera_center_px: Tuple[float, float],
        pixel_pitch_deg: float,
        target_detected: bool,
        predicted_pixel_pos: Optional[Tuple[float, float]],
        processing_time_s: float,
    ) -> bool:
        """Records telemetry for one simulation frame.
        
        Returns:
            Current lock status (True if locked).
        """
        self.timestamps.append(sim_time)
        self.processing_times_s.append(processing_time_s)
        self.detected_flags.append(target_detected)

        # 1. Calculate true pointing error relative to boresight (optical axis)
        if gt_pixel_pos is not None:
            err_px = float(np.hypot(gt_pixel_pos[0] - camera_center_px[0], gt_pixel_pos[1] - camera_center_px[1]))
            err_deg = err_px * pixel_pitch_deg
        else:
            err_px = float("nan")
            err_deg = float("nan")

        self.tracking_errors_px.append(err_px)
        self.tracking_errors_deg.append(err_deg)

        # 2. Prediction accuracy (how close predicted position is to future ground truth)
        if predicted_pixel_pos is not None and gt_pixel_pos is not None:
            pred_err = float(np.hypot(predicted_pixel_pos[0] - gt_pixel_pos[0], predicted_pixel_pos[1] - gt_pixel_pos[1]))
            self.prediction_errors_px.append(pred_err)

        # 3. Transparent lock condition:
        # Target must be detected and within coarse alignment error threshold
        is_locked_instant = target_detected and (not np.isnan(err_px)) and (err_px <= self.lock_threshold_px)

        if is_locked_instant:
            self.consecutive_locked += 1
        else:
            self.consecutive_locked = 0

        was_locked = self.is_currently_locked
        self.is_currently_locked = self.consecutive_locked >= self.min_consecutive
        self.locked_flags.append(self.is_currently_locked)

        # Track first acquisition time
        if self.is_currently_locked and self.initial_acquisition_time_s is None:
            self.initial_acquisition_time_s = sim_time

        # Track lock-loss and reacquisition events
        if was_locked and not self.is_currently_locked:
            self.lock_loss_count += 1
            self._unlock_timestamp = sim_time
        elif not was_locked and self.is_currently_locked and self._unlock_timestamp is not None:
            reacq_time = sim_time - self._unlock_timestamp
            self.reacquisition_times.append(reacq_time)
            self._unlock_timestamp = None

        return self.is_currently_locked

    def compute_summary(self) -> Dict[str, Any]:
        """Calculates final scientific performance metrics."""
        valid_errors = [e for e in self.tracking_errors_px if not np.isnan(e)]
        valid_pred_errors = [p for p in self.prediction_errors_px if not np.isnan(p)]

        total_frames = len(self.timestamps)
        locked_frames = sum(1 for lock in self.locked_flags if lock)
        lock_retention_pct = (locked_frames / total_frames * 100.0) if total_frames > 0 else 0.0

        mean_proc_time_s = float(np.mean(self.processing_times_s)) if self.processing_times_s else 0.0
        fps = (1.0 / mean_proc_time_s) if mean_proc_time_s > 0 else 0.0

        summary = {
            "scenario_id": self.scenario_id,
            "method": self.method,
            "total_simulation_time_s": round(self.timestamps[-1], 3) if self.timestamps else 0.0,
            "total_frames": total_frames,
            "acquisition_time_s": round(self.initial_acquisition_time_s, 4) if self.initial_acquisition_time_s is not None else None,
            "mean_tracking_error_px": round(float(np.mean(valid_errors)), 2) if valid_errors else None,
            "max_tracking_error_px": round(float(np.max(valid_errors)), 2) if valid_errors else None,
            "std_tracking_error_px": round(float(np.std(valid_errors)), 2) if valid_errors else None,
            "mean_prediction_error_px": round(float(np.mean(valid_pred_errors)), 2) if valid_pred_errors else None,
            "lock_retention_percent": round(lock_retention_pct, 2),
            "lock_loss_count": self.lock_loss_count,
            "mean_reacquisition_time_s": round(float(np.mean(self.reacquisition_times)), 4) if self.reacquisition_times else None,
            "mean_processing_time_ms": round(mean_proc_time_s * 1000.0, 3),
            "measured_fps": round(fps, 1),
            "lock_threshold_px": self.lock_threshold_px,
        }
        return summary

    def export_json(self, output_path: str) -> None:
        """Saves run summary to JSON format."""
        summary = self.compute_summary()
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    def export_csv(self, output_path: str) -> None:
        """Saves full step-by-step telemetry history to CSV."""
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp_s",
                "tracking_error_px",
                "tracking_error_deg",
                "target_detected",
                "is_locked",
                "processing_time_ms",
            ])
            for i in range(len(self.timestamps)):
                writer.writerow([
                    f"{self.timestamps[i]:.4f}",
                    f"{self.tracking_errors_px[i]:.2f}" if not np.isnan(self.tracking_errors_px[i]) else "NaN",
                    f"{self.tracking_errors_deg[i]:.4f}" if not np.isnan(self.tracking_errors_deg[i]) else "NaN",
                    self.detected_flags[i],
                    self.locked_flags[i],
                    f"{self.processing_times_s[i] * 1000.0:.2f}",
                ])
