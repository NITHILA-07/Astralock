"""Staged reacquisition strategy for recovering lost optical beacon lock."""

from typing import Optional, Tuple, Dict, Any


class ReacquisitionManager:
    """Manages staged visual search after beacon loss: Local ROI -> Uncertainty expansion -> Full-frame."""

    def __init__(
        self,
        timeout_s: float = 0.25,
        initial_roi_half_size: int = 40,
        expansion_rate_px_s: float = 80.0,
        max_roi_half_size: int = 240,
        frame_width: int = 640,
        frame_height: int = 480,
    ):
        self.timeout_s = timeout_s
        self.initial_roi_half_size = initial_roi_half_size
        self.expansion_rate = expansion_rate_px_s
        self.max_roi_half_size = max_roi_half_size
        self.frame_width = frame_width
        self.frame_height = frame_height

        # State tracking
        self.is_searching = False
        self.time_lost_start: Optional[float] = None
        self.last_known_pos: Optional[Tuple[float, float]] = None
        self.total_reacquisitions = 0
        self.last_reacquisition_time_s = 0.0

    def reset(self) -> None:
        """Resets reacquisition state."""
        self.is_searching = False
        self.time_lost_start = None
        self.last_known_pos = None
        self.total_reacquisitions = 0
        self.last_reacquisition_time_s = 0.0

    def update(
        self,
        target_detected: bool,
        current_time: float,
        last_estimated_pos: Optional[Tuple[float, float]],
        uncertainty_sigma_px: float = 10.0,
    ) -> Dict[str, Any]:
        """Updates reacquisition state machine and calculates search bounding box.
        
        Returns:
            Dict with search state, ROI bounding box (or None for full-frame), and reacquisition metrics.
        """
        if target_detected:
            if self.is_searching and self.time_lost_start is not None:
                # Successfully reacquired!
                self.last_reacquisition_time_s = current_time - self.time_lost_start
                self.total_reacquisitions += 1
                self.is_searching = False
                self.time_lost_start = None

            if last_estimated_pos is not None:
                self.last_known_pos = last_estimated_pos

            return {
                "state": "LOCKED_OR_TRACKING",
                "is_searching": False,
                "search_roi": None,
                "reacquisition_time_s": self.last_reacquisition_time_s,
                "total_reacquisitions": self.total_reacquisitions,
            }

        # Target NOT detected in current frame
        if not self.is_searching:
            self.is_searching = True
            self.time_lost_start = current_time

        duration_lost = current_time - self.time_lost_start if self.time_lost_start else 0.0

        # Anchor search around last known/predicted position, or center if none
        anchor_x = self.last_known_pos[0] if self.last_known_pos else (self.frame_width / 2.0)
        anchor_y = self.last_known_pos[1] if self.last_known_pos else (self.frame_height / 2.0)

        # Stage 1: Local ROI based on uncertainty
        # Stage 2: Time-expanded search region
        half_size = self.initial_roi_half_size + (2.0 * uncertainty_sigma_px) + (self.expansion_rate * duration_lost)

        if half_size >= self.max_roi_half_size:
            # Stage 3: Full-frame global sweep
            search_state = "FULL_FRAME_SEARCH"
            roi = None
        elif duration_lost > self.timeout_s:
            search_state = "EXPANDING_ROI_SEARCH"
            roi = (
                max(0, int(anchor_x - half_size)),
                max(0, int(anchor_y - half_size)),
                min(self.frame_width, int(anchor_x + half_size)),
                min(self.frame_height, int(anchor_y + half_size)),
            )
        else:
            search_state = "LOCAL_ROI_SEARCH"
            roi = (
                max(0, int(anchor_x - half_size)),
                max(0, int(anchor_y - half_size)),
                min(self.frame_width, int(anchor_x + half_size)),
                min(self.frame_height, int(anchor_y + half_size)),
            )

        return {
            "state": search_state,
            "is_searching": True,
            "search_roi": roi,
            "duration_lost_s": duration_lost,
            "reacquisition_time_s": self.last_reacquisition_time_s,
            "total_reacquisitions": self.total_reacquisitions,
        }
