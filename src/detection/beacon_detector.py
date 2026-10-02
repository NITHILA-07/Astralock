"""Robust optical beacon detection module using thresholding and sub-pixel moment centroid extraction."""

import numpy as np
import cv2
from typing import Optional, Tuple, Dict, Any


class BeaconDetector:
    """Detects optical beacon spots in camera sensor frames via intensity thresholding and moment analysis."""

    def __init__(
        self,
        threshold_value: int = 120,
        min_area_px: int = 4,
        max_area_px: int = 800,
    ):
        self.threshold_value = threshold_value
        self.min_area_px = min_area_px
        self.max_area_px = max_area_px

    def detect(
        self,
        frame: np.ndarray,
        roi: Optional[Tuple[int, int, int, int]] = None,
    ) -> Dict[str, Any]:
        """Performs beacon extraction on the given frame or sub-region.
        
        Args:
            frame: 2D uint8 grayscale image sensor frame.
            roi: Optional bounding box (x_min, y_min, x_max, y_max) for gated reacquisition.
            
        Returns:
            Dict containing detection flag, sub-pixel centroid coordinates, area, peak intensity, and SNR.
        """
        h, w = frame.shape[:2]

        # Crop to ROI if provided, else full frame
        if roi is not None:
            x0 = max(0, int(roi[0]))
            y0 = max(0, int(roi[1]))
            x1 = min(w, int(roi[2]))
            y1 = min(h, int(roi[3]))
            if x1 <= x0 or y1 <= y0:
                roi_frame = frame
                offset_x, offset_y = 0, 0
            else:
                roi_frame = frame[y0:y1, x0:x1]
                offset_x, offset_y = x0, y0
        else:
            roi_frame = frame
            offset_x, offset_y = 0, 0

        # Background statistics for SNR calculation
        bg_mean = float(np.mean(roi_frame))
        bg_std = float(np.std(roi_frame)) + 1e-4

        # Intensity thresholding
        effective_thresh = max(self.threshold_value, int(bg_mean + 3.0 * bg_std))
        _, binary = cv2.threshold(roi_frame, effective_thresh, 255, cv2.THRESH_BINARY)

        # Connected component / contour analysis
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_cand = None
        max_peak = -1.0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if self.min_area_px <= area <= self.max_area_px:
                # Calculate subpixel centroid via raw spatial moments
                m = cv2.moments(cnt)
                if m["m00"] > 0:
                    cx = (m["m10"] / m["m00"]) + offset_x
                    cy = (m["m01"] / m["m00"]) + offset_y
                    
                    # Bounding box
                    bx, by, bw, bh = cv2.boundingRect(cnt)
                    patch = roi_frame[by:by+bh, bx:bx+bw]
                    peak_val = float(np.max(patch)) if patch.size > 0 else 0.0

                    if peak_val > max_peak:
                        max_peak = peak_val
                        best_cand = {
                            "centroid": (float(cx), float(cy)),
                            "area_px": float(area),
                            "peak_intensity": float(peak_val),
                            "bbox": (int(bx + offset_x), int(by + offset_y), int(bw), int(bh)),
                        }

        if best_cand is not None:
            # Genuine physical SNR (dB) = 20 * log10((peak - mean) / std)
            signal_diff = max(best_cand["peak_intensity"] - bg_mean, 1e-3)
            snr_db = 20.0 * np.log10(signal_diff / bg_std)

            return {
                "detected": True,
                "centroid": best_cand["centroid"],
                "area_px": best_cand["area_px"],
                "peak_intensity": best_cand["peak_intensity"],
                "snr_db": float(snr_db),
                "bbox": best_cand["bbox"],
                "roi_used": (offset_x, offset_y, offset_x + roi_frame.shape[1], offset_y + roi_frame.shape[0]),
            }

        return {
            "detected": False,
            "centroid": None,
            "area_px": 0.0,
            "peak_intensity": float(np.max(roi_frame)),
            "snr_db": 0.0,
            "bbox": None,
            "roi_used": (offset_x, offset_y, offset_x + roi_frame.shape[1], offset_y + roi_frame.shape[0]),
        }
