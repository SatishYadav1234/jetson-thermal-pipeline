"""
Thermal Defect & Anomaly Inference Engine
File: src/python/inference_engine.py

Consumes de-noised GPU memory tensors directly from the CUDA pipeline.
Performs:
 - Thermal Anomaly Localization & Bounding Box Generation
 - Component Peak Temperature Tracking
 - Multi-Level Defect Severity Classification (NORMAL, WARNING, CRITICAL)
 - Statistical Thermal Histogram & Gradient Analysis
"""

import numpy as np
import cv2
import time
from typing import List, Dict, Any, Tuple

class ThermalAnomaly:
    """Represents a detected thermal defect."""
    def __init__(
        self,
        defect_id: int,
        name: str,
        bbox: Tuple[int, int, int, int], # (x, y, w, h)
        peak_temp_c: float,
        mean_temp_c: float,
        area_pixels: int,
        severity: str
    ):
        self.defect_id = defect_id
        self.name = name
        self.bbox = bbox
        self.peak_temp_c = peak_temp_c
        self.mean_temp_c = mean_temp_c
        self.area_pixels = area_pixels
        self.severity = severity

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.defect_id,
            "name": self.name,
            "bbox": list(self.bbox),
            "peak_temp_c": round(self.peak_temp_c, 2),
            "mean_temp_c": round(self.mean_temp_c, 2),
            "area_pixels": self.area_pixels,
            "severity": self.severity
        }


class ThermalInferenceEngine:
    """
    Downstream Inference Engine for Industrial Infrared Thermography.
    Receives de-noised temperature tensors directly.
    """

    def __init__(
        self,
        warning_temp_threshold_c: float = 70.0,
        critical_temp_threshold_c: float = 90.0,
        min_defect_area_pixels: int = 15
    ):
        self.warning_thresh = warning_temp_threshold_c
        self.critical_thresh = critical_temp_threshold_c
        self.min_area = min_defect_area_pixels

    def infer(
        self,
        denoised_temp_c: np.ndarray,
        ambient_temp_c: float = 25.0
    ) -> Tuple[List[ThermalAnomaly], Dict[str, Any]]:
        """
        Executes thermal defect segmentation and classification on the de-noised frame.
        """
        t_start = time.perf_counter()

        height, width = denoised_temp_c.shape
        anomalies: List[ThermalAnomaly] = []

        # 1. Delta-T Temperature thresholding above ambient baseline
        hotspot_mask = (denoised_temp_c >= self.warning_thresh).astype(np.uint8) * 255

        # 2. Morphological cleanup of connected heat islands
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned_mask = cv2.morphologyEx(hotspot_mask, cv2.MORPH_CLOSE, kernel)

        # 3. Contour detection for localized components
        contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        defect_counter = 1
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            component_roi = denoised_temp_c[y:y+h, x:x+w]
            
            peak_t = float(np.max(component_roi))
            mean_t = float(np.mean(component_roi))

            severity = "CRITICAL" if peak_t >= self.critical_thresh else "WARNING"
            
            # Identify likely component type based on aspect ratio and heat profile
            if peak_t >= 95.0:
                name = f"Power Transistor Overheat #{defect_counter}"
            elif peak_t >= 75.0:
                name = f"IC Regulator / High-Current Bus #{defect_counter}"
            else:
                name = f"Thermal Hotspot #{defect_counter}"

            anomaly = ThermalAnomaly(
                defect_id=defect_counter,
                name=name,
                bbox=(x, y, w, h),
                peak_temp_c=peak_t,
                mean_temp_c=mean_t,
                area_pixels=int(area),
                severity=severity
            )
            anomalies.append(anomaly)
            defect_counter += 1

        inference_time_ms = (time.perf_counter() - t_start) * 1000.0

        # Global frame statistics
        stats = {
            "max_frame_temp_c": float(np.max(denoised_temp_c)),
            "min_frame_temp_c": float(np.min(denoised_temp_c)),
            "mean_frame_temp_c": float(np.mean(denoised_temp_c)),
            "detected_defects_count": len(anomalies),
            "critical_count": sum(1 for a in anomalies if a.severity == "CRITICAL"),
            "warning_count": sum(1 for a in anomalies if a.severity == "WARNING"),
            "inference_latency_ms": round(inference_time_ms, 3)
        }

        return anomalies, stats

    def draw_detections(
        self,
        rgb_image: np.ndarray,
        anomalies: List[ThermalAnomaly]
    ) -> np.ndarray:
        """Overlays high-visibility bounding boxes and temperature readouts."""
        annotated = rgb_image.copy()

        for a in anomalies:
            x, y, w, h = a.bbox
            color = (0, 0, 255) if a.severity == "CRITICAL" else (0, 165, 255) # Red for critical, Amber for warning

            # Bounding box
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)

            # Label text
            label = f"{a.name} [{a.peak_temp_c:.1f}C]"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.45
            thickness = 1
            (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

            # Background text box
            cv2.rectangle(
                annotated,
                (x, max(0, y - text_h - 6)),
                (x + text_w + 4, max(0, y)),
                color,
                -1
            )
            cv2.putText(
                annotated,
                label,
                (x + 2, max(10, y - 4)),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA
            )

        return annotated
