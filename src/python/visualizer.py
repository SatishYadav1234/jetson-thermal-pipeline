"""
High-Performance Thermal Visualizer & Dashboard
File: src/python/visualizer.py

Generates 4-panel side-by-side diagnostic displays:
 [1] Noisy Raw Thermal Input (Simulated Sensor Hiss & Dead Pixels)
 [2] Custom CUDA Shared-Memory Filtered (Edge-Preserving Smooth)
 [3] Radiometric Temperature Heatmap (High-Contrast Colormap)
 [4] Inference Engine Defect Localization & Telemetry (FPS, Latency)
"""

import numpy as np
import cv2
from typing import List, Dict, Any, Optional

class ThermalVisualizer:
    """Creates visualization layouts for both GUI display and file export."""

    @staticmethod
    def create_side_by_side_dashboard(
        noisy_raw_16bit: np.ndarray,
        denoised_temp_c: np.ndarray,
        clean_ground_truth_c: Optional[np.ndarray],
        annotated_inference_rgb: np.ndarray,
        telemetry: Dict[str, Any],
        anomalies_count: int,
        title: str = "CUDA Industrial Thermal De-noising Pipeline"
    ) -> np.ndarray:
        """
        Assembles a 2x2 multi-panel diagnostic dashboard.
        """
        # 1. Normalize Noisy Raw for display
        raw_c = (noisy_raw_16bit.astype(np.float32) * 0.01) - 273.15
        min_c, max_c = 0.0, 120.0
        noisy_norm = np.clip((raw_c - min_c) / (max_c - min_c) * 255.0, 0, 255).astype(np.uint8)
        panel_noisy = cv2.applyColorMap(noisy_norm, cv2.COLORMAP_INFERNO)

        # 2. Denoised Colormap
        denoised_norm = np.clip((denoised_temp_c - min_c) / (max_c - min_c) * 255.0, 0, 255).astype(np.uint8)
        panel_denoised = cv2.applyColorMap(denoised_norm, cv2.COLORMAP_INFERNO)

        # 3. Clean Ground Truth (or Turbo Colormap)
        if clean_ground_truth_c is not None:
            clean_norm = np.clip((clean_ground_truth_c - min_c) / (max_c - min_c) * 255.0, 0, 255).astype(np.uint8)
            panel_ground_truth = cv2.applyColorMap(clean_norm, cv2.COLORMAP_TURBO)
        else:
            panel_ground_truth = cv2.applyColorMap(denoised_norm, cv2.COLORMAP_TURBO)

        # 4. Inference Output is already RGB
        panel_inference = annotated_inference_rgb

        # Resize all to uniform panel resolution for display (e.g. 640x360 each)
        pw, ph = 640, 360
        p1 = cv2.resize(panel_noisy, (pw, ph))
        p2 = cv2.resize(panel_denoised, (pw, ph))
        p3 = cv2.resize(panel_ground_truth, (pw, ph))
        p4 = cv2.resize(panel_inference, (pw, ph))

        # Add Titles & Overlays to each panel
        ThermalVisualizer._add_panel_label(p1, "[1] Raw Sensor Input (Noisy + Dead Pixels)", (0, 0, 255))
        ThermalVisualizer._add_panel_label(p2, "[2] CUDA Custom Shared-Memory Filtered", (0, 255, 0))
        ThermalVisualizer._add_panel_label(p3, "[3] Radiometric Heatmap (Turbo LUT)", (255, 200, 0))
        ThermalVisualizer._add_panel_label(p4, f"[4] Inference Engine ({anomalies_count} Defects)", (0, 165, 255))

        # Stack into 2x2 grid
        top_row = np.hstack([p1, p2])
        bottom_row = np.hstack([p3, p4])
        grid = np.vstack([top_row, bottom_row])

        # Add Top Header & Telemetry Banner
        banner_h = 70
        banner = np.zeros((banner_h, grid.shape[1], 3), dtype=np.uint8)
        banner[:] = (25, 25, 25) # Dark charcoal background

        # Title
        cv2.putText(banner, title, (20, 30), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 1, cv2.LINE_AA)

        # Telemetry metrics
        fps_val = telemetry.get("fps", 0.0)
        lat_val = telemetry.get("total_ms", 0.0)
        kernel_val = telemetry.get("kernel_ms", 0.0)
        
        telemetry_str = f"Throughput: {fps_val:.1f} FPS | Total Latency: {lat_val:.2f}ms | CUDA Kernel: {kernel_val:.2f}ms | Hardware: Jetson/CUDA"
        cv2.putText(banner, telemetry_str, (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 180), 1, cv2.LINE_AA)

        full_dashboard = np.vstack([banner, grid])
        return full_dashboard

    @staticmethod
    def _add_panel_label(img: np.ndarray, text: str, accent_color: tuple):
        """Draws clean HUD banner on top of each subpanel."""
        # Top banner background
        cv2.rectangle(img, (0, 0), (img.shape[1], 28), (15, 15, 15), -1)
        # Left color pill
        cv2.rectangle(img, (0, 0), (6, 28), accent_color, -1)
        # Text
        cv2.putText(img, text, (15, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (240, 240, 240), 1, cv2.LINE_AA)
