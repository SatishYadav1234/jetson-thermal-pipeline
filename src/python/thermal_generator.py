"""
Synthetic High-Resolution Industrial Thermal Image Generator
File: src/python/thermal_generator.py

Simulates realistic 16-bit radiometric infrared thermography data:
 - Microbolometer sensor noise (Gaussian NETD)
 - Fixed-Pattern Noise (FPN) and column striping
 - Stuck/dead pixels (impulse thermal noise)
 - Industrial scenarios: PCB inspection, Motor bearing friction, Solar panel defect, Electrical panel
"""

import numpy as np
import cv2
from typing import Tuple, Dict, Any, Optional

class IndustrialThermalScene:
    """Industrial thermography scenes generator."""

    @staticmethod
    def generate_pcb_inspection(width: int, height: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Simulates an electronic PCB under load.
        Hot components: Power MOSFET (95°C), Microcontroller (65°C), Voltage Regulator (80°C).
        Ambient: 25°C.
        """
        # Base temperature: Ambient 25°C (298.15K)
        temp_map = np.full((height, width), 25.0, dtype=np.float32)

        # Subtle ambient heat dissipation gradient
        y_coords, x_coords = np.mgrid[0:height, 0:width]
        temp_map += 2.0 * np.sin(x_coords / width * np.pi) * np.cos(y_coords / height * np.pi)

        ground_truth_defects = []

        # 1. Overheated Power MOSFET (Defect: 105°C / Limit: 85°C)
        cx1, cy1 = int(width * 0.35), int(height * 0.40)
        sigma1 = min(width, height) * 0.05
        dist1_sq = (x_coords - cx1)**2 + (y_coords - cy1)**2
        temp_map += 80.0 * np.exp(-dist1_sq / (2.0 * sigma1**2))
        ground_truth_defects.append({
            "name": "Overheated MOSFET Q1",
            "bbox": [int(cx1 - sigma1*1.5), int(cy1 - sigma1*1.5), int(sigma1*3), int(sigma1*3)],
            "max_temp_c": 105.0,
            "severity": "CRITICAL"
        })

        # 2. Linear Voltage Regulator U3 (78°C)
        cx2, cy2 = int(width * 0.70), int(height * 0.30)
        sigma2 = min(width, height) * 0.04
        dist2_sq = (x_coords - cx2)**2 + (y_coords - cy2)**2
        temp_map += 53.0 * np.exp(-dist2_sq / (2.0 * sigma2**2))
        ground_truth_defects.append({
            "name": "Regulator U3 High Load",
            "bbox": [int(cx2 - sigma2*1.5), int(cy2 - sigma2*1.5), int(sigma2*3), int(sigma2*3)],
            "max_temp_c": 78.0,
            "severity": "WARNING"
        })

        # 3. High-current trace warming
        trace_mask = (np.abs(y_coords - int(height * 0.40)) < 3) & (x_coords > cx1) & (x_coords < int(width * 0.65))
        temp_map[trace_mask] += 25.0

        # 4. Background passive components (35°C - 45°C)
        np.random.seed(42)
        for _ in range(8):
            px = np.random.randint(int(width*0.1), int(width*0.9))
            py = np.random.randint(int(height*0.1), int(height*0.9))
            rad = np.random.randint(4, 12)
            temp_map[max(0, py-rad):min(height, py+rad), max(0, px-rad):min(width, px+rad)] += np.random.uniform(10.0, 20.0)

        metadata = {
            "scene": "PCB Electronic Inspection",
            "ambient_c": 25.0,
            "max_temp_c": float(np.max(temp_map)),
            "min_temp_c": float(np.min(temp_map)),
            "defects": ground_truth_defects
        }
        return temp_map, metadata

    @staticmethod
    def generate_motor_bearing(width: int, height: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Simulates an industrial electric motor with bearing friction fault.
        """
        temp_map = np.full((height, width), 28.0, dtype=np.float32)
        y_coords, x_coords = np.mgrid[0:height, 0:width]

        # Motor casing cylinder temperature (50°C)
        cx, cy = int(width * 0.5), int(height * 0.5)
        rx, ry = int(width * 0.35), int(height * 0.25)
        ellipse_mask = ((x_coords - cx)**2 / rx**2 + (y_coords - cy)**2 / ry**2) <= 1.0
        temp_map[ellipse_mask] = 48.0 + 5.0 * np.cos((x_coords[ellipse_mask] - cx) / rx * np.pi)

        # Bearing Housing Defect (Friction Overheat: 112°C)
        bx, by = int(width * 0.22), int(height * 0.5)
        dist_b = (x_coords - bx)**2 + (y_coords - by)**2
        sigma_b = min(width, height) * 0.04
        temp_map += 64.0 * np.exp(-dist_b / (2.0 * sigma_b**2))

        defects = [{
            "name": "Drive-End Bearing Overheat",
            "bbox": [int(bx - sigma_b*2), int(by - sigma_b*2), int(sigma_b*4), int(sigma_b*4)],
            "max_temp_c": 112.0,
            "severity": "CRITICAL"
        }]

        metadata = {
            "scene": "Motor Bearing Friction",
            "ambient_c": 28.0,
            "max_temp_c": float(np.max(temp_map)),
            "min_temp_c": float(np.min(temp_map)),
            "defects": defects
        }
        return temp_map, metadata


class ThermalSensorNoiseModel:
    """
    Simulates real-world uncooled microbolometer sensor noise:
      - Noise Equivalent Temperature Difference (NETD): Gaussian noise (e.g. 40mK - 80mK)
      - Fixed Pattern Noise (FPN): Column non-uniformity & spatial gain drift
      - Dead / Stuck Pixels: Defective microbolometer elements (hot/cold pixels)
    """

    def __init__(
        self,
        netd_std_c: float = 2.5,        # Thermal sensitivity noise std in °C
        fpn_column_std_c: float = 1.2,   # Column fixed pattern noise std in °C
        dead_pixel_ratio: float = 0.003  # 0.3% dead/stuck pixels
    ):
        self.netd_std_c = netd_std_c
        self.fpn_column_std_c = fpn_column_std_c
        self.dead_pixel_ratio = dead_pixel_ratio

    def corrupt(self, clean_temp_c: np.ndarray, seed: Optional[int] = None) -> np.ndarray:
        """Applies realistic thermal sensor degradation."""
        if seed is not None:
            np.random.seed(seed)

        height, width = clean_temp_c.shape
        noisy_temp = clean_temp_c.copy()

        # 1. Gaussian NETD Sensor Noise
        gaussian_noise = np.random.normal(0.0, self.netd_std_c, (height, width)).astype(np.float32)
        noisy_temp += gaussian_noise

        # 2. Column-Wise Fixed Pattern Noise (FPN striping)
        col_offsets = np.random.normal(0.0, self.fpn_column_std_c, (1, width)).astype(np.float32)
        noisy_temp += col_offsets

        # 3. Dead / Stuck Microbolometer Sensor Pixels
        num_dead = int(height * width * self.dead_pixel_ratio)
        if num_dead > 0:
            dead_y = np.random.randint(0, height, num_dead)
            dead_x = np.random.randint(0, width, num_dead)
            
            # 50% stuck cold (0°C), 50% stuck hot (150°C)
            stuck_values = np.random.choice([0.0, 150.0], size=num_dead).astype(np.float32)
            noisy_temp[dead_y, dead_x] = stuck_values

        return noisy_temp


def convert_temp_to_raw_radiometric_16bit(temp_celsius: np.ndarray) -> np.ndarray:
    """
    Converts Celsius temperature map to raw 16-bit radiometric centi-Kelvin
    (Standard FLIR/Lepton radiometric format: 25.0°C -> 298.15K -> 29815 counts).
    """
    temp_kelvin = temp_celsius + 273.15
    raw_16bit = np.clip(temp_kelvin * 100.0, 0, 65535).astype(np.uint16)
    return raw_16bit


def convert_raw_16bit_to_temp_celsius(raw_16bit: np.ndarray) -> np.ndarray:
    """Decodes 16-bit radiometric centi-Kelvin back to Celsius."""
    temp_kelvin = raw_16bit.astype(np.float32) * 0.01
    return temp_kelvin - 273.15


def generate_thermal_dataset(
    width: int = 1280,
    height: int = 720,
    scene_type: str = "pcb",
    noise_level: str = "medium"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Convenience factory to generate full dataset pair:
    Returns: (clean_celsius, noisy_celsius, raw_16bit_radiometric, metadata)
    """
    if scene_type == "motor":
        clean_temp, meta = IndustrialThermalScene.generate_motor_bearing(width, height)
    else:
        clean_temp, meta = IndustrialThermalScene.generate_pcb_inspection(width, height)

    noise_presets = {
        "low": {"netd": 1.0, "fpn": 0.5, "dead": 0.001},
        "medium": {"netd": 2.5, "fpn": 1.2, "dead": 0.003},
        "high": {"netd": 4.5, "fpn": 2.5, "dead": 0.008}
    }
    cfg = noise_presets.get(noise_level, noise_presets["medium"])
    noise_model = ThermalSensorNoiseModel(
        netd_std_c=cfg["netd"],
        fpn_column_std_c=cfg["fpn"],
        dead_pixel_ratio=cfg["dead"]
    )

    noisy_temp = noise_model.corrupt(clean_temp, seed=123)
    raw_16bit = convert_temp_to_raw_radiometric_16bit(noisy_temp)

    return clean_temp, noisy_temp, raw_16bit, meta


if __name__ == "__main__":
    clean, noisy, raw16, meta = generate_thermal_dataset(1280, 720, "pcb", "medium")
    print(f"[Thermal Generator] Generated {meta['scene']} at {raw16.shape[1]}x{raw16.shape[0]}")
    print(f"  Clean Temp Range: {meta['min_temp_c']:.1f}°C to {meta['max_temp_c']:.1f}°C")
    print(f"  Raw 16-bit Range: {np.min(raw16)} to {np.max(raw16)} counts")
    print(f"  Ground Truth Defects: {len(meta['defects'])}")
