"""
Unit Tests for CUDA Thermal De-noising & Inference Pipeline
File: tests/test_kernels.py
"""

import os
import sys
try:
    import pytest
except ImportError:
    pytest = None

# Adjust module path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../src/python"))

from thermal_generator import (
    generate_thermal_dataset,
    convert_temp_to_raw_radiometric_16bit,
    convert_raw_16bit_to_temp_celsius
)
from cuda_denoiser import CudaThermalDenoiser, CudaDenoiseConfig
from inference_engine import ThermalInferenceEngine

def calculate_psnr(clean: np.ndarray, test_img: np.ndarray, max_val: float = 120.0) -> float:
    mse = np.mean((clean - test_img) ** 2)
    if mse < 1e-10:
        return 100.0
    return float(20 * np.log10(max_val / np.sqrt(mse)))

def test_radiometric_conversion():
    """Verify exact round-trip conversion between Celsius and 16-bit centi-Kelvin."""
    original_celsius = np.array([[25.0, 100.0], [0.0, -10.0]], dtype=np.float32)
    raw_16bit = convert_temp_to_raw_radiometric_16bit(original_celsius)
    reconstructed_celsius = convert_raw_16bit_to_temp_celsius(raw_16bit)

    # 0.01 centi-Kelvin precision tolerance
    assert np.allclose(original_celsius, reconstructed_celsius, atol=0.02), "Radiometric conversion loss exceeded tolerance"

def test_psnr_improvement():
    """Verify that the CUDA pipeline yields significant PSNR improvement over noisy baseline."""
    clean, noisy, raw16, meta = generate_thermal_dataset(640, 512, "pcb", "medium")
    
    denoiser = CudaThermalDenoiser()
    denoised_c, _, _ = denoiser.denoise_frame(raw16)

    psnr_noisy = calculate_psnr(clean, noisy)
    psnr_denoised = calculate_psnr(clean, denoised_c)

    psnr_gain = psnr_denoised - psnr_noisy
    print(f"\n[Test PSNR] Noisy: {psnr_noisy:.2f} dB -> Denoised: {psnr_denoised:.2f} dB (Gain: +{psnr_gain:.2f} dB)")
    assert psnr_gain >= 5.0, f"Expected PSNR gain of at least 5.0 dB, got {psnr_gain:.2f} dB"

def test_hotspot_temperature_preservation():
    """Verify that edge-preserving filtering retains critical defect hotspot peak temperatures."""
    clean, noisy, raw16, meta = generate_thermal_dataset(640, 512, "pcb", "medium")
    
    denoiser = CudaThermalDenoiser()
    denoised_c, _, _ = denoiser.denoise_frame(raw16)

    gt_max = meta["max_temp_c"]
    denoised_max = np.max(denoised_c)

    temp_error = abs(gt_max - denoised_max)
    print(f"\n[Test Peak Temp] Ground Truth: {gt_max:.2f}°C vs Denoised: {denoised_max:.2f}°C (Error: {temp_error:.2f}°C)")
    assert temp_error <= 3.5, f"Peak hotspot temperature degraded by {temp_error:.2f}°C (Limit: 3.5°C)"

def test_dead_pixel_suppression():
    """Verify that isolated dead/stuck microbolometer pixels are removed."""
    clean_temp = np.full((100, 100), 30.0, dtype=np.float32)
    # Inject deliberate stuck cold and hot pixel spikes
    clean_temp[50, 50] = 150.0 # Stuck hot
    clean_temp[20, 20] = 0.0   # Stuck cold
    
    raw16 = convert_temp_to_raw_radiometric_16bit(clean_temp)
    denoiser = CudaThermalDenoiser()
    denoised_c, _, _ = denoiser.denoise_frame(raw16)

    assert denoised_c[50, 50] < 45.0, "Stuck hot pixel was not suppressed"
    assert denoised_c[20, 20] > 20.0, "Stuck cold pixel was not suppressed"

def test_inference_defect_detection():
    """Verify that the downstream inference engine flags critical defects."""
    clean, noisy, raw16, meta = generate_thermal_dataset(640, 512, "pcb", "medium")
    denoiser = CudaThermalDenoiser()
    denoised_c, _, _ = denoiser.denoise_frame(raw16)

    engine = ThermalInferenceEngine(warning_temp_threshold_c=70.0, critical_temp_threshold_c=90.0)
    anomalies, stats = engine.infer(denoised_c, ambient_temp_c=25.0)

    assert len(anomalies) >= 1, "Inference engine failed to detect ground truth thermal defects"
    assert stats["critical_count"] >= 1, "Failed to classify MOSFET overheat as CRITICAL"

import numpy as np

if __name__ == "__main__":
    print("Running CUDA Thermal Pipeline Unit Tests...")
    test_radiometric_conversion()
    print("  [PASS] test_radiometric_conversion")
    test_psnr_improvement()
    print("  [PASS] test_psnr_improvement")
    test_hotspot_temperature_preservation()
    print("  [PASS] test_hotspot_temperature_preservation")
    test_dead_pixel_suppression()
    print("  [PASS] test_dead_pixel_suppression")
    test_inference_defect_detection()
    print("  [PASS] test_inference_defect_detection")
    print("\nALL 5 TEST CASES PASSED SUCCESSFULLY!")
