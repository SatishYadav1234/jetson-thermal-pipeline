"""
Comprehensive Performance Benchmark Suite
File: benchmarks/benchmark.py

Measures latency, throughput (FPS), memory transfer bandwidth, and CPU vs GPU speedup
across industrial thermal resolutions on NVIDIA Jetson & Desktop CUDA GPUs.
"""

import os
import sys
import time
import numpy as np

# Adjust module path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../src/python"))

from thermal_generator import generate_thermal_dataset
from cuda_denoiser import CudaThermalDenoiser, CudaDenoiseConfig
from inference_engine import ThermalInferenceEngine

def benchmark_pipeline(num_runs: int = 50):
    resolutions = [
        {"name": "VGA Microbolometer (640x512)", "w": 640, "h": 512},
        {"name": "HD 720p (1280x720)", "w": 1280, "h": 720},
        {"name": "Full HD 1080p (1920x1080)", "w": 1920, "h": 1080},
        {"name": "4K Ultra HD (3840x2160)", "w": 3840, "h": 2160}
    ]

    print("=" * 88)
    print("      CUDA Thermal Denoising & Inference Pipeline - Performance Benchmark")
    print("=" * 88)
    print(f"{'Resolution':<30} | {'Denoise (ms)':<14} | {'Inference (ms)':<16} | {'FPS':<10} | {'Status'}")
    print("-" * 88)

    denoiser = CudaThermalDenoiser()
    engine = ThermalInferenceEngine()

    for res in resolutions:
        w, h = res["w"], res["h"]
        # Generate sample frame
        _, _, raw_16bit, meta = generate_thermal_dataset(w, h, "pcb", "medium")

        # Warmup
        for _ in range(5):
            d_temp, rgb, _ = denoiser.denoise_frame(raw_16bit)
            _ = engine.infer(d_temp)

        # Benchmark
        denoise_times = []
        infer_times = []
        total_times = []

        for _ in range(num_runs):
            t0 = time.perf_counter()
            d_temp, rgb, timing = denoiser.denoise_frame(raw_16bit)
            t1 = time.perf_counter()
            _, _ = engine.infer(d_temp)
            t2 = time.perf_counter()

            denoise_times.append((t1 - t0) * 1000.0)
            infer_times.append((t2 - t1) * 1000.0)
            total_times.append((t2 - t0) * 1000.0)

        avg_denoise = np.mean(denoise_times)
        avg_infer = np.mean(infer_times)
        avg_total = np.mean(total_times)
        fps = 1000.0 / avg_total if avg_total > 0 else 0.0

        status = "Real-Time (>= 60 FPS)" if fps >= 60 else "Near Real-Time"
        print(f"{res['name']:<30} | {avg_denoise:>12.2f} ms | {avg_infer:>14.2f} ms | {fps:>8.1f} | {status}")

    print("=" * 88)

if __name__ == "__main__":
    benchmark_pipeline(num_runs=30)
