"""
End-to-End Thermal De-noising & Inference Pipeline Runner
File: src/python/pipeline_runner.py

Orchestrates:
 1. Thermal image acquisition / simulation (16-bit raw radiometric)
 2. Custom CUDA shared-memory de-noising
 3. Zero-copy / direct GPU tensor handoff to Inference Engine
 4. Telemetry logging and side-by-side dashboard generation
"""

import os
import sys
import time
import argparse
import cv2
import numpy as np

# Adjust module path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from thermal_generator import generate_thermal_dataset
from cuda_denoiser import CudaThermalDenoiser, CudaDenoiseConfig
from inference_engine import ThermalInferenceEngine
from visualizer import ThermalVisualizer

def calculate_psnr(clean: np.ndarray, noisy_or_denoised: np.ndarray, max_val: float = 120.0) -> float:
    """Calculates Peak Signal-to-Noise Ratio in dB."""
    mse = np.mean((clean - noisy_or_denoised) ** 2)
    if mse < 1e-10:
        return 100.0
    return float(20 * np.log10(max_val / np.sqrt(mse)))

def run_pipeline(
    resolution: tuple = (1280, 720),
    scene_type: str = "pcb",
    noise_level: str = "medium",
    num_frames: int = 10,
    output_dir: str = "output",
    show_gui: bool = False
):
    os.makedirs(output_dir, exist_ok=True)
    w, h = resolution

    print(f"\n=======================================================")
    print(f"  CUDA Thermal Denoising & Inference Pipeline Runner   ")
    print(f"=======================================================")
    print(f" [Config] Resolution: {w}x{h}")
    print(f" [Config] Scene: {scene_type.upper()} | Noise: {noise_level.upper()}")
    print(f" [Config] Frame Count: {num_frames}")
    print(f" [Config] Output Directory: {output_dir}\n")

    # Initialize Modules
    denoise_cfg = CudaDenoiseConfig(
        sigma_spatial=2.5,
        sigma_range=0.08,
        filter_radius=2,
        min_temp_celsius=0.0,
        max_temp_celsius=160.0,
        colormap="inferno"
    )
    denoiser = CudaThermalDenoiser(denoise_cfg)
    inference_engine = ThermalInferenceEngine(
        warning_temp_threshold_c=70.0,
        critical_temp_threshold_c=90.0
    )

    latencies = []
    fps_list = []
    psnr_noisy_list = []
    psnr_denoised_list = []

    print(f"Processing frames...")

    for frame_idx in range(num_frames):
        # 1. Acquire / Generate Thermal Frame
        clean_c, noisy_c, raw_16bit, meta = generate_thermal_dataset(
            width=w, height=h, scene_type=scene_type, noise_level=noise_level
        )

        # 2. CUDA Shared-Memory Denoising
        denoised_c, rgb_disp, timing = denoiser.denoise_frame(raw_16bit, return_rgb=True)

        # 3. Downstream Inference Engine
        anomalies, stats = inference_engine.infer(denoised_c, ambient_temp_c=meta["ambient_c"])
        annotated_rgb = inference_engine.draw_detections(rgb_disp, anomalies)

        # 4. Metrics
        psnr_noisy = calculate_psnr(clean_c, noisy_c)
        psnr_denoised = calculate_psnr(clean_c, denoised_c)
        
        latencies.append(timing["total_ms"])
        fps_list.append(timing["fps"])
        psnr_noisy_list.append(psnr_noisy)
        psnr_denoised_list.append(psnr_denoised)

        # 5. Dashboard Generation
        dashboard = ThermalVisualizer.create_side_by_side_dashboard(
            noisy_raw_16bit=raw_16bit,
            denoised_temp_c=denoised_c,
            clean_ground_truth_c=clean_c,
            annotated_inference_rgb=annotated_rgb,
            telemetry=timing,
            anomalies_count=len(anomalies),
            title=f"CUDA Thermal Pipeline | Frame #{frame_idx+1} | {meta['scene']}"
        )

        # Save sample frame
        if frame_idx == 0 or frame_idx == num_frames - 1:
            out_path = os.path.join(output_dir, f"thermal_dashboard_frame_{frame_idx+1}.jpg")
            cv2.imwrite(out_path, dashboard)
            print(f" [Frame #{frame_idx+1:02d}] Saved dashboard image to: {out_path}")

        print(f" [Frame #{frame_idx+1:02d}] Latency: {timing['total_ms']:.2f}ms | FPS: {timing['fps']:.1f} | PSNR Gain: +{(psnr_denoised - psnr_noisy):.2f} dB | Defects: {len(anomalies)}")

        if show_gui:
            cv2.imshow("CUDA Thermal Pipeline (Jetson / Desktop)", dashboard)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    if show_gui:
        cv2.destroyAllWindows()

    avg_lat = np.mean(latencies)
    avg_fps = np.mean(fps_list)
    avg_psnr_gain = np.mean(psnr_denoised_list) - np.mean(psnr_noisy_list)

    print(f"\n=======================================================")
    print(f"                  PIPELINE SUMMARY                     ")
    print(f"=======================================================")
    print(f" Average Latency:      {avg_lat:.2f} ms")
    print(f" Average Throughput:   {avg_fps:.1f} FPS")
    print(f" Baseline Noisy PSNR:  {np.mean(psnr_noisy_list):.2f} dB")
    print(f" Denoised PSNR:        {np.mean(psnr_denoised_list):.2f} dB")
    print(f" Net PSNR Improvement: +{avg_psnr_gain:.2f} dB")
    print(f"=======================================================\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CUDA Thermal Denoise Pipeline")
    parser.add_argument("--width", type=int, default=1280, help="Frame width")
    parser.add_argument("--height", type=int, default=720, help="Frame height")
    parser.add_argument("--scene", type=str, default="pcb", choices=["pcb", "motor"], help="Thermal scene")
    parser.add_argument("--noise", type=str, default="medium", choices=["low", "medium", "high"], help="Noise level")
    parser.add_argument("--frames", type=int, default=5, help="Number of frames to process")
    parser.add_argument("--output", type=str, default="output", help="Output directory")
    parser.add_argument("--gui", action="store_true", help="Display GUI window")
    args = parser.parse_args()

    run_pipeline(
        resolution=(args.width, args.height),
        scene_type=args.scene,
        noise_level=args.noise,
        num_frames=args.frames,
        output_dir=args.output,
        show_gui=args.gui
    )
