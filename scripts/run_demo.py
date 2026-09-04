"""
1-Click End-to-End Demonstration Script
File: scripts/run_demo.py

Runs the complete CUDA thermal denoising & inference pipeline,
generates side-by-side diagnostic visual dashboards, and benchmarks performance.
"""

import os
import sys

# Add project root to sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(root_dir, "src/python"))

from pipeline_runner import run_pipeline

def main():
    print("=" * 70)
    print("  CUDA-Accelerated Image De-noising Pipeline for Jetson Orion Nano")
    print("=" * 70)
    print(" Running comprehensive demonstration on synthetic industrial data...\n")

    output_dir = os.path.join(root_dir, "output")

    # Run on PCB scenario
    print("[1/2] Processing PCB Inspection Scene (1280x720 HD)...")
    run_pipeline(
        resolution=(1280, 720),
        scene_type="pcb",
        noise_level="medium",
        num_frames=3,
        output_dir=output_dir,
        show_gui=False
    )

    # Run on Motor Bearing scenario
    print("\n[2/2] Processing Motor Bearing Friction Scene (1280x720 HD)...")
    run_pipeline(
        resolution=(1280, 720),
        scene_type="motor",
        noise_level="medium",
        num_frames=3,
        output_dir=output_dir,
        show_gui=False
    )

    print("\n" + "=" * 70)
    print(" [DEMO COMPLETE] Visual outputs generated successfully!")
    print(f" Check results in: {output_dir}")
    print("=" * 70)

if __name__ == "__main__":
    main()
