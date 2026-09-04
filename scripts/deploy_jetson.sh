#!/bin/bash
# ==============================================================================
# One-Click Deployment Script for NVIDIA Jetson Orion Nano
# Hardware: NVIDIA Jetson Orion Nano Developer Kit (Ampere sm_87, aarch64)
# JetPack Version: JetPack 5.1+ / JetPack 6.0+
# ==============================================================================

set -e

echo "================================================================="
echo "   NVIDIA Jetson Orion Nano - CUDA Thermal Pipeline Deployer    "
echo "================================================================="

# 1. Check Platform Architecture
ARCH=$(uname -m)
if [ "$ARCH" != "aarch64" ]; then
    echo "[WARNING] Detected architecture '$ARCH' is not aarch64 (Jetson)."
    echo "          Proceeding with standard Linux compilation."
else
    echo "[INFO] Verified target architecture: aarch64 (NVIDIA Jetson)."
fi

# 2. Configure Max Power & Performance Mode on Jetson
if command -v nvpmodel &> /dev/null; then
    echo "[STEP 1/6] Setting Jetson Orion Nano to MAXN Power Mode..."
    sudo nvpmodel -m 0 || true
fi

if command -v jetson_clocks &> /dev/null; then
    echo "[STEP 2/6] Locking Clocks to Maximum via jetson_clocks..."
    sudo jetson_clocks || true
fi

# 3. Verify CUDA Installation & nvcc
echo "[STEP 3/6] Verifying CUDA Toolkit..."
if ! command -v nvcc &> /dev/null; then
    echo "[INFO] Adding default CUDA paths to environment..."
    export PATH=/usr/local/cuda/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
fi

if ! command -v nvcc &> /dev/null; then
    echo "[ERROR] 'nvcc' not found. Please ensure NVIDIA JetPack / CUDA is installed."
    exit 1
fi
echo "[INFO] Found CUDA Compiler: $(nvcc --version | grep 'release')"

# 4. Install System & Python Dependencies
echo "[STEP 4/6] Installing dependencies..."
sudo apt-get update -y
sudo apt-get install -y build-essential cmake git python3-pip python3-dev libopencv-dev

python3 -m pip install --upgrade pip
python3 -m pip install numpy opencv-python pillow matplotlib pytest

# 5. Build Native CUDA Shared Library & C++ Executable
echo "[STEP 5/6] Building CUDA Kernels for sm_87 (Jetson Orion Nano)..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$ROOT_DIR/build"

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

cmake .. -DCMAKE_BUILD_TYPE=Release -DCUDA_ARCH="87"
make -j$(nproc)

echo "[INFO] Compilation successful: libdenoise.so & cuda_thermal_benchmark generated!"

# 6. Run Sanity Check
echo "[STEP 6/6] Running verification demo..."
cd "$ROOT_DIR"
python3 scripts/run_demo.py

echo "================================================================="
echo " [SUCCESS] Deployment completed successfully on Jetson Orion Nano!"
echo " To run live visual pipeline with monitor display:"
echo "   python3 src/python/pipeline_runner.py --gui"
echo "================================================================="
