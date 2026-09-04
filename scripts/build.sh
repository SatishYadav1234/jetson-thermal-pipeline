#!/bin/bash
# Linux Build Script
set -e

mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
echo "[INFO] Build completed successfully!"
