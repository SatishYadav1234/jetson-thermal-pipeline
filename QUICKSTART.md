# ⚡ 5-Minute Quickstart Guide

This guide gets your **CUDA-Accelerated Thermal De-noising Pipeline** running immediately on both your local development machine and your **NVIDIA Jetson Orion Nano**.

---

## 💻 1. Running on Local PC (Windows / Linux)

### Step 1: Install Python Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run 1-Click Demonstration
```bash
python scripts/run_demo.py
```
> **What this does**: Automatically generates synthetic 1280x720 HD radiometric thermal scenes (PCB & Motor Bearing), runs the de-noising pipeline, executes the thermal anomaly inference engine, and saves diagnostic dashboards to `output/`.

### Step 3: Run Unit Tests & Verification
```bash
pytest tests/test_kernels.py -v
```

---

## 🟢 2. Deploying on NVIDIA Jetson Orion Nano

### Hardware Checklist:
* NVIDIA Jetson Orion Nano Developer Kit (JetPack 5.1+ or 6.0+)
* DisplayPort-to-HDMI Cable connected to your Monitor
* USB Keyboard & Mouse
* USB Pendrive **OR** Ethernet cable connected to your local network

---

### Deployment Option A: Via USB Flash Drive (Pendrive)
1. **On your PC**: Copy the `cuda-thermal-denoise-jetson` folder to your USB flash drive.
2. **On Jetson Orion Nano**: Plug the USB drive into the Jetson.
3. Open Terminal (`Ctrl + Alt + T`) and copy to home directory:
   ```bash
   cp -r /media/$USER/*/cuda-thermal-denoise-jetson ~/cuda-thermal-denoise-jetson
   cd ~/cuda-thermal-denoise-jetson
   ```
4. Run the automated deployment script:
   ```bash
   chmod +x scripts/deploy_jetson.sh
   ./scripts/deploy_jetson.sh
   ```

---

### Deployment Option B: Via GitHub & Ethernet
1. Connect Ethernet cable to Jetson.
2. Open Terminal and clone your GitHub repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/cuda-thermal-denoise-jetson.git
   cd cuda-thermal-denoise-jetson
   chmod +x scripts/deploy_jetson.sh
   ./scripts/deploy_jetson.sh
   ```

---

## 🖥️ 3. Running Live on Jetson Display

### Launch Live Interactive Visualizer (Monitor Output)
```bash
python3 src/python/pipeline_runner.py --gui --scene pcb --width 1280 --height 720
```

### Run High-Performance Standalone C++ Benchmark
```bash
cd build
./cuda_thermal_benchmark
```

### Run Multi-Resolution Python Benchmark
```bash
python3 benchmarks/benchmark.py
```
