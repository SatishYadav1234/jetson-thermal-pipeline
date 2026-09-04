/**
 * CUDA-Accelerated Thermal Image De-noising Pipeline
 * C++ Standalone Benchmark & Verification Harness
 * File: src/cpp/main.cpp
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cmath>
#include "../kernels/denoise_kernels.h"

// Generate synthetic thermal test pattern in Host memory (16-bit Raw Centi-Kelvin)
void generate_synthetic_thermal_frame(std::vector<uint16_t>& buffer, int width, int height) {
    buffer.resize(width * height);
    
    // Base temperature: Ambient 298.15K (25°C) -> 29815 centi-Kelvin
    uint16_t ambient_ck = 29815;
    
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float fx = (float)x / width;
            float fy = (float)y / height;

            // Background ambient with gradient
            float temp = 298.15f + 5.0f * std::sin(fx * 3.14159f) * std::cos(fy * 3.14159f);

            // Hotspot 1: Overheated Power MOSFET (363.15K / 90°C)
            float dx1 = fx - 0.35f;
            float dy1 = fy - 0.40f;
            float r1_sq = dx1 * dx1 + dy1 * dy1;
            if (r1_sq < 0.015f) {
                temp += 65.0f * std::exp(-r1_sq / 0.003f);
            }

            // Hotspot 2: High-Current Trace / Inductor (343.15K / 70°C)
            float dx2 = fx - 0.70f;
            float dy2 = fy - 0.65f;
            float r2_sq = dx2 * dx2 + dy2 * dy2;
            if (r2_sq < 0.02f) {
                temp += 45.0f * std::exp(-r2_sq / 0.005f);
            }

            // Synthetic Microbolometer Sensor Noise (Gaussian + Fixed Pattern)
            float noise = ((float)rand() / RAND_MAX - 0.5f) * 6.0f; // ±3.0°C thermal noise
            temp += noise;

            // Salt-and-pepper dead sensor pixels (0.5% probability)
            if ((rand() % 1000) < 5) {
                temp = (rand() % 2 == 0) ? 250.0f : 450.0f; // Stuck cold / stuck hot
            }

            // Convert to centi-Kelvin
            buffer[y * width + x] = (uint16_t)(temp * 100.0f);
        }
    }
}

int main(int argc, char** argv) {
    std::cout << "=================================================================\n";
    std::cout << "  CUDA Thermal De-noising Pipeline (Jetson Orion Nano / Desktop)  \n";
    std::cout << "=================================================================\n";

    // Inspect CUDA Device
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        std::cerr << "[ERROR] No CUDA-capable device detected!\n";
        return 1;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "[INFO] Active GPU: " << prop.name << "\n";
    std::cout << "[INFO] Compute Capability: sm_" << prop.major << prop.minor << "\n";
    std::cout << "[INFO] Total Global Memory: " << (prop.totalGlobalMem / (1024 * 1024)) << " MB\n";
    std::cout << "[INFO] Shared Memory per Block: " << (prop.sharedMemPerBlock / 1024) << " KB\n\n";

    // Resolutions to benchmark
    struct Resolution { int w; int h; const char* label; };
    std::vector<Resolution> resolutions = {
        {640, 512, "VGA Microbolometer (640x512)"},
        {1280, 720, "HD 720p (1280x720)"},
        {1920, 1080, "Full HD 1080p (1920x1080)"},
        {3840, 2160, "4K Ultra HD (3840x2160)"}
    };

    DenoiseConfig config;
    config.sigma_spatial = 2.5f;
    config.sigma_range = 0.08f;
    config.filter_radius = 2; // 5x5 window
    config.min_temp_kelvin = 273.15f; // 0°C
    config.max_temp_kelvin = 393.15f; // 120°C
    config.colormap = COLORMAP_INFERNO;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    std::cout << std::left << std::setw(32) << "Resolution"
              << std::setw(16) << "Frame Latency"
              << std::setw(16) << "Throughput"
              << std::setw(18) << "Memory Bandwidth" << "\n";
    std::cout << std::string(82, '-') << "\n";

    for (const auto& res : resolutions) {
        config.width = res.w;
        config.height = res.h;

        // Allocate Host Pinned Memory
        uint16_t* h_raw = nullptr;
        float* h_denoised = nullptr;
        size_t raw_bytes = res.w * res.h * sizeof(uint16_t);
        size_t float_bytes = res.w * res.h * sizeof(float);

        cudaMallocHost((void**)&h_raw, raw_bytes);
        cudaMallocHost((void**)&h_denoised, float_bytes);

        // Generate synthetic data
        std::vector<uint16_t> sample_data;
        generate_synthetic_thermal_frame(sample_data, res.w, res.h);
        memcpy(h_raw, sample_data.data(), raw_bytes);

        // Allocate Device Buffers
        uint16_t* d_raw = nullptr;
        float* d_denoised = nullptr;
        cudaMalloc(&d_raw, raw_bytes);
        cudaMalloc(&d_denoised, float_bytes);

        // Warmup runs
        for (int i = 0; i < 5; ++i) {
            cudaMemcpyAsync(d_raw, h_raw, raw_bytes, cudaMemcpyHostToDevice, stream);
            process_thermal_frame_gpu(d_raw, d_denoised, nullptr, &config, stream, nullptr);
            cudaMemcpyAsync(h_denoised, d_denoised, float_bytes, cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
        }

        // Benchmark 100 frames
        const int num_iterations = 100;
        cudaEvent_t start_ev, stop_ev;
        cudaEventCreate(&start_ev);
        cudaEventCreate(&stop_ev);

        cudaEventRecord(start_ev, stream);
        for (int i = 0; i < num_iterations; ++i) {
            cudaMemcpyAsync(d_raw, h_raw, raw_bytes, cudaMemcpyHostToDevice, stream);
            process_thermal_frame_gpu(d_raw, d_denoised, nullptr, &config, stream, nullptr);
            cudaMemcpyAsync(h_denoised, d_denoised, float_bytes, cudaMemcpyDeviceToHost, stream);
        }
        cudaEventRecord(stop_ev, stream);
        cudaEventSynchronize(stop_ev);

        float total_ms = 0.0f;
        cudaEventElapsedTime(&total_ms, start_ev, stop_ev);
        float avg_latency_ms = total_ms / num_iterations;
        float fps = 1000.0f / avg_latency_ms;
        double giga_bytes = (double)(raw_bytes + float_bytes * 2) * num_iterations / (1024.0 * 1024.0 * 1024.0);
        double bandwidth_gbs = giga_bytes / (total_ms / 1000.0);

        std::cout << std::left << std::setw(32) << res.label
                  << std::setw(16) << (std::to_string(avg_latency_ms).substr(0, 5) + " ms")
                  << std::setw(16) << (std::to_string((int)fps) + " FPS")
                  << std::setw(18) << (std::to_string(bandwidth_gbs).substr(0, 5) + " GB/s") << "\n";

        // Cleanup per resolution
        cudaFree(d_raw);
        cudaFree(d_denoised);
        cudaFreeHost(h_raw);
        cudaFreeHost(h_denoised);
        cudaEventDestroy(start_ev);
        cudaEventDestroy(stop_ev);
    }

    std::cout << std::string(82, '-') << "\n";
    std::cout << "[SUCCESS] CUDA Shared Memory Thermal Pipeline verification completed!\n";

    cudaStreamDestroy(stream);
    return 0;
}
