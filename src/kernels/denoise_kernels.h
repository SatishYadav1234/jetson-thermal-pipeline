/**
 * CUDA-Accelerated Thermal Image De-noising Pipeline
 * Header: denoise_kernels.h
 * 
 * Hardware Target: NVIDIA Jetson Orion Nano (sm_87) & Desktop CUDA (sm_75 - sm_89)
 */

#ifndef DENOISE_KERNELS_H
#define DENOISE_KERNELS_H

#include <cuda_runtime.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Tile dimensions for Shared Memory CUDA blocks
#define TILE_WIDTH 16
#define TILE_HEIGHT 16
#define MAX_FILTER_RADIUS 4
#define SHARED_MEM_WIDTH (TILE_WIDTH + 2 * MAX_FILTER_RADIUS)
#define SHARED_MEM_HEIGHT (TILE_HEIGHT + 2 * MAX_FILTER_RADIUS)

// Thermal Colormap Modes
typedef enum {
    COLORMAP_INFERNO = 0,
    COLORMAP_IRONBOW = 1,
    COLORMAP_TURBO   = 2,
    COLORMAP_GRAYSCALE = 3
} ThermalColormapMode;

// Denoising Configuration Parameters
typedef struct {
    int width;
    int height;
    float sigma_spatial;     // Spatial domain variance (distance weighting)
    float sigma_range;       // Radiometric range domain variance (temperature intensity weighting)
    int filter_radius;       // Radius of the filter window (e.g. 2 for 5x5, 3 for 7x7)
    float min_temp_kelvin;   // Dynamic range min for radiometric conversion (e.g. 263.15K / -10°C)
    float max_temp_kelvin;   // Dynamic range max for radiometric conversion (e.g. 393.15K / +120°C)
    ThermalColormapMode colormap;
} DenoiseConfig;

// Performance Telemetry Timers
typedef struct {
    float h2d_ms;       // Host to Device async copy time
    float kernel_ms;    // Custom CUDA shared-memory kernel execution time
    float d2h_ms;       // Device to Host copy time (optional for zero-copy)
    float total_ms;     // Total pipeline frame processing time
    float fps;          // Effective throughput
} PipelineTelemetry;

/**
 * Executes the 2D Shared-Memory Bilateral Filter on raw 16-bit / 32-bit thermal data.
 * Utilizes custom shared memory cache with apron padding to eliminate redundant global memory reads.
 */
cudaError_t launch_shared_memory_bilateral_filter(
    const float* d_input,
    float* d_output,
    int width,
    int height,
    int radius,
    float sigma_spatial,
    float sigma_range,
    cudaStream_t stream
);

/**
 * Executes a fast Shared-Memory Median Filter (3x3 or 5x5) using register sorting networks
 * to eliminate dead/stuck microbolometer sensor pixels in high-res thermal imagery.
 */
cudaError_t launch_shared_memory_median_filter(
    const float* d_input,
    float* d_output,
    int width,
    int height,
    int radius,
    cudaStream_t stream
);

/**
 * Converts 16-bit Raw ADC Thermal Microbolometer Data (FLIR / Lepton / Seek format)
 * to Normalized Float32 Temperature and applies on-GPU Colormap LUT.
 */
cudaError_t launch_radiometric_thermal_conversion(
    const uint16_t* d_raw_adc,
    float* d_normalized_temp,
    uint8_t* d_rgb_output,
    int width,
    int height,
    float min_temp,
    float max_temp,
    ThermalColormapMode colormap,
    cudaStream_t stream
);

/**
 * End-to-end C/C++ pipeline invocation combining raw 16-bit conversion,
 * dead pixel median filtering, shared-memory bilateral filtering, and zero-copy tensor preparation.
 */
cudaError_t process_thermal_frame_gpu(
    const uint16_t* d_raw_input,
    float* d_denoised_tensor,
    uint8_t* d_rgb_display,
    const DenoiseConfig* config,
    cudaStream_t stream,
    PipelineTelemetry* telemetry
);

#ifdef __cplusplus
}
#endif

#endif // DENOISE_KERNELS_H
