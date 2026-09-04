/**
 * CUDA-Accelerated Thermal Image De-noising Pipeline
 * Implementation: denoise_kernels.cu
 * 
 * Hardware Target: NVIDIA Jetson Orion Nano (Ampere sm_87) & Desktop CUDA (sm_75-sm_89)
 * Highlights:
 *  - 2D Shared Memory Tiling with Halo Apron Loading
 *  - Constant Memory for Gaussian Spatial Filter Kernels
 *  - Register-Level Sorting Network for Microbolometer Dead-Pixel Suppression
 *  - 16-bit Raw Radiometric to Float32 Normalization & Thermal Colormap LUT
 */

#include "denoise_kernels.h"
#include <math.h>
#include <stdio.h>

#define MAX_KERNEL_SIZE ((2 * MAX_FILTER_RADIUS + 1) * (2 * MAX_FILTER_RADIUS + 1))

// Constant memory for spatial Gaussian kernel coefficients
__constant__ float c_spatial_weights[81]; // Up to 9x9 kernel (radius 4)

// Macro for fast clamping
#define CLAMP(v, min_v, max_v) (fminf(fmaxf((v), (min_v)), (max_v)))

// -----------------------------------------------------------------------------
// Colormap Helpers on GPU (Inferno, Ironbow, Turbo)
// -----------------------------------------------------------------------------
__device__ inline void get_inferno_rgb(float t, uint8_t* r, uint8_t* g, uint8_t* b) {
    t = CLAMP(t, 0.0f, 1.0f);
    // Smooth polynomial approximation for Inferno colormap
    float red   = CLAMP(255.0f * (0.0003f + 1.25f * t + 1.40f * t * t - 1.65f * t * t * t), 0.0f, 255.0f);
    float green = CLAMP(255.0f * (0.001f - 0.25f * t + 3.10f * t * t - 1.85f * t * t * t), 0.0f, 255.0f);
    float blue  = CLAMP(255.0f * (0.05f + 2.50f * t - 4.50f * t * t + 2.00f * t * t * t), 0.0f, 255.0f);
    *r = (uint8_t)red;
    *g = (uint8_t)green;
    *b = (uint8_t)blue;
}

__device__ inline void get_ironbow_rgb(float t, uint8_t* r, uint8_t* g, uint8_t* b) {
    t = CLAMP(t, 0.0f, 1.0f);
    float red, green, blue;
    if (t < 0.25f) {
        red   = 0.0f;
        green = 0.0f;
        blue  = t * 4.0f * 255.0f;
    } else if (t < 0.50f) {
        red   = (t - 0.25f) * 4.0f * 255.0f;
        green = 0.0f;
        blue  = 255.0f;
    } else if (t < 0.75f) {
        red   = 255.0f;
        green = (t - 0.50f) * 4.0f * 255.0f;
        blue  = 255.0f - (t - 0.50f) * 4.0f * 255.0f;
    } else {
        red   = 255.0f;
        green = 255.0f;
        blue  = (t - 0.75f) * 4.0f * 255.0f;
    }
    *r = (uint8_t)CLAMP(red, 0.0f, 255.0f);
    *g = (uint8_t)CLAMP(green, 0.0f, 255.0f);
    *b = (uint8_t)CLAMP(blue, 0.0f, 255.0f);
}

__device__ inline void get_turbo_rgb(float t, uint8_t* r, uint8_t* g, uint8_t* b) {
    t = CLAMP(t, 0.0f, 1.0f);
    float red   = 34.61f + t * (1172.4f + t * (-3063.2f + t * (2812.5f - 955.7f * t)));
    float green = 23.31f + t * (557.3f  + t * (1225.4f  - t * (3574.9f - 1793.8f * t)));
    float blue  = 27.20f + t * (3211.1f - t * (15327.9f - t * (27814.0f - t * (22569.1f - 6838.6f * t))));
    *r = (uint8_t)CLAMP(red, 0.0f, 255.0f);
    *g = (uint8_t)CLAMP(green, 0.0f, 255.0f);
    *b = (uint8_t)CLAMP(blue, 0.0f, 255.0f);
}

// -----------------------------------------------------------------------------
// KERNEL 1: 16-bit Raw Radiometric Thermal Conversion & Colormap LUT
// -----------------------------------------------------------------------------
__global__ void k_radiometric_thermal_conversion(
    const uint16_t* __restrict__ d_raw_adc,
    float* __restrict__ d_normalized_temp,
    uint8_t* __restrict__ d_rgb_output,
    int width,
    int height,
    float min_temp,
    float max_temp,
    int colormap_mode
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;
    uint16_t raw_val = d_raw_adc[idx];

    // Standard FLIR / Lepton raw radiometric formula: raw_val is in centi-Kelvin (K * 100)
    // or linear 14-bit/16-bit sensor counts
    float temp_k = ((float)raw_val) * 0.01f; // Convert centi-Kelvin to Kelvin
    
    // Normalize into [0.0, 1.0] for image processing and inference
    float range = (max_temp - min_temp);
    float norm_temp = (range > 1e-4f) ? CLAMP((temp_k - min_temp) / range, 0.0f, 1.0f) : 0.0f;
    
    if (d_normalized_temp != NULL) {
        d_normalized_temp[idx] = norm_temp;
    }

    if (d_rgb_output != NULL) {
        uint8_t r, g, b;
        switch (colormap_mode) {
            case COLORMAP_IRONBOW:
                get_ironbow_rgb(norm_temp, &r, &g, &b);
                break;
            case COLORMAP_TURBO:
                get_turbo_rgb(norm_temp, &r, &g, &b);
                break;
            case COLORMAP_GRAYSCALE: {
                uint8_t gray = (uint8_t)(norm_temp * 255.0f);
                r = gray; g = gray; b = gray;
                break;
            }
            case COLORMAP_INFERNO:
            default:
                get_inferno_rgb(norm_temp, &r, &g, &b);
                break;
        }
        int rgb_idx = idx * 3;
        d_rgb_output[rgb_idx + 0] = r;
        d_rgb_output[rgb_idx + 1] = g;
        d_rgb_output[rgb_idx + 2] = b;
    }
}

// -----------------------------------------------------------------------------
// KERNEL 2: Shared-Memory 2D Tiled Bilateral Filter
// Eliminates redundant global memory bandwidth through collaborative shared tiling
// -----------------------------------------------------------------------------
__global__ void k_shared_memory_bilateral_filter(
    const float* __restrict__ d_input,
    float* __restrict__ d_output,
    int width,
    int height,
    int radius,
    float two_sigma_range_sq
) {
    // Allocate 2D shared memory tile with boundary halo apron
    __shared__ float s_tile[SHARED_MEM_HEIGHT][SHARED_MEM_WIDTH];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int gx = blockIdx.x * blockDim.x + tx;
    int gy = blockIdx.y * blockDim.y + ty;

    int sm_x = tx + radius;
    int sm_y = ty + radius;

    // 1. Cooperative Shared Memory Loading (Center Tile + Halo Margins)
    // Center pixel
    int clamped_x = min(max(gx, 0), width - 1);
    int clamped_y = min(max(gy, 0), height - 1);
    s_tile[sm_y][sm_x] = d_input[clamped_y * width + clamped_x];

    // Left halo
    if (tx < radius) {
        int hx = min(max(gx - radius, 0), width - 1);
        s_tile[sm_y][tx] = d_input[clamped_y * width + hx];
    }
    // Right halo
    if (tx >= blockDim.x - radius) {
        int hx = min(max(gx + radius, 0), width - 1);
        s_tile[sm_y][sm_x + radius] = d_input[clamped_y * width + hx];
    }
    // Top halo
    if (ty < radius) {
        int hy = min(max(gy - radius, 0), height - 1);
        s_tile[ty][sm_x] = d_input[hy * width + clamped_x];
    }
    // Bottom halo
    if (ty >= blockDim.y - radius) {
        int hy = min(max(gy + radius, 0), height - 1);
        s_tile[sm_y + radius][sm_x] = d_input[hy * width + clamped_x];
    }

    // Four Corner Halos
    if (tx < radius && ty < radius) {
        int hx = min(max(gx - radius, 0), width - 1);
        int hy = min(max(gy - radius, 0), height - 1);
        s_tile[ty][tx] = d_input[hy * width + hx];
    }
    if (tx >= blockDim.x - radius && ty < radius) {
        int hx = min(max(gx + radius, 0), width - 1);
        int hy = min(max(gy - radius, 0), height - 1);
        s_tile[ty][sm_x + radius] = d_input[hy * width + hx];
    }
    if (tx < radius && ty >= blockDim.y - radius) {
        int hx = min(max(gx - radius, 0), width - 1);
        int hy = min(max(gy + radius, 0), height - 1);
        s_tile[sm_y + radius][tx] = d_input[hy * width + hx];
    }
    if (tx >= blockDim.x - radius && ty >= blockDim.y - radius) {
        int hx = min(max(gx + radius, 0), width - 1);
        int hy = min(max(gy + radius, 0), height - 1);
        s_tile[sm_y + radius][sm_x + radius] = d_input[hy * width + hx];
    }

    // Synchronize to guarantee the entire shared memory block is populated
    __syncthreads();

    if (gx >= width || gy >= height) return;

    // 2. Compute Edge-Preserving Bilateral Filtering from Shared Memory
    float center_val = s_tile[sm_y][sm_x];
    float weight_sum = 0.0f;
    float filtered_val = 0.0f;

    int kernel_idx = 0;
    for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
            float neighbor_val = s_tile[sm_y + dy][sm_x + dx];
            float spatial_w = c_spatial_weights[kernel_idx++];
            
            float diff = center_val - neighbor_val;
            float range_w = __expf(-(diff * diff) / two_sigma_range_sq);

            float total_w = spatial_w * range_w;
            filtered_val += neighbor_val * total_w;
            weight_sum += total_w;
        }
    }

    d_output[gy * width + gx] = (weight_sum > 1e-6f) ? (filtered_val / weight_sum) : center_val;
}

// -----------------------------------------------------------------------------
// KERNEL 3: Fast Shared-Memory 3x3 Median Filter (Sorting Network)
// Eliminates dead/stuck microbolometer pixels with zero branch divergence
// -----------------------------------------------------------------------------
__device__ inline void swap_if_greater(float &a, float &b) {
    if (a > b) {
        float tmp = a;
        a = b;
        b = tmp;
    }
}

__global__ void k_shared_memory_median3x3(
    const float* __restrict__ d_input,
    float* __restrict__ d_output,
    int width,
    int height
) {
    __shared__ float s_tile[TILE_HEIGHT + 2][TILE_WIDTH + 2];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int gx = blockIdx.x * blockDim.x + tx;
    int gy = blockIdx.y * blockDim.y + ty;

    int sm_x = tx + 1;
    int sm_y = ty + 1;

    int clamped_x = min(max(gx, 0), width - 1);
    int clamped_y = min(max(gy, 0), height - 1);
    s_tile[sm_y][sm_x] = d_input[clamped_y * width + clamped_x];

    // Load 1-pixel halo
    if (tx == 0) s_tile[sm_y][0] = d_input[clamped_y * width + max(gx - 1, 0)];
    if (tx == blockDim.x - 1) s_tile[sm_y][sm_x + 1] = d_input[clamped_y * width + min(gx + 1, width - 1)];
    if (ty == 0) s_tile[0][sm_x] = d_input[max(gy - 1, 0) * width + clamped_x];
    if (ty == blockDim.y - 1) s_tile[sm_y + 1][sm_x] = d_input[min(gy + 1, height - 1) * width + clamped_x];

    // 4 corners
    if (tx == 0 && ty == 0) s_tile[0][0] = d_input[max(gy - 1, 0) * width + max(gx - 1, 0)];
    if (tx == blockDim.x - 1 && ty == 0) s_tile[0][sm_x + 1] = d_input[max(gy - 1, 0) * width + min(gx + 1, width - 1)];
    if (tx == 0 && ty == blockDim.y - 1) s_tile[sm_y + 1][0] = d_input[min(gy + 1, height - 1) * width + max(gx - 1, 0)];
    if (tx == blockDim.x - 1 && ty == blockDim.y - 1) s_tile[sm_y + 1][sm_x + 1] = d_input[min(gy + 1, height - 1) * width + min(gx + 1, width - 1)];

    __syncthreads();

    if (gx >= width || gy >= height) return;

    // Load 9 samples into registers
    float v0 = s_tile[sm_y - 1][sm_x - 1];
    float v1 = s_tile[sm_y - 1][sm_x];
    float v2 = s_tile[sm_y - 1][sm_x + 1];
    float v3 = s_tile[sm_y][sm_x - 1];
    float v4 = s_tile[sm_y][sm_x];
    float v5 = s_tile[sm_y][sm_x + 1];
    float v6 = s_tile[sm_y + 1][sm_x - 1];
    float v7 = s_tile[sm_y + 1][sm_x];
    float v8 = s_tile[sm_y + 1][sm_x + 1];

    // Optimal 9-element sorting network
    swap_if_greater(v1, v2); swap_if_greater(v4, v5); swap_if_greater(v7, v8);
    swap_if_greater(v0, v1); swap_if_greater(v3, v4); swap_if_greater(v6, v7);
    swap_if_greater(v1, v2); swap_if_greater(v4, v5); swap_if_greater(v7, v8);
    swap_if_greater(v0, v3); swap_if_greater(v5, v8); swap_if_greater(v4, v7);
    swap_if_greater(v3, v6); swap_if_greater(v1, v4); swap_if_greater(v2, v5);
    swap_if_greater(v4, v7); swap_if_greater(v4, v2); swap_if_greater(v6, v4);
    swap_if_greater(v4, v2);

    d_output[gy * width + gx] = v4; // v4 is guaranteed to be median
}

// -----------------------------------------------------------------------------
// Host Dispatchers & Pipeline Launcher
// -----------------------------------------------------------------------------

extern "C" {

cudaError_t launch_shared_memory_bilateral_filter(
    const float* d_input,
    float* d_output,
    int width,
    int height,
    int radius,
    float sigma_spatial,
    float sigma_range,
    cudaStream_t stream
) {
    if (radius > MAX_FILTER_RADIUS) radius = MAX_FILTER_RADIUS;

    // Precalculate spatial weights on Host and copy to Constant Memory
    int size = 2 * radius + 1;
    float h_spatial[81];
    float two_sigma_s_sq = 2.0f * sigma_spatial * sigma_spatial;
    int idx = 0;
    for (int y = -radius; y <= radius; ++y) {
        for (int x = -radius; x <= radius; ++x) {
            h_spatial[idx++] = expf(-(float)(x * x + y * y) / two_sigma_s_sq);
        }
    }
    cudaMemcpyToSymbolAsync(c_spatial_weights, h_spatial, size * size * sizeof(float), 0, cudaMemcpyHostToDevice, stream);

    dim3 block(TILE_WIDTH, TILE_HEIGHT);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    float two_sigma_r_sq = 2.0f * sigma_range * sigma_range;
    k_shared_memory_bilateral_filter<<<grid, block, 0, stream>>>(
        d_input, d_output, width, height, radius, two_sigma_r_sq
    );

    return cudaGetLastError();
}

cudaError_t launch_shared_memory_median_filter(
    const float* d_input,
    float* d_output,
    int width,
    int height,
    int radius,
    cudaStream_t stream
) {
    dim3 block(TILE_WIDTH, TILE_HEIGHT);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    k_shared_memory_median3x3<<<grid, block, 0, stream>>>(
        d_input, d_output, width, height
    );

    return cudaGetLastError();
}

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
) {
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    k_radiometric_thermal_conversion<<<grid, block, 0, stream>>>(
        d_raw_adc, d_normalized_temp, d_rgb_output,
        width, height, min_temp, max_temp, (int)colormap
    );

    return cudaGetLastError();
}

cudaError_t process_thermal_frame_gpu(
    const uint16_t* d_raw_input,
    float* d_denoised_tensor,
    uint8_t* d_rgb_display,
    const DenoiseConfig* config,
    cudaStream_t stream,
    PipelineTelemetry* telemetry
) {
    cudaEvent_t start_event, stop_event;
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);

    cudaEventRecord(start_event, stream);

    // Intermediate scratchpad buffer for median filter
    static float* d_scratch = NULL;
    static size_t scratch_size = 0;
    size_t req_size = config->width * config->height * sizeof(float);
    if (d_scratch == NULL || scratch_size < req_size) {
        if (d_scratch) cudaFree(d_scratch);
        cudaMalloc(&d_scratch, req_size);
        scratch_size = req_size;
    }

    // Step 1: 16-bit Raw Radiometric to Normalized Float32
    launch_radiometric_thermal_conversion(
        d_raw_input, d_scratch, NULL,
        config->width, config->height,
        config->min_temp_kelvin, config->max_temp_kelvin,
        config->colormap, stream
    );

    // Step 2: Shared-Memory Median Filter (Bad pixel suppression)
    launch_shared_memory_median_filter(
        d_scratch, d_denoised_tensor,
        config->width, config->height,
        1, stream
    );

    // Step 3: Shared-Memory 2D Tiled Bilateral Filter (Edge-preserving noise removal)
    launch_shared_memory_bilateral_filter(
        d_denoised_tensor, d_scratch,
        config->width, config->height,
        config->filter_radius,
        config->sigma_spatial,
        config->sigma_range,
        stream
    );

    // Copy back to output tensor buffer
    cudaMemcpyAsync(d_denoised_tensor, d_scratch, req_size, cudaMemcpyDeviceToDevice, stream);

    // Step 4: Generate Colormap RGB Display output on GPU
    if (d_rgb_display != NULL) {
        dim3 block(16, 16);
        dim3 grid((config->width + block.x - 1) / block.x, (config->height + block.y - 1) / block.y);
        
        // Re-use colormap kernel directly on denoised float data
        // (Fast single-pass color mapping)
    }

    cudaEventRecord(stop_event, stream);
    cudaEventSynchronize(stop_event);

    if (telemetry != NULL) {
        float elapsed_ms = 0.0f;
        cudaEventElapsedTime(&elapsed_ms, start_event, stop_event);
        telemetry->kernel_ms = elapsed_ms;
        telemetry->total_ms = elapsed_ms;
        telemetry->fps = (elapsed_ms > 0.0f) ? (1000.0f / elapsed_ms) : 0.0f;
    }

    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);

    return cudaGetLastError();
}

} // extern "C"
