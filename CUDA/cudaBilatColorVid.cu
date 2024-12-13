#include <cuda_runtime.h>
#include <stdint.h>

#define MAXRADIUS 16
#define COLORSPACE 4

__constant__ float const_spatial_weights[(2 * MAXRADIUS + 1) * (2 * MAXRADIUS + 1)];

__device__ inline float lorentzian(float x_sq, float factor) {
    // Approximate the Gaussian function with the Lorentizian function
    return 1.0f / (1.0f + x_sq * factor);
}

__global__ void bilateralFilterKernel(
    uint32_t* input_pad, uint32_t* output,
    const int rows, const int cols, const int colorspace,
    const int radius, const float intensity_factor) {

    // Padded Thread block
    __shared__ uint32_t s_input_pad[32][32];

    // Calculate global and local thread indices
    const int global_c_pad = threadIdx.x + blockIdx.x * (blockDim.x - 2 * radius);
    const int global_r_pad = threadIdx.y + blockIdx.y * (blockDim.y - 2 * radius);

    const int local_c = threadIdx.x;
    const int local_r = threadIdx.y;

    const int cols_pad = cols + 2 * radius;
    const int rows_pad = rows + 2 * radius;

    uint32_t centered_pixel;
    // Each thread loads one element from global memeory to shared memory
    if (global_r_pad < rows_pad && global_c_pad < cols_pad) {
        centered_pixel = input_pad[global_r_pad * cols_pad + global_c_pad];
        s_input_pad[local_r][local_c] = centered_pixel;
    }
    __syncthreads();

    // Extract color channels
    uint8_t red = (centered_pixel >> 24) & 0xFF;
    uint8_t green = (centered_pixel >> 16) & 0xFF;
    uint8_t blue = (centered_pixel >> 8) & 0xFF;
    uint8_t alpha = centered_pixel & 0xFF;

    // Only inner threads process pixels
    if ((local_r >= radius) && (local_r < blockDim.y - radius)
        && (local_c >= radius) && (local_c < blockDim.x - radius)
        && (global_r_pad >= radius) && (global_r_pad < rows + radius)
        && (global_c_pad >= radius) && (global_c_pad < cols + radius)) {

        float Wp = 0.0f;
        float filtered_pixels[COLORSPACE] = { 0.0f };

        // Iterate over neighborhood
        for (int wi = -radius; wi <= radius; wi++) {
            for (int wj = -radius; wj <= radius; wj++) {
                int nbor_i = local_r + wi;
                int nbor_j = local_c + wj;

                uint32_t nbor_values = s_input_pad[nbor_i][nbor_j];

                // Extra nbor color channels
                uint8_t nbor_red = (nbor_values >> 24) & 0xFF;
                uint8_t nbor_green = (nbor_values >> 16) & 0xFF;
                uint8_t nbor_blue = (nbor_values >> 8) & 0xFF;
                uint8_t nbor_alpha = nbor_values & 0xFF;

                float intensity_dist_sq = (float)(nbor_red - red) * (nbor_red - red)
                    + (float)(nbor_green - green) * (nbor_green - green)
                    + (float)(nbor_blue - blue) * (nbor_blue - blue)
                    + (float)(nbor_alpha - alpha) * (nbor_alpha - alpha);

                float weight = const_spatial_weights[(wi + radius) * (2 * radius + 1) + (wj + radius)]
                    * lorentzian(intensity_dist_sq, intensity_factor);

                filtered_pixels[0] += weight * nbor_red;
                filtered_pixels[1] += weight * nbor_green;
                filtered_pixels[2] += weight * nbor_blue;
                filtered_pixels[3] += weight * nbor_alpha;

                Wp += weight;
            }
        }

        float Wp_inv = 1.0f / Wp;
        uint32_t packed_filtered_pixel = ((uint8_t)(filtered_pixels[0] / Wp_inv) << 24)
            | ((uint8_t)(filtered_pixels[1] * Wp_inv) << 16)
            | ((uint8_t)(filtered_pixels[2] * Wp_inv) << 8)
            | (uint8_t)(filtered_pixels[3] * Wp_inv);

        output[(global_r_pad - radius) * cols + (global_c_pad - radius)] = packed_filtered_pixel;
    }
}

extern "C" {
void filter(
    uint8_t p_padded_frames[], uint8_t p_filtered_frames[], const int num_frames,
    const int frame_height, const int frame_width, const int colorspace,
    const int radius, const float intensity_factor, const float* spatial_weights) {

    cudaError(cudaSetDevice(1));

    int pad_img_size = (frame_height + 2 * radius) * (frame_width + 2 * radius) * colorspace * sizeof(uint8_t);
    int img_size = frame_height * frame_width * colorspace * sizeof(uint8_t);

    int frame_height_pad = frame_height + 2 * radius;
    int frame_width_pad = frame_width + 2 * radius;

    // Copy spatial kernel to constant cache
    cudaError(cudaMemcpyToSymbol(const_spatial_weights, spatial_weights, (2 * radius + 1) * (2 * radius + 1) * sizeof(float), 0, cudaMemcpyDefault));

    const dim3 block(32, 32);
    const dim3 grid(
        ((frame_width + (block.x - 2 * radius) - 1) / (block.x - 2 * radius)),
        ((frame_height + (block.y - 2 * radius) - 1) / (block.y - 2 * radius)));

    // Number of Streams
    int num_streams = 2;

    // Streams
    cudaStream_t streams[num_streams];

    // DEVICE buffers
    uint32_t* d_frames[num_streams];
    uint32_t* d_filtered_frames[num_streams];

    for (int stream_idx = 0; stream_idx < num_streams; stream_idx++) {
        cudaStreamCreate(&streams[stream_idx]);
        cudaMalloc(&d_frames[stream_idx], pad_img_size);
        cudaMalloc(&d_filtered_frames[stream_idx], img_size);
    }

    for (int idx = 0; idx < num_frames; idx++) {
        uint8_t* p_frame = &p_padded_frames[idx * (frame_height_pad * frame_width_pad * colorspace)];
        uint8_t* p_filtered_frame = &p_filtered_frames[idx * (frame_height * frame_width * colorspace)];

        int stream_idx = idx % num_streams;

        // Async copy to device
        cudaMemcpyAsync(d_frames[stream_idx], p_frame, pad_img_size, cudaMemcpyHostToDevice, streams[stream_idx]);

        // Run the bilateral filter kernel
        bilateralFilterKernel<<<grid, block, 0, streams[stream_idx]>>>(
            d_frames[stream_idx], d_filtered_frames[stream_idx],
            frame_height, frame_width, colorspace,
            radius, intensity_factor);

        // Async copy filtered frame back to pinned memory
        cudaMemcpyAsync(p_filtered_frame, d_filtered_frames[stream_idx], img_size, cudaMemcpyDeviceToHost, streams[stream_idx]);

        // Synchronize previous streams
        if (idx >= num_streams - 1) {
            cudaStreamSynchronize(streams[(stream_idx - 1 + num_streams) % num_streams]); // Ensure proper indexing
        }
    }

    // Free DEVICE Memory
    for (int stream_idx = 0; stream_idx < num_streams; stream_idx++) {
        cudaFree(d_frames[stream_idx]);
        cudaFree(d_filtered_frames[stream_idx]);
        cudaStreamDestroy(streams[stream_idx]);
    }
}
}
