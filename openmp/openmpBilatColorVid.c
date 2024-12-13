#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define MAXRADIUS 16
#define COLORSPACE 4

inline float vec4_dist_sq_uint8(uint8_t v[], uint8_t w[]) {
    return (w[0] - v[0]) * (w[0] - v[0])
        + (w[1] - v[1]) * (w[1] - v[1])
        + (w[2] - v[2]) * (w[2] - v[2])
        + (w[3] - v[3]) * (w[3] - v[3]);
}

inline float fast_gaussian(float x_sq, float factor) {
    // Approximate the Gaussian function as 1 / (1 + x^2)
    return 1.0f / (1.0f + x_sq * factor);
}

void filter(
    uint8_t padded_frames[], uint8_t filtered_frames[], const int num_frames,
    const int frame_height, const int frame_width, const int colorspace,
    const int radius, const float intensity_factor, const float* spatial_weights,
    const int num_threads) {

    // Set the number of threads
    omp_set_dynamic(0);
    omp_set_num_threads(num_threads);

    int pad_img_size = (frame_height + 2 * radius) * (frame_width + 2 * radius) * colorspace * sizeof(uint8_t);
    int img_size = frame_height * frame_width * colorspace * sizeof(uint8_t);

    int frame_height_pad = frame_height + 2 * radius;
    int frame_width_pad = frame_width + 2 * radius;

    for (int idx = 0; idx < num_frames; idx++) {
        uint8_t* pad_frame = &padded_frames[idx * (frame_height_pad * frame_width_pad * colorspace)];
        uint8_t* filtered_frame = &filtered_frames[idx * (frame_height * frame_width * colorspace)];

        // Ensure local variables for thread safety
        int local_frame_height_pad = frame_height_pad;
        int local_frame_width_pad = frame_width_pad;
        int local_colorspace = colorspace;
        float local_intensity_factor = intensity_factor;
        const float* local_spatial_weights = spatial_weights;

#pragma omp parallel for
        for (int r = 0; r < frame_height; r++) {
            for (int c = 0; c < frame_width; c++) {

                const int r_pad = r + radius;
                const int c_pad = c + radius;

                float Wp = 0.0f;
                float filtered_pixel[COLORSPACE] = { 0.0f };

                uint8_t* centered_pixel = &pad_frame[(r_pad * local_frame_width_pad + c_pad) * local_colorspace];

                for (int wi = -radius; wi <= radius; wi++) {
                    for (int wj = -radius; wj <= radius; wj++) {
                        int nbor_i = r_pad + wi;
                        int nbor_j = c_pad + wj;

                        uint8_t* nbor_value = &pad_frame[(nbor_i * local_frame_width_pad + nbor_j) * local_colorspace];

                        float intensity_dist_sq = vec4_dist_sq_uint8(nbor_value, centered_pixel);

                        float weight = local_spatial_weights[(wi + radius) * (2 * radius + 1) + (wj + radius)] * fast_gaussian(intensity_dist_sq, local_intensity_factor);

                        filtered_pixel[0] += weight * nbor_value[0];
                        filtered_pixel[1] += weight * nbor_value[1];
                        filtered_pixel[2] += weight * nbor_value[2];
                        filtered_pixel[3] += weight * nbor_value[3];

                        Wp += weight;
                    }
                }
                float Wp_inv = 1 / Wp;
                filtered_frame[(r * frame_width + c) * colorspace + 0] = (uint8_t)(filtered_pixel[0] * Wp_inv);
                filtered_frame[(r * frame_width + c) * colorspace + 1] = (uint8_t)(filtered_pixel[1] * Wp_inv);
                filtered_frame[(r * frame_width + c) * colorspace + 2] = (uint8_t)(filtered_pixel[2] * Wp_inv);
                filtered_frame[(r * frame_width + c) * colorspace + 3] = (uint8_t)(filtered_pixel[3] * Wp_inv);
            }
        }
    }
}
