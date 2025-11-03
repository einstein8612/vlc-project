#include <Arduino.h>

#define WINDOW_SIZE 31    // Must be odd
#define HALF_WINDOW (WINDOW_SIZE / 2)

void moving_mean_filter(uint16_t* data[4], uint16_t* out[4],
                        int n_samples) {
    for (int ch = 0; ch < 4; ch++) {
        for (int i = 0; i < n_samples; i++) {
            uint32_t sum = 0;  // use 32-bit to avoid overflow

            for (int k = -HALF_WINDOW; k <= HALF_WINDOW; k++) {
                int idx = i + k;

                // Reflect edges
                if (idx < 0) idx = -idx;
                if (idx >= n_samples) idx = 2 * n_samples - idx - 2;

                sum += data[idx * 4 + ch];
            }

            out[i * 4 + ch] = sum / WINDOW_SIZE;
        }
    }
}