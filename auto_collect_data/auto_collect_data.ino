#include <Arduino.h>
#include "WiseChipHUD.h"
#include "ssd1306_utils.h"

// =====================
// Configuration
// =====================
#define TRIGGER_PIN A7
#define PD_1 A11
#define PD_2 A10
#define PD_3 A9
#define PD_4 A8

#define SAMPLE_LENGTH 2   // seconds of full capture
#define FS 1000            // Hz sampling frequency
#define TOTAL_SAMPLES (SAMPLE_LENGTH * FS)
#define TRIGGER_SAMPLES (FS / 2)  // 0.5 seconds of trigger buffer

#define WINDOW_SIZE 25     // moving average window size
#define HALF_WINDOW (WINDOW_SIZE / 2)
#define N_CHANNELS 4

// =====================
// Buffers
// =====================
uint16_t prev_trigger_buffer[TRIGGER_SAMPLES][N_CHANNELS];
uint16_t trigger_buffer[TRIGGER_SAMPLES][N_CHANNELS];
uint16_t trigger_smoothed[TRIGGER_SAMPLES][N_CHANNELS];
uint16_t pd_values[TOTAL_SAMPLES][N_CHANNELS];

// =====================
// HUD & Timing
// =====================
WiseChipHUD hud;
uint32_t delay_us = 0;

// =====================
// Moving Mean Filter
// =====================
void moving_mean_filter(const uint16_t *data, uint16_t *out,
                        int n_samples, int n_channels) {
    for (int ch = 0; ch < n_channels; ch++) {
        for (int i = 0; i < n_samples; i++) {
            uint32_t sum = 0;
            for (int k = -HALF_WINDOW; k <= HALF_WINDOW; k++) {
                int idx = i + k;
                if (idx < 0) idx = -idx;  // reflect
                if (idx >= n_samples) idx = 2 * n_samples - idx - 2;
                sum += data[idx * n_channels + ch];
            }
            out[i * n_channels + ch] = sum / WINDOW_SIZE;
        }
    }
}

// =====================
// Trigger Logic
// =====================
bool detect_gesture(const uint16_t *filtered, int n_samples, int n_channels) {
    // Simple gesture detection: check if any channel shows a sudden increase
    for (int ch = 0; ch < n_channels; ch++) {
        uint16_t min_val = 65535, max_val = 0;
        for (int i = 0; i < n_samples; i++) {
            uint16_t val = filtered[i * n_channels + ch];
            if (val < min_val) min_val = val;
            if (val > max_val) max_val = val;
        }
        if (max_val - min_val > 75) { // adjust threshold experimentally
            return true;
        }
    }
    return false;
}

// =====================
// Setup
// =====================
void setup() {
    hud.begin();
    hud.allOn();
    setup_ssd1306();
    write_happy_monkey();

    SerialUSB.begin(115200);
    while (!SerialUSB) {
        ; // wait for the USB connection to be established
    }
    while (SerialUSB.available() > 0)
        SerialUSB.read();

    // Test analogRead time
    unsigned long startTime = micros();
    analogRead(PD_1);
    analogRead(PD_2);
    analogRead(PD_3);
    analogRead(PD_4);
    uint8_t duration = (uint8_t)(micros() - startTime);
    
    uint16_t fs = (uint16_t)FS;
    uint8_t sample_length = (uint8_t)SAMPLE_LENGTH;

    // compute delay_us
    delay_us = (1000000 / FS) - (long)duration;

    SerialUSB.write(duration);
    SerialUSB.write((uint8_t *)&fs, sizeof(fs));
    SerialUSB.write(sample_length);
    SerialUSB.write((uint8_t *)&delay_us, sizeof(delay_us));

    pinMode(LED_BUILTIN, OUTPUT);
    digitalWrite(LED_BUILTIN, LOW); // Turn the LED off
}

// =====================
// Main Loop
// =====================
void loop() {
    // 1. Continuously collect trigger buffer
    for (int i = 0; i < TRIGGER_SAMPLES; i++) {
        trigger_buffer[i][0] = analogRead(PD_1);
        trigger_buffer[i][1] = analogRead(PD_2);
        trigger_buffer[i][2] = analogRead(PD_3);
        trigger_buffer[i][3] = analogRead(PD_4);

        delayMicroseconds(delay_us);
    }

    // 2. Smooth data
    moving_mean_filter((uint16_t *)trigger_buffer, (uint16_t *)trigger_smoothed,
                       TRIGGER_SAMPLES, N_CHANNELS);

    // 3. Detect gesture
    bool gesture_detected = detect_gesture((uint16_t *)trigger_smoothed,
                                           TRIGGER_SAMPLES, N_CHANNELS);

    if (!gesture_detected) {
        memcpy(prev_trigger_buffer, trigger_buffer, sizeof(trigger_buffer));
        return; // no gesture → keep scanning
    }

    // 4. Gesture detected → record main buffer
    digitalWrite(LED_BUILTIN, HIGH);
    write_thinking_monkey();

    // Copy trigger buffer as first part
    memcpy(pd_values, prev_trigger_buffer, sizeof(trigger_buffer));
    memcpy(pd_values[TRIGGER_SAMPLES], trigger_buffer, sizeof(trigger_buffer));
    memcpy(prev_trigger_buffer, trigger_buffer, sizeof(trigger_buffer));

    for (int i = 2*TRIGGER_SAMPLES; i < TOTAL_SAMPLES; i++) {
        pd_values[i][0] = analogRead(PD_1);
        pd_values[i][1] = analogRead(PD_2);
        pd_values[i][2] = analogRead(PD_3);
        pd_values[i][3] = analogRead(PD_4);
        delayMicroseconds(delay_us);
    }

    // 5. Send through USB serial
    SerialUSB.write((uint8_t *)pd_values, sizeof(pd_values));
    SerialUSB.flush();

    digitalWrite(LED_BUILTIN, LOW);
    write_happy_monkey();
}
