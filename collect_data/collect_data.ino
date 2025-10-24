#include <Arduino.h>

#define TRIGGER_PIN A0
#define PD_1 A1
#define PD_2 A2
#define PD_3 A3
#define PD_4 A4

#define SAMPLE_LENGTH 3 // Sample length in seconds
#define FS 1000         // Sampling frequency in Hz
#define TOTAL_SAMPLES (SAMPLE_LENGTH * FS)

uint32_t delay_us = 0;
uint16_t pd_values[TOTAL_SAMPLES][4];

bool is_trigger_pressed() {
    int trigger_value = analogRead(TRIGGER_PIN);
    return trigger_value > 512; // Assuming a threshold for pressed state
}

void setup()
{
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

void loop()
{
    if (!is_trigger_pressed())
    {
        return;
    }

    digitalWrite(LED_BUILTIN, HIGH);

    for (int i = 0; i < TOTAL_SAMPLES; i++)
    {
        int value1 = analogRead(PD_1);
        int value2 = analogRead(PD_2);
        int value3 = analogRead(PD_3);
        int value4 = analogRead(PD_4);

        pd_values[i][0] = value1;
        pd_values[i][1] = value2;
        pd_values[i][2] = value3;
        pd_values[i][3] = value4;

        delayMicroseconds(delay_us);
    }

    // Send data over SerialUSB
    for (int i = 0; i < TOTAL_SAMPLES; i++)
    {
        SerialUSB.write((uint8_t *)&pd_values[i][0], sizeof(uint16_t) * 4);
    }
    SerialUSB.flush();

    digitalWrite(LED_BUILTIN, LOW);

    while (SerialUSB.available() > 0)
        SerialUSB.read(); // Clear the input buffer
}
