#include <Arduino.h>

#include "ssd1306_utils.h"

void setup()
{
    setup_ssd1306();
    write_header_ssd1306("SSD1306 Test");
}

int i = 0;

void loop()
{
    write_big_number_ssd1306(i);
    i = (i + 1) % 10000;
    delay(10);
}
