#include "ssd1306_utils.h"

#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

// SSD1306 configuration
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define OLED_RESET -1
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);

int setup_ssd1306()
{
    if (!display.begin(SSD1306_SWITCHCAPVCC, 0x3C))
    {
        return -1;
    }

    display.clearDisplay();
    display.setTextColor(SSD1306_WHITE);
    return 0;
}

void write_header_ssd1306(const char *header)
{
    display.setTextSize(1);
    display.setCursor(0, 0);
    display.println(header);
    display.display();
}

void write_big_number_ssd1306(int number)
{
    display.setTextSize(5);    // big size
    display.setCursor(5, 20); // adjust for centering
    display.fillRect(0, 16, SCREEN_WIDTH, SCREEN_HEIGHT - 16, SSD1306_BLACK); // Clear previous number area

    char buf[5]; // 4 digits + null terminator
    sprintf(buf, "%04d", number);  // e.g., "0042"

    display.println(buf);
    display.display();
}
