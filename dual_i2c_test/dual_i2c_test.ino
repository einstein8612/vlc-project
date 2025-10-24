#include "WiseChipHUD.h"
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

// WiseChipHUD instance
WiseChipHUD hud;

// SSD1306 configuration
#define SCREEN_WIDTH 128 // OLED width in pixels
#define SCREEN_HEIGHT 64 // OLED height in pixels
#define OLED_RESET    -1 // No reset pin
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);

void setup() {
  // Start serial for debugging
  Serial.begin(115200);

  // Initialize HUD
  hud.begin();
  // Turn all LEDs on HUD
  hud.allOn();

  // Initialize SSD1306
  if (!display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) { // default I2C address 0x3C
    Serial.println(F("SSD1306 allocation failed"));
    for (;;); // stop here
  }

  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(SSD1306_WHITE);
  display.setCursor(0, 0);
  display.println("HUD + SSD1306 ready!");
  display.display();
}

void loop() {
  ;
}
