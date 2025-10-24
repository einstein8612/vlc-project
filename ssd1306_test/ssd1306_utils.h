#ifndef SSD1306_UTILS_H
#define SSD1306_UTILS_H

int setup_ssd1306();
void write_header_ssd1306(const char* header);
void write_big_number_ssd1306(int number);

#endif // SSD1306_UTILS_H