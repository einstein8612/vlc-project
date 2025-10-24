#include <Arduino.h>

#define PD_1 A1
#define PD_2 A2
#define PD_3 A3
#define PD_4 A4

void setup()
{
    Serial.begin(115200);
    Serial.println("Hello, PD Test!");
}

void loop()
{
    int button1 = analogRead(A0);
    int value1 = analogRead(PD_1);
    int value2 = analogRead(PD_2);
    int value3 = analogRead(PD_3);
    int value4 = analogRead(PD_4);

    Serial.print("Button1: ");
    Serial.print(button1);
    Serial.print("; PD_1: ");
    Serial.print(value1);
    Serial.print("; PD_2: ");
    Serial.print(value2);
    Serial.print("; PD_3: ");
    Serial.print(value3);
    Serial.print("; PD_4: ");
    Serial.println(value4);

    delay(1000);
}