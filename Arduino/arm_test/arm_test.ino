#include <ATX2.h>
void setup() {
  // put your setup code here, to run once:
  OK(); // wait for OK button
  // glcdClear(); // clear LCD screen

}

void loop() {
  // put your main code here, to run repeatedly:
  servo(1,45);
  delay(1000);
  servo(2,45);
  delay(1000);
  servo(3,45);
  delay(1000);
  servo(4,45);
  delay(1000);
  servo(5,45);
  delay(1000);
  servo(6,20);
  delay(1000);
  // 
  // servo(1,90);
  // delay(1000);
  // servo(2,90);
  // delay(1000);
  // servo(3,90);
  // delay(1000);
  // servo(4,90);
  // delay(1000);
  // servo(5,90);
  // delay(1000);
  // servo(6,90);
}
