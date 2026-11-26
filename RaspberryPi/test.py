from time import sleep

from gpiozero import Servo, DigitalInputDevice, Button

servo1 = Servo(17) # check the yellow wire "GP" number
servo2 = Servo(4)

servo1.min()
servo2.min()
sleep(1)
servo1.mid()
servo2.mid()
sleep(1)
servo1.max()
servo2.max()
sleep(1)

sensor = DigitalInputDevice(24)
button = Button(18)

while True:
    if sensor.is_active:
        print("No obstacle detected")
    else:
        print("Obstacle detected")
    if button.is_pressed:
      print("Button is pressed") 
    else:
      print("Button is not pressed")
    sleep(0.5) 
