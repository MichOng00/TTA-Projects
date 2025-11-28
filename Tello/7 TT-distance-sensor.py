from tello_sdk import Tello
import time

tel = Tello.Tello(tips=False)

print(f"Battery: {tel.get_battery()}")
print(f"SDK version: {tel.get_sdk()}") # if not 30, update firmware using ipad Tello app

#### get and print distance continuously
# while True:
#     try:
#         dist = tel.get_tof()[4:] # remove "tof: "
#         print(dist, end="  \r")
#         time.sleep(1)
#     except KeyboardInterrupt:
#         tel.end()
#         break

#### keep a constant distance from an obstacle
tel.takeoff()
target_distance = 50
speed = 20
while True:
    try:
        try:
            dist = int(tel.get_tof()[4:]) # remove "tof: "
        except ValueError: # command timed out returns "error" so can't convert type
            # drone stays still
            dist = target_distance
        print(dist, end="  \r")
        if dist > target_distance + 20:
            tel.rc(0, speed, 0, 0)
        elif dist < target_distance - 20:
            tel.rc(0, -speed, 0, 0)
        else:
            tel.rc(0, 0, 0, 0)
        
        # time.sleep(0.1)
    except KeyboardInterrupt:
        tel.land()
        tel.end()
        break

