from djitellopy import tello
import time

tel = tello.Tello()
tel.connect()

print(f"Battery: {tel.get_battery()}")
print(f"SDK version: {tel.query_sdk_version()}") # if not 30, update firmware using ipad Tello app

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
target_distance = 200
speed = 20
while True:
    try:
        # try:
        #     dist = int(tel.get_tof()[4:]) # remove "tof: "
        # except ValueError: # command timed out returns "error" so can't convert type
        #     # drone stays still
        #     dist = target_distance

        dist = tel.get_distance_tof()
        
        if dist > target_distance + 50:
            tel.rc(0, 0, 0, -speed)
        elif dist < target_distance - 50:
            tel.rc(0, 0, 0, speed)
        else:
            tel.rc(0, 0, 0, 0)
        
        # time.sleep(0.1)
    except KeyboardInterrupt:
        tel.land()
        tel.end()
        break

