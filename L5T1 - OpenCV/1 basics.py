"""Minimal example to connect, take off and land"""
from djitellopy import tello
import time

tel = tello.Tello()
tel.connect()
print(f"battery: {tel.get_battery()}")

tel.takeoff()
time.sleep(2)
tel.land()