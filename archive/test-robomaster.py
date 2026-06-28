"""
NB: Python version must be between 3.6.6 and 3.8.9, tested using 3.7.1
Create a fresh venv / conda env, pip install robomaster.
The documentation on the DJI website is outdated/unclear, but the examples on the github work well:
https://github.com/dji-sdk/robomaster-sdk
"""
import robomaster
# from robomaster import robot, robotic_arm, camera, led, vision
from robomaster import *
import time
import cv2

markers = []
distance = None

def detect_markers(marker_info):
    """Returns information of the first marker detected."""
    # x, y, w, h, info
    markers.clear()
    for i in range(len(marker_info)):
        markers.append(marker_info[i])

def distance_handler(sub_info):
    global distance
    distance = sub_info[0]

if __name__ == '__main__':
    ep_robot = robot.Robot()
    try:
        ep_robot.initialize(conn_type='ap')

        ##############################################
        # LED
        ep_led = ep_robot.led
        brightness = 1
        ep_led.set_led(comp=led.COMP_ALL, r=255, g=0, b=255)
        time.sleep(1)
        ##############################################
        # # arm
        # ep_arm = ep_robot.robotic_arm
        # # DO NOT use ep_arm.recenter, sets to (0,0) which is not possible
        # ep_arm.moveto(x=50, y=70).wait_for_completed()
        # ep_arm.move(x=60).wait_for_completed()
        # ep_arm.move(x=-60).wait_for_completed()
        # ep_arm.move(y=60).wait_for_completed()
        # ep_arm.move(y=-60).wait_for_completed()
        # # gripper
        # ep_gripper = ep_robot.gripper
        # ep_gripper.open()
        # time.sleep(1)
        # ep_gripper.close()
        ##############################################
        # # hot cross buns
        # for i in range(2):
        #     ep_robot.play_sound(robot.SOUND_ID_1E).wait_for_completed()
        #     ep_robot.play_sound(robot.SOUND_ID_1D).wait_for_completed()
        #     ep_robot.play_sound(robot.SOUND_ID_1C).wait_for_completed()
        #     time.sleep(1)
        # ep_robot.play_sound(robot.SOUND_ID_1C).wait_for_completed()
        # ep_robot.play_sound(robot.SOUND_ID_1C).wait_for_completed()
        # ep_robot.play_sound(robot.SOUND_ID_1D).wait_for_completed()
        # ep_robot.play_sound(robot.SOUND_ID_1D).wait_for_completed()
        # ep_robot.play_sound(robot.SOUND_ID_1E).wait_for_completed()
        # ep_robot.play_sound(robot.SOUND_ID_1D).wait_for_completed()
        # ep_robot.play_sound(robot.SOUND_ID_1C).wait_for_completed()
        ##############################################
        # # basic movement
        # ep_chassis = ep_robot.chassis
        # # move(self, x=0, y=0, z=0, xy_speed=0.5, z_speed=30)
        # """ Control chassis movement when the position is specified, and the origin of the coordinate axis is the current position.

        # :param x: float: [-5,5], x-axis movement distance, unit m
        # :param y: float: [-5,5], y-axis movement distance, unit m
        # :param z: float: [-1800,1800], z-axis rotation angle, unit °
        # :param xy_speed: float: [0.5,2], xy axis movement speed, unit m/s
        # :param z_speed: float: [10,540], z-axis rotation speed, unit °/s
        # :return: Return action object
        # """
        # ep_chassis.move(x=1).wait_for_completed()
        # ep_chassis.move(x=-1).wait_for_completed()        
        # ep_chassis.move(y=1).wait_for_completed()
        # ep_chassis.move(y=-1).wait_for_completed()
        ##############################################
        # TOF sensor
        # ep_sensor = ep_robot.sensor
        # while True:
        #     distance_result = ep_sensor.sub_distance(callback=distance_handler)
        #     if distance <= 30:
        #         print("close")
        #         break
        ##############################################
        # camera, rc, vision
        ep_camera = ep_robot.camera
        # display = True does not work
        ep_camera.start_video_stream(display=False)
        
        ep_vision = ep_robot.vision
        while True:
            img = ep_camera.read_cv2_image(strategy="newest")
            vision_result = ep_vision.sub_detect_info(name="marker", callback=detect_markers)

            if vision_result:
                # cv2.putText(img, f"{markers}", (20, 50), 1, 3, (255,255,255), 1, cv2.LINE_AA)
                for marker in markers:
                    top_x_normal = int((marker[0] - marker[2]/2) * 1280)
                    top_y_normal = int((marker[1] - marker[3]/2) * 720)
                    bottom_x_normal = int((marker[0] + marker[2]/2) * 1280)
                    bottom_y_normal = int((marker[1] + marker[3]/2) * 720)
                    cv2.rectangle(img, (top_x_normal, top_y_normal), (bottom_x_normal, bottom_y_normal), (0, 255, 0), 2)
                    cv2.putText(img, f"Marker {marker[4]}", (top_x_normal, top_y_normal-5), 0, 1, (0,255,0), 1, cv2.LINE_AA)
            
            cv2.imshow("Robomaster", img)

            key = cv2.waitKey(1) & 0xFF

            # x_speed = 0 # m/s
            # y_speed = 0 # m/s
            # z_speed = 0 # degrees/s

            # if key == ord('w'):
            #     x_speed = 0.2
            # elif key == ord('s'):
            #     x_speed = -0.2
            # elif key == ord('d'):
            #     y_speed = 0.2
            # elif key == ord('a'):
            #     y_speed = -0.2
            # elif key == ord('e'):
            #     z_speed = 30
            # elif key == ord('q'):
            #     z_speed = -30

            # ep_chassis.drive_speed(x_speed, y_speed, z_speed)

            if key==27: #ESC
                break

        result = ep_vision.unsub_detect_info(name="marker")
        ep_camera.stop_video_stream()
        ##############################################

    finally:
        cv2.destroyAllWindows()
        ep_robot.close()

