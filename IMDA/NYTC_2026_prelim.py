"""
NYTC 2026 preliminary robot control scripts.

This module provides simple utilities and example sequences for the UGOT
robot used in the NYTC 2026 demonstration. It exposes convenience functions
for showing the UGOT camera, following a line, and locating an AprilTag.

The top-level script runs a sample robot behaviour using pose-based
control (`pose_yolo.run_pose_control`) followed by line following and
AprilTag interaction. Functions are written for clarity and educational use
so that students can reuse and extend them during workshops.

Notes:
- Assumes an accessible UGOT robot at the configured IP address.
- Requires UGOT SDK, OpenCV, NumPy and the `pose_yolo` helper module.
"""

import cv2
import numpy as np
import time
from ugot import ugot
from pose_yolo import run_pose_control

# Initialize UGOT
got = ugot.UGOT()
got.initialize('192.168.1.164')

got.load_models(['line_recognition', 'face_recognition', 'apriltag_qrcode'])

got.set_track_recognition_line(0)

got.open_camera()

print("Initialisation successful!")

def display_camera(got=got):
    """
    Read a single frame from the UGOT camera and display it.

    This helper decodes the raw camera bytes provided by the UGOT SDK,
    converts them to an OpenCV image and displays the frame in a window
    titled "UGOT Camera". The function performs a non-blocking key poll
    (`cv2.waitKey(1)`) so it is suitable to call repeatedly inside a loop.

    Args:
        got (ugot.UGOT): Initialized UGOT instance (default: module-level `got`).
    """
    frame = got.read_camera_data()
    if frame is not None:
        nparr = np.frombuffer(frame, np.uint8)
        data = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        cv2.imshow("UGOT Camera", data)
        cv2.waitKey(1)

def line_follow(got=got, mult=0.25, speed=20):
    """
    Perform a single line-following update using UGOT sensor data.

    The function reads line-tracking information from the robot, converts
    the measured offset into a rotation speed (using `mult` as gain), and
    commands the robot's mecanum drive accordingly.

    Args:
        got (ugot.UGOT): Initialized UGOT instance (default: module-level `got`).
        mult (float): Gain applied to the line offset to compute rotation speed.
        speed (int): Forward movement speed to use while following the line.

    Returns:
        tuple: `(type, x, y)` where `type` is the detected line type and
               `x`, `y` are the reported normalized line coordinates.
    """
    offset, type, x, y = got.get_single_track_total_info()
    rotation_speed = int(offset * mult)

    got.mecanum_move_xyz(x_speed=0, y_speed=speed, z_speed=rotation_speed)

    return type, x, y

def find_apriltag(got=got, id=1):
    """
    Search for and centre the robot on a specific AprilTag.

    This blocking routine continuously reads the UGOT camera and AprilTag detection
    information. When the AprilTag with the requested `id` is observed, the
    function issues mecanum drive commands to correct bearing and horizontal
    offset until the tag appears centered in the camera frame.
    If the requested AprilTag is not seen, the robot will continue rotating to scan.
    The function returns once the AprilTag is considered centered.

    Args:
        got (ugot.UGOT): Initialized UGOT instance (default: module-level `got`).
        id (int): The AprilTag ID to search for (default: 1).
    """
    while True:
        # display image
        display_camera()

        # apriltag
        AP_info = got.get_apriltag_total_info()
        index = None
        # print(AP_info)
        if AP_info:
            # Iterate through list to see if target apriltag seen
            for i, lst in enumerate(AP_info):
                if lst[0] == id:
                    index = i
                    break
            # If target seen, centralise
            if index is not None:
                centre_x = AP_info[index][1]
                bearing_h = AP_info[index][10]
                # print(f"centre x:{centre_x}, bearing h:{bearing_h}")

                # Correct angle
                if bearing_h < -0.02:
                    got.mecanum_move_xyz(5, 0, 5)
                elif bearing_h > 0.02:
                    got.mecanum_move_xyz(-5, 0, -5)
                else:
                    # Centre horizontally
                    if centre_x < 280:
                        got.mecanum_move_xyz(-7, 0, 0)
                    elif centre_x > 360:
                        got.mecanum_move_xyz(7, 0, 0)
                    else:
                        got.mecanum_stop()
                        break
            else:
                got.mecanum_turn_speed(3, 20)
        else:
            got.mecanum_turn_speed(3, 20)
    print("Apriltag is centralized squarely.")

def pickup_ap(got=got):
    """Helper function to pick up token."""
    got.mechanical_clamp_release()
    time.sleep(1)
    got.mechanical_joint_control(0, 45, 45, 500)
    time.sleep(1)
    got.mechanical_joint_control(0, 0, -70, 800) #down - for apriltag
    time.sleep(1)
    got.mechanical_clamp_close()
    time.sleep(1)
    got.mechanical_joint_control(0, 60, -50, 800) #up
    time.sleep(1)
    got.mechanical_joint_control(-90, 60, -50, 800)
    time.sleep(1)

def put_ap(got=got):
    """Helper function to put down token."""
    got.mechanical_joint_control(-90, 30, -50, 800)
    time.sleep(1)
    got.mechanical_joint_control(-90, -20, -30, 800)
    time.sleep(1)
    got.mechanical_clamp_release()
    time.sleep(1)

if __name__ == "__main__":
    TOLERANCE = 10
    try:
        # Initial values
        got.screen_display_background(0)
        got.mechanical_joint_control(-90, 45, 45, 500)

        # Pose control to go through obstacles and pick up
        run_pose_control(
        forward_speed=30,
        backward_speed=30,
        turn_speed=45,
        camera_index=0,
        model_path="yolov8n-pose.pt",
        up_margin_factor=0.7,
        down_margin_factor=0.7,
        min_conf=0.3,
        enable_robot=True,
        debounce_frames=2,
        # max_frames=None,
        got=got
        )

        # Start following line after picking up token
        line_threshold = 0
        while line_threshold < TOLERANCE:
            # display image
            display_camera()
                
            # follow line
            line_type, x, y = line_follow(got=got, mult=0.25, speed=18)
            
            if line_type == 0:
                line_threshold += 1
            else:
                line_threshold = 0
        got.mecanum_stop()

        # Move forward slightly
        # got.mecanum_move_speed_times(0, 20, 30, 1)

        # Find and approach apriltag
        find_apriltag(got, 5)

        # Move forward to pick up apriltag
        while True:
            AP_info = got.get_apriltag_total_info()
            if AP_info:
                distance = AP_info[0][6]
                got.mecanum_move_speed(0, 15)
                if distance < 0.15:
                    got.mecanum_stop()
                    break
        pickup_ap()
        time.sleep(1)
        got.mecanum_move_speed_times(0, 20, 20, 1)

        # follow line again
        line_threshold = 0
        while line_threshold < TOLERANCE:
            # display image
            display_camera()
                
            # follow line
            line_type, x, y = line_follow(got=got, mult=0.25, speed=18)
            
            if line_type == 0:
                line_threshold += 1
            else:
                line_threshold = 0

        # turn onto expressway
        # got.mecanum_translate_speed_times(0, 20, 20, 1)
        while line_type != 1:
            got.mecanum_turn_speed(2, 20)
            line_type = got.get_single_track_total_info()[2]
        # time.sleep(0.5)

        # follow line again
        line_threshold = 0
        while line_threshold < TOLERANCE:
            # display image
            display_camera()
                
            # follow line
            line_type, x, y = line_follow(got=got, mult=0.25, speed=18)
            
            if line_type == 0:
                line_threshold += 1
            else:
                line_threshold = 0
        got.mecanum_stop()

    finally:
        got.mecanum_stop()
        time.sleep(1)
        put_ap()
        cv2.destroyAllWindows()
        print("Done")