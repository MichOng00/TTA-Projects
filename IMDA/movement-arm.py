from ugot import ugot
import time

# Instantiate the UGOT robot controller
got = ugot.UGOT()

def forward():
    """Move forward indefinitely."""
    got.mecanum_move_xyz(0, 20, 0)

def backward():
    """Move backward indefinitely."""
    got.mecanum_move_xyz(0, -20, 0)

def left():
    """Move left indefinitely."""
    got.mecanum_move_xyz(-20, 0, 0)

def right():   
    """Move right indefinitely."""
    got.mecanum_move_xyz(20, 0, 0)

def approach():
    """Move forward until distance to object is less than 10 cm."""
    while True:
        got.mecanum_move_xyz(0, 20, 0)
        distance = got.read_distance_data(51) # check port
        if distance < 10:
            got.mecanum_stop()
            break

def scan():
    "Pans left until apriltag is detected, then picks up."
    scanned=False

    while not scanned:
        results=got.get_qrcode_apriltag_total_info()
        if results[1] != -1:
            got.mecanum_stop()
            got.screen_display_background(6)  # green background
            time.sleep(1)
            approach()
            time.sleep(1)
            pick_up()

            scanned=True

        
        got.mecanum_translate_speed(angle=90,speed=20)

def pick_up():
    """
    Attempt to pick up an object directly beneath the gripper.
    """
    # Halt all wheel motion
    got.mecanum_stop() 
    # Lower the arm to approach object
    got.mechanical_joint_control(0, -20, -40, 500)
    time.sleep(1)
    # Close gripper on object
    got.mechanical_clamp_close()
    time.sleep(1)
    # Lift the arm back up
    got.mechanical_joint_control(0, 30, 30, 500)
    time.sleep(1)

def main():
    """Initialize systems, perform search-and-fetch routine, then return home."""
    got.initialize('192.168.88.1')           # Connect to robot over network
    # Pre-position arm and open gripper
    # got.mechanical_joint_control(0, 0, -20, 500)
    # got.mechanical_clamp_release()
    # got.screen_clear()                        # Clear any previous display

    # scan()
    got.screen_display_background(6)
    time.sleep(2)
    got.screen_clear()



if __name__ == "__main__":
    main()