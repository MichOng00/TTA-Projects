# Pose-Based Drone Controller using Body Detection
# Controls drone movement based on upper and lower body detection from webcam
import cv2
import numpy as np
from djitellopy import tello
import time

def detect_body_parts(image):
    """
    Detect upper body, lower body, and full body in the image
    Returns positions and areas of detected body parts
    """
    upperbody_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_upperbody.xml")
    lowerbody_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_lowerbody.xml")
    fullbody_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_fullbody.xml")
    
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    upperbody = upperbody_cascade.detectMultiScale(image_gray, 1.1, 4)
    lowerbody = lowerbody_cascade.detectMultiScale(image_gray, 1.1, 4)
    fullbody = fullbody_cascade.detectMultiScale(image_gray, 1.1, 4)
    
    body_data = {
        'upperbody': [],
        'lowerbody': [],
        'fullbody': []
    }
    
    # Process upper body detections
    for (x, y, w, h) in upperbody:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        center_x = x + w // 2
        center_y = y + h // 2
        area = w * h
        cv2.circle(image, (center_x, center_y), 5, (0, 255, 0), cv2.FILLED)
        cv2.putText(image, "Upper", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        body_data['upperbody'].append({'center': [center_x, center_y], 'area': area, 'rect': (x, y, w, h)})
    
    # Process lower body detections
    for (x, y, w, h) in lowerbody:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        center_x = x + w // 2
        center_y = y + h // 2
        area = w * h
        cv2.circle(image, (center_x, center_y), 5, (255, 0, 0), cv2.FILLED)
        cv2.putText(image, "Lower", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        body_data['lowerbody'].append({'center': [center_x, center_y], 'area': area, 'rect': (x, y, w, h)})
    
    # Process full body detections
    for (x, y, w, h) in fullbody:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        center_x = x + w // 2
        center_y = y + h // 2
        area = w * h
        cv2.circle(image, (center_x, center_y), 5, (0, 0, 255), cv2.FILLED)
        cv2.putText(image, "Full", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        body_data['fullbody'].append({'center': [center_x, center_y], 'area': area, 'rect': (x, y, w, h)})
    
    return image, body_data

def control_drone_by_pose(body_data, image_width, image_height, tel, pid_constants, previous_error):
    """
    Control drone based on detected body parts
    - Upper body horizontal position: controls yaw (left/right rotation)
    - Upper body vertical position: controls pitch (forward/backward)
    - Upper vs Lower body distance: controls throttle (up/down)
    """
    lr_speed = 0
    fb_speed = 0
    ud_speed = 0
    yaw_speed = 0
    
    # Use full body if available, otherwise use upper body
    if body_data['fullbody']:
        main_body = body_data['fullbody'][0]
    elif body_data['upperbody']:
        main_body = body_data['upperbody'][0]
    else:
        main_body = None
    
    if main_body:
        center_x, center_y = main_body['center']
        
        # Yaw control: left-right movement based on horizontal position
        error = center_x - image_width // 2
        yaw_speed = int(pid_constants[0] * error + pid_constants[1] * (error - previous_error))
        yaw_speed = int(np.clip(yaw_speed, -100, 100))
        
        # Forward-backward control: based on vertical position
        # Top of screen = move forward, bottom = move backward
        v_error = center_y - image_height // 2
        fb_speed = int(pid_constants[0] * v_error / 2)
        fb_speed = int(np.clip(fb_speed, -100, 100))
        
        # Up-down control: based on distance between upper and lower body
        if body_data['upperbody'] and body_data['lowerbody']:
            upper_y = body_data['upperbody'][0]['center'][1]
            lower_y = body_data['lowerbody'][0]['center'][1]
            distance = lower_y - upper_y
            
            # Define desired distance range (person size)
            desired_distance_range = [80, 150]
            
            if distance > desired_distance_range[1]:
                ud_speed = -30  # Move up if person is too low in frame
            elif distance < desired_distance_range[0]:
                ud_speed = 30   # Move down if person is too high in frame
            else:
                ud_speed = 0
        else:
            ud_speed = 0
    
    # Send control commands to drone
    tel.send_rc_control(lr_speed, fb_speed, ud_speed, yaw_speed)
    
    previous_error = error if main_body else 0
    return previous_error

def main():
    # Initialize drone connection
    tel = tello.Tello()
    try:
        tel.connect()
        print(f"Battery: {tel.get_battery()}%")
    except Exception as e:
        print(f"Could not connect to drone: {e}")
        print("Running in webcam-only mode (no drone control)")
        tel = None
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    image_width = 720
    image_height = 480
    
    # PID constants for smooth tracking
    pid_constants = [0.4, 0.2, 0]  # [P, D, I]
    previous_error = 0
    
    # Drone config
    if tel:
        try:
            tel.streamon()
            tel.takeoff()
            time.sleep(2)
            print("Drone takeoff complete. Move your body to control!")
            print("Move left/right to rotate drone")
            print("Move up/down in frame to move forward/backward")
            print("Distance between upper and lower body controls altitude")
            print("Press 'q' to land and exit")
        except Exception as e:
            print(f"Error during takeoff: {e}")
            tel = None
    else:
        print("\nRunning in webcam preview mode")
        print("Move your body to see drone control values")
        print("Press 'q' to exit")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame!")
                break
            
            # Flip frame for selfie-view
            frame = cv2.flip(frame, 1)
            
            # Detect body parts
            frame, body_data = detect_body_parts(frame)
            
            # Control drone based on pose
            if tel:
                previous_error = control_drone_by_pose(
                    body_data, image_width, image_height, 
                    tel, pid_constants, previous_error
                )
            else:
                # Still show control values even without drone
                if body_data['fullbody'] or body_data['upperbody']:
                    main_body = body_data['fullbody'][0] if body_data['fullbody'] else body_data['upperbody'][0]
                    center_x, center_y = main_body['center']
                    error = center_x - image_width // 2
                    yaw_speed = int(pid_constants[0] * error)
                    v_error = center_y - image_height // 2
                    fb_speed = int(pid_constants[0] * v_error / 2)
                    
                    cv2.putText(frame, f"Yaw: {yaw_speed}", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(frame, f"Forward/Back: {fb_speed}", (10, 70), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Add crosshair to show center
            center_x = image_width // 2
            center_y = image_height // 2
            cv2.circle(frame, (center_x, center_y), 10, (255, 255, 0), 2)
            cv2.line(frame, (center_x - 20, center_y), (center_x + 20, center_y), (255, 255, 0), 1)
            cv2.line(frame, (center_x, center_y - 20), (center_x, center_y + 20), (255, 255, 0), 1)
            
            cv2.imshow("Pose-Based Drone Controller", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        # Cleanup
        if tel:
            try:
                tel.land()
                time.sleep(2)
                tel.streamoff()
            except:
                pass
            finally:
                tel.end()
        
        cap.release()
        cv2.destroyAllWindows()
        print("Controller shutdown complete")

if __name__ == "__main__":
    main()
