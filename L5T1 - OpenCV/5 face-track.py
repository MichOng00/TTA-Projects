import cv2
import numpy as np
from djitellopy import tello
import time

def find_face(image):
    face_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")
    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(image_gray, 1.2, 8)
    
    face_centers = []
    face_areas = []
    
    if len(faces) != 0:
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            center_x = x + w // 2
            center_y = y + h // 2
            area = w * h
            cv2.circle(image, (center_x, center_y), 5, (0, 255, 0), cv2.FILLED)
            face_centers.append([center_x, center_y])
            face_areas.append(area)
        
        max_area_index = face_areas.index(max(face_areas))
        return image, [face_centers[max_area_index], face_areas[max_area_index]]
    
    else:
        return image, [[0, 0], 0]

def track_face(face_info, image_width, pid_constants, previous_error, tel):
    area = face_info[1]
    x, y = face_info[0]
    pitch = 0
    
    error = x - image_width // 2
    yaw = pid_constants[0] * error + pid_constants[1] * (error - previous_error)
    yaw = int(np.clip(yaw, -100, 100))
    
    if area > area_range[0] and area < area_range[1]:
        pitch = 0
    elif area > area_range[1]:
        pitch = -20
    elif area < area_range[0] and area != 0:
        pitch = 20
    
    if x == 0:
        yaw = 0
        error = 0
    
    tel.send_rc_control(0, pitch, 0, yaw)
    return error

if __name__ == "__main__":
    # Connect to the drone and take off
    tel = tello.Tello()
    tel.connect()
    print(f"Battery: {tel.get_battery()}")
    tel.streamon()
    tel.takeoff()

    # Move the drone up to face height
    tel.send_rc_control(0, 0, 50, 0)
    time.sleep(1)
    
    image_width, image_height = 720, 480
    area_range = [6200, 6800]

    # PID constants - only K_p and K_d used
    pid_constants = [0.4, 0.4, 0]
    previous_error = 0
    
    while True:
        image = tel.get_frame_read().frame
        image = cv2.resize(image, (image_width, image_height))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        image, face_info = find_face(image)
        previous_error = track_face(face_info, image_width, pid_constants, previous_error, tel)
        
        cv2.imshow("Output", image)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    tel.land()
    tel.streamoff()
    tel.end()
    cv2.destroyAllWindows()
