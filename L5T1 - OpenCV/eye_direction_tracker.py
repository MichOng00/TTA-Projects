# Eye Direction Tracker
# Detects eyes and determines gaze direction in real-time
import cv2
import numpy as np

def detect_eyes_and_gaze(frame):
    """
    Detect eyes in the frame and estimate gaze direction
    Returns frame with annotations and eye data
    """
    face_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_eye.xml")
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces first
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    eye_data = []
    
    for face in faces:
        x, y, w, h = face
        # Draw face rectangle
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        # Region of interest (ROI) for eyes is in upper half of face
        roi_gray = gray[y:y + h // 2, x:x + w]
        roi_color = frame[y:y + h // 2, x:x + w]
        
        # Detect eyes in the face ROI
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 4)
        
        if len(eyes) >= 2:
            # Sort eyes left to right
            eyes = sorted(eyes, key=lambda e: e[0])
            
            for eye_idx, eye in enumerate(eyes[:2]):  # Only process first 2 eyes
                ex, ey, ew, eh = eye
                
                # Adjust coordinates to full frame
                eye_x = x + ex
                eye_y = y + ey
                
                # Draw eye rectangle
                cv2.rectangle(frame, (eye_x, eye_y), (eye_x + ew, eye_y + eh), (0, 255, 0), 2)
                
                # Get eye region
                eye_region = roi_color[ey:ey + eh, ex:ex + ew]
                
                if eye_region.size > 0:
                    # Convert to grayscale for iris detection
                    eye_region_gray = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
                    
                    # Apply Gaussian blur to smooth
                    blurred = cv2.GaussianBlur(eye_region_gray, (5, 5), 0)
                    
                    # Threshold to find dark iris
                    _, thresh = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY_INV)
                    
                    # Find contours (iris)
                    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    
                    if contours:
                        # Find largest contour (iris)
                        iris = max(contours, key=cv2.contourArea)
                        (iris_x, iris_y), iris_radius = cv2.minEnclosingCircle(iris)
                        
                        # Adjust to full frame coordinates
                        iris_center = (int(eye_x + iris_x), int(eye_y + iris_y))
                        
                        # Draw iris
                        cv2.circle(frame, iris_center, int(iris_radius), (0, 0, 255), 2)
                        cv2.circle(frame, iris_center, 3, (0, 255, 255), -1)
                        
                        # Calculate gaze direction
                        eye_center = (eye_x + ew // 2, eye_y + eh // 2)
                        
                        # Vector from eye center to iris
                        gaze_x = iris_center[0] - eye_center[0]
                        gaze_y = iris_center[1] - eye_center[1]
                        
                        # Determine gaze direction
                        gaze_direction = determine_gaze_direction(gaze_x, gaze_y, ew, eh)
                        
                        # Draw gaze direction arrow
                        arrow_length = 60
                        angle = np.arctan2(gaze_y, gaze_x)
                        end_x = int(eye_center[0] + arrow_length * np.cos(angle))
                        end_y = int(eye_center[1] + arrow_length * np.sin(angle))
                        cv2.arrowedLine(frame, eye_center, (end_x, end_y), (0, 255, 255), 3, tipLength=0.3)
                        
                        # Label with gaze direction
                        label = f"Eye {eye_idx + 1}: {gaze_direction}"
                        cv2.putText(frame, label, (eye_x, eye_y - 5), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        eye_data.append({
                            'center': eye_center,
                            'iris': iris_center,
                            'gaze_direction': gaze_direction,
                            'gaze_vector': (gaze_x, gaze_y)
                        })
    
    return frame, eye_data

def determine_gaze_direction(gaze_x, gaze_y, eye_width, eye_height):
    """
    Determine gaze direction based on iris position relative to eye center
    """
    # Threshold for determining direction (roughly 1/4 of eye region)
    x_threshold = eye_width * 0.15
    y_threshold = eye_height * 0.05
    
    # Determine horizontal direction
    if gaze_x < -x_threshold:
        h_direction = "LEFT"
    elif gaze_x > x_threshold:
        h_direction = "RIGHT"
    else:
        h_direction = "CENTER"
    
    # Determine vertical direction
    if gaze_y < -y_threshold:
        v_direction = "UP"
    elif gaze_y > y_threshold:
        v_direction = "DOWN"
    else:
        v_direction = ""
    
    # Combine directions
    if v_direction:
        return f"{v_direction}-{h_direction}"
    else:
        return h_direction

def draw_gaze_grid(frame):
    """
    Draw a grid to help visualize gaze zones
    """
    height, width = frame.shape[:2]
    
    # Draw vertical lines
    cv2.line(frame, (width // 3, 0), (width // 3, height), (200, 200, 200), 1)
    cv2.line(frame, (2 * width // 3, 0), (2 * width // 3, height), (200, 200, 200), 1)
    
    # Draw horizontal lines
    cv2.line(frame, (0, height // 3), (width, height // 3), (200, 200, 200), 1)
    cv2.line(frame, (0, 2 * height // 3), (width, 2 * height // 3), (200, 200, 200), 1)
    
    # Add zone labels
    zones = [
        ("UP-LEFT", width // 6, height // 6),
        ("UP-CENTER", width // 2, height // 6),
        ("UP-RIGHT", 5 * width // 6, height // 6),
        ("LEFT", width // 6, height // 2),
        ("CENTER", width // 2, height // 2),
        ("RIGHT", 5 * width // 6, height // 2),
        ("DOWN-LEFT", width // 6, 5 * height // 6),
        ("DOWN-CENTER", width // 2, 5 * height // 6),
        ("DOWN-RIGHT", 5 * width // 6, 5 * height // 6),
    ]
    
    for zone_label, x, y in zones:
        cv2.putText(frame, zone_label, (x - 40, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    return frame

def main():
    """
    Main function to run eye direction tracker
    """
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # FPS counter
    fps_start = cv2.getTickCount()
    fps_count = 0
    
    print("Eye Direction Tracker")
    print("=" * 50)
    print("Tracking eye gaze direction in real-time")
    print("Green boxes = Eyes detected")
    print("Red circle = Iris position")
    print("Yellow arrow = Gaze direction")
    print("Press 'q' to quit")
    print("=" * 50)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame!")
                break
            
            # Flip for selfie view
            frame = cv2.flip(frame, 1)
            
            # Draw gaze zones
            frame = draw_gaze_grid(frame)
            
            # Detect eyes and gaze
            frame, eye_data = detect_eyes_and_gaze(frame)
            
            # Draw overall statistics
            stats_y = 30
            cv2.putText(frame, f"Eyes Detected: {len(eye_data)}", (10, stats_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            if eye_data:
                for idx, eye in enumerate(eye_data):
                    direction = eye['gaze_direction']
                    cv2.putText(frame, f"Eye {idx + 1} Gaze: {direction}", 
                               (10, stats_y + 30 * (idx + 1)), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # FPS counter
            fps_count += 1
            fps_elapsed = (cv2.getTickCount() - fps_start) / cv2.getTickFrequency()
            if fps_elapsed >= 1.0:
                fps = fps_count / fps_elapsed
                fps_start = cv2.getTickCount()
                fps_count = 0
            else:
                fps = fps_count / fps_elapsed
            
            cv2.putText(frame, f"FPS: {fps:.1f}", 
                       (frame.shape[1] - 150, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("Eye Direction Tracker", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\nTracker shutdown complete")

if __name__ == "__main__":
    main()
