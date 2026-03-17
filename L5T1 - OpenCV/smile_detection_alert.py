# Smile Detection Alert System
# Detects smiles and triggers alerts, captures photos, and tracks mood
import cv2
import numpy as np
import time
import winsound
from datetime import datetime
import os

def detect_faces_and_smiles(frame):
    """
    Detect faces and smiles in the frame
    Returns frame with annotations and detection data
    """
    face_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")
    smile_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_smile.xml")
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    detection_data = {
        'faces': [],
        'smiles': [],
        'smile_detected': False
    }
    
    for face in faces:
        x, y, w, h = face
        
        # Draw face rectangle
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, "Face", (x, y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Face center
        face_center = (x + w // 2, y + h // 2)
        detection_data['faces'].append({
            'rect': (x, y, w, h),
            'center': face_center,
            'area': w * h
        })
        
        # Detect smiles in lower half of face (where mouth is)
        roi_gray = gray[y + h // 2:y + h, x:x + w]
        roi_color = frame[y + h // 2:y + h, x:x + w]
        
        smiles = smile_cascade.detectMultiScale(roi_gray, 
                                               scaleFactor=1.8, 
                                               minNeighbors=20, 
                                               minSize=(25, 25))
        
        if len(smiles) > 0:
            detection_data['smile_detected'] = True
            
            for smile in smiles:
                sx, sy, sw, sh = smile
                
                # Adjust coordinates to full frame
                smile_x = x + sx
                smile_y = y + h // 2 + sy
                
                # Draw smile rectangle with bright color
                cv2.rectangle(frame, (smile_x, smile_y), 
                            (smile_x + sw, smile_y + sh), (0, 255, 0), 3)
                cv2.putText(frame, "SMILE!", (smile_x, smile_y - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 3)
                
                # Draw circle around smile detection
                smile_center = (smile_x + sw // 2, smile_y + sh // 2)
                cv2.circle(frame, smile_center, int(max(sw, sh) // 1.5), (0, 255, 0), 2)
                
                detection_data['smiles'].append({
                    'rect': (smile_x, smile_y, sw, sh),
                    'center': smile_center,
                    'confidence': len(smiles)  # More detections = higher confidence
                })
    
    return frame, detection_data

def play_alert_sound(frequency=1000, duration=200):
    """
    Play a beep alert sound
    frequency: Hz
    duration: milliseconds
    """
    try:
        winsound.Beep(frequency, duration)
    except Exception as e:
        print(f"Could not play sound: {e}")

def capture_smile_photo(frame, capture_dir="smile_captures"):
    """
    Save a photo of the detected smile
    """
    # Create directory if it doesn't exist
    if not os.path.exists(capture_dir):
        os.makedirs(capture_dir)
    
    # Create filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(capture_dir, f"smile_{timestamp}.jpg")
    
    # Save the frame
    cv2.imwrite(filename, frame)
    return filename

def main():
    """
    Main function to run smile detection alert system
    """
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # Configuration
    smile_cooldown = 0.5  # Minimum seconds between alerts
    last_smile_time = 0
    smile_count = 0
    
    # Mode selection
    alert_mode = "beep"  # "beep", "photo", or "both"
    auto_capture = False
    
    print("Smile Detection Alert System")
    print("=" * 50)
    print("Detects smiles and triggers alerts")
    print("\nControls:")
    print("  'q' = Quit")
    print("  'b' = Beep alert mode")
    print("  'p' = Photo capture mode")
    print("  'a' = Auto-capture both (beep + photo)")
    print("  'c' = Toggle auto-capture")
    print("  'r' = Reset smile counter")
    print("=" * 50)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame!")
                break
            
            # Flip for selfie view
            frame = cv2.flip(frame, 1)
            
            # Detect faces and smiles
            frame, detection_data = detect_faces_and_smiles(frame)
            
            # Handle smile detection
            current_time = time.time()
            
            if detection_data['smile_detected']:
                # Check cooldown to avoid rapid-fire alerts
                if current_time - last_smile_time > smile_cooldown:
                    smile_count += 1
                    last_smile_time = current_time
                    
                    # Trigger alert based on mode
                    if alert_mode == "beep" or alert_mode == "both":
                        play_alert_sound(frequency=1000, duration=200)
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] Smile #{smile_count} detected! (Beep)")
                    
                    if alert_mode == "photo" or alert_mode == "both":
                        filename = capture_smile_photo(frame)
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] Smile #{smile_count} detected! Photo saved: {filename}")
                    
                    if alert_mode == "beep":
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] Smile #{smile_count} detected!")
                
                # Visual feedback: flash background
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), 
                            (0, 255, 0), -1)
                frame = cv2.addWeighted(overlay, 0.1, frame, 0.9, 0)
            
            # Draw statistics
            stats_y = 30
            cv2.putText(frame, f"Faces Detected: {len(detection_data['faces'])}", 
                       (10, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, f"Smiles Detected This Session: {smile_count}", 
                       (10, stats_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Mode indicator
            mode_text = f"Mode: {alert_mode.upper()}"
            cv2.putText(frame, mode_text, (10, stats_y + 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 200, 255), 2)
            
            # Current time
            current_time_str = datetime.now().strftime("%H:%M:%S")
            cv2.putText(frame, current_time_str, 
                       (frame.shape[1] - 200, stats_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Smile detection indicator (big red/green circle)
            circle_color = (0, 255, 0) if detection_data['smile_detected'] else (0, 0, 255)
            circle_radius = 30 if detection_data['smile_detected'] else 25
            cv2.circle(frame, (frame.shape[1] - 50, 50), circle_radius, circle_color, 3)
            
            # Help text
            help_text = "Press keys: b(beep) p(photo) a(both) c(toggle) r(reset) q(quit)"
            cv2.putText(frame, help_text, (10, frame.shape[0] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            cv2.imshow("Smile Detection Alert System", frame)
            
            # Keyboard controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('b'):
                alert_mode = "beep"
                print("Alert mode: BEEP")
            elif key == ord('p'):
                alert_mode = "photo"
                print("Alert mode: PHOTO CAPTURE")
            elif key == ord('a'):
                alert_mode = "both"
                print("Alert mode: BOTH (Beep + Photo)")
            elif key == ord('c'):
                auto_capture = not auto_capture
                print(f"Auto-capture: {'ON' if auto_capture else 'OFF'}")
            elif key == ord('r'):
                smile_count = 0
                print("Smile counter reset!")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\n" + "=" * 50)
        print(f"Session complete! Total smiles detected: {smile_count}")
        print("=" * 50)

if __name__ == "__main__":
    main()
