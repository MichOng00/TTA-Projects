import cv2
from ugot import ugot
import numpy as np
import threading
import time

# ============================================================================
# DUAL CAMERA DISPLAY - UGOT and Webcam Side-by-Side
# ============================================================================

# Initialize UGOT camera
got = ugot.UGOT()
got.initialize("192.168.1.230")  # Change IP based on robot
got.open_camera()

# Initialize webcam
webcam = cv2.VideoCapture(0)

# Get frame dimensions
# For webcam
webcam_width = int(webcam.get(cv2.CAP_PROP_FRAME_WIDTH))
webcam_height = int(webcam.get(cv2.CAP_PROP_FRAME_HEIGHT))

# UGOT camera will determine its own dimensions from first frame
ugot_frame = None
ugot_width = None
ugot_height = None

# Threading setup
frame_lock = threading.Lock()
latest_ugot_frame = None
latest_webcam_frame = None
stop_threads = False

def read_ugot_frames():
    """Read frames from UGOT camera in a separate thread."""
    global latest_ugot_frame, ugot_width, ugot_height
    
    while not stop_threads:
        try:
            frame_data = got.read_camera_data()
            if frame_data is not None:
                # Decode camera frame from bytes
                nparr = np.frombuffer(frame_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if img is not None:
                    # Set dimensions from first frame
                    if ugot_width is None:
                        ugot_height, ugot_width = img.shape[:2]
                    
                    with frame_lock:
                        latest_ugot_frame = img
        except Exception as e:
            print(f"Error reading UGOT frame: {e}")
        
        time.sleep(0.01)  # Small delay to prevent CPU overload

def read_webcam_frames():
    """Read frames from webcam in a separate thread."""
    global latest_webcam_frame
    
    while not stop_threads:
        ret, frame = webcam.read()
        if ret:
            with frame_lock:
                latest_webcam_frame = frame
        else:
            print("Error reading webcam frame")
        
        time.sleep(0.01)  # Small delay to prevent CPU overload

def main():
    """Display both camera feeds side-by-side."""
    global stop_threads, ugot_width, ugot_height
    
    print("Starting dual camera display...")
    print("Press 'Q' to quit")
    
    # Start reading threads
    ugot_thread = threading.Thread(target=read_ugot_frames, daemon=True)
    webcam_thread = threading.Thread(target=read_webcam_frames, daemon=True)
    
    ugot_thread.start()
    webcam_thread.start()
    
    # Give threads time to start reading frames
    time.sleep(1)
    
    try:
        while True:
            with frame_lock:
                ugot_img = latest_ugot_frame.copy() if latest_ugot_frame is not None else None
                webcam_img = latest_webcam_frame.copy() if latest_webcam_frame is not None else None
            
            # If we don't have frames yet, wait
            if ugot_img is None or webcam_img is None:
                print("Waiting for camera frames...")
                time.sleep(0.1)
                continue
            
            # Ensure both frames have the same height for side-by-side display
            target_height = min(ugot_img.shape[0], webcam_img.shape[0])
            
            # Resize frames to same height while maintaining aspect ratio
            ugot_aspect = ugot_img.shape[1] / ugot_img.shape[0]
            webcam_aspect = webcam_img.shape[1] / webcam_img.shape[0]
            
            ugot_resized = cv2.resize(ugot_img, (int(target_height * ugot_aspect), target_height))
            webcam_resized = cv2.resize(webcam_img, (int(target_height * webcam_aspect), target_height))
            
            # Combine frames side-by-side horizontally
            combined = np.hstack((ugot_resized, webcam_resized))
            
            # Add labels
            cv2.putText(combined, "UGOT Camera", (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(combined, "Webcam", (ugot_resized.shape[1] + 20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display combined frame
            cv2.imshow("Dual Camera Display - UGOT & Webcam", combined)
            
            # Check for quit key
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                print("Exiting...")
                break
    
    except KeyboardInterrupt:
        print("Interrupted by user")
    
    finally:
        # Cleanup
        stop_threads = True
        ugot_thread.join(timeout=1)
        webcam_thread.join(timeout=1)
        
        webcam.release()
        cv2.destroyAllWindows()
        print("Cleanup complete")

if __name__ == "__main__":
    main()
