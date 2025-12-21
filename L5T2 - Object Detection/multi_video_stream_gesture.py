import cv2
from ugot import ugot
import numpy as np
import threading
import time

# ============================================================================
# DUAL CAMERA DISPLAY - UGOT and Webcam Side-by-Side
# ============================================================================

class DualCameraDisplay:
    """
    Manages dual camera streams (UGOT and Webcam) with separate reading threads.
    Displays both feeds side-by-side in a single window.
    """
    
    def __init__(self, ugot_ip="192.168.1.230", webcam_index=0, target_height=480,
                 ugot_processor=None, webcam_processor=None):
        """
        Initialize the dual camera system.
        
        Args:
            ugot_ip (str): IP address of the UGOT robot
            webcam_index (int): Index of the webcam (0 for default)
            target_height (int): Height for resizing frames (maintains aspect ratio)
        """
        self.ugot_ip = ugot_ip
        self.target_height = target_height
        
        # Camera objects
        self.ugot = None
        self.webcam = None
        
        # Frame storage
        self.latest_ugot_frame = None
        self.latest_webcam_frame = None
        
        # Threading
        self.frame_lock = threading.Lock()
        self.stop_threads = False
        self.ugot_thread = None
        self.webcam_thread = None
        
        # Initialize cameras
        self._init_ugot()
        self._init_webcam(webcam_index)
        # Optional processors that annotate/process frames
        # Each should be a callable that accepts a BGR image and returns an annotated BGR image.
        self.ugot_processor = ugot_processor
        self.webcam_processor = webcam_processor
    
    def _init_ugot(self):
        """Initialize UGOT camera connection."""
        try:
            self.ugot = ugot.UGOT()
            self.ugot.initialize(self.ugot_ip)
            self.ugot.open_camera()
            print(f"✓ UGOT camera initialized at {self.ugot_ip}")
            self.ugot.balance_start_balancing()
        except Exception as e:
            print(f"✗ Error initializing UGOT: {e}")
            self.ugot = None
    
    def _init_webcam(self, index):
        """Initialize webcam connection."""
        try:
            self.webcam = cv2.VideoCapture(index)
            if not self.webcam.isOpened():
                raise RuntimeError("Could not open webcam")
            print(f"✓ Webcam initialized (index {index})")
        except Exception as e:
            print(f"✗ Error initializing webcam: {e}")
            self.webcam = None
    
    def _read_ugot_frames(self):
        """Read frames from UGOT camera in a separate thread."""
        while not self.stop_threads:
            if self.ugot is None:
                time.sleep(0.1)
                continue
            
            try:
                frame_data = self.ugot.read_camera_data()
                if frame_data is not None:
                    # Decode camera frame from bytes
                    nparr = np.frombuffer(frame_data, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    if img is not None:
                        with self.frame_lock:
                            self.latest_ugot_frame = img
            except Exception as e:
                print(f"Error reading UGOT frame: {e}")
            
            time.sleep(0.01)
    
    def _read_webcam_frames(self):
        """Read frames from webcam in a separate thread."""
        while not self.stop_threads:
            if self.webcam is None:
                time.sleep(0.1)
                continue
            
            try:
                ret, frame = self.webcam.read()
                if ret:
                    with self.frame_lock:
                        self.latest_webcam_frame = frame
                else:
                    print("Error reading webcam frame")
            except Exception as e:
                print(f"Error reading webcam: {e}")
            
            time.sleep(0.01)
    
    def start(self):
        """Start the camera reading threads."""
        self.stop_threads = False
        self.ugot_thread = threading.Thread(target=self._read_ugot_frames, daemon=True)
        self.webcam_thread = threading.Thread(target=self._read_webcam_frames, daemon=True)
        
        self.ugot_thread.start()
        self.webcam_thread.start()
        
        # Give threads time to start reading frames
        time.sleep(1)
        print("Camera threads started")
    
    def get_combined_frame(self):
        """
        Get the current combined frame (both cameras side-by-side).
        
        Returns:
            np.ndarray or None: Combined frame with both cameras, or None if not ready
        """
        with self.frame_lock:
            ugot_img = self.latest_ugot_frame.copy() if self.latest_ugot_frame is not None else None
            webcam_img = self.latest_webcam_frame.copy() if self.latest_webcam_frame is not None else None
        
        if ugot_img is None or webcam_img is None:
            return None
        
        # Allow optional per-frame processing (e.g., gesture HUD, YOLO HUD)
        try:
            if ugot_img is not None and self.ugot_processor is not None:
                # processors expected to return an annotated BGR image
                ugot_img = self.ugot_processor(ugot_img)
        except Exception as e:
            print(f"Error in UGOT processor: {e}")

        try:
            if webcam_img is not None and self.webcam_processor is not None:
                webcam_img = self.webcam_processor(webcam_img)
        except Exception as e:
            print(f"Error in webcam processor: {e}")

        # Ensure both frames have the same height
        target_height = min(ugot_img.shape[0], webcam_img.shape[0], self.target_height)
        
        # Resize frames to same height while maintaining aspect ratio
        ugot_aspect = ugot_img.shape[1] / ugot_img.shape[0]
        webcam_aspect = webcam_img.shape[1] / webcam_img.shape[0]
        
        ugot_resized = cv2.resize(ugot_img, (int(target_height * ugot_aspect), target_height))
        webcam_resized = cv2.resize(webcam_img, (int(target_height * webcam_aspect), target_height))
        
        # Combine frames side-by-side
        combined = np.hstack((ugot_resized, webcam_resized))
        
        # Add labels
        cv2.putText(combined, "UGOT Camera", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(combined, "Webcam", (ugot_resized.shape[1] + 20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return combined
    
    def display(self, window_name="Dual Camera Display"):
        """
        Display the combined frames in a window. Press 'Q' to quit.
        
        Args:
            window_name (str): Name of the display window
        """
        print(f"Displaying cameras... Press 'Q' to quit")
        
        try:
            while True:
                combined = self.get_combined_frame()
                
                if combined is None:
                    print("Waiting for camera frames...")
                    time.sleep(0.1)
                    continue
                
                cv2.imshow(window_name, combined)
                
                # Check for quit key
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == ord('Q'):
                    print("Exiting...")
                    break
        
        except KeyboardInterrupt:
            print("Interrupted by user")
    
    def stop(self):
        """Stop the camera reading threads and clean up resources."""
        print("Stopping cameras...")
        self.stop_threads = True
        
        if self.ugot_thread:
            self.ugot_thread.join(timeout=1)
        if self.webcam_thread:
            self.webcam_thread.join(timeout=1)
        
        if self.webcam:
            self.webcam.release()
        
        cv2.destroyAllWindows()
        print("Cleanup complete")


def main():
    """Main entry point."""
    # Create and run the dual camera display
    display = DualCameraDisplay(ugot_ip="192.168.1.230", target_height=480)
    
    try:
        display.start()
        display.display()
    finally:
        display.stop()


if __name__ == "__main__":
    main()
