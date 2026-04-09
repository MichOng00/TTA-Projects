"""
Session 01: Basic Webcam Display
===============================
The simplest starting point - just display the webcam feed.

Learning objectives:
- Basic OpenCV camera capture
- Window creation and display
- Basic event loop
- Clean shutdown

Requirements:
    pip install opencv-python

Controls:
    Q / ESC — Quit
"""

import cv2
import sys

def main():
    print("Session 01: Basic Webcam Display")
    print("=================================")

    # Create video capture object
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR: Could not open camera")
        sys.exit(1)

    # Get camera properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"Camera opened: {width}x{height} @ {fps} FPS")

    # Create window
    window_name = "Session 01: Webcam Feed"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, width, height)

    print("Press Q or ESC to quit")
    print("Camera feed should be visible in the window")

    while True:
        # Capture frame
        ret, frame = cap.read()

        if not ret:
            print("ERROR: Could not read frame")
            break

        # Flip horizontally for mirror effect
        frame = cv2.flip(frame, 1)

        # Display frame
        cv2.imshow(window_name, frame)

        # Check for quit key
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):  # Q or ESC
            break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    print("Session 01 completed!")

if __name__ == "__main__":
    main()