"""Minimal webcam viewer using OpenCV"""
import cv2

# Open webcam (0 = default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot open webcam")
    exit()

print("Webcam opened. Press ESC or 'q' to exit.")

try:
    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            break
        
        # Display frame
        cv2.imshow("Webcam", frame)
        
        # Exit on ESC or 'q'
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam closed.")
