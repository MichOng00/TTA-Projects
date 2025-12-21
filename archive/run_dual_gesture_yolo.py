from multi_video_stream_gesture import DualCameraDisplay
from SB_gesture_YOLO import GestureYOLOProcessor
import time
import cv2


def main():
    proc = GestureYOLOProcessor(ugot_ip="192.168.1.230", yolo_path="../IMDA/best_coffee.pt")

    # Start in gesture mode: webcam shows gesture HUD, UGOT view is plain
    display = DualCameraDisplay(
        ugot_ip="192.168.1.230",
        webcam_index=0,
        target_height=480,
        ugot_processor=None,
        webcam_processor=proc.process_webcam,
    )

    try:
        display.start()

        mode = "gesture"  # or "object"
        print("Mode: gesture (press 'o' to switch to object detection, 'g' for gesture, 'q' to quit)")

        while True:
            combined = display.get_combined_frame()
            if combined is None:
                time.sleep(0.05)
                continue

            # Overlay current mode label
            cv2.putText(combined, f"MODE: {mode.upper()}", (10, combined.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            cv2.imshow("Dual Gesture + YOLO", combined)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('o'):
                # Switch to object detection: enable UGOT processor, disable webcam processor
                display.ugot_processor = proc.process_ugot
                display.webcam_processor = None
                # stop any gesture movement
                try:
                    proc.control_robot("None", 0)
                except Exception:
                    pass
                mode = "object"
                print("Switched to object detection mode")
            elif key == ord('g'):
                # Switch to gesture control: enable webcam processor, disable UGOT processor
                display.webcam_processor = proc.process_webcam
                display.ugot_processor = None
                # ensure robot is stopped initially
                try:
                    proc.control_robot("None", 0)
                except Exception:
                    pass
                mode = "gesture"
                print("Switched to gesture control mode")

    finally:
        display.stop()


if __name__ == "__main__":
    main()
