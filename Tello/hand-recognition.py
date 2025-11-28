### NOT DONE
# compared to remote-control.py, video does not pause
from djitellopy import tello
from threading import Thread
import cv2, time

class VideoStream:
    def __init__(self, tello_instance):
        self.tel = tello_instance
        self.frame_read = self.tel.get_frame_read()
        self.running = True
        self.key = None

    def start(self):
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
        return self

    def update(self):
        while self.running:
            img = self.frame_read.frame

            #Draw a red rectangle
            cv2.rectangle(img, (50, 50), (200, 200), (0, 0, 255), 2)

            # Draw text
            cv2.putText(img, "Hello OpenCV", (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            cv2.imshow("Drone camera", img)
            self.key = cv2.waitKey(1) & 0xFF  # Update key based on user input

    def stop(self):
        self.running = False
        self.thread.join()
        cv2.destroyAllWindows()

def remote_control(tel):
    try:
        video_stream = VideoStream(tel)
        video_stream.start()
        # tel.takeoff()

        while True:
            key = video_stream.key  # Read the key press from the video stream

            if key == 27:  # ESC key
                break
            # elif key == ord('w'):
            #     tel.move_forward(30)
            # elif key == ord('s'):
            #     tel.move_back(30)
            # elif key == ord('a'):
            #     tel.move_left(30)
            # elif key == ord('d'):
            #     tel.move_right(30)
            # elif key == ord('e'):
            #     tel.rotate_clockwise(30)
            # elif key == ord('q'):
            #     tel.rotate_counter_clockwise(30)
            # elif key == ord('r'):
            #     tel.move_up(30)
            # elif key == ord('f'):
            #     tel.move_down(30)

            time.sleep(0.01)  # Add small delay to prevent CPU overload

    finally:
        video_stream.stop()
        tel.land()
        tel.streamoff()
        tel.end()

if __name__ == "__main__":
    tel = tello.Tello()
    tel.connect()
    print(f"battery: {tel.get_battery()}")

    tel.streamon()
    remote_control(tel)
