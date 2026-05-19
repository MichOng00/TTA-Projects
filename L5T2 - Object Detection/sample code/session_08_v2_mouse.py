import cv2
import time
import random
from dataclasses import dataclass


@dataclass
class Note:
    lane: int
    y: float
    speed: float = 200.0
    active: bool = True

    def update(self, dt: float) -> None:
        self.y += self.speed * dt
        if self.y > 540:
            self.active = False


class RhythmGameMouse:
    """Intermediate rhythm game: move mouse to choose lane, press space to hit."""
    def __init__(self):
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise RuntimeError("Unable to open webcam")

        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)

        self.lane_count = 3
        self.lane_x = [int((i + 0.5) * self.frame_width / self.lane_count) for i in range(self.lane_count)]
        self.hit_zone_y = int(self.frame_height * 0.82)
        self.hit_window = 80

        self.spawn_interval = 0.8
        self.last_spawn_time = time.time()

        self.notes = []
        self.score = 0
        self.combo = 0
        self.health = 6

        self.cursor_x = self.frame_width // 2

    def spawn_notes(self, now):
        if now - self.last_spawn_time < self.spawn_interval:
            return
        lane = random.randint(0, self.lane_count - 1)
        self.notes.append(Note(lane=lane, y=-50.0, speed=180.0))
        self.last_spawn_time = now

    def hit(self):
        # attempt to hit any note in the cursor lane
        for note in self.notes:
            if not note.active:
                continue
            if abs(self.cursor_x - self.lane_x[note.lane]) < self.frame_width // (self.lane_count * 2):
                if abs(note.y - self.hit_zone_y) < self.hit_window:
                    note.active = False
                    distance = abs(note.y - self.hit_zone_y)
                    points = 120 if distance < 30 else 90 if distance < 55 else 60
                    self.score += points + self.combo * 5
                    self.combo += 1
                    return

    def update_notes(self, now):
        for note in self.notes:
            if not note.active:
                continue
            note.update(now - self.last_spawn_time)
            if note.y > self.hit_zone_y + self.hit_window:
                note.active = False
                self.combo = 0
                self.health -= 1

        self.notes = [n for n in self.notes if n.active]

    def draw(self, frame):
        for x in self.lane_x:
            cv2.line(frame, (x, 0), (x, self.frame_height), (180, 180, 180), 1)
        cv2.line(frame, (0, self.hit_zone_y), (self.frame_width, self.hit_zone_y), (0, 190, 255), 2)

        for note in self.notes:
            cv2.circle(frame, (self.lane_x[note.lane], int(note.y)), 28, (90, 220, 90), -1)

        # draw cursor
        cv2.circle(frame, (self.cursor_x, self.hit_zone_y), 18, (220, 80, 80), -1)

        cv2.putText(frame, f"Score: {self.score}", (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(frame, "Move mouse and press SPACE to hit", (16, self.frame_height - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2)

    def run(self):
        prev = time.time()
        cv2.namedWindow("Hand Rhythm Game")

        def mouse_cb(event, x, y, flags, param):
            if event == cv2.EVENT_MOUSEMOVE:
                self.cursor_x = x

        cv2.setMouseCallback("Hand Rhythm Game", mouse_cb)

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            now = time.time()
            dt = now - prev
            prev = now

            frame = cv2.flip(frame, 1)
            self.spawn_notes(now)
            key = cv2.waitKey(1) & 0xFF

            if key == ord(' '):
                self.hit()

            self.update_notes(now)
            self.draw(frame)

            if self.health <= 0:
                cv2.putText(frame, "GAME OVER", (self.frame_width//2 - 120, self.frame_height//2), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0,0,255), 3)

            cv2.imshow("Hand Rhythm Game", frame)
            if key == ord('q') or key == 27:
                break

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    g = RhythmGameMouse()
    g.run()
