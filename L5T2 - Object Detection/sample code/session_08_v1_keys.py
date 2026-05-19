import cv2
import time
import random
from dataclasses import dataclass


@dataclass
class Note:
    lane: int
    y: float
    speed: float = 40.0
    active: bool = True

    def update(self, dt: float) -> None:
        self.y += self.speed * dt
        if self.y > 540:
            self.active = False


class RhythmGameSimple:
    """Very small playable rhythm game.
    Controls: press 1/2/3 to hit corresponding lanes.
    """
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

    def spawn_notes(self, now):
        if now - self.last_spawn_time < self.spawn_interval:
            return
        lane = random.randint(0, self.lane_count - 1)
        self.notes.append(Note(lane=lane, y=-50.0, speed=30.0))
        self.last_spawn_time = now

    def update_notes(self, now, key):
        for note in self.notes:
            if not note.active:
                continue
            note.update(now - self.last_spawn_time)

            if note.y > self.hit_zone_y + self.hit_window:
                note.active = False
                self.combo = 0
                self.health -= 1
                continue

            # simple keyboard hit: keys 1,2,3 map to lanes
            if key in (ord('1'), ord('2'), ord('3')):
                pressed_lane = key - ord('1')
                if pressed_lane == note.lane and abs(note.y - self.hit_zone_y) < self.hit_window:
                    note.active = False
                    distance = abs(note.y - self.hit_zone_y)
                    points = 120 if distance < 30 else 90 if distance < 55 else 60
                    self.score += points + self.combo * 5
                    self.combo += 1

        self.notes = [n for n in self.notes if n.active]
        if self.health <= 0:
            return True
        return False

    def draw(self, frame):
        for x in self.lane_x:
            cv2.line(frame, (x, 0), (x, self.frame_height), (180, 180, 180), 1)
        cv2.line(frame, (0, self.hit_zone_y), (self.frame_width, self.hit_zone_y), (0, 190, 255), 2)

        for note in self.notes:
            cv2.circle(frame, (self.lane_x[note.lane], int(note.y)), 28, (90, 220, 90), -1)

        cv2.putText(frame, f"Score: {self.score}", (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(frame, "Press 1/2/3 to hit lanes", (16, self.frame_height - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2)

    def run(self):
        prev = time.time()
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
            game_over = self.update_notes(now, key)
            self.draw(frame)
            if game_over:
                cv2.putText(frame, "GAME OVER", (self.frame_width//2 - 120, self.frame_height//2), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0,0,255), 3)

            cv2.imshow("Hand Rhythm Game", frame)
            if key == ord('q') or key == 27:
                break

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    g = RhythmGameSimple()
    g.run()
