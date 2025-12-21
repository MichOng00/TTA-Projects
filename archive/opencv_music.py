import cv2
import mediapipe as mp
import time
import numpy as np
import pygame

# ------------------------------
# AUDIO SETUP (robust mono/stereo)
# ------------------------------
pygame.mixer.pre_init(44100, -16, 1, 512)
pygame.init()
try:
    pygame.mixer.quit()
except Exception:
    pass
pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=512)

def make_tone(freq=440.0, duration=0.18, sr=44100, vol=0.35):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    wave = (np.sin(2 * np.pi * freq * t) * (32767 * vol)).astype(np.int16)

    init = pygame.mixer.get_init()
    if init is None:
        raise RuntimeError("pygame mixer not initialized")
    channels = init[2]
    if channels == 2:
        wave = np.column_stack((wave, wave))

    return pygame.sndarray.make_sound(wave)

# ------------------------------
# NOTES (RIGHT HAND PIANO MAP)
# ------------------------------
NOTE_FREQ = {
    "C": 261.63,  # Middle C
    "D": 293.66,
    "E": 329.63,
    "F": 349.23,
    "G": 392.00,
}

tones = {name: make_tone(freq) for name, freq in NOTE_FREQ.items()}

# ------------------------------
# MEDIAPIPE SETUP
# ------------------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def fingers_up(lm, hand_label):
    """
    Returns [thumb, index, middle, ring, pinky] as 0/1.
    """
    out = [0, 0, 0, 0, 0]

    # Thumb (x-direction test)
    thumb_tip = lm[4]
    thumb_mcp = lm[2]
    if hand_label == "Right":
        out[0] = 1 if thumb_tip.x < thumb_mcp.x else 0
    else:
        out[0] = 1 if thumb_tip.x > thumb_mcp.x else 0

    # Other fingers (tip above PIP)
    pairs = [(8, 6), (12, 10), (16, 14), (20, 18)]
    for i, (tip, pip) in enumerate(pairs, start=1):
        out[i] = 1 if lm[tip].y < lm[pip].y else 0

    return out

def pick_notes_from_right(right5):
    """
    Map raised right-hand fingers to piano notes (chords allowed).
    """
    finger_to_note = {
        0: "C",  # thumb
        1: "D",  # index
        2: "E",  # middle
        3: "F",  # ring
        4: "G",  # pinky
    }
    return [finger_to_note[i] for i, up in enumerate(right5) if not up]

# ------------------------------
# MAIN LOOP
# ------------------------------
def main():
    print("Mixer init:", pygame.mixer.get_init())

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open webcam")
        return

    prev_left_thumb = 0
    last_play_t = 0.0
    cooldown = 0.12

    with mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5
    ) as hands:

        print("🎹 AIR PIANO")
        print("Left thumb 0→1 = PLAY")
        print("Right hand: Thumb=C Index=D Middle=E Ring=F Pinky=G")
        print("Press Q to quit")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Camera read failed")
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            left5  = [0, 0, 0, 0, 0]
            right5 = [0, 0, 0, 0, 0]
            left_seen = right_seen = False

            if results.multi_hand_landmarks and results.multi_handedness:
                for hand_landmarks, handedness in zip(
                        results.multi_hand_landmarks,
                        results.multi_handedness):

                    label = handedness.classification[0].label
                    lm = hand_landmarks.landmark

                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                    )

                    ups = fingers_up(lm, label)
                    if label == "Left":
                        left5 = ups
                        left_seen = True
                    else:
                        right5 = ups
                        right_seen = True

            notes = pick_notes_from_right(right5)

            left_thumb = left5[0]
            now = time.time()

            # PLAY trigger (rising edge)
            if left_thumb == 1 and prev_left_thumb == 0 and (now - last_play_t) > cooldown:
                for n in notes:
                    tones[n].play()
                last_play_t = now

            prev_left_thumb = left_thumb

            # ------------------------------
            # UI
            # ------------------------------
            cv2.putText(
                frame, f"L bits: {''.join(map(str,left5))}  thumb={left_thumb}",
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2
            )
            cv2.putText(
                frame, f"R bits: {''.join(map(str,right5))}  notes: {' '.join(notes) if notes else '-'}",
                (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2
            )
            cv2.putText(
                frame, "Left thumb 0→1 = PLAY | Q to quit",
                (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2
            )

            if not left_seen:
                cv2.putText(frame, "Left hand: not seen", (10, 160),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180,180,180), 2)
            if not right_seen:
                cv2.putText(frame, "Right hand: not seen", (10, 190),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180,180,180), 2)

            cv2.imshow("Air Piano", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    pygame.quit()
    print("✅ Closed.")

if __name__ == "__main__":
    main()
