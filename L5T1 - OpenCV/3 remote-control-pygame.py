"""Keyboard control of Tello drone flight using pygame for input/UI
Still displays the OpenCV camera frame in a separate window.
"""
import sys
import time
import cv2
import pygame
from djitellopy import tello


def main():
    pygame.init()
    pygame.display.set_caption("Tello Remote (pygame)")
    # start with a small window; we'll resize once we get a video frame
    screen = pygame.display.set_mode((400, 200))
    font = pygame.font.SysFont(None, 22)
    clock = pygame.time.Clock()

    # video frame dims (set on first received frame)
    frame_w = None
    frame_h = None
    ui_height = 120

    tel = tello.Tello()
    tel.connect()
    # start video stream and get a single FrameRead (threaded) object
    tel.streamon()

    frame_read = tel.get_frame_read()
    battery = tel.get_battery()
    print(f"battery: {battery}")

    try:
        tel.takeoff()

        running = True
        while running:
            # Get latest camera frame from the threaded FrameRead
            img = frame_read.frame
            if img is None:
                # If no frame yet, wait a bit
                time.sleep(0.02)
            else:
                # convert BGR -> RGB for pygame
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # initialize window size on first frame
                if frame_w is None:
                    frame_h, frame_w = img.shape[:2]
                    win_w = max(400, frame_w)
                    win_h = frame_h + ui_height
                    screen = pygame.display.set_mode((win_w, win_h))

                # create a pygame surface from the RGB buffer and blit into window
                # try:
                video_surf = pygame.image.frombuffer(img_rgb.tobytes(), (frame_w, frame_h), 'RGB')
                # except Exception:
                #     # fallback: create from copied buffer
                #     video_surf = pygame.image.fromstring(img_rgb.tobytes(), (frame_w, frame_h), 'RGB')

                screen.fill((0, 0, 0))
                screen.blit(video_surf, (0, 0))

            # Pygame event handling for remote control
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                #     elif event.key == pygame.K_w:
                #         tel.move_forward(30)
                #     elif event.key == pygame.K_s:
                #         tel.move_back(30)
                #     elif event.key == pygame.K_a:
                #         tel.move_left(30)
                #     elif event.key == pygame.K_d:
                #         tel.move_right(30)
                #     elif event.key == pygame.K_l:
                #         tel.rotate_clockwise(30)
                #     elif event.key == pygame.K_j:
                #         tel.rotate_counter_clockwise(30)
                #     elif event.key == pygame.K_i:
                #         tel.move_up(30)
                #     elif event.key == pygame.K_k:
                #         tel.move_down(30)

            # Also allow holding keys for continuous control (optional)
            keys = pygame.key.get_pressed()
            # Example: hold arrow keys to move small steps
            if keys[pygame.K_UP]:
                tel.move_forward(20)
            if keys[pygame.K_DOWN]:
                tel.move_back(20)
            if keys[pygame.K_LEFT]:
                tel.move_left(20)
            if keys[pygame.K_RIGHT]:
                tel.move_right(20)


            # Draw simple UI in pygame window below the video
            lines = [
                f"Battery: {battery}",
                "Controls:",
                "W/S/A/D: forward/back/left/right",
                "I/K: up/down    J/L: rotate",
                "Arrow keys (hold): small moves",
                "Esc / window close: quit",
            ]
            # compute UI start y (leave some padding if frame not yet available)
            ui_y = frame_h + 8 if frame_h is not None else 8
            y = ui_y
            for ln in lines:
                surf = font.render(ln, True, (220, 220, 220))
                screen.blit(surf, (8, y))
                y += 26

            # process display
            pygame.display.flip()

            clock.tick(20)

    finally:
        tel.land()
        tel.streamoff()
        tel.end()
        cv2.destroyAllWindows()
        pygame.quit()


if __name__ == "__main__":
    main()
