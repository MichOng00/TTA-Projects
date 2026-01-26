from ugot import ugot
got = ugot.UGOT()
got.initialize("192.168.1.205")
import time
import pygame
import cv2
import numpy as np

def forward():
    got.mecanum_move_speed(0, 30)

def backward():
    got.mecanum_move_speed(1, 30)

def turn_left():
    got.mecanum_turn_speed(2, 30)

def turn_right():
    got.mecanum_turn_speed(3, 30)

if __name__ == "__main__":
    pygame.init()
    got.open_camera()
    screen = None

    running = True
    while running:
        frame = got.read_camera_data()
        if not frame:
            break

        nparr = np.frombuffer(frame, np.uint8)
        data = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        flipped = cv2.flip(data, 1)
        frame_rgb = cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB)

        h, w = frame_rgb.shape[:2]
        if screen is None: # only create window on first frame
            screen = pygame.display.set_mode((w,h))
            pygame.display.set_caption("UGOT control")

        surface = pygame.image.frombuffer(frame_rgb.tobytes(), (w,h), "RGB")
        screen.blit(surface, (0, 0))
        pygame.display.flip()

        for evt in pygame.event.get():
            if evt.type == pygame.QUIT:
                running = False

            # move on keypress, stop when key released
            elif evt.type == pygame.KEYDOWN:
                if evt.key == pygame.K_w:
                    forward()
                elif evt.key == pygame.K_s:
                    backward()
                elif evt.key == pygame.K_a:
                    turn_left()
                elif evt.key == pygame.K_d:
                    turn_right()
            
            elif evt.type == pygame.KEYUP:
                got.mecanum_stop()