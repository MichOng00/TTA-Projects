from djitellopy import tello
import time
import cv2
import pygame

def main():
    pygame.init()
    pygame.display.set_caption("Tello pygame")
    screen = pygame.display.set_mode((400, 200))
    frame_w = None
    frame_h = None

    tel = tello.Tello()
    tel.connect()
    print(f"battery: {tel.get_battery()}")

    # open camera
    tel.streamon()
    frame_read = tel.get_frame_read()

    try:
        tel.takeoff()
        running = True
        while running:
            img = frame_read.frame
            if img is None:
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # initialize window size on first frame
            if frame_w is None:
                frame_h, frame_w = img.shape[:2]
                screen = pygame.display.set_mode((frame_w, frame_h))
            
            video_surface = pygame.image.frombuffer(
                img_rgb.tobytes(), (frame_w, frame_h), "RGB")
            screen.blit(video_surface, (0,0))

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]:
                tel.move_forward(20)
            if keys[pygame.K_DOWN]:
                tel.move_back(20)
            if keys[pygame.K_LEFT]:
                tel.move_left(20)
            if keys[pygame.K_RIGHT]:
                tel.move_right(20)

            pygame.display.flip()

    finally:
        tel.land()
        tel.streamoff()
        cv2.destroyAllWindows()
        pygame.quit()

if __name__ == "__main__":
    main()