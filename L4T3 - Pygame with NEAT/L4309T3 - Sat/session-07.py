import pygame
import math
import sys
import neat

SCREEN_WIDTH = 900
SCREEN_HEIGHT = 735

SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

TRACK = pygame.image.load("Assets/not_car/track.png")
TRACK = pygame.transform.scale(TRACK, (SCREEN_WIDTH, SCREEN_HEIGHT))
START_POS = (350, 600)

GRASS_COLOR = pygame.Color(2, 105, 31, 255)

pygame.init()

font = pygame.font.Font(None, 36)

class Car(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.original_image = pygame.image.load("Assets/not_car/car.png")
        self.image = self.original_image
        self.rect = self.image.get_rect(center = START_POS)
        self.drive_state = False
        self.vel_vector = pygame.math.Vector2(0.8, 0)
        self.angle = 0
        self.rotation_vel = 5
        self.direction = 0 # 0: straight, 1/-1: left/right
        self.gear = 1

    def update(self):
        self.drive()
        self.rotate()

    def drive(self):
        if self.drive_state:
            self.rect.center += self.vel_vector * (6 + self.gear)

    def rotate(self):
        if self.direction == 1:
            self.angle -= self.rotation_vel
            self.vel_vector.rotate_ip(self.rotation_vel)
        if self.direction == -1:
            self.angle += self.rotation_vel
            self.vel_vector.rotate_ip(-self.rotation_vel)
        self.image = pygame.transform.rotozoom(self.original_image, self.angle, 0.1)
        self.rect = self.image.get_rect(center = self.rect.center)

    def check_collision(self):
        x, y  = int(self.rect.centerx), int(self.rect.centery)
        if 0 <= x < SCREEN_WIDTH and 0 <= y < SCREEN_HEIGHT:
            color_at_car = SCREEN.get_at((x, y))
            if color_at_car == GRASS_COLOR:
                print("touch grass")
                self.respawn()

    def respawn(self):
        self.rect.center = START_POS
        self.angle = 0
        self.vel_vector = pygame.math.Vector2(0.8, 0)
        self.direction = 0
        self.drive_state = False

car = pygame.sprite.GroupSingle(Car())

def main():
    shifted = False
    run = True
    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LSHIFT:
                    car.sprite.gear += 1
                    car.sprite.gear = min(car.sprite.gear, 5)
                if event.key == pygame.K_LCTRL:
                    car.sprite.gear -= 1
                    car.sprite.gear = max(car.sprite.gear, 1)

        SCREEN.blit(TRACK, (0,0))

        user_input = pygame.key.get_pressed()
        if sum(user_input) <= 1:
            car.sprite.drive_state = False
            car.sprite.direction = 0
        if user_input[pygame.K_UP]:
            car.sprite.drive_state = True
        if user_input[pygame.K_RIGHT]:
            car.sprite.direction = 1
        if user_input[pygame.K_LEFT]:
            car.sprite.direction = -1

        car.update()
        car.sprite.check_collision()
        car.draw(SCREEN)

        gear_text = f"Gear: {car.sprite.gear}"
        gear_surface = font.render(gear_text, True, (255,255,255))
        SCREEN.blit(gear_surface, (20, 20))

        pygame.display.update()

    pygame.quit()
    sys.exit()
main()