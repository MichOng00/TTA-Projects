'''Setup and display car'''

import pygame
import math
import sys
import neat

# Constants
SCREEN_WIDTH = 900
SCREEN_HEIGHT = 735
SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

TRACK = pygame.image.load('Assets/not_car/track.png')
TRACK = pygame.transform.scale(TRACK, (SCREEN_WIDTH, SCREEN_HEIGHT))
START_POS = (350, 600)

pygame.init()

class Car(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        # This is the image of your car
        self.original_image = pygame.image.load('Assets/not_car/car.png')
        self.image = self.original_image
        self.rect = self.image.get_rect(center=START_POS)
        self.drive_state = False
        self.vel_vector = pygame.math.Vector2(0.8, 0)
        self.angle = 0
        self.rotation_vel = 5
        self.direction = 0

    def update(self):
        self.drive()
        self.rotate()

    def drive(self):
        if self.drive_state:
            self.rect.center += self.vel_vector * 6

    def rotate(self):
        if self.direction == 1:
            self.angle -= self.rotation_vel
            self.vel_vector.rotate_ip(self.rotation_vel)
        if self.direction == -1:
            self.angle += self.rotation_vel
            self.vel_vector.rotate_ip(-self.rotation_vel)
        self.image = pygame.transform.rotozoom(self.original_image, self.angle, 0.1)
        self.rect = self.image.get_rect(center=self.rect.center)

car = pygame.sprite.GroupSingle(Car())

def eval_genomes():
    run = True
    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        SCREEN.blit(TRACK, (0, 0))

        # User Input
        user_input = pygame.key.get_pressed()
        if sum(pygame.key.get_pressed()) <= 1:
            car.sprite.drive_state = False
            car.sprite.direction = 0

        # Drive
        if user_input[pygame.K_UP]:
            car.sprite.drive_state = True

        # Steering time
        if user_input[pygame.K_RIGHT]:
            car.sprite.direction = 1
        if user_input[pygame.K_LEFT]:
            car.sprite.direction = -1

        # Update everything
        car.update()
        car.draw(SCREEN)
        pygame.display.update()

eval_genomes()