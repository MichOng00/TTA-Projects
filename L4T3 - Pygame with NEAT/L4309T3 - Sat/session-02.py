import pygame
import os
import random
import sys

pygame.init()

# constants
SCREEN_HEIGHT = 600
SCREEN_WIDTH = 1100
SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

RUNNING = [pygame.image.load("Assets/Dino/DinoRun1.png"),
           pygame.image.load("Assets/Dino/DinoRun2.png")]
JUMPING = pygame.image.load("Assets/Dino/DinoJump.png")

BG = pygame.image.load("Assets/Other/Track.png")

FONT = pygame.font.Font("freesansbold.ttf", 20)

SMALL_CACTUS = [pygame.transform.scale(pygame.image.load(f"Assets/Cactus/SmallCactus{i}.png"), (50,60)) for i in range(1,4)]
LARGE_CACTUS = [pygame.transform.scale(pygame.image.load(f"Assets/Cactus/LargeCactus{i}.png"), (50,70)) for i in range(1,4)]

class Dinosaur:
    X_POS = 80 # class attribute
    Y_POS = 300
    JUMP_VEL = 8

    def __init__(self, img=RUNNING[0]):
        self.image = img # instance attribute
        self.dino_run = True
        self.dino_jump = False
        self.jump_vel = self.JUMP_VEL
        self.rect = pygame.Rect(self.X_POS, self.Y_POS, img.get_width(), img.get_height())
        self.step_index = 0

    def update(self):
        if self.dino_run:
            self.run()
        if self.dino_jump:
            self.jump()
        if self.step_index >= 10:
            self.step_index = 0

    def jump(self):
        self.image = JUMPING
        if self.dino_jump:
            self.rect.y -= self.jump_vel * 4
            self.jump_vel -= 0.8
        if self.jump_vel <= -self.JUMP_VEL:
            self.dino_jump = False
            self.dino_run = True
            self.jump_vel = self.JUMP_VEL

    def run(self):
        self.image = RUNNING[self.step_index // 5]
        self.rect.x = self.X_POS
        self.rect.y = self.Y_POS
        self.step_index += 1

    def draw(self, SCREEN):
        SCREEN.blit(self.image, (self.rect.x, self.rect.y))

# class Obstacle:
#     def __init__(self, image, number_of_cacti):
#         self.image = image
#         self.type = number_of_cacti
#         self.rect = self.image[self.type].get_rect()
#         self.rect.x = SCREEN_WIDTH # start at the right edge of window

#     def update(self):
#         self.rect.x -= game_speed
#         if self.rect.x < -self.rect.width: # delete obstacle after going off screen
#             obstacles.pop()

#     def draw(self, SCREEN):
#         SCREEN.blit(self.image[self.type], self.rect)

def main():
    clock = pygame.time.Clock()
    dinosaurs = [Dinosaur()]
    run = True
    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        SCREEN.fill((255, 255, 255))

        for dino in dinosaurs:
            dino.update()
            dino.draw(SCREEN)

        user_input = pygame.key.get_pressed()

        for i, dino in enumerate(dinosaurs):
            if user_input[pygame.K_SPACE] and not dino.dino_jump:
                dino.dino_jump = True
                dino.dino_run = False

        clock.tick(60)
        pygame.display.update()

main()
# tinyurl.com/27j32jp3
# pastebin.com/Kf7mZGzh

