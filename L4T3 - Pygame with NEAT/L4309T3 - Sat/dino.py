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
BG = pygame.image.load("Assets/Other/Track.png")

class Dinosaur:
    X_POS = 80
    Y_POS = 300
    JUMP_VEL = 8

    def __init__(self, img=RUNNING[0]):
        self.image = img
        self.rect = pygame.Rect(self.X_POS, self.Y_POS, img.get_width(), img.get_height())

    def draw(self, SCREEN):
        SCREEN.blit(self.image, (self.rect.x, self.rect.y))

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
            dino.draw(SCREEN)

        clock.tick(60)
        pygame.display.update()

main()
# tinyurl.com/27j32jp3