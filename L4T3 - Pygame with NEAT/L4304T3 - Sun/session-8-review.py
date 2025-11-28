# FROM MEMORY, write a class for a dino.
# Load any image from the Dino folder and scale it to 2X size.
# Write a main function that runs a pygame loop.
# In the loop, blit ONE instance of the dino on the screen.
# Write code to make the dino move using arrow keys OR WASD.
# Limit the dino's movement to the pygame window - dino should be fully visible at all times.
# Write one word in the middle of the screen.
# Make the background white.

# Time: 40 mins

import pygame
import sys

pygame.init()

SCREEN = pygame.display.set_mode((600, 600))

class Dino:
    def __init__(self, coord):
        self.img = pygame.image.load("Assets/Dino/DinoRun1.png")
        self.img = pygame.transform.scale(self.img, (self.img.get_width()*2, self.img.get_height()*2))
        self.rect = pygame.rect.Rect(*coord, self.img.get_width(), self.img.get_height())

    def display(self):
        SCREEN.blit(self.img, (self.rect.x, self.rect.y))

def main():
    dino1 = Dino((50, 50))
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        keys = pygame.key.get_pressed()
        if keys[pygame.K_w] and dino1.rect.y >= 0:
            dino1.rect.y -= 1
        if keys[pygame.K_s] and dino1.rect.y + dino1.img.get_height() < 600:
            dino1.rect.y += 1
        if keys[pygame.K_a] and dino1.rect.x >= 0:
            dino1.rect.x -= 1
        if keys[pygame.K_d] and dino1.rect.x + dino1.img.get_width() < 600:
            dino1.rect.x += 1

        SCREEN.fill((255, 255, 255))

        dino1.display()

        pygame.display.update()

main()

# pastebin.com/MzwNpswJ