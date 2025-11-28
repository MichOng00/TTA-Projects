# Use your dino code from session 8.
# Add a class for a cactus.
# Blit one cactus in the middle of the screen.
# Draw the outline for both the dino and cactus in green. Use pygame.draw.rect.
# If the dino and cactus overlap, the outline for the dino should turn red.

# Time: 30 mins

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

class Cactus:
    def __init__(self, coord):
        self.img = pygame.image.load("Assets/Cactus/SmallCactus1.png")
        # self.img = pygame.transform.scale(self.img, (self.img.get_width()*2, self.img.get_height()*2))
        self.rect = pygame.rect.Rect(*coord, self.img.get_width(), self.img.get_height())

    def display(self):
        SCREEN.blit(self.img, (self.rect.x, self.rect.y))    

def main():
    dino1 = Dino((50, 50))
    cactus = Cactus((40, 40))
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
        cactus.display()

        pygame.draw.rect(SCREEN, (0, 255, 0), cactus.rect, 2)

        # collision
        if dino1.rect.colliderect(cactus.rect):
            pygame.draw.rect(SCREEN, (255, 0, 0), dino1.rect, 2)
        else:
            pygame.draw.rect(SCREEN, (0, 255, 0), dino1.rect, 2)

        pygame.display.update()

main()
