# FROM MEMORY, write a class for a dino.
# Load any image from the Dino folder and scale it to 2X size.
# Write a main function that runs a pygame loop.
# In the loop, blit ONE instance of the dino on the screen.
# Write code to make the dino move using arrow keys OR WASD.
# Limit the dino's movement to the pygame window - dino should be fully visible at all times.
# Write one word in the middle of the screen.
# Make the background white.

# Time: 20 mins

import pygame
import sys

pygame.init()

SCREEN = pygame.display.set_mode((600, 600))
FONT = pygame.font.Font(None, 20)

class Dino:
    def __init__(self, coord):
        self.img = pygame.image.load("Assets/Dino/DinoRun1.png")
        self.img = pygame.transform.scale(self.img, (self.img.get_width()*2, self.img.get_height()*2))
        self.rect = pygame.rect.Rect(*coord, self.img.get_width(), self.img.get_height())

    def display(self):
        SCREEN.blit(self.img, (self.rect.x, self.rect.y))

def main():
    dino1 = Dino((50, 50))
    clock = pygame.time.Clock()
    speed = 6
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        
        user_input = pygame.key.get_pressed()
        if user_input[pygame.K_w]:
            dino1.rect.y -= speed
        if user_input[pygame.K_s]:
            dino1.rect.y += speed
        if user_input[pygame.K_a]:
            dino1.rect.x -= speed
        if user_input[pygame.K_d]:
            dino1.rect.x += speed

        if dino1.rect.y < 0:
            dino1.rect.y = 0
        elif dino1.rect.y > 600 - dino1.rect.w:
            dino1.rect.y = 600 - dino1.rect.w
        if dino1.rect.x < 0:
            dino1.rect.x = 0
        elif dino1.rect.x > 600 - dino1.rect.h:
            dino1.rect.x = 600 - dino1.rect.h

        SCREEN.fill((255, 255, 255))

        dino1.display()
        
        text = FONT.render("Hello", True, (255, 0, 0))
        SCREEN.blit(text, ((600-text.get_width())//2, 300))

        pygame.display.update()
        clock.tick(60)

main()