# FROM MEMORY, write a class for a dino
# load any image from the Dino folder
# write a main function that runs a pygame loop
# in the loop, blit two instances of the dino on the screen at different positions
# make the background white

# Time: 50 mins
import pygame
import sys

pygame.init()

SCREEN = pygame.display.set_mode((600, 600))

class Dino:
    def __init__(self, coord):
        self.img = pygame.image.load("Assets/Dino/DinoRun1.png")
        self.rect = pygame.rect.Rect(*coord, self.img.get_width(), self.img.get_height())

    def display(self):
        SCREEN.blit(self.img, (self.rect.x, self.rect.y))

def main():
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        dino1 = Dino((50, 50))
        dino2 = Dino((400, 400))

        SCREEN.fill((255, 255, 255))

        dino1.display()
        dino2.display()

        pygame.display.update()

main()