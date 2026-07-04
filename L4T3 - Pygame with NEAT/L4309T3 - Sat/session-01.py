# Classes
class Cat:
    def __init__(self, name, age):
        self.name = name # attribute (property)
        self.age = age

    def meow(self):
        print(f"{self.name} says meow")

    # create a method to say happy birthday (increase the age)
    def happy_birthday(self):
        self.age += 1
        print(f"{self.name} is now {self.age} years old")

my_cat = Cat("ginger", 5) # creating a Cat instance
print(type(my_cat))
print(my_cat.name) # accessing an attribute
print(my_cat.age) # accessing an attribute
my_cat.meow()     # using a method

# exercise: create your_cat and make it meow
your_cat = Cat("loaf", 2)
your_cat.meow()
your_cat.happy_birthday()
your_cat.happy_birthday()

# creating a subclass
class Persian(Cat): # subclass of Cat
    def __init__(self, name, age):
        super().__init__(name, age)
        self.toy = "laser pointer"

fancy_cat = Persian("fluffy", 1)
print(fancy_cat.toy)
# print(my_cat.toy) # AttributeError

#############################################################
# Images at:
# tinyurl.com/27j32jp3
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
