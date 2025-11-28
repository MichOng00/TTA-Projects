'''Create the dino class and make it jump'''

import pygame
import os
import random
import sys

pygame.init()

# Constants
SCREEN_HEIGHT = 600
SCREEN_WIDTH = 1100
SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

RUNNING = [pygame.image.load('Assets/Dino/DinoRun1.png'),
           pygame.image.load('Assets/Dino/DinoRun2.png')]

JUMPING = pygame.image.load('Assets/Dino/DinoJump.png')

BG = pygame.image.load('Assets/Other/Track.png')

FONT = pygame.font.Font('freesansbold.ttf', 20)

# SMALL_CACTUS = [pygame.image.load('Assets/Cactus/SmallCactus1.png'),
#                 pygame.image.load('Assets/Cactus/SmallCactus2.png'),
#                 pygame.image.load('Assets/Cactus/SmallCactus3.png')]

# LARGE_CACTUS = [pygame.image.load('Assets/Cactus/LargeCactus1.png'),
#                 pygame.image.load('Assets/Cactus/LargeCactus2.png'),
#                 pygame.image.load('Assets/Cactus/LargeCactus3.png')]

# Scale the cacti down slightly
SMALL_CACTUS = [pygame.transform.scale(pygame.image.load(f'Assets/Cactus/SmallCactus{i}.png'), (50, 60)) for i in range(1, 4)]
LARGE_CACTUS = [pygame.transform.scale(pygame.image.load(f'Assets/Cactus/LargeCactus{i}.png'), (50, 70)) for i in range(1, 4)]

# Dino class
class Dinosaur:
    X_POS = 80
    Y_POS = 300
    JUMP_VEL = 8

    def __init__(self, img=RUNNING[0]):
        self.image = img
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

class Obstacle:
    def __init__(self, image, number_of_cacti):
        self.image = image
        self.type = number_of_cacti
        self.rect = self.image[self.type].get_rect()
        self.rect.x = SCREEN_WIDTH

    def update(self):
        self.rect.x -= game_speed
        if self.rect.x < -self.rect.width:
            obstacles.pop()

    def draw(self, SCREEN):
        SCREEN.blit(self.image[self.type], self.rect)

class SmallCactus(Obstacle):
    def __init__(self, image, number_of_cacti):
        super().__init__(image, number_of_cacti)
        self.rect.y = 330

class LargeCactus(Obstacle):
    def __init__(self, image, number_of_cacti):
        super().__init__(image, number_of_cacti)
        self.rect.y = 320

def remove(index):
    dinosaurs.pop(index)

def main():
    global game_speed, x_pos_bg, y_pos_bg, points, dinosaurs, obstacles
    clock = pygame.time.Clock()
    points = 0

    obstacles = []
    dinosaurs = [Dinosaur()]

    x_pos_bg = 0
    y_pos_bg = 380
    game_speed = 20

    def score():
        global points, game_speed
        points += 1
        if points % 100 == 0:
            game_speed += 1
        text = FONT.render("Points: " + str(points), True, (0, 255, 191))
        SCREEN.blit(text, (950, 50))

    def background():
        global x_pos_bg, y_pos_bg
        image_width = BG.get_width()
        SCREEN.blit(BG, (x_pos_bg, y_pos_bg))
        SCREEN.blit(BG, (image_width + x_pos_bg, y_pos_bg))
        if x_pos_bg <= -image_width:
            x_pos_bg = 0
        x_pos_bg -= game_speed

    run = True
    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        SCREEN.fill((255, 255, 255))

        for i, dinosaur in enumerate(dinosaurs):
            dinosaur.update()
            dinosaur.draw(SCREEN)

        if len(dinosaurs) == 0:
            break

        if len(obstacles) == 0:
            if random.randint(0, 1) == 0:
                obstacles.append(SmallCactus(SMALL_CACTUS, random.randint(0, 2)))
            else:
                obstacles.append(LargeCactus(LARGE_CACTUS, random.randint(0, 2)))

        for obstacle in obstacles:
            obstacle.draw(SCREEN)
            obstacle.update()
            for i, dinosaur in enumerate(dinosaurs):
                if dinosaur.rect.colliderect(obstacle.rect):
                    remove(i)

        user_input = pygame.key.get_pressed()

        for i, dinosaur in enumerate(dinosaurs):
            if user_input[pygame.K_SPACE] and not dinosaur.dino_jump:
                dinosaur.dino_jump = True
                dinosaur.dino_run = False
        
        # FPS
        score()
        background()
        clock.tick(60)
        pygame.display.update()

main()
print(f"You scored {points} points!")