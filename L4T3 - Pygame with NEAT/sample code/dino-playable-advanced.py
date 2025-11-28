import pygame
import os
import random
import sys

pygame.init()

# tinyurl.com/4yxvas9n

SCREEN_HEIGHT = 600
SCREEN_WIDTH = 1100
SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

RUNNING = [pygame.image.load("Assets/Dino/DinoRun1.png"),
           pygame.image.load("Assets/Dino/DinoRun2.png")]

JUMPING = pygame.image.load("Assets/Dino/DinoJump.png")
DUCKING = [pygame.image.load("Assets/Dino/DinoDuck1.png"),
           pygame.image.load("Assets/Dino/DinoDuck2.png")]
DUCKING = [pygame.transform.scale_by(img, 0.2) for img in DUCKING]
BG = pygame.image.load("Assets/Other/Track.png")

FONT = pygame.font.Font("freesansbold.ttf", 20)

SMALL_CACTUS = [pygame.image.load("Assets/Cactus/SmallCactus1.png"),
           pygame.image.load("Assets/Cactus/SmallCactus2.png"),
           pygame.image.load("Assets/Cactus/SmallCactus3.png")]

LARGE_CACTUS = [pygame.image.load("Assets/Cactus/LargeCactus1.png"),
           pygame.image.load("Assets/Cactus/LargeCactus2.png"),
           pygame.image.load("Assets/Cactus/LargeCactus3.png")]

SMALL_CACTUS = [pygame.transform.scale(pygame.image.load(f"Assets/Cactus/SmallCactus{i}.png"), (50, 60)) for i in range(1,4)]
LARGE_CACTUS = [pygame.transform.scale(pygame.image.load(f"Assets/Cactus/LargeCactus{i}.png"), (50, 70)) for i in range(1,4)]
PTERODACTYL = [pygame.transform.scale_by(pygame.image.load("Assets/Dino/Pterodactyl.png"), 0.5)]

class Dinosaur:
    X_POS = 80
    Y_POS = 310
    JUMP_VEL = 10
    Y_POS_DUCK = 340

    def __init__(self, img = RUNNING[0]):
        self.image = img
        self.dino_run = True
        self.dino_jump = False
        self.dino_duck = False
        self.jump_vel = self.JUMP_VEL
        self.rect = pygame.Rect(self.X_POS, self.Y_POS, img.get_width(), img.get_height())
        self.step_index = 0

    def update(self):
        if self.dino_run:
            self.run()
        elif self.dino_jump:
            self.jump()
        elif self.dino_duck:
            self.duck()
        if self.step_index >= 10:
            self.step_index = 0

    def jump(self):
        self.image = JUMPING
        self.rect.w = self.image.get_width()
        self.rect.h = self.image.get_height()
        if self.dino_jump:
            self.rect.y -= self.jump_vel * 4
            self.jump_vel -= 0.8
        if self.jump_vel <= -self.JUMP_VEL:
            self.dino_jump = False
            self.jump_vel = self.JUMP_VEL
            self.rect.y = self.Y_POS

    def run(self):
        self.image = RUNNING[self.step_index // 5]
        self.rect.w = self.image.get_width()
        self.rect.h = self.image.get_height()
        self.rect.y = self.Y_POS
        self.step_index += 1

    def duck(self):
        self.image = DUCKING[self.step_index // 5]
        self.rect.w = self.image.get_width()
        self.rect.h = self.image.get_height()
        self.rect.y = self.Y_POS_DUCK
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

class Pterodactyl(Obstacle):
    def __init__(self, image):
        super().__init__(image, 0)
        self.rect.y = random.choice([200, 250])  # Flying height

def remove(index):
    dinosaurs.pop(index)

def main():
    global game_speed, dinosaurs, obstacles, points, x_pos_bg, y_pos_bg
    clock = pygame.time.Clock()
    points = 0

    dinosaurs = [Dinosaur()]
    obstacles = []

    x_pos_bg = 0
    y_pos_bg = 380
    game_speed = 10

    def score():
        global points, game_speed
        points += 1
        if points % 100 == 0:
            game_speed += 1
        text = FONT.render(f"Points: {points}", True, (0, 255, 255))
        SCREEN.blit(text, (950, 50))

    def background():
        global x_pos_bg, y_pos_bg
        image_width = BG.get_width()
        SCREEN.blit(BG, (x_pos_bg, y_pos_bg))
        SCREEN.blit(BG, (x_pos_bg + image_width, y_pos_bg))
        if x_pos_bg <= -image_width:
            x_pos_bg = 0
        x_pos_bg -= game_speed
        # after adding background, ask them to save a copy

    run = True
    game_over = False
    win = False
    while run:
        while game_over:
            SCREEN.fill((255, 255, 255))
            if not win:
                text = FONT.render(f"Game over! You scored {points} points. Press 'R' to play again or 'Q' to quit.", True, (255, 0, 0))
            else:
                text = FONT.render(f"You win! You scored {points} points. Press 'R' to play again or 'Q' to quit.", True, (0, 255, 0))
            SCREEN.blit(text, (SCREEN_WIDTH // 2 - text.get_width() // 2, SCREEN_HEIGHT // 2))
            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        pygame.quit()
                        sys.exit()
                    if event.key == pygame.K_r:
                        game_over = False
                        points = 0
                        dinosaurs = [Dinosaur()]
                        obstacles = []
                        game_speed = 10
                        win = False
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        SCREEN.fill((255, 255, 255))

        for dinosaur in dinosaurs:
            dinosaur.update()
            dinosaur.draw(SCREEN)

        if len(dinosaurs) == 0:
            game_over = True
        elif points >= 1000:
            win = True
            game_over = True

        if len(obstacles) == 0:
            choice = random.randint(0, 2)
            if choice == 0:
                obstacles.append(SmallCactus(SMALL_CACTUS, random.randint(0, 2)))
            elif choice == 1:
                obstacles.append(LargeCactus(LARGE_CACTUS, random.randint(0, 2)))
            else:
                obstacles.append(Pterodactyl(PTERODACTYL))

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
                dinosaur.dino_duck = False
            elif user_input[pygame.K_DOWN] and not dinosaur.dino_jump:
                dinosaur.dino_duck = True
                dinosaur.dino_run = False
            elif not dinosaur.dino_jump:
                dinosaur.dino_run = True
                dinosaur.dino_duck = False


        score()
        background()
        clock.tick(60)
        pygame.display.update()

main()
