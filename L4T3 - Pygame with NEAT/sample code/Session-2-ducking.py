import pygame
import os 
import random
import sys

pygame.init()

SCREEN_HEIGHT = 600
SCREEN_WIDTH = 1100
SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

# Load assets
RUNNING = [pygame.image.load('Assets/Dino/DinoRun1.png'),
           pygame.image.load('Assets/Dino/DinoRun2.png')]
JUMPING = pygame.image.load('Assets/Dino/DinoJump.png')
DUCKING = [pygame.image.load('Assets/Dino/DinoDuck1.png'),
           pygame.image.load('Assets/Dino/DinoDuck2.png')]
DUCKING = [pygame.transform.scale_by(img, 0.2) for img in DUCKING]

BG = pygame.image.load('Assets/Other/Track.png')
FONT = pygame.font.Font('freesansbold.ttf', 20)

SMALL_CACTUS = [pygame.image.load('Assets/Cactus/SmallCactus1.png'),
                pygame.image.load('Assets/Cactus/SmallCactus2.png'),
                pygame.image.load('Assets/Cactus/SmallCactus3.png')]
LARGE_CACTUS = [pygame.image.load('Assets/Cactus/LargeCactus1.png'),
                pygame.image.load('Assets/Cactus/LargeCactus2.png'),
                pygame.image.load('Assets/Cactus/LargeCactus3.png')]
PTERODACTYL = [pygame.transform.scale_by(pygame.image.load('Assets/Dino/Pterodactyl.png'), 0.5)]

class Dinosaur:
    JUMP_VEL = 15
    X_POS = 80
    Y_POS = 310
    Y_POS_DUCK = 340

    def __init__(self, img=RUNNING[0]):
        self.image = img
        self.dino_run = True
        self.dino_jump = False
        self.dino_duck = False
        self.jump_vel = self.JUMP_VEL
        self.rect = pygame.Rect(self.X_POS, self.Y_POS, img.get_width(), img.get_height())
        self.step_index = 0

    def draw(self, SCREEN):
        SCREEN.blit(self.image, (self.rect.x, self.rect.y))

    def update(self, user_input):
        if self.dino_jump:
            self.jump()
        elif self.dino_duck:
            self.duck()
        else:
            self.run()

        if self.step_index >= 10:
            self.step_index = 0

        if user_input[pygame.K_SPACE] and not self.dino_jump:
            self.dino_jump = True
            self.dino_run = False
            self.dino_duck = False
        elif user_input[pygame.K_DOWN] and not self.dino_jump:
            self.dino_duck = True
            self.dino_run = False
        elif not self.dino_jump:
            self.dino_run = True
            self.dino_duck = False

    def jump(self):
        self.image = JUMPING
        self.rect.y -= self.jump_vel * 4
        self.jump_vel -= 0.8
        if self.jump_vel <= -self.JUMP_VEL:
            self.dino_jump = False
            self.jump_vel = self.JUMP_VEL
            self.rect.y = self.Y_POS

    def run(self):
        self.image = RUNNING[self.step_index // 5]
        self.rect = pygame.Rect(self.X_POS, self.Y_POS, self.image.get_width(), self.image.get_height())
        self.step_index += 1

    def duck(self):
        self.image = DUCKING[self.step_index // 5]
        self.rect = pygame.Rect(self.X_POS, self.Y_POS_DUCK, self.image.get_width(), self.image.get_height())
        self.step_index += 1

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
        self.rect.y = 325

class LargeCactus(Obstacle):
    def __init__(self, image, number_of_cacti):
        super().__init__(image, number_of_cacti)
        self.rect.y = 300

class Pterodactyl(Obstacle):
    def __init__(self, image):
        self.image = image
        self.type = 0
        self.rect = self.image[self.type].get_rect()
        self.rect.x = SCREEN_WIDTH
        self.rect.y = random.choice([200, 250])  # Flying height

    def update(self):
        self.rect.x -= game_speed
        if self.rect.x < -self.rect.width:
            obstacles.pop()

    def draw(self, SCREEN):
        SCREEN.blit(self.image[self.type], self.rect)

def main():
    global x_pos_bg, y_pos_bg, game_speed, dinosaurs, obstacles, points
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
        text = FONT.render(f"Points: {points}", True, (0, 255, 191))
        SCREEN.blit(text, (950, 50))

    def remove(index):
        dinosaurs.pop(index)

    def background():
        global x_pos_bg, y_pos_bg, game_speed
        image_width = BG.get_width()
        SCREEN.blit(BG, (x_pos_bg, y_pos_bg))
        SCREEN.blit(BG, (image_width + x_pos_bg, y_pos_bg))
        if x_pos_bg <= -image_width:
            x_pos_bg = 0
        x_pos_bg -= game_speed

    run = True
    game_over = False
    while run:
        while game_over:
            SCREEN.fill((255, 255, 255))
            text = FONT.render(f"Game over! You scored {points} points. Press 'R' to play again or 'Q' to quit.", True, (0, 0, 0))
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
                        x_pos_bg = 0
                        y_pos_bg = 380
                        game_speed = 20

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        SCREEN.fill((255, 255, 255))
        user_input = pygame.key.get_pressed()

        for dinosaur in dinosaurs:
            dinosaur.update(user_input)
            dinosaur.draw(SCREEN)

        if len(dinosaurs) == 0:
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

        score()
        background()
        clock.tick(60)
        pygame.display.update()

main()
print(f"You scored {points} points!")
