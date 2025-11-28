# tinyurl.com/4yxvas9n

import pygame
import os
import random
import sys
import neat

pygame.init()

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

SMALL_CACTUS = [pygame.transform.scale(
    pygame.image.load(f"Assets/Cactus/SmallCactus{i}.png"), (50, 60)) for i in range(1,4)]

LARGE_CACTUS = [pygame.transform.scale(
    pygame.image.load(f"Assets/Cactus/LargeCactus{i}.png"), (50, 70)) for i in range(1,4)]

FLY = [pygame.transform.scale_by(pygame.image.load("Assets/Dino/Pterodactyl.png"), 0.5)]

class Dinosaur:
    X_POS = 80
    Y_POS = 300
    JUMP_VEL = 8
    Y_POS_DUCK = 340

    def __init__(self, img = RUNNING[0]): # initialize
        self.image = img
        self.dino_run = True
        self.dino_jump = False
        self.jump_vel = self.JUMP_VEL
        self.rect = pygame.Rect(self.X_POS, self.Y_POS, img.get_width(), img.get_height())
        self.step_index = 0
        self.color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))

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
        if self.jump_vel <= -self.JUMP_VEL: # dino has landed
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
        pygame.draw.rect(SCREEN, self.color, (self.rect.x, self.rect.y, self.rect.width, self.rect.height), 2)
        for obstacle in obstacles:
            pygame.draw.line(SCREEN, self.color, (self.rect.x + 54, self.rect.y + 12), obstacle.rect.center, 2)

class Obstacle:
    def __init__(self, image, number_of_cacti):
        self.image = image
        self.type = number_of_cacti
        self.rect = self.image[self.type].get_rect()
        self.rect.x = SCREEN_WIDTH # start cactus position on right side of window

    def update(self):
        self.rect.x -= game_speed
        if self.rect.x < -self.rect.width:
            obstacles.pop() # remove the obstacle from the list

    def draw(self, SCREEN):
        SCREEN.blit(self.image[self.type], self.rect)

class SmallCactus(Obstacle): # subclass of Obstacle class
    def __init__(self, image, number_of_cacti):
        super().__init__(image, number_of_cacti) # copy init method of superclass
        self.rect.y = 330

class LargeCactus(Obstacle): # subclass of Obstacle class
    def __init__(self, image, number_of_cacti):
        super().__init__(image, number_of_cacti) # copy init method of superclass
        self.rect.y = 320

def remove(index):
    dinosaurs.pop(index)
    ge.pop(index)
    nets.pop(index)

def distance(pos_a, pos_b): # pos_a = (x,y)
    dx = pos_a[0] - pos_b[0]
    dy = pos_a[1] - pos_b[1]
    return (dx ** 2 + dy ** 2) ** 0.5

def eval_genomes(genomes, config):
    # global shares variables with other functions
    global dinosaurs, obstacles, game_speed, points, x_pos_bg, y_pos_bg, ge, nets
    x_pos_bg = 0
    y_pos_bg = 380
    points = 0
    clock = pygame.time.Clock()
    dinosaurs = []
    obstacles = []
    ge = [] # genomes
    nets = [] # neural networks
    game_speed = 10

    for genome_id, genome in genomes:
        dinosaurs.append(Dinosaur())
        ge.append(genome)
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        genome.fitness = 0 # points

    def score():
        global points, game_speed
        points += 1
        if points % 100 == 0:
            game_speed += 1
        text = FONT.render(f"Points: {points}", True, (0, 255, 191))
        SCREEN.blit(text, (950, 50))

    def background():
        global x_pos_bg, y_pos_bg
        image_width = BG.get_width()
        SCREEN.blit(BG, (x_pos_bg, y_pos_bg))
        SCREEN.blit(BG, (x_pos_bg + image_width, y_pos_bg)) # image copy next to first image
        if x_pos_bg <= -image_width:
            x_pos_bg = 0
        x_pos_bg -= game_speed

    def statistics():
        global dinosaurs, game_speed, ge
        text_1 = FONT.render(f"Dinosaurs: {len(dinosaurs)}", True, (0, 0, 0))
        text_2 = FONT.render(f"Game speed: {game_speed}", True, (0, 0, 0))
        text_3 = FONT.render(f"Generation: {pop.generation + 1}", True, (0, 0, 0))

        SCREEN.blit(text_1, (50, 50))
        SCREEN.blit(text_2, (50, 80))
        SCREEN.blit(text_3, (50, 110))

    run = True
    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        SCREEN.fill((255, 255, 255))

        for i, dinosaur in enumerate(dinosaurs):
            ge[i].fitness = points
            dinosaur.update()
            dinosaur.draw(SCREEN)

        if len(dinosaurs) == 0: # all the dinos have died :(
            break

        if len(obstacles) == 0: # no more obstacles
            if random.randint(0, 1) == 0: # randomly pick between small and large cactus
                obstacles.append(SmallCactus(SMALL_CACTUS, random.randint(0,2)))
            else:
                obstacles.append(LargeCactus(LARGE_CACTUS, random.randint(0,2)))

        for obs in obstacles:
            obs.draw(SCREEN)
            obs.update()
            for i, dino in enumerate(dinosaurs):
                if dino.rect.colliderect(obs.rect):
                    remove(i)

        # user_input = pygame.key.get_pressed()
        for i, dinosaur in enumerate(dinosaurs):
            output = nets[i].activate((dinosaur.rect.y,
                                       distance((dinosaur.rect.x, dinosaur.rect.y), obs.rect.midtop),
                                       game_speed))
            if output[0] > 0.5 and not dinosaur.dino_jump:
                dinosaur.dino_jump = True
                dinosaur.dino_run = False

        statistics()
        score()
        background()
        clock.tick(60)
        pygame.display.update()

def run(config_path):
    global pop, stats
    config = neat.config.Config(
        neat.DefaultGenome, 
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )

    pop = neat.Population(config)

    pop.run(eval_genomes, 50)

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-dino.txt")
    run(config_path)