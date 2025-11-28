import pygame
import os 
import random
import sys
import neat

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
    JUMP_VEL = 10
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

    def update(self):
        if self.dino_jump:
            self.jump()
        elif self.dino_duck:
            self.duck()
        else:
            self.run()

        if self.step_index >= 10:
            self.step_index = 0

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

def eval_genomes(genomes, config):
    global x_pos_bg, y_pos_bg, game_speed, dinosaurs, obstacles, points, ge, nets
    clock = pygame.time.Clock()
    points = 0
    dinosaurs = []
    obstacles = []
    ge = []
    nets = []

    x_pos_bg = 0
    y_pos_bg = 380
    game_speed = 10

    for genome_id, genome in genomes:
        dinosaurs.append(Dinosaur())
        ge.append(genome)
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        genome.fitness = 0

    def score():
        global points, game_speed
        points += 1
        if points % 100 == 0:
            game_speed += 1
        text = FONT.render(f"Points: {points}", True, (0, 255, 191))
        SCREEN.blit(text, (950, 50))

    def remove(index):
        dinosaurs.pop(index)
        ge.pop(index)
        nets.pop(index)

    def distance(pos_a, pos_b):
        dx = pos_a[0] - pos_b[0]
        dy = pos_a[1] - pos_b[1]
        return (dx ** 2 + dy ** 2) ** 0.5

    def background():
        global x_pos_bg, y_pos_bg, game_speed
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

        for dinosaur in dinosaurs:
            dinosaur.update()
            dinosaur.draw(SCREEN)

        if len(dinosaurs) == 0:
            break

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
                    ge[i].fitness -= 1
                    remove(i)

        for i, dinosaur in enumerate(dinosaurs):
            # output = nets[i].activate((dinosaur.rect.y,
            #                            distance(
            #                                (dinosaur.rect.x, dinosaur.rect.y),
            #                                obstacle.rect.midtop
            #                            ),
            #                            obstacle.rect.centery,
            #                            game_speed
            #                            ))
            # # clear state
            # dinosaur.dino_duck = False
            # dinosaur.dino_run = True

            # if output[0] > 0.5 and dinosaur.rect.y == dinosaur.Y_POS:
            #     dinosaur.dino_jump = True
            #     dinosaur.dino_run = False
            # elif output[1] > 0.5 and not dinosaur.dino_jump:
            #     dinosaur.dino_duck = True
            #     dinosaur.dino_run = False

            output = nets[i].activate((dinosaur.rect.centerx, dinosaur.rect.centery, obstacle.rect.centerx, obstacle.rect.centery, abs(dinosaur.rect.centerx - obstacle.rect.centerx), abs(dinosaur.rect.centery - obstacle.rect.centery)))

            if output[0] > 0.5: #the AI first decides whether to make a move
                if output[1] > 0.5 and not dinosaur.dino_duck: #the AI then decides whether to jump 
                    dinosaur.dino_jump = True
                    dinosaur.dino_run = False
                elif output[2] > 0.5 and not dinosaur.dino_jump:
                    dinosaur.dino_duck = True
            else:
                dinosaur.dino_jump = False
                dinosaur.dino_duck = False

        score()
        background()
        clock.tick(60)
        pygame.display.update()


def run(config_path):
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
    config_path = os.path.join(local_dir, "config-dino-duck.txt")
    run(config_path)
