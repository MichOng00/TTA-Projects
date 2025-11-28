# pastebin.com/gXLbD6GJ
import neat.config
import pygame
import os
import random
import sys
import neat
import pickle
import matplotlib.pyplot as plt

pygame.init()

# tinyurl.com/4yxvas9n

SCREEN_HEIGHT = 600
SCREEN_WIDTH = 1100
SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

RUNNING = [pygame.image.load("Assets/Dino/DinoRun1.png"),
           pygame.image.load("Assets/Dino/DinoRun2.png")]

JUMPING = pygame.image.load("Assets/Dino/DinoJump.png")

# DUCKING = [pygame.image.load("Assets/Dino/DinoDuck1.png"),
#            pygame.image.load("Assets/Dino/DinoDuck2.png")]
# DUCKING = [pygame.transform.scale_by(img, 0.2) for img in DUCKING]

BG = pygame.image.load("Assets/Other/Track.png")

FONT = pygame.font.Font("freesansbold.ttf", 20)

SMALL_CACTUS = [pygame.transform.scale(pygame.image.load(f"Assets/Cactus/SmallCactus{i}.png"), (50, 60)) for i in range(1,4)]
LARGE_CACTUS = [pygame.transform.scale(pygame.image.load(f"Assets/Cactus/LargeCactus{i}.png"), (50, 70)) for i in range(1,4)]
# PTERODACTYL = [pygame.transform.scale_by(pygame.image.load("Assets/Dino/Pterodactyl.png"), 0.5)]

class Dinosaur:
    X_POS = 80
    Y_POS = 310
    JUMP_VEL = 10
    Y_POS_DUCK = 340

    def __init__(self, img = RUNNING[0]):
        self.image = img
        self.dino_run = True
        self.dino_jump = False
        # self.dino_duck = False
        self.jump_vel = self.JUMP_VEL
        self.rect = pygame.Rect(self.X_POS, self.Y_POS, img.get_width(), img.get_height())
        self.step_index = 0
        self.color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    def update(self):
        if self.dino_run:
            self.run()
        elif self.dino_jump:
            self.jump()
        # elif self.dino_duck:
        #     self.duck()
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
            self.dino_run = True
            self.jump_vel = self.JUMP_VEL

    def run(self):
        self.image = RUNNING[self.step_index // 5]
        self.rect.w = self.image.get_width()
        self.rect.h = self.image.get_height()
        self.rect.x = self.X_POS
        self.rect.y = self.Y_POS
        self.step_index += 1

    # def duck(self):
    #     self.image = DUCKING[self.step_index // 5]
    #     self.rect.w = self.image.get_width()
    #     self.rect.h = self.image.get_height()
    #     self.rect.y = self.Y_POS_DUCK
    #     self.step_index += 1

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

# class Pterodactyl(Obstacle):
#     def __init__(self, image, number_of_cacti):
#         super().__init__(image, number_of_cacti)
#         self.type = 0
#         self.rect = self.image[self.type].get_rect()
#         self.rect.x = SCREEN_WIDTH
#         self.rect.y = random.choice([200, 250]) # Flying height

def remove(index):
    dinosaurs.pop(index)
    ge.pop(index)
    nets.pop(index)

def distance(pos_a, pos_b):
    dx = pos_a[0] - pos_b[0]
    dy = pos_a[1] - pos_b[1]
    return (dx ** 2 + dy ** 2) ** 0.5

def eval_genomes(genomes, config, show_winner=False):
    global game_speed, dinosaurs, obstacles, points, x_pos_bg, y_pos_bg, ge, nets, generation
    clock = pygame.time.Clock()
    points = 0
    generation = pop.generation + 1

    dinosaurs = []
    obstacles = []
    ge = []
    nets = []

    x_pos_bg = 0
    y_pos_bg = 380
    game_speed = 20

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

    def statistics():
        global dinosaurs, game_speed, ge
        text_1 = FONT.render(f"Dinosaurs: {len(dinosaurs)}", True, (0,0,0))
        text_2 = FONT.render(f"Game speed: {game_speed}", True, (0,0,0))
        text_3 = FONT.render(f"Generation: {pop.generation + 1}", True, (0,0,0))

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

        for i, dinosaur in enumerate(dinosaurs): # dinosaurs[i] = dinosaur
            ge[i].fitness = points
            dinosaur.update()
            dinosaur.draw(SCREEN)

        if len(dinosaurs) == 0:
            break

        if len(obstacles) == 0:
            choice = random.randint(0, 1)
            if choice == 0:
                obstacles.append(SmallCactus(SMALL_CACTUS, random.randint(0, 2)))
            elif choice == 1:
                obstacles.append(LargeCactus(LARGE_CACTUS, random.randint(0, 2)))

        for obstacle in obstacles:
            obstacle.draw(SCREEN)
            obstacle.update()
            for i, dinosaur in enumerate(dinosaurs):
                if dinosaur.rect.colliderect(obstacle.rect):
                    remove(i)

        # user_input = pygame.key.get_pressed()

        for i, dinosaur in enumerate(dinosaurs):
            output = nets[i].activate((dinosaur.rect.y,
                                       distance((dinosaur.rect.x, dinosaur.rect.y), obstacle.rect.midtop),
                                       game_speed))
            if output[0] > 0.5 and dinosaur.rect.y == dinosaur.Y_POS: #TODO: change
                dinosaur.dino_jump = True
                dinosaur.dino_run = False

        # for i, dinosaur in enumerate(dinosaurs):
        #     if user_input[pygame.K_SPACE] and not dinosaur.dino_jump:
        #         dinosaur.dino_jump = True
        #         dinosaur.dino_run = False
        #         dinosaur.dino_duck = False
        #     elif user_input[pygame.K_DOWN] and not dinosaur.dino_jump:
        #         dinosaur.dino_duck = True
        #         dinosaur.dino_run = False
        #     elif not dinosaur.dino_jump:
        #         dinosaur.dino_run = True
        #         dinosaur.dino_duck = False

        if show_winner:
            winner_text = FONT.render("Winner genome run", True, (255, 0, 0))
            fitness_text = FONT.render(f"Achieved fitness {winner.fitness} in {generation} generations", True, (0, 0, 0))
        
            SCREEN.blit(winner_text, (SCREEN_WIDTH // 2 - winner_text.get_width() // 2, 20))
            SCREEN.blit(fitness_text, (SCREEN_WIDTH // 2 - fitness_text.get_width() // 2, 50))
        else:
            statistics()
        score()
        background()
        clock.tick(60)
        pygame.display.update()

def run(config_path):
    global pop, stats, logger
    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )

    pop = neat.Population(config)

    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)

    logger = FitnessLogger()
    pop.add_reporter(logger)

    pop.run(eval_genomes, 50)

def update_config(config_path, pop_size):
    #file = open(config_path, "r")
    # ...
    # file.close()
    with open(config_path, "r") as file: # r - reading
        lines = file.readlines()

    with open(config_path, "w") as file: # w - writing
        for line in lines:
            if line.startswith("pop_size"):
                file.write(f"pop_size = {pop_size}\n")
            else:
                file.write(line)

def replay_genome(config_path, genome_path = "winner-dino.pkl"):
    with open(genome_path, "rb") as file: # rb - reading binary
        genome = pickle.load(file)
    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )

    genomes = [(1, genome)]
    eval_genomes(genomes, config, show_winner = True)

# replay_genome("path")
# replay_genome("path", "winner.pkl")

class FitnessLogger(neat.reporting.BaseReporter):
    def __init__(self):
        self.fitnesses = []
    
    def post_evaluate(self, best_genome):
        self.fitnesses.append(best_genome.fitness)

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-dino.txt")
    pop_size = input("What population size? >")
    update_config(config_path, pop_size)
    run(config_path)

    winner = stats.best_genome()
    print(f"\nBest genome:\nfitness {winner.fitness}\n{winner}")
    with open("winner-dino.pkl", "wb") as file: # wb - writing binary
        pickle.dump(winner, file)

    # graph of top fitness per generation
    plt.plot(logger.fitnesses)
    plt.title("Top fitness per generation")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.grid(True)
    plt.show()

    replay_genome(config_path)

# limewire.com/d/g6Hr5#3VNAHGvSi2