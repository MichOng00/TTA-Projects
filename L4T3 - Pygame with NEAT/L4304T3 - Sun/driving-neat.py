# pastebin.com/PSHjDv8J
# pastebin.com/0g1BtrXu
import pygame
import math
import sys
import neat.config
import neat.population
import time

import pickle
import visualize


pygame.init()

TIME_LIMIT = 120
FONT = pygame.font.Font("freesansbold.ttf", 20)

SCREEN_WIDTH = 900
SCREEN_HEIGHT = 735
SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

TRACK = pygame.image.load("Assets/not_car/track.png")
TRACK = pygame.transform.scale(TRACK, (SCREEN_WIDTH, SCREEN_HEIGHT))
START_POS = (350, 600)

GRASS_COLOR = pygame.Color(2, 105, 31, 255)



class Car(pygame.sprite.Sprite): # subclass of pygame Sprite class
    def __init__(self): # method
        super().__init__()
        # attributes
        self.original_image = pygame.image.load("Assets/not_car/car.png")
        self.image = self.original_image
        self.rect = self.image.get_rect(center = START_POS)
        # self.drive_state = 0
        self.drive_state = True
        self.vel_vector = pygame.math.Vector2(0.8, 0) # start driving to the right 
        self.angle = 0
        self.rotation_vel = 5 # turning speed
        self.direction = 0
        self.time_since_death = 0
        self.alive = True
        self.radars = []
        self.driving_speed = 6

    def update(self):
        self.radars.clear()
        self.drive()
        self.rotate()
        for radar_angle in (-60, -30, 0, 30, 60):
            self.radar(radar_angle)
        self.check_collision()
        self.collision()
        self.data()
        

    def drive(self):
        if self.drive_state:
            self.rect.center += self.vel_vector * self.driving_speed
        # if self.drive_state == 1:
        #     self.rect.center += self.vel_vector * 6
        # if self.drive_state == -1:
        #     self.rect.center -= self.vel_vector * 6

    def collision(self):
        length = 30
        right_x = int(self.rect.center[0] + math.cos(math.radians(self.angle + 18)) * length)
        right_y = int(self.rect.center[1] - math.sin(math.radians(self.angle + 18)) * length)

        left_x = int(self.rect.center[0] + math.cos(math.radians(self.angle - 18)) * length)
        left_y = int(self.rect.center[1] - math.sin(math.radians(self.angle - 18)) * length)

        right = (right_x, right_y)
        left = (left_x, left_y)

        if 0 <= right_x < SCREEN_WIDTH and 0 <= right_y < SCREEN_HEIGHT:
            if SCREEN.get_at(right) == GRASS_COLOR:
                self.alive = False

        if 0 <= left_x < SCREEN_WIDTH and 0 <= left_y < SCREEN_HEIGHT:
            if SCREEN.get_at(left) == GRASS_COLOR:
                self.alive = False

        pygame.draw.circle(SCREEN, (0, 255, 255), right, 4)
        pygame.draw.circle(SCREEN, (0, 255, 255), left, 4)
    
    def rotate(self):
        if self.direction == 1: # turn right
            self.angle -= self.rotation_vel
            self.vel_vector.rotate_ip(self.rotation_vel)
        if self.direction == -1: # turn left
            self.angle += self.rotation_vel
            self.vel_vector.rotate_ip(-self.rotation_vel)
        self.image = pygame.transform.rotozoom(self.original_image, self.angle, 0.1)
        self.rect = self.image.get_rect(center = self.rect.center)

    def radar(self, radar_angle):
        length = 0
        x, y = int(self.rect.centerx), int(self.rect.centery)
        while (
            0 <= x < SCREEN_WIDTH and
            0 <= y < SCREEN_HEIGHT and
            not SCREEN.get_at((x, y)) == GRASS_COLOR and
            length < 200
        ):
            length += 1
            x = int(self.rect.center[0] + math.cos(math.radians(self.angle + radar_angle)) * length)
            y = int(self.rect.center[1] - math.sin(math.radians(self.angle + radar_angle)) * length)

        pygame.draw.line(SCREEN, (255, 255, 255), self.rect.center, (x, y), 1)
        pygame.draw.circle(SCREEN, GRASS_COLOR, (x, y), 3)

        dist = int(math.sqrt((self.rect.center[0]-x)**2 + (self.rect.center[1] - y)**2))

        self.radars.append([radar_angle, dist])

    def data(self):
        input = [0, 0, 0, 0, 0]
        for i, radar in enumerate(self.radars):
            input[i] = int(radar[1])
        return input

    def check_collision(self):
        x, y = int(self.rect.centerx), int(self.rect.centery)
        if 0 <= x < SCREEN_WIDTH and 0 <= y < SCREEN_HEIGHT:
            color_at_car = SCREEN.get_at((x, y))
            if color_at_car == GRASS_COLOR:
                self.alive = False

    def respawn(self):
        self.time_since_death = pygame.time.get_ticks()
        self.rect.center = START_POS
        self.drive_state = False
        self.vel_vector = pygame.math.Vector2(0.8, 0) 
        self.angle = 0
        self.rotation_vel = 5
        self.direction = 0

def remove(i):
    cars.pop(i)
    ge.pop(i)
    nets.pop(i)

def eval_genomes(genomes, config):
    global cars, ge, nets, time_left
    cars = []
    ge = []
    nets = []
    
    for genome_id, genome in genomes:
        cars.append(pygame.sprite.GroupSingle(Car()))
        ge.append(genome)
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        genome.fitness = 0 # the number the AI uses to tell how well it's doing

    timer = time.time()
    last_speedup = 1

    def statistics():
        global ge, time_left

        text_1 = FONT.render(f"Cars: {len(cars)}", True, (0, 0, 0))
        text_2 = FONT.render(f"Generation: {pop.generation + 1}", True, (0, 0, 0))
        time_left = round(TIME_LIMIT - (time.time() - timer), 2)
        text_3 = FONT.render(f"Time left: {time_left}", True, (0, 0, 0))

        SCREEN.blit(text_1, (50, 50))
        SCREEN.blit(text_2, (50, 80))
        SCREEN.blit(text_3, (50, 110))

    run = True
    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        SCREEN.blit(TRACK, (0,0))

        if len(cars) == 0 or time.time() - timer > TIME_LIMIT:
            break

        for i, car in enumerate(cars):
            ge[i].fitness += 1
            if not car.sprite.alive:
                remove(i)

        for i, car in enumerate(cars):
            output = nets[i].activate(car.sprite.data()) # inputs: radars
            if output[0] > 0.7:
                car.sprite.direction = 1
            if output[1] > 0.7:
                car.sprite.direction = -1
            if output[0] <= 0.7 and output[1] <= 0.7:
                car.sprite.direction = 0

            car.update()
            # car.sprite.check_collision()
            car.draw(SCREEN)

        statistics()

        pygame.display.update()


# pastebin.com/pV6VWrMf
# Setup NEAT
def run(config_path):
    global pop
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
 
    winner = pop.run(eval_genomes, 50)

    # with open("best.pickle", "wb") as f:
    #     pickle.dump(winner, f)

    # with open("best.pickle", "rb") as f:
    #     winner = pickle.load(f)

    # node_names = {-5: "left", -4: "left diag", -3: "forward", -2: "right diag", -1: "right", 0: "turn right", 1:"turn left"}

    # visualize.draw_net(config, winner, True, node_names=node_names, prune_unused=True)
 
if __name__ == "__main__":
    config_path = "./config-car.txt"
    run(config_path)