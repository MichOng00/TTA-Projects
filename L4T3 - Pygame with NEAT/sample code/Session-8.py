'''Setup and display car'''
import neat.config
import neat.population
import pygame
import math
import sys
import neat
import time

pygame.init()

# Constants
SCREEN_WIDTH = 900
SCREEN_HEIGHT = 735
SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
TIME_LIMIT = 500

TRACK = pygame.image.load('Assets/not_car/track.png')
TRACK = pygame.transform.scale(TRACK, (SCREEN_WIDTH, SCREEN_HEIGHT))
START_POS = (350, 600)

FONT = pygame.font.Font('freesansbold.ttf', 20)
GRASS_COLOR = pygame.Color(2, 105, 31, 255)

class Car(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.original_image = pygame.image.load('Assets/not_car/car.png')
        self.image = self.original_image
        self.rect = self.image.get_rect(center=START_POS)
        self.vel_vector = pygame.math.Vector2(0.8, 0)
        self.angle = 0
        self.rotation_vel = 5 # rotation speed
        self.direction = 0
        self.alive = True
        self.radars = []
        self.driving_speed = 6

    def update(self):
        self.radars.clear()
        self.drive()
        self.rotate()
        for radar_angle in (-60, -30, 0, 30, 60):
            self.radar(radar_angle)
        self.collision()
        self.data()

    def drive(self):
        self.rect.center += self.vel_vector * self.driving_speed # driving speed

    def collision(self):
        length = 30
        coll_right_x = int(self.rect.center[0] + math.cos(math.radians(self.angle + 18)) * length)
        coll_right_y = int(self.rect.center[1] - math.sin(math.radians(self.angle + 18)) * length)
        coll_left_x = int(self.rect.center[0] + math.cos(math.radians(self.angle - 18)) * length)
        coll_left_y = int(self.rect.center[1] - math.sin(math.radians(self.angle - 18)) * length)

        collision_point_right = (coll_right_x, coll_right_y)
        collision_point_left = (coll_left_x, coll_left_y)
        
        if 0 <= coll_right_x < SCREEN_WIDTH and 0 <= coll_right_y < SCREEN_HEIGHT:
            if SCREEN.get_at((coll_right_x, coll_right_y)) == GRASS_COLOR:
                self.alive = False

        if 0 <= coll_left_x < SCREEN_WIDTH and 0 <= coll_left_y < SCREEN_HEIGHT:
            if SCREEN.get_at((coll_left_x, coll_left_y)) == GRASS_COLOR:
                self.alive = False
            

        # Draw collision points
        pygame.draw.circle(SCREEN, (0, 255, 255, 0), collision_point_right, 4)
        pygame.draw.circle(SCREEN, (0, 255, 255, 0), collision_point_left, 4)

    def rotate(self):
        if self.direction == 1:
            self.angle -= self.rotation_vel
            self.vel_vector.rotate_ip(self.rotation_vel)
        if self.direction == -1:
            self.angle += self.rotation_vel
            self.vel_vector.rotate_ip(-self.rotation_vel)
        self.image = pygame.transform.rotozoom(self.original_image, self.angle, 0.2)
        self.rect = self.image.get_rect(center=self.rect.center)

    def radar(self, radar_angle):
        length = 0
        x = int(self.rect.center[0])
        y = int(self.rect.center[1])

        while (
            0 <= x < SCREEN_WIDTH and
            0 <= y < SCREEN_HEIGHT and
            not SCREEN.get_at((x, y)) == GRASS_COLOR and
            length < 200
        ):
            length += 1
            x = int(self.rect.center[0] + math.cos(math.radians(self.angle + radar_angle)) * length)
            y = int(self.rect.center[1] - math.sin(math.radians(self.angle + radar_angle)) * length)

        # Visualise radar
        pygame.draw.line(SCREEN, (255, 255, 255, 255), self.rect.center, (x, y), 1)
        pygame.draw.circle(SCREEN, (0, 255, 0, 0), (x, y), 3)

        dist = int(math.sqrt(math.pow(self.rect.center[0] - x, 2) + math.pow(self.rect.center[1] - y, 2)))

        self.radars.append([radar_angle, dist])

    def data(self):
        input = [0, 0, 0, 0, 0]
        for i, radar in enumerate(self.radars):
            input[i] = int(radar[1])
        return input

def remove(index):
    cars.pop(index)
    ge.pop(index)
    nets.pop(index)

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
        genome.fitness = 0

    timer = time.time()
    last_speedup = 1

    def statistics():
        global ge, time_left

        text_1 = FONT.render(f'Cars: {len(cars)}', True, (0, 0, 0))
        text_2 = FONT.render(f'Generation: {pop.generation + 1}', True, (0, 0, 0))
        time_left = round(TIME_LIMIT - (time.time() - timer), 2)
        text_3 = FONT.render(f'Time Left: {time_left}', True, (0, 0, 0))

        SCREEN.blit(text_1, (50, 50))
        SCREEN.blit(text_2, (50, 80))
        SCREEN.blit(text_3, (50, 110))

    run = True
    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        SCREEN.blit(TRACK, (0, 0))

        if len(cars) == 0 or time.time() - timer > TIME_LIMIT:
            break

        for i, car in enumerate(cars):
            ge[i].fitness += 1
            if not car.sprite.alive:
                remove(i)

        for i, car in enumerate(cars):
            output = nets[i].activate(car.sprite.data())
            if output[0] > 0.7:
                car.sprite.direction = 1
            if output[1] > 0.7:
                car.sprite.direction = -1
            if output[0] <= 0.7 and output[1] <= 0.7:
                car.sprite.direction = 0
        # Update
        for car in cars:
            car.update()
            car.draw(SCREEN)
        caption = (f"Time left: {round(TIME_LIMIT - (time.time() - timer), 2)}s")
        pygame.display.set_caption(caption)
        statistics()

        elapsed = int(time_left)
        if elapsed % 15 == 0 and elapsed != last_speedup:
            for car in cars:
                car.sprite.driving_speed += 1
                car.sprite.rotation_vel += 1
            last_speedup = elapsed
        pygame.display.update()

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

    pop.run(eval_genomes, 50)

if __name__ == "__main__":
    config_path = "./config-car.txt"
    run(config_path)