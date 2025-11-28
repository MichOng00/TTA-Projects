import pygame
import math
import sys
import neat.config
import neat.population
import time

pygame.init()

SCREEN_WIDTH = 900
SCREEN_HEIGHT = 735
SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

TRACK = pygame.image.load("Assets/not_car/track.png")
TRACK = pygame.transform.scale(TRACK, (SCREEN_WIDTH, SCREEN_HEIGHT))
START_POS = (350, 600) # starting position of the car

GRASS_COLOR = (2, 105, 31)
TIME_LIMIT = 120
FONT = pygame.font.Font("freesansbold.ttf", 20)

class Car(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.original_image = pygame.image.load("Assets/not_car/car.png")
        self.image = self.original_image
        self.rect = self.image.get_rect(center = START_POS)
        self.drive_state = False
        self.vel_vector = pygame.math.Vector2(0.8, 0) # moving to the right at speed 0.8
        self.angle = 0 # where car is facing at the moment
        self.rotation_vel = 5 # how fast car rotates
        self.direction = 0 # whether you are turning left or right or going straight
        self.time_since_death = 0
        self.alive = True
        self.radars = []

    def update(self):
        self.radars.clear() # delete all previous radars
        self.drive()
        self.rotate()
        for radar_angle in (-60, -30, 0, 30, 60):
            self.radar(radar_angle)
        self.collision()
        self.check_collision()
        self.data()

    def collision(self):
        length = 30
        right_x = int(self.rect.center[0] + math.cos(math.radians(self.angle + 18)) * length)
        left_x = int(self.rect.center[0] + math.cos(math.radians(self.angle - 18)) * length)

        right_y = int(self.rect.center[1] - math.sin(math.radians(self.angle + 18)) * length)
        left_y = int(self.rect.center[1] - math.sin(math.radians(self.angle - 18)) * length)

        right = (right_x, right_y)
        left = (left_x, left_y)

        if 0 <= right_x < SCREEN_WIDTH and 0 <= right_y < SCREEN_HEIGHT:
            if SCREEN.get_at(right) == GRASS_COLOR:
                self.alive = False
                print("dead")

        if 0 <= left_x < SCREEN_WIDTH and 0 <= left_y < SCREEN_HEIGHT:
            if SCREEN.get_at(left) == GRASS_COLOR:
                self.alive = False
                print("dead")

        pygame.draw.circle(SCREEN, (0, 255, 255, 0), right, 4)
        pygame.draw.circle(SCREEN, (0, 255, 255, 0), left, 4)


    def drive(self):
        self.rect.center += self.vel_vector * 6

    def rotate(self):
        # check if turning left or right
        if self.direction == 1:
            self.angle -= self.rotation_vel
            self.vel_vector.rotate_ip(self.rotation_vel)
        elif self.direction == -1:
            self.angle += self.rotation_vel
            self.vel_vector.rotate_ip(-self.rotation_vel)
        # rotate image and update position
        self.image = pygame.transform.rotozoom(self.original_image, self.angle, 0.1)
        self.rect = self.image.get_rect(center = self.rect.center)

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

        pygame.draw.line(SCREEN, (255, 255, 255), self.rect.center, (x, y), 1)
        pygame.draw.circle(SCREEN, (0, 255, 0), (x, y), 3)

        dist = int(math.sqrt((self.rect.center[0] - x)**2 + (self.rect.center[1] - y)**2))

        self.radars.append([radar_angle, dist])

    def data(self):
        input = [0,0,0,0,0]
        for i, radar in enumerate(self.radars):
            input[i] = int(radar[1])
        return input

    def check_collision(self):
        x, y = int(self.rect.centerx), int(self.rect.centery)
        if 0 <= x < SCREEN_WIDTH and 0 <= y < SCREEN_HEIGHT:
            color_at_car = SCREEN.get_at((x,y)) # color under car centre
            if color_at_car == GRASS_COLOR: # you die
                print("You touched grass! Respawning...")
                self.respawn()

    def respawn(self):
        self.time_since_death = pygame.time.get_ticks()
        self.rect.center = START_POS
        self.angle = 0
        self.vel_vector = pygame.math.Vector2(0.8, 0)
        self.direction = 0
        self.drive_state = False

    def draw_info(self, start_time):
        font = pygame.font.Font(None, 36)

        elapsed_time = (pygame.time.get_ticks() - start_time) // 1000
        death_time = (pygame.time.get_ticks() - self.time_since_death) // 1000
        minutes = elapsed_time // 60
        seconds = elapsed_time % 60
        death_minutes = death_time // 60
        death_seconds = death_time % 60
        timer_text = f"Time: {minutes:02}:{seconds:02}"
        lap_text = f"Lap Time: {death_minutes:02}:{death_seconds:02}"

        # draw background box
        info_box = pygame.Rect(SCREEN_WIDTH - 220, 10, 210, 70)
        pygame.draw.rect(SCREEN, (0,0,0), info_box, border_radius=5)
        pygame.draw.rect(SCREEN, (255,255,255), info_box, 2, border_radius=5)

        # blit text
        timer_surface = font.render(timer_text, True, (255, 255, 255))
        lap_surface = font.render(lap_text, True, (255, 255, 255))
        SCREEN.blit(timer_surface, (SCREEN_WIDTH - 210, 20))
        SCREEN.blit(lap_surface, (SCREEN_WIDTH - 210, 40))

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

        text_1 = FONT.render(f"Cars: {len(cars)}", True, (0, 0, 0))
        text_2 = FONT.render(f"Generation: {pop.generation + 1}", True, (0, 0, 0))
        time_left = round(TIME_LIMIT - (time.time() - timer), 2)
        text_3 = FONT.render(f"Time left: {max(0, time_left)}", True, (0, 0, 0))

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
            break # all your cars are dead OR you ran out of time

        for i, car in enumerate(cars):
            ge[i].fitness += 1 # the longer the car survives, the higher the fitness
            if not car.sprite.alive:
                remove(i)

        for i, car in enumerate(cars):
            output = nets[i].activate(car.sprite.data()) # give data as inputs

            if output[0] > 0.7:
                car.sprite.direction = 1 # turn right
            if output[1] > 0.7:
                car.sprite.direction = -1 # turn left
            if output[0] <= 0.7 and output[1] <= 0.7:
                car.sprite.direction = 0 # straight

        for car in cars:
            car.update()
            # car.sprite.check_collision()
            car.draw(SCREEN) # comes from pygame.sprite.Sprite
            # car.sprite.draw_info(start_time)

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